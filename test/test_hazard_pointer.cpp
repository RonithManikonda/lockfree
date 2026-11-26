#include <lockfree/hazard_pointer.hpp>

#include <gtest/gtest.h>

#include "stress_harness.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <thread>
#include <vector>

using lockfree::HazardPointer;
using lockfree::detail::ScopedTestHook;

namespace {

// Counts its destructions, so a test can tell exactly when a scan freed it.
struct Obj {
    static inline std::atomic<int> freed{0};
    explicit Obj(int v) : value(v) {}
    ~Obj() { freed.fetch_add(1, std::memory_order_relaxed); }
    int value;
};

class HazardPointerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Earlier tests on this thread may have left entries behind.
        HazardPointer::scan();
        ASSERT_EQ(HazardPointer::retired_count(), 0u);
        Obj::freed = 0;
    }
};

int freed() { return Obj::freed.load(std::memory_order_relaxed); }

} // namespace

TEST_F(HazardPointerTest, ProtectReturnsTheCurrentValue) {
    Obj a{1};
    lockfree::detail::atomic<Obj*> src{&a};
    HazardPointer hp;
    EXPECT_EQ(hp.protect(src), &a);

    lockfree::detail::atomic<Obj*> null_src{nullptr};
    EXPECT_EQ(hp.protect(null_src), nullptr);
}

TEST_F(HazardPointerTest, UnprotectedRetireIsFreedByTheNextScan) {
    HazardPointer::retire(new Obj{1});
    EXPECT_EQ(HazardPointer::retired_count(), 1u);
    EXPECT_EQ(freed(), 0) << "retire must defer, not delete";
    HazardPointer::scan();
    EXPECT_EQ(freed(), 1);
    EXPECT_EQ(HazardPointer::retired_count(), 0u);
}

TEST_F(HazardPointerTest, ProtectedObjectSurvivesScansUntilReset) {
    auto* p = new Obj{7};
    lockfree::detail::atomic<Obj*> src{p};
    HazardPointer hp;
    Obj* got = hp.protect(src);
    ASSERT_EQ(got, p);

    // Unlink and retire, as a structure would.
    src.store(nullptr);
    HazardPointer::retire(p);
    HazardPointer::scan();
    HazardPointer::scan();
    EXPECT_EQ(freed(), 0);
    EXPECT_EQ(got->value, 7); // still valid memory (ASan would object otherwise)
    EXPECT_EQ(HazardPointer::retired_count(), 1u);

    hp.reset();
    HazardPointer::scan();
    EXPECT_EQ(freed(), 1);
    EXPECT_EQ(HazardPointer::retired_count(), 0u);
}

TEST_F(HazardPointerTest, ReprotectingMovesTheHazard) {
    auto* a = new Obj{1};
    auto* b = new Obj{2};
    lockfree::detail::atomic<Obj*> src{a};
    HazardPointer hp;
    hp.protect(src);
    src.store(b);
    hp.protect(src); // now guards b only

    HazardPointer::retire(a);
    HazardPointer::retire(b);
    HazardPointer::scan();
    EXPECT_EQ(freed(), 1) << "a should go, b should stay";
    EXPECT_EQ(HazardPointer::retired_count(), 1u);

    hp.reset();
    HazardPointer::scan();
    EXPECT_EQ(freed(), 2);
}

TEST_F(HazardPointerTest, DestroyingTheHazardPointerReleasesItsProtection) {
    auto* p = new Obj{3};
    lockfree::detail::atomic<Obj*> src{p};
    {
        HazardPointer hp;
        hp.protect(src);
        HazardPointer::retire(p);
        HazardPointer::scan();
        EXPECT_EQ(freed(), 0);
    }
    HazardPointer::scan();
    EXPECT_EQ(freed(), 1);
}

TEST_F(HazardPointerTest, ReleasedRecordsAreReused) {
    { HazardPointer warm_up; }
    const std::size_t before = HazardPointer::record_count();
    for (int i = 0; i < 1000; ++i) {
        HazardPointer a;
        HazardPointer b;
    }
    // One extra at most: the loop needs two at once, warm_up made only one.
    EXPECT_LE(HazardPointer::record_count(), before + 1);
}

TEST_F(HazardPointerTest, RetireScansOnItsOwnPastTheThreshold) {
    // No explicit scan: the retired list has to bound itself.
    const std::size_t threshold =
        lockfree::detail::hazard::kScanBase + 2 * HazardPointer::record_count();
    for (std::size_t i = 0; i < threshold; ++i) HazardPointer::retire(new Obj{0});
    EXPECT_EQ(freed(), static_cast<int>(threshold));
    EXPECT_EQ(HazardPointer::retired_count(), 0u);
}

namespace {
std::atomic<int> custom_deletes{0};
struct CountingDelete {
    void operator()(Obj* p) const {
        custom_deletes.fetch_add(1, std::memory_order_relaxed);
        delete p;
    }
};
void free_int(void* p) {
    custom_deletes.fetch_add(1, std::memory_order_relaxed);
    delete static_cast<int*>(p);
}
} // namespace

TEST_F(HazardPointerTest, RetireUsesTheGivenDeleter) {
    custom_deletes = 0;
    HazardPointer::retire(new Obj{1}, CountingDelete{});
    HazardPointer::retire(new int{5}, &free_int);
    HazardPointer::scan();
    EXPECT_EQ(custom_deletes.load(), 2);
    EXPECT_EQ(freed(), 1);
}

namespace {
// A node that owns another one and retires it when it goes, the way a
// structure's node might hold a second allocation.
struct Owner {
    Obj* child;
    ~Owner() { HazardPointer::retire(child); }
};
} // namespace

TEST_F(HazardPointerTest, ADeleterMayRetireMore) {
    HazardPointer::retire(new Owner{new Obj{9}});
    HazardPointer::scan();
    EXPECT_EQ(freed(), 0) << "the child is only retired during the scan";
    EXPECT_EQ(HazardPointer::retired_count(), 1u);
    HazardPointer::scan();
    EXPECT_EQ(freed(), 1);
}

TEST_F(HazardPointerTest, ThreadExitHandsLeftoversToTheNextScan) {
    auto* p = new Obj{4};
    lockfree::detail::atomic<Obj*> src{p};
    HazardPointer hp;
    hp.protect(src);

    // The other thread retires p and exits while we still protect it. Its
    // retired list dies with it, so p must be passed on rather than leaked
    // or freed.
    std::thread([p] { HazardPointer::retire(p); }).join();
    EXPECT_EQ(freed(), 0);

    hp.reset();
    HazardPointer::scan(); // adopts the orphan
    EXPECT_EQ(freed(), 1);
}

// ---- The protect / validate window ---------------------------------------------

TEST_F(HazardPointerTest, ProtectRetriesWhenTheSourceChangesBeforeValidation) {
    auto* a = new Obj{1};
    auto* b = new Obj{2};
    lockfree::detail::atomic<Obj*> src{a};
    HazardPointer hp;

    // Act as another thread in the gap between publishing the hazard and
    // re-reading src: replace a with b. Only the first pass through the gap
    // does it; the retry has to settle on b.
    int calls = 0;
    Obj* got = nullptr;
    {
        ScopedTestHook hook([&](std::string_view where) {
            if (where == "hazard_protect_before_validate" && calls++ == 0) src.store(b);
        });
        got = hp.protect(src);
    }
    EXPECT_EQ(calls, 2) << "publish, see the change, publish again, validate";
    EXPECT_EQ(got, b);

    // The hazard followed: a is unprotected now, b isn't.
    HazardPointer::retire(a);
    HazardPointer::retire(b);
    HazardPointer::scan();
    EXPECT_EQ(freed(), 1);
    hp.reset();
    HazardPointer::scan();
    EXPECT_EQ(freed(), 2);
}

// ---- Threaded --------------------------------------------------------------------

namespace {

// Clears its canary on destruction, so a reader that got hold of a freed
// object sees a wrong value even without ASan. ASan is the real check: it
// reports the read itself.
struct Canary {
    static constexpr unsigned kAlive = 0x5AFE;
    static inline std::atomic<std::uint64_t> freed{0};

    explicit Canary(std::uint64_t v) : value(v) {}
    ~Canary() {
        canary = 0;
        freed.fetch_add(1, std::memory_order_relaxed);
    }

    unsigned canary = kAlive;
    std::uint64_t value;
};

class HazardPointerStress : public lockfree::test::StressTest {
protected:
    void SetUp() override {
        HazardPointer::scan();
        Canary::freed = 0;
    }
};

} // namespace

TEST_F(HazardPointerStress, ProtectedObjectOutlivesAnotherThreadsScans) {
    auto* x = new Canary{42};
    lockfree::detail::atomic<Canary*> slot{x};
    std::atomic<int> phase{0};
    unsigned seen_canary = 0;
    std::uint64_t seen_value = 0;
    std::uint64_t freed_while_held = 0;

    std::thread reader([&] {
        HazardPointer hp;
        Canary* p = hp.protect(slot);
        phase.store(1, std::memory_order_release);
        while (phase.load(std::memory_order_acquire) != 2) std::this_thread::yield();
        seen_canary = p->canary;
        seen_value = p->value;
        hp.reset();
        phase.store(3, std::memory_order_release);
    });

    while (phase.load(std::memory_order_acquire) != 1) std::this_thread::yield();
    // Unlink x, then retire enough unrelated objects to force several scans.
    slot.store(nullptr);
    HazardPointer::retire(x);
    for (int i = 0; i < 2000; ++i) HazardPointer::retire(new Canary{0});
    HazardPointer::scan();
    freed_while_held = Canary::freed.load();
    phase.store(2, std::memory_order_release);
    while (phase.load(std::memory_order_acquire) != 3) std::this_thread::yield();
    reader.join();

    EXPECT_EQ(freed_while_held, 2000u) << "everything but x should have gone";
    EXPECT_EQ(seen_canary, Canary::kAlive);
    EXPECT_EQ(seen_value, 42u);
    HazardPointer::scan();
    EXPECT_EQ(Canary::freed.load(), 2001u);
}

// Writers keep swapping fresh objects into one slot and retiring what they
// swapped out; readers protect the slot and read through the pointer. Any
// reclamation bug is a read of a freed object, which ASan reports and the
// canary shows. Every object must be freed exactly once by the end.
TEST_F(HazardPointerStress, ReadersNeverSeeAFreedObject) {
    constexpr std::size_t kWriters = 2;
    constexpr std::size_t kReaders = 4;
    const std::uint64_t swaps = lockfree::test::scaled(20000);

    lockfree::detail::atomic<Canary*> slot{new Canary{0}};
    std::atomic<std::size_t> writers_left{kWriters};
    std::atomic<std::uint64_t> bad{0};
    std::atomic<std::uint64_t> reads{0};

    lockfree::test::run_stress(kWriters + kReaders, [&](std::size_t index,
                                                        lockfree::test::StressRng& rng) {
        if (index < kWriters) {
            for (std::uint64_t i = 1; i <= swaps; ++i) {
                // acq_rel: the new object's fields are published by the
                // release, and the old one's are ours before we retire it.
                Canary* old = slot.exchange(new Canary{i}, std::memory_order_acq_rel);
                HazardPointer::retire(old);
                lockfree::test::maybe_delay(rng);
            }
            writers_left.fetch_sub(1, std::memory_order_release);
        } else {
            HazardPointer hp;
            std::uint64_t n = 0;
            std::uint64_t local_bad = 0;
            while (writers_left.load(std::memory_order_acquire) != 0) {
                Canary* p = hp.protect(slot);
                if (p->canary != Canary::kAlive || p->value > swaps) ++local_bad;
                lockfree::test::maybe_delay(rng);
                // Read again after the delay: a scan may have run meanwhile.
                if (p->canary != Canary::kAlive) ++local_bad;
                hp.reset();
                ++n;
            }
            bad.fetch_add(local_bad, std::memory_order_relaxed);
            reads.fetch_add(n, std::memory_order_relaxed);
        }
    });

    EXPECT_EQ(bad.load(), 0u);
    EXPECT_GT(reads.load(), 0u);

    // The writers' leftovers were orphaned when they exited; nobody holds a
    // hazard now, so one scan here frees all of it.
    HazardPointer::retire(slot.load());
    HazardPointer::scan();
    EXPECT_EQ(Canary::freed.load(), kWriters * swaps + 1);
    EXPECT_EQ(HazardPointer::retired_count(), 0u);
}

// Each reader holds two hazards at once and walks a two-node chain, the shape
// the Michael-Scott queue's dequeue has (protect head, then head->next).
TEST_F(HazardPointerStress, TwoHazardsPerReaderOverAChain) {
    struct Link {
        lockfree::detail::atomic<Canary*> second{nullptr};
        Canary first{0};
    };
    // Links are retired as a unit; the Canary inside counts the frees.
    constexpr std::size_t kThreads = 6;
    const std::uint64_t swaps = lockfree::test::scaled(10000);

    auto make = [](std::uint64_t i) {
        auto* l = new Link;
        l->first.value = i;
        l->second.store(new Canary{i}, std::memory_order_relaxed);
        return l;
    };
    lockfree::detail::atomic<Link*> head{make(0)};
    std::atomic<std::size_t> writers_left{2};
    std::atomic<std::uint64_t> bad{0};

    lockfree::test::run_stress(kThreads, [&](std::size_t index, lockfree::test::StressRng& rng) {
        if (index < 2) {
            for (std::uint64_t i = 1; i <= swaps; ++i) {
                Link* old = head.exchange(make(i), std::memory_order_acq_rel);
                // Unlink the inner object first, then retire both. A reader
                // that protected `old` but not yet `second` re-validates and
                // sees nullptr.
                Canary* inner = old->second.exchange(nullptr, std::memory_order_acq_rel);
                HazardPointer::retire(inner);
                HazardPointer::retire(old);
                lockfree::test::maybe_delay(rng);
            }
            writers_left.fetch_sub(1, std::memory_order_release);
        } else {
            HazardPointer hp_link;
            HazardPointer hp_inner;
            std::uint64_t local_bad = 0;
            while (writers_left.load(std::memory_order_acquire) != 0) {
                Link* l = hp_link.protect(head);
                Canary* c = hp_inner.protect(l->second);
                if (l->first.canary != Canary::kAlive) ++local_bad;
                lockfree::test::maybe_delay(rng);
                if (c != nullptr && (c->canary != Canary::kAlive || c->value != l->first.value))
                    ++local_bad;
                hp_inner.reset();
                hp_link.reset();
            }
            bad.fetch_add(local_bad, std::memory_order_relaxed);
        }
    });

    EXPECT_EQ(bad.load(), 0u);
    Link* last = head.load();
    HazardPointer::retire(last->second.load());
    HazardPointer::retire(last);
    HazardPointer::scan();
    // Each link frees two canaries: its own `first` and the inner one.
    EXPECT_EQ(Canary::freed.load(), 2 * (2 * swaps + 1));
}
