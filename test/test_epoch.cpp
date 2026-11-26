#include <lockfree/epoch.hpp>

#include "stress_harness.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>

using lockfree::Epoch;
using lockfree::EpochGuard;

namespace {

struct Obj {
    static inline std::atomic<int> freed{0};
    explicit Obj(int v) : value(v) {}
    ~Obj() { freed.fetch_add(1, std::memory_order_relaxed); }
    int value;
};

int freed() { return Obj::freed.load(std::memory_order_relaxed); }

// Three collections move the epoch two steps past anything retired before
// them, which empties every bag (nobody else is pinned in these tests).
void drain() {
    for (int i = 0; i < 3; ++i) Epoch::collect();
}

class EpochTest : public ::testing::Test {
protected:
    void SetUp() override {
        drain();
        ASSERT_EQ(Epoch::retired_count(), 0u);
        Obj::freed = 0;
    }
};

} // namespace

TEST_F(EpochTest, RetiredObjectWaitsForTwoAdvances) {
    Epoch::retire(new Obj{1});
    EXPECT_EQ(Epoch::retired_count(), 1u);
    EXPECT_EQ(freed(), 0);

    Epoch::collect(); // epoch + 1: a reader pinned at the tag may still be around
    EXPECT_EQ(freed(), 0);
    Epoch::collect(); // epoch + 2: everyone from then has unpinned
    EXPECT_EQ(freed(), 1);
    EXPECT_EQ(Epoch::retired_count(), 0u);
}

TEST_F(EpochTest, AdvanceNeedsEveryPinnedThreadInTheCurrentEpoch) {
    EpochGuard guard; // pinned at the current epoch, e
    EXPECT_TRUE(Epoch::try_advance()) << "we're pinned at e, so e -> e + 1 is fine";
    const std::uint64_t e1 = Epoch::current();
    EXPECT_FALSE(Epoch::try_advance()) << "we're still at e, so e + 1 -> e + 2 has to wait";
    EXPECT_EQ(Epoch::current(), e1);
}

TEST_F(EpochTest, OwnPinKeepsItsRetireAlive) {
    {
        EpochGuard guard;
        Epoch::retire(new Obj{1});
        for (int i = 0; i < 5; ++i) Epoch::collect();
        EXPECT_EQ(freed(), 0) << "freed while the retiring thread itself was still pinned";
    }
    drain();
    EXPECT_EQ(freed(), 1);
}

TEST_F(EpochTest, GuardsNest) {
    EpochGuard outer;
    {
        EpochGuard inner;
    }
    // The inner guard ending must not unpin: the outer one still reads.
    EXPECT_TRUE(Epoch::try_advance());
    EXPECT_FALSE(Epoch::try_advance());
}

TEST_F(EpochTest, UnpinnedThreadsDontHoldTheEpoch) {
    // A thread that pinned once and then unpinned (or exited) must not count.
    std::thread([] { EpochGuard g; }).join();
    const std::uint64_t e = Epoch::current();
    EXPECT_TRUE(Epoch::try_advance());
    EXPECT_TRUE(Epoch::try_advance());
    EXPECT_EQ(Epoch::current(), e + 2);
}

TEST_F(EpochTest, RetireCollectsOnItsOwn) {
    // No explicit collect: periodic collection inside retire must keep the
    // backlog bounded when nobody is pinned.
    for (int i = 0; i < 10'000; ++i) Epoch::retire(new Obj{i});
    EXPECT_GT(freed(), 9'000);
    EXPECT_LT(Epoch::retired_count(), 3 * lockfree::detail::epoch::kCollectEvery);
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

struct Owner {
    Obj* child;
    ~Owner() { Epoch::retire(child); }
};
} // namespace

TEST_F(EpochTest, RetireUsesTheGivenDeleter) {
    custom_deletes = 0;
    Epoch::retire(new Obj{1}, CountingDelete{});
    Epoch::retire(new int{5}, &free_int);
    drain();
    EXPECT_EQ(custom_deletes.load(), 2);
    EXPECT_EQ(freed(), 1);
}

TEST_F(EpochTest, ADeleterMayRetireMore) {
    Epoch::retire(new Owner{new Obj{9}});
    drain();
    EXPECT_EQ(freed(), 0) << "the child is only retired when its owner is freed";
    EXPECT_EQ(Epoch::retired_count(), 1u);
    drain();
    EXPECT_EQ(freed(), 1);
}

TEST_F(EpochTest, ThreadExitHandsItsBagsOn) {
    std::thread([] { Epoch::retire(new Obj{4}); }).join();
    EXPECT_EQ(freed(), 0);
    drain(); // adopts the orphaned bag once it has expired
    EXPECT_EQ(freed(), 1);
}

// ---- Threaded --------------------------------------------------------------------

namespace {

// Same canary type as the hazard pointer stress tests: cleared on
// destruction, so a reader holding a freed object sees it even without ASan.
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

class EpochStress : public lockfree::test::StressTest {
protected:
    void SetUp() override {
        drain();
        Canary::freed = 0;
    }
};

} // namespace

// The weakness EBR is known for, shown on purpose: one thread that stays
// pinned stops the epoch for everybody, and nothing retired after it pinned
// can be freed, however much piles up.
TEST_F(EpochStress, StalledPinnedThreadBlocksReclamation) {
    std::atomic<int> phase{0};
    std::thread stalled([&] {
        EpochGuard guard;
        phase.store(1, std::memory_order_release);
        while (phase.load(std::memory_order_acquire) != 2) std::this_thread::yield();
    });
    while (phase.load(std::memory_order_acquire) != 1) std::this_thread::yield();

    const std::uint64_t pinned_at = Epoch::current();
    constexpr std::uint64_t kRetired = 10'000;
    for (std::uint64_t i = 0; i < kRetired; ++i) Epoch::retire(new Canary{i});
    for (int i = 0; i < 10; ++i) Epoch::collect();

    const std::uint64_t freed_while_stalled = Canary::freed.load();
    const std::size_t backlog = Epoch::retired_count();
    const std::uint64_t epoch_while_stalled = Epoch::current();

    phase.store(2, std::memory_order_release);
    stalled.join();
    drain();

    EXPECT_EQ(freed_while_stalled, 0u);
    EXPECT_EQ(backlog, kRetired) << "every retired object is stuck in limbo";
    EXPECT_LE(epoch_while_stalled, pinned_at + 1) << "the epoch can move once, never twice";
    EXPECT_EQ(Canary::freed.load(), kRetired) << "and it all goes as soon as the thread unpins";
}

TEST_F(EpochStress, PinnedReaderOutlivesAnotherThreadsCollects) {
    auto* x = new Canary{42};
    lockfree::detail::atomic<Canary*> slot{x};
    std::atomic<int> phase{0};
    unsigned seen_canary = 0;
    std::uint64_t seen_value = 0;

    std::thread reader([&] {
        EpochGuard guard;
        Canary* p = slot.load(std::memory_order_acquire);
        phase.store(1, std::memory_order_release);
        while (phase.load(std::memory_order_acquire) != 2) std::this_thread::yield();
        seen_canary = p->canary;
        seen_value = p->value;
    });

    while (phase.load(std::memory_order_acquire) != 1) std::this_thread::yield();
    slot.store(nullptr);
    Epoch::retire(x);
    for (int i = 0; i < 10; ++i) Epoch::collect();
    const std::uint64_t freed_while_pinned = Canary::freed.load();
    phase.store(2, std::memory_order_release);
    reader.join();

    EXPECT_EQ(freed_while_pinned, 0u);
    EXPECT_EQ(seen_canary, Canary::kAlive);
    EXPECT_EQ(seen_value, 42u);
    drain();
    EXPECT_EQ(Canary::freed.load(), 1u);
}

// Writers swap fresh objects into one slot and retire the old ones; readers
// pin, load and read through the pointer. A reclamation bug is a read of a
// freed object: ASan reports it, the canary shows it.
TEST_F(EpochStress, ReadersNeverSeeAFreedObject) {
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
                Canary* old = slot.exchange(new Canary{i}, std::memory_order_acq_rel);
                Epoch::retire(old);
                lockfree::test::maybe_delay(rng);
            }
            writers_left.fetch_sub(1, std::memory_order_release);
        } else {
            std::uint64_t n = 0;
            std::uint64_t local_bad = 0;
            while (writers_left.load(std::memory_order_acquire) != 0) {
                EpochGuard guard;
                Canary* p = slot.load(std::memory_order_acquire);
                if (p->canary != Canary::kAlive || p->value > swaps) ++local_bad;
                lockfree::test::maybe_delay(rng);
                if (p->canary != Canary::kAlive) ++local_bad;
                ++n;
            }
            bad.fetch_add(local_bad, std::memory_order_relaxed);
            reads.fetch_add(n, std::memory_order_relaxed);
        }
    });

    EXPECT_EQ(bad.load(), 0u);
    EXPECT_GT(reads.load(), 0u);
    Epoch::retire(slot.load());
    drain();
    EXPECT_EQ(Canary::freed.load(), kWriters * swaps + 1);
    EXPECT_EQ(Epoch::retired_count(), 0u);
}

// A two-link chain read under one pin, the shape of a queue's dequeue
// (head, then head->next). Unlike hazard pointers there's nothing per
// pointer to get wrong: one guard covers every load inside it.
TEST_F(EpochStress, OneGuardCoversAChain) {
    struct Link {
        lockfree::detail::atomic<Canary*> second{nullptr};
        Canary first{0};
    };
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
                Canary* inner = old->second.exchange(nullptr, std::memory_order_acq_rel);
                Epoch::retire(inner);
                Epoch::retire(old);
                lockfree::test::maybe_delay(rng);
            }
            writers_left.fetch_sub(1, std::memory_order_release);
        } else {
            std::uint64_t local_bad = 0;
            while (writers_left.load(std::memory_order_acquire) != 0) {
                EpochGuard guard;
                Link* l = head.load(std::memory_order_acquire);
                Canary* c = l->second.load(std::memory_order_acquire);
                if (l->first.canary != Canary::kAlive) ++local_bad;
                lockfree::test::maybe_delay(rng);
                if (c != nullptr && (c->canary != Canary::kAlive || c->value != l->first.value))
                    ++local_bad;
            }
            bad.fetch_add(local_bad, std::memory_order_relaxed);
        }
    });

    EXPECT_EQ(bad.load(), 0u);
    Link* last = head.load();
    Epoch::retire(last->second.load());
    Epoch::retire(last);
    drain();
    EXPECT_EQ(Canary::freed.load(), 2 * (2 * swaps + 1));
}
