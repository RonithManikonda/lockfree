#include <lockfree/hazard_pointer.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <thread>
#include <vector>

using lockfree::HazardPointer;

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
