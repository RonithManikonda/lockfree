#include <lockfree/epoch.hpp>

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
