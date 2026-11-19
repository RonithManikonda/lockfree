#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "lockfree/detail/atomic_ref.hpp"

// Built twice: test_atomic_shim (normal) and test_atomic_shim_sched (with
// LOCKFREE_SCHEDULER). The semantic tests run in both; the yield tests only
// exist in the scheduler build.

using lockfree::detail::atomic;

namespace {

#if !defined(LOCKFREE_SCHEDULER)

// The whole point of the normal build: no wrapper, nothing to pay for.
static_assert(std::is_same_v<atomic<int>, std::atomic<int>>);
static_assert(std::is_same_v<atomic<void*>, std::atomic<void*>>);

#endif

TEST(AtomicShim, BehavesLikeStdAtomic) {
    atomic<std::size_t> a{5};
    EXPECT_EQ(a.load(std::memory_order_acquire), 5u);
    a.store(7, std::memory_order_release);
    EXPECT_EQ(a.load(), 7u);
    EXPECT_EQ(a.exchange(9, std::memory_order_acq_rel), 7u);
    EXPECT_EQ(a.fetch_add(3, std::memory_order_relaxed), 9u);
    EXPECT_EQ(a.fetch_sub(2), 12u);
    EXPECT_EQ(a.load(), 10u);

    std::size_t expected = 4;
    EXPECT_FALSE(a.compare_exchange_strong(expected, 1));
    EXPECT_EQ(expected, 10u); // failure reports what was there
    EXPECT_TRUE(a.compare_exchange_strong(expected, 1, std::memory_order_acq_rel,
                                          std::memory_order_acquire));
    EXPECT_EQ(a.load(), 1u);

    // weak may fail spuriously, so retry like real code does.
    expected = 1;
    while (!a.compare_exchange_weak(expected, 2, std::memory_order_release,
                                    std::memory_order_relaxed)) {
        ASSERT_EQ(expected, 1u);
    }
    EXPECT_EQ(a.load(), 2u);
}

TEST(AtomicShim, HoldsPointersAndSmallStructs) {
    int x = 0;
    int y = 0;
    atomic<int*> p{&x};
    int* want = &x;
    EXPECT_TRUE(p.compare_exchange_strong(want, &y));
    EXPECT_EQ(p.load(), &y);

    struct Pair {
        void* ptr;
        std::uint64_t tag;
    };
    atomic<Pair> w{Pair{nullptr, 0}};
    Pair cur = w.load();
    EXPECT_TRUE(w.compare_exchange_strong(cur, Pair{&x, cur.tag + 1}));
    EXPECT_EQ(w.load().ptr, &x);
    EXPECT_EQ(w.load().tag, 1u);
}

TEST(AtomicShim, CountsUpFromManyThreads) {
    atomic<std::uint64_t> n{0};
    constexpr int threads = 4;
    constexpr int per_thread = 10'000;
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) {
        pool.emplace_back([&] {
            for (int i = 0; i < per_thread; ++i) n.fetch_add(1, std::memory_order_relaxed);
        });
    }
    for (auto& th : pool) th.join();
    EXPECT_EQ(n.load(), std::uint64_t{threads} * per_thread);
}

#if defined(LOCKFREE_SCHEDULER)

using lockfree::detail::AtomicOp;
using lockfree::detail::yield_hook;
using lockfree::detail::YieldPoint;

// What the hook saw. Single-threaded tests only, so plain globals are fine.
std::vector<YieldPoint> seen;
// The value the atomic held when its yield point ran, to prove the hook runs
// *before* the access (that's what lets a scheduler preempt in front of it).
const atomic<int>* watched = nullptr;
std::vector<int> value_at_yield;

void record(const YieldPoint& p) noexcept {
    seen.push_back(p);
    if (watched != nullptr && p.address == watched) {
        // Read the current value with the hook off, so it doesn't recurse.
        yield_hook() = nullptr;
        value_at_yield.push_back(watched->load());
        yield_hook() = &record;
    }
}

class AtomicShimYield : public ::testing::Test {
protected:
    void SetUp() override {
        seen.clear();
        value_at_yield.clear();
        watched = nullptr;
        yield_hook() = &record;
    }
    void TearDown() override { yield_hook() = nullptr; }
};

static_assert(!std::is_same_v<atomic<int>, std::atomic<int>>);
static_assert(sizeof(atomic<int>) == sizeof(std::atomic<int>));

TEST_F(AtomicShimYield, EveryOperationYieldsOnceWithItsKind) {
    atomic<int> a{0};
    int e = 0;
    a.load();
    a.store(1);
    a.exchange(2);
    e = 2;
    a.compare_exchange_strong(e, 3);
    e = 3;
    a.compare_exchange_strong(e, 4, std::memory_order_acq_rel, std::memory_order_acquire);
    e = 4;
    while (!a.compare_exchange_weak(e, 5)) {
    }
    a.fetch_add(1);
    a.fetch_sub(1);

    const std::vector<AtomicOp> want = {
        AtomicOp::load,      AtomicOp::store,     AtomicOp::exchange,
        AtomicOp::compare_exchange, AtomicOp::compare_exchange,
    };
    ASSERT_GE(seen.size(), want.size() + 3);
    for (std::size_t i = 0; i < want.size(); ++i) {
        EXPECT_EQ(seen[i].op, want[i]) << i;
        EXPECT_EQ(seen[i].address, &a) << i;
    }
    // The weak CAS may have retried; fetch_add and fetch_sub are the last two.
    EXPECT_EQ(seen[seen.size() - 2].op, AtomicOp::fetch_add);
    EXPECT_EQ(seen.back().op, AtomicOp::fetch_sub);
    for (std::size_t i = want.size(); i + 2 < seen.size(); ++i) {
        EXPECT_EQ(seen[i].op, AtomicOp::compare_exchange) << i;
    }
}

TEST_F(AtomicShimYield, YieldRunsBeforeTheAccess) {
    atomic<int> a{10};
    watched = &a;
    a.store(20);
    a.exchange(30);
    a.fetch_add(1);
    ASSERT_EQ(value_at_yield.size(), 3u);
    EXPECT_EQ(value_at_yield[0], 10);
    EXPECT_EQ(value_at_yield[1], 20);
    EXPECT_EQ(value_at_yield[2], 30);
}

TEST_F(AtomicShimYield, SiteIsTheCallingLine) {
    atomic<int> a{0};
    const auto line = std::source_location::current().line() + 1;
    a.load(std::memory_order_relaxed);
    ASSERT_EQ(seen.size(), 1u);
    EXPECT_EQ(seen[0].where.line(), line);
    EXPECT_NE(std::string(seen[0].where.file_name()).find("test_atomic_shim.cpp"),
              std::string::npos);
}

TEST_F(AtomicShimYield, NoHookNoCalls) {
    yield_hook() = nullptr;
    atomic<int> a{0};
    a.store(1);
    a.load();
    EXPECT_TRUE(seen.empty());
}

#endif

} // namespace
