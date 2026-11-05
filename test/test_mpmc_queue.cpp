#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>

#include "lockfree/mpmc_queue.hpp"
#include "stress_harness.hpp"

using lockfree::MpmcQueue;

// Smoke tests for the core algorithm. The full single-threaded contract and
// the per-producer ordering stress live with the later MPMC steps.

TEST(MpmcQueue, PushThenPopRoundTrips) {
    MpmcQueue<int, 8> q;
    static_assert(MpmcQueue<int, 8>::capacity() == 8);
    EXPECT_TRUE(q.empty_approx());
    int v = -1;
    EXPECT_FALSE(q.try_pop(v));

    EXPECT_TRUE(q.try_push(7));
    EXPECT_EQ(q.size_approx(), 1u);
    ASSERT_TRUE(q.try_pop(v));
    EXPECT_EQ(v, 7);
    EXPECT_TRUE(q.empty_approx());
}

TEST(MpmcQueue, FillsEverySlotAndReportsFull) {
    MpmcQueue<int, 4> q;
    for (int i = 0; i < 4; ++i) ASSERT_TRUE(q.try_push(i));
    EXPECT_FALSE(q.try_push(99));
    for (int i = 0; i < 4; ++i) {
        auto v = q.try_pop();
        ASSERT_TRUE(v.has_value());
        EXPECT_EQ(*v, i);
    }
    EXPECT_FALSE(q.try_pop().has_value());
}

TEST(MpmcQueue, NonTrivialElementsSurviveLaps) {
    MpmcQueue<std::string, 2> q;
    for (int i = 0; i < 100; ++i) {
        ASSERT_TRUE(q.try_push(std::string(40, static_cast<char>('a' + i % 26))));
        auto v = q.try_pop();
        ASSERT_TRUE(v.has_value());
        EXPECT_EQ(*v, std::string(40, static_cast<char>('a' + i % 26)));
    }
    // Leave one behind so the destructor has something to drain under ASan.
    ASSERT_TRUE(q.try_push(std::string(64, 'z')));
}

class MpmcStress : public lockfree::test::StressTest {};

// Every thread both pushes and pops. A sum and count check catches lost or
// duplicated elements; ordering checks come with the dedicated stress step.
TEST_F(MpmcStress, MixedPushPopKeepsSumAndCount) {
    constexpr std::size_t threads = 4;
    const std::uint64_t per_thread = lockfree::test::scaled(50000);
    MpmcQueue<std::uint64_t, 64> q;
    std::atomic<std::uint64_t> popped_sum{0};
    std::atomic<std::uint64_t> popped_count{0};

    lockfree::test::run_stress(threads, [&](std::size_t index, lockfree::test::StressRng& rng) {
        std::uint64_t sum = 0, count = 0;
        for (std::uint64_t i = 0; i < per_thread; ++i) {
            const std::uint64_t value = index * per_thread + i + 1;
            while (!q.try_push(value)) {
                // Full: help drain so pushers can't all wedge on a full queue.
                if (auto v = q.try_pop()) { sum += *v; ++count; }
            }
            if (auto v = q.try_pop()) { sum += *v; ++count; }
            lockfree::test::maybe_delay(rng);
        }
        popped_sum.fetch_add(sum, std::memory_order_relaxed);
        popped_count.fetch_add(count, std::memory_order_relaxed);
    });

    // Drain what's left single-threaded.
    std::uint64_t sum = popped_sum.load(), count = popped_count.load();
    while (auto v = q.try_pop()) { sum += *v; ++count; }

    const std::uint64_t n = threads * per_thread;
    EXPECT_EQ(count, n);
    EXPECT_EQ(sum, n * (n + 1) / 2);
}
