#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <utility>

#include "lockfree/mpmc_queue.hpp"
#include "stress_harness.hpp"

using lockfree::MpmcQueue;

// Single-threaded contract tests. With one thread nothing is in flight, so
// the "approx" queries are exact and can be asserted directly.

TEST(MpmcQueue, StartsEmpty) {
    MpmcQueue<int, 8> q;
    static_assert(MpmcQueue<int, 8>::capacity() == 8);
    EXPECT_TRUE(q.empty_approx());
    EXPECT_EQ(q.size_approx(), 0u);
    int v = -1;
    EXPECT_FALSE(q.try_pop(v));
    EXPECT_EQ(v, -1) << "failed pop must not touch the output";
    EXPECT_FALSE(q.try_pop().has_value());
}

TEST(MpmcQueue, PushThenPopRoundTrips) {
    MpmcQueue<int, 8> q;
    EXPECT_TRUE(q.try_push(7));
    EXPECT_EQ(q.size_approx(), 1u);
    int v = -1;
    ASSERT_TRUE(q.try_pop(v));
    EXPECT_EQ(v, 7);
    EXPECT_TRUE(q.empty_approx());
}

TEST(MpmcQueue, UsesEverySlot) {
    MpmcQueue<int, 4> q;
    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(q.try_push(i));
        EXPECT_EQ(q.size_approx(), static_cast<std::size_t>(i + 1));
    }
    EXPECT_FALSE(q.empty_approx());
}

TEST(MpmcQueue, PushOnFullFailsAndKeepsContents) {
    MpmcQueue<int, 4> q;
    for (int i = 0; i < 4; ++i) ASSERT_TRUE(q.try_push(i));
    EXPECT_FALSE(q.try_push(99));
    EXPECT_FALSE(q.try_push(100));
    // A failed push never claims a position, so the count doesn't creep.
    EXPECT_EQ(q.size_approx(), 4u);

    for (int i = 0; i < 4; ++i) {
        auto v = q.try_pop();
        ASSERT_TRUE(v.has_value());
        EXPECT_EQ(*v, i);
    }
    EXPECT_FALSE(q.try_pop().has_value());
}

TEST(MpmcQueue, PopsInPushOrder) {
    MpmcQueue<int, 16> q;
    for (int i = 0; i < 10; ++i) ASSERT_TRUE(q.try_push(i * 3));
    int v = 0;
    for (int i = 0; i < 10; ++i) {
        ASSERT_TRUE(q.try_pop(v));
        EXPECT_EQ(v, i * 3);
    }
    EXPECT_TRUE(q.empty_approx());
}

// Each round leaves every cell's seq one lap further on; a mistake in the
// "p + N means free again" step shows up as the second round reporting full.
TEST(MpmcQueue, FullThenDrainThenFullAgain) {
    MpmcQueue<int, 8> q;
    int v = 0;
    for (int round = 0; round < 3; ++round) {
        for (int i = 0; i < 8; ++i) ASSERT_TRUE(q.try_push(round * 100 + i)) << "round " << round;
        ASSERT_FALSE(q.try_push(-1));
        for (int i = 0; i < 8; ++i) {
            ASSERT_TRUE(q.try_pop(v));
            EXPECT_EQ(v, round * 100 + i);
        }
        ASSERT_FALSE(q.try_pop(v));
    }
}

// Keep the queue partly full while the positions lap the buffer many times,
// so each cell is reused with the enqueue and dequeue positions at every
// offset from each other.
TEST(MpmcQueue, WrapsAroundManyTimes) {
    MpmcQueue<int, 4> q;
    int next_in = 0;
    int next_out = 0;
    int v = 0;
    for (int round = 0; round < 200; ++round) {
        while (q.try_push(next_in)) ++next_in;
        ASSERT_EQ(next_in - next_out, 4);
        ASSERT_EQ(q.size_approx(), 4u);
        const int take = 1 + round % 4;
        for (int i = 0; i < take; ++i) {
            ASSERT_TRUE(q.try_pop(v));
            ASSERT_EQ(v, next_out++);
        }
        if (take == 4) {
            ASSERT_FALSE(q.try_pop(v));
        }
    }
}

TEST(MpmcQueue, SmallestCapacity) {
    MpmcQueue<int, 2> q;
    int v = 0;
    for (int i = 0; i < 50; ++i) {
        ASSERT_TRUE(q.try_push(2 * i));
        ASSERT_TRUE(q.try_push(2 * i + 1));
        ASSERT_FALSE(q.try_push(-1));
        ASSERT_TRUE(q.try_pop(v));
        EXPECT_EQ(v, 2 * i);
        ASSERT_TRUE(q.try_pop(v));
        EXPECT_EQ(v, 2 * i + 1);
        ASSERT_FALSE(q.try_pop(v));
    }
}

// Popping from an empty queue must not claim a position either; otherwise
// the next push would land behind a dequeue position that already passed it.
TEST(MpmcQueue, FailedPopsDontSkipAhead) {
    MpmcQueue<int, 4> q;
    for (int i = 0; i < 5; ++i) EXPECT_FALSE(q.try_pop().has_value());
    ASSERT_TRUE(q.try_push(42));
    auto v = q.try_pop();
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(*v, 42);
}

TEST(MpmcQueue, BothPopFormsShareOneOrder) {
    MpmcQueue<int, 8> q;
    for (int i = 0; i < 6; ++i) ASSERT_TRUE(q.try_push(i));
    int v = -1;
    for (int i = 0; i < 6; ++i) {
        if (i % 2 == 0) {
            ASSERT_TRUE(q.try_pop(v));
            EXPECT_EQ(v, i);
        } else {
            auto o = q.try_pop();
            ASSERT_TRUE(o.has_value());
            EXPECT_EQ(*o, i);
        }
    }
}

TEST(MpmcQueue, EmplaceBuildsInPlace) {
    MpmcQueue<std::pair<int, std::string>, 4> q;
    ASSERT_TRUE(q.try_emplace(3, "three"));
    ASSERT_TRUE(q.try_emplace(4, std::string(4, 'x')));
    auto a = q.try_pop();
    ASSERT_TRUE(a.has_value());
    EXPECT_EQ(a->first, 3);
    EXPECT_EQ(a->second, "three");
    auto b = q.try_pop();
    ASSERT_TRUE(b.has_value());
    EXPECT_EQ(b->second, "xxxx");
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
