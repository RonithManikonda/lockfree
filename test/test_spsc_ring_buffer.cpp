#include <gtest/gtest.h>

#include "lockfree/spsc_ring_buffer.hpp"

using lockfree::SpscRingBuffer;

// Single-threaded contract tests. With only one thread the "approx" queries
// are exact, so they can be asserted directly.

TEST(SpscRingBuffer, StartsEmpty) {
    SpscRingBuffer<int, 8> q;
    EXPECT_TRUE(q.empty_approx());
    EXPECT_EQ(q.size_approx(), 0u);
    int v = -1;
    EXPECT_FALSE(q.try_pop(v));
    EXPECT_EQ(v, -1) << "failed pop must not touch the output";
}

TEST(SpscRingBuffer, UsesEverySlot) {
    SpscRingBuffer<int, 4> q;
    static_assert(SpscRingBuffer<int, 4>::capacity() == 4);
    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(q.try_push(i));
        EXPECT_EQ(q.size_approx(), static_cast<std::size_t>(i + 1));
    }
    EXPECT_FALSE(q.empty_approx());
}

TEST(SpscRingBuffer, PushOnFullFailsAndKeepsContents) {
    SpscRingBuffer<int, 4> q;
    for (int i = 0; i < 4; ++i) ASSERT_TRUE(q.try_push(i));
    EXPECT_FALSE(q.try_push(99));
    EXPECT_FALSE(q.try_push(100));
    EXPECT_EQ(q.size_approx(), 4u);

    // The rejected values must not have overwritten anything.
    int v = 0;
    for (int i = 0; i < 4; ++i) {
        ASSERT_TRUE(q.try_pop(v));
        EXPECT_EQ(v, i);
    }
    EXPECT_FALSE(q.try_pop(v));
}

TEST(SpscRingBuffer, PopsInPushOrder) {
    SpscRingBuffer<int, 16> q;
    for (int i = 0; i < 10; ++i) ASSERT_TRUE(q.try_push(i * 3));
    int v = 0;
    for (int i = 0; i < 10; ++i) {
        ASSERT_TRUE(q.try_pop(v));
        EXPECT_EQ(v, i * 3);
    }
    EXPECT_TRUE(q.empty_approx());
}

TEST(SpscRingBuffer, FullThenDrainThenFullAgain) {
    SpscRingBuffer<int, 8> q;
    int v = 0;
    for (int round = 0; round < 3; ++round) {
        for (int i = 0; i < 8; ++i) ASSERT_TRUE(q.try_push(round * 100 + i));
        ASSERT_FALSE(q.try_push(-1));
        for (int i = 0; i < 8; ++i) {
            ASSERT_TRUE(q.try_pop(v));
            EXPECT_EQ(v, round * 100 + i);
        }
        ASSERT_FALSE(q.try_pop(v));
    }
}

// Keep the queue partly full while the indices lap the buffer many times,
// so every slot is reused with the read and write positions at every
// possible offset from each other.
TEST(SpscRingBuffer, WrapsAroundManyTimes) {
    SpscRingBuffer<int, 8> q;
    int next_push = 0;
    int next_pop = 0;
    int v = 0;
    for (int step = 0; step < 10'000; ++step) {
        // Push 3, pop 2: the fill level ratchets up, then the full check
        // starts rejecting pushes and the queue oscillates near capacity.
        for (int k = 0; k < 3; ++k) {
            if (q.try_push(next_push)) ++next_push;
        }
        for (int k = 0; k < 2; ++k) {
            if (q.try_pop(v)) {
                ASSERT_EQ(v, next_pop);
                ++next_pop;
            }
        }
        ASSERT_LE(q.size_approx(), q.capacity());
        ASSERT_EQ(q.size_approx(), static_cast<std::size_t>(next_push - next_pop));
    }
    while (q.try_pop(v)) ASSERT_EQ(v, next_pop++);
    EXPECT_EQ(next_pop, next_push);
    EXPECT_GT(next_push, 8 * 1000) << "indices should have lapped the buffer";
}

TEST(SpscRingBuffer, SmallestCapacity) {
    SpscRingBuffer<int, 2> q;
    static_assert(SpscRingBuffer<int, 2>::capacity() == 2);
    int v = 0;
    for (int i = 0; i < 100; ++i) {
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

TEST(SpscRingBuffer, InterleavedSinglePushPop) {
    // One in, one out: the occupied slot walks around the ring.
    SpscRingBuffer<long, 4> q;
    long v = 0;
    for (long i = 0; i < 1000; ++i) {
        ASSERT_TRUE(q.try_push(i));
        EXPECT_EQ(q.size_approx(), 1u);
        ASSERT_TRUE(q.try_pop(v));
        EXPECT_EQ(v, i);
        EXPECT_TRUE(q.empty_approx());
    }
}
