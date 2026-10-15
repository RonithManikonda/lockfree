#include <gtest/gtest.h>

#include "lockfree/spsc_ring_buffer.hpp"

using lockfree::SpscRingBuffer;

// Smoke tests for the basic contract; the full single-threaded suite and
// the two-thread test build on these.

TEST(SpscRingBuffer, UsesEverySlot) {
    SpscRingBuffer<int, 4> q;
    static_assert(SpscRingBuffer<int, 4>::capacity() == 4);
    for (int i = 0; i < 4; ++i) EXPECT_TRUE(q.try_push(i));
    EXPECT_FALSE(q.try_push(99));
    EXPECT_EQ(q.size_approx(), 4u);
}

TEST(SpscRingBuffer, PopsInPushOrder) {
    SpscRingBuffer<int, 4> q;
    EXPECT_TRUE(q.empty_approx());
    q.try_push(1);
    q.try_push(2);
    int v = 0;
    ASSERT_TRUE(q.try_pop(v));
    EXPECT_EQ(v, 1);
    ASSERT_TRUE(q.try_pop(v));
    EXPECT_EQ(v, 2);
    EXPECT_FALSE(q.try_pop(v));
    EXPECT_TRUE(q.empty_approx());
}
