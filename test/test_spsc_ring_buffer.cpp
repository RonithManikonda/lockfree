#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

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

// ---- Element types beyond int ----------------------------------------------

TEST(SpscRingBuffer, MoveOnlyElements) {
    SpscRingBuffer<std::unique_ptr<int>, 4> q;
    ASSERT_TRUE(q.try_push(std::make_unique<int>(1)));
    ASSERT_TRUE(q.try_emplace(new int(2)));

    std::unique_ptr<int> out;
    ASSERT_TRUE(q.try_pop(out));
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(*out, 1);

    std::optional<std::unique_ptr<int>> opt = q.try_pop();
    ASSERT_TRUE(opt.has_value());
    EXPECT_EQ(**opt, 2);
    EXPECT_FALSE(q.try_pop().has_value());
}

TEST(SpscRingBuffer, NoDefaultConstructorNeeded) {
    struct Pair {
        Pair(int a, std::string b) : a(a), b(std::move(b)) {}
        int a;
        std::string b;
    };
    static_assert(!std::is_default_constructible_v<Pair>);

    SpscRingBuffer<Pair, 2> q;
    ASSERT_TRUE(q.try_emplace(1, "one"));
    ASSERT_TRUE(q.try_emplace(2, std::string(40, 'x'))); // past SSO, heap-allocated
    EXPECT_FALSE(q.try_emplace(3, "three"));

    auto first = q.try_pop();
    ASSERT_TRUE(first);
    EXPECT_EQ(first->a, 1);
    EXPECT_EQ(first->b, "one");
    auto second = q.try_pop();
    ASSERT_TRUE(second);
    EXPECT_EQ(second->b.size(), 40u);
}

TEST(SpscRingBuffer, ThrowingConstructorLeavesQueueUnchanged) {
    struct Picky {
        explicit Picky(int v) : v(v) {
            if (v < 0) throw std::invalid_argument("negative");
        }
        int v;
    };

    SpscRingBuffer<Picky, 4> q;
    ASSERT_TRUE(q.try_emplace(7));
    EXPECT_THROW(q.try_emplace(-1), std::invalid_argument);
    EXPECT_EQ(q.size_approx(), 1u);

    // The slot the failed emplace targeted is still usable.
    ASSERT_TRUE(q.try_emplace(8));
    EXPECT_EQ(q.try_pop()->v, 7);
    EXPECT_EQ(q.try_pop()->v, 8);
    EXPECT_FALSE(q.try_pop());
}

// ---- Two threads -----------------------------------------------------------

namespace {

// Two fields that must always agree. If the consumer ever saw a slot before
// the producer's write to it was published, it could observe a mix of old
// and new contents; TSan would also flag the slot access as a race.
struct Item {
    std::uint64_t seq;
    std::uint64_t check;
};

template <std::size_t N>
void run_producer_consumer(std::uint64_t count) {
    SpscRingBuffer<Item, N> q;

    std::thread producer([&] {
        for (std::uint64_t i = 0; i < count; ++i) {
            while (!q.try_push(Item{i, ~i})) std::this_thread::yield();
        }
    });

    // Consumer runs on this thread. Record the first mismatch instead of
    // asserting in the loop, so a failure doesn't leave the producer
    // spinning on a full queue forever.
    std::uint64_t expected = 0;
    bool ok = true;
    Item got{};
    Item bad{};
    while (expected < count) {
        if (!q.try_pop(got)) {
            std::this_thread::yield();
            continue;
        }
        if (got.seq != expected || got.check != ~expected) {
            ok = false;
            bad = got;
            break;
        }
        ++expected;
    }
    if (!ok) {
        // Drain so the producer can finish and the thread can be joined.
        for (std::uint64_t left = count - expected - 1; left > 0;) {
            if (q.try_pop(got)) --left;
            else std::this_thread::yield();
        }
    }
    producer.join();

    ASSERT_TRUE(ok) << "at index " << expected << ": got seq=" << bad.seq
                    << " check=" << bad.check;
    EXPECT_FALSE(q.try_pop(got));
}

// Modest so the TSan build stays quick; the stress harness scales it later.
constexpr std::uint64_t kItems = 200'000;

} // namespace

TEST(SpscRingBufferThreads, ExactSequenceTinyQueue) {
    // N = 2 keeps the queue full or empty almost all the time, so nearly
    // every operation hits the cross-thread index check.
    run_producer_consumer<2>(kItems);
}

TEST(SpscRingBufferThreads, ExactSequenceMediumQueue) {
    run_producer_consumer<64>(kItems);
}

TEST(SpscRingBufferThreads, ExactSequenceLargeQueue) {
    run_producer_consumer<4096>(kItems);
}
