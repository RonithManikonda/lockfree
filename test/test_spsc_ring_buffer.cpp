#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include "lockfree/spsc_ring_buffer.hpp"
#include "stress_harness.hpp"

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

// Each side caches the other's counter and only reloads it when the cache says
// full/empty. These check that a stale cache gets refreshed instead of
// reporting a full or empty queue that has since changed.

TEST(SpscRingBuffer, ProducerSeesSlotFreedAfterFailedPush) {
    SpscRingBuffer<int, 4> q;
    for (int i = 0; i < 4; ++i) ASSERT_TRUE(q.try_push(i));
    ASSERT_FALSE(q.try_push(-1));  // producer's cache now says full
    int v = 0;
    ASSERT_TRUE(q.try_pop(v));
    EXPECT_EQ(v, 0);
    EXPECT_TRUE(q.try_push(4)) << "push must reload read_ when the cache says full";
    EXPECT_FALSE(q.try_push(-1));
    for (int want = 1; want <= 4; ++want) {
        ASSERT_TRUE(q.try_pop(v));
        EXPECT_EQ(v, want);
    }
}

TEST(SpscRingBuffer, ConsumerSeesItemsPushedAfterFailedPop) {
    SpscRingBuffer<int, 8> q;
    ASSERT_FALSE(q.try_pop().has_value());  // consumer's cache now says empty
    for (int i = 0; i < 3; ++i) ASSERT_TRUE(q.try_push(i));
    // Only the first pop should need a reload; the next two run off the cache.
    for (int i = 0; i < 3; ++i) {
        auto v = q.try_pop();
        ASSERT_TRUE(v.has_value()) << "pop must reload write_ when the cache says empty";
        EXPECT_EQ(*v, i);
    }
    EXPECT_FALSE(q.try_pop().has_value());
}

TEST(SpscRingBuffer, CachesTrackAcrossLapsAtFullAndEmpty) {
    // Alternate between hitting full and hitting empty so both caches go
    // stale and get refreshed many times, across the wrap point.
    SpscRingBuffer<int, 4> q;
    int next_in = 0;
    int next_out = 0;
    int v = 0;
    for (int round = 0; round < 200; ++round) {
        while (q.try_push(next_in)) ++next_in;
        ASSERT_EQ(next_in - next_out, 4);
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

// ---- Object lifetimes -------------------------------------------------------

namespace {

// Counts live instances and catches double destruction: each object carries
// a canary that its destructor clears, so destroying it twice (or destroying
// a slot that never held an object) trips the check.
struct Tracked {
    static inline int live = 0;
    static inline int bad_destroys = 0;
    static constexpr unsigned kAlive = 0xA11CE;

    explicit Tracked(int v) : value(v) { ++live; }
    Tracked(const Tracked& o) : value(o.value) { ++live; }
    Tracked(Tracked&& o) noexcept : value(o.value) { ++live; }
    Tracked& operator=(const Tracked&) = default;
    Tracked& operator=(Tracked&&) noexcept = default;
    ~Tracked() {
        if (canary != kAlive) ++bad_destroys;
        canary = 0;
        --live;
    }

    int value;
    unsigned canary = kAlive;
};

class SpscLifetime : public ::testing::Test {
protected:
    void SetUp() override {
        Tracked::live = 0;
        Tracked::bad_destroys = 0;
    }
    void TearDown() override {
        EXPECT_EQ(Tracked::live, 0) << "leaked or over-destroyed elements";
        EXPECT_EQ(Tracked::bad_destroys, 0);
    }
};

} // namespace

TEST_F(SpscLifetime, EmptyQueueConstructsNothing) {
    SpscRingBuffer<Tracked, 64> q;
    EXPECT_EQ(Tracked::live, 0);
}

TEST_F(SpscLifetime, OneLiveObjectPerQueuedElement) {
    SpscRingBuffer<Tracked, 8> q;
    for (int i = 0; i < 5; ++i) ASSERT_TRUE(q.try_emplace(i));
    EXPECT_EQ(Tracked::live, 5);

    Tracked out(-1); // +1 live, held by the test
    ASSERT_TRUE(q.try_pop(out));
    EXPECT_EQ(out.value, 0);
    EXPECT_EQ(Tracked::live, 5); // 4 queued + out

    {
        auto opt = q.try_pop();
        ASSERT_TRUE(opt);
        EXPECT_EQ(opt->value, 1);
        EXPECT_EQ(Tracked::live, 5); // 3 queued + out + opt
    }
    EXPECT_EQ(Tracked::live, 4);
}

TEST_F(SpscLifetime, PushCopiesAndMovesWithoutLeaking) {
    SpscRingBuffer<Tracked, 4> q;
    Tracked a(1);
    ASSERT_TRUE(q.try_push(a));            // copy
    ASSERT_TRUE(q.try_push(Tracked(2)));   // move from a temporary
    EXPECT_EQ(Tracked::live, 3);
    EXPECT_EQ(a.value, 1);
    EXPECT_EQ(q.try_pop()->value, 1);
    EXPECT_EQ(q.try_pop()->value, 2);
    EXPECT_EQ(Tracked::live, 1);
}

TEST_F(SpscLifetime, DestructorDrainsRemainingElements) {
    {
        SpscRingBuffer<Tracked, 8> q;
        for (int i = 0; i < 8; ++i) ASSERT_TRUE(q.try_emplace(i));
        for (int i = 0; i < 3; ++i) ASSERT_TRUE(q.try_pop());
        EXPECT_EQ(Tracked::live, 5);
    }
    // TearDown checks live == 0 and no double destroys.
}

TEST_F(SpscLifetime, DestructorDrainsAcrossTheWrapPoint) {
    // Leave elements straddling the end of the buffer so the drain loop has
    // to follow the masked index back around to slot 0.
    {
        SpscRingBuffer<Tracked, 4> q;
        for (int i = 0; i < 3; ++i) ASSERT_TRUE(q.try_emplace(i));
        for (int i = 0; i < 3; ++i) ASSERT_TRUE(q.try_pop());
        for (int i = 0; i < 4; ++i) ASSERT_TRUE(q.try_emplace(10 + i)); // slots 3,0,1,2
        EXPECT_EQ(Tracked::live, 4);
    }
}

TEST_F(SpscLifetime, ManyLapsKeepCountsBalanced) {
    SpscRingBuffer<Tracked, 4> q;
    for (int i = 0; i < 1000; ++i) {
        ASSERT_TRUE(q.try_emplace(i));
        if (i % 3 != 0) ASSERT_TRUE(q.try_pop());
        ASSERT_LE(Tracked::live, 4);
        if (q.size_approx() == 4) {
            while (q.try_pop()) {}
        }
    }
}

// ---- Two threads -----------------------------------------------------------

using lockfree::test::maybe_delay;
using lockfree::test::run_stress;
using lockfree::test::scaled;
using lockfree::test::StressRng;

namespace {

class SpscRingBufferThreads : public lockfree::test::StressTest {};

// Two fields that must always agree. If the consumer ever saw a slot before
// the producer's write to it was published, it could observe a mix of old
// and new contents; TSan would also flag the slot access as a race.
struct Item {
    std::uint64_t seq;
    std::uint64_t check;
};

// Thread 0 produces, thread 1 consumes. Both sides insert random delays so
// the queue spends time full, empty and in between. The consumer never stops
// early: it pops all `count` items and remembers the first mismatch, so the
// producer can always finish and the threads can be joined.
template <std::size_t N>
void run_producer_consumer(std::uint64_t count) {
    SpscRingBuffer<Item, N> q;

    std::uint64_t popped = 0;
    std::uint64_t first_bad_index = 0;
    Item bad{};
    bool ok = true;

    run_stress(2, [&](std::size_t index, StressRng& rng) {
        if (index == 0) {
            for (std::uint64_t i = 0; i < count; ++i) {
                maybe_delay(rng);
                while (!q.try_push(Item{i, ~i})) std::this_thread::yield();
            }
            return;
        }
        Item got{};
        while (popped < count) {
            maybe_delay(rng);
            if (!q.try_pop(got)) {
                std::this_thread::yield();
                continue;
            }
            if (ok && (got.seq != popped || got.check != ~popped)) {
                ok = false;
                first_bad_index = popped;
                bad = got;
            }
            ++popped;
        }
    });

    ASSERT_TRUE(ok) << "at index " << first_bad_index << ": got seq=" << bad.seq
                    << " check=" << bad.check;
    Item extra{};
    EXPECT_FALSE(q.try_pop(extra));
}

// Modest so the TSan build stays quick; LOCKFREE_STRESS_SCALE raises it.
const std::uint64_t kItems = scaled(200'000);

} // namespace

TEST_F(SpscRingBufferThreads, ExactSequenceTinyQueue) {
    // N = 2 keeps the queue full or empty almost all the time, so nearly
    // every operation hits the cross-thread index check.
    run_producer_consumer<2>(kItems);
}

TEST_F(SpscRingBufferThreads, ExactSequenceMediumQueue) {
    run_producer_consumer<64>(kItems);
}

TEST_F(SpscRingBufferThreads, ExactSequenceLargeQueue) {
    run_producer_consumer<4096>(kItems);
}


TEST_F(SpscRingBufferThreads, MoveOnlyAcrossThreads) {
    constexpr int kCount = 50'000;
    SpscRingBuffer<std::unique_ptr<int>, 16> q;

    std::thread producer([&] {
        for (int i = 0; i < kCount; ++i) {
            auto p = std::make_unique<int>(i);
            // try_push(T&&) only moves from p on success, so retrying is safe.
            while (!q.try_push(std::move(p))) std::this_thread::yield();
        }
    });

    int expected = 0;
    bool ok = true;
    while (expected < kCount) {
        auto got = q.try_pop();
        if (!got) {
            std::this_thread::yield();
            continue;
        }
        if (!*got || **got != expected) ok = false;
        ++expected;
    }
    producer.join();
    EXPECT_TRUE(ok);
    EXPECT_FALSE(q.try_pop());
}
