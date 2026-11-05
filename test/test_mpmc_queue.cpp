#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <utility>
#include <vector>

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
// duplicated elements; the producer/consumer split below checks ordering.
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

// ---- Producers x consumers ---------------------------------------------------

// Each value carries (producer id, per-producer sequence number). What a
// linearizable FIFO guarantees here:
//   - every value comes out exactly once;
//   - one consumer sees each producer's values in increasing order. A
//     producer's pushes claim increasing positions, and a consumer's pops
//     claim increasing positions too, so whatever one consumer gets from one
//     producer arrives in push order. Across consumers there's no order to
//     check: two consumers can finish their pops in either order.
namespace {

constexpr std::uint64_t make_tagged(std::uint64_t producer, std::uint64_t seq) {
    return producer << 32 | seq;
}
constexpr std::uint64_t tag_producer(std::uint64_t v) { return v >> 32; }
constexpr std::uint64_t tag_seq(std::uint64_t v) { return v & 0xffffffffu; }

struct Shape {
    std::size_t producers;
    std::size_t consumers;
};

std::string shape_name(const ::testing::TestParamInfo<Shape>& info) {
    return std::to_string(info.param.producers) + "p" + std::to_string(info.param.consumers) + "c";
}

std::vector<Shape> all_shapes() {
    std::vector<Shape> shapes;
    for (std::size_t p = 1; p <= 4; ++p) {
        for (std::size_t c = 1; c <= 4; ++c) shapes.push_back({p, c});
    }
    return shapes;
}

} // namespace

class MpmcProducersConsumers : public lockfree::test::StressTest,
                               public ::testing::WithParamInterface<Shape> {};

TEST_P(MpmcProducersConsumers, ExactlyOnceAndPerProducerOrder) {
    const auto [producers, consumers] = GetParam();
    const std::uint64_t per_producer = lockfree::test::scaled(20000);
    const std::uint64_t total = producers * per_producer;

    // Small on purpose: producers keep hitting full and consumers keep
    // hitting empty, so the "lap behind" and "not pushed yet" paths both run.
    MpmcQueue<std::uint64_t, 16> q;

    // seen[p][s] counts how often (p, s) was popped. Each entry is written by
    // whichever consumer popped it, hence atomic.
    std::vector<std::vector<std::atomic<std::uint8_t>>> seen(producers);
    for (auto& row : seen) row = std::vector<std::atomic<std::uint8_t>>(per_producer);

    std::atomic<std::uint64_t> consumed{0};
    std::atomic<std::uint64_t> order_violations{0};
    std::atomic<std::uint64_t> bad_values{0};

    lockfree::test::run_stress(producers + consumers, [&](std::size_t index,
                                                          lockfree::test::StressRng& rng) {
        if (index < producers) {
            for (std::uint64_t s = 0; s < per_producer; ++s) {
                while (!q.try_push(make_tagged(index, s))) std::this_thread::yield();
                lockfree::test::maybe_delay(rng);
            }
            return;
        }

        // last[p] = one past the highest seq this consumer got from producer p.
        std::vector<std::uint64_t> last(producers, 0);
        while (consumed.load(std::memory_order_relaxed) < total) {
            auto v = q.try_pop();
            if (!v) {
                std::this_thread::yield();
                continue;
            }
            consumed.fetch_add(1, std::memory_order_relaxed);
            const std::uint64_t p = tag_producer(*v), s = tag_seq(*v);
            if (p >= producers || s >= per_producer) {
                bad_values.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            if (s < last[p]) order_violations.fetch_add(1, std::memory_order_relaxed);
            last[p] = s + 1;
            seen[p][s].fetch_add(1, std::memory_order_relaxed);
            lockfree::test::maybe_delay(rng);
        }
    });

    EXPECT_EQ(bad_values.load(), 0u);
    EXPECT_EQ(order_violations.load(), 0u);
    EXPECT_EQ(consumed.load(), total);
    EXPECT_TRUE(q.empty_approx());

    std::uint64_t missing = 0, duplicated = 0;
    for (const auto& row : seen) {
        for (const auto& n : row) {
            const auto c = n.load(std::memory_order_relaxed);
            if (c == 0) ++missing;
            if (c > 1) ++duplicated;
        }
    }
    EXPECT_EQ(missing, 0u);
    EXPECT_EQ(duplicated, 0u);
}

INSTANTIATE_TEST_SUITE_P(Shapes, MpmcProducersConsumers, ::testing::ValuesIn(all_shapes()),
                         shape_name);
