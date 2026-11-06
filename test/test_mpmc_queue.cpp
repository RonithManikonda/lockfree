#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "lockfree/mpmc_queue.hpp"
#include "stress_harness.hpp"
#include "tracked.hpp"

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

// ---- Element types beyond int ------------------------------------------------

TEST(MpmcQueue, MoveOnlyElements) {
    MpmcQueue<std::unique_ptr<int>, 4> q;
    ASSERT_TRUE(q.try_push(std::make_unique<int>(1)));
    ASSERT_TRUE(q.try_emplace(new int(2)));
    q.push(std::make_unique<int>(3));

    std::unique_ptr<int> out;
    ASSERT_TRUE(q.try_pop(out));
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(*out, 1);

    std::optional<std::unique_ptr<int>> opt = q.try_pop();
    ASSERT_TRUE(opt.has_value());
    EXPECT_EQ(**opt, 2);
    EXPECT_EQ(*q.pop(), 3);
    EXPECT_FALSE(q.try_pop().has_value());
}

TEST(MpmcQueue, NoDefaultConstructorNeeded) {
    struct Pair {
        Pair(int a, std::string b) : a(a), b(std::move(b)) {}
        int a;
        std::string b;
    };
    static_assert(!std::is_default_constructible_v<Pair>);

    MpmcQueue<Pair, 2> q;
    ASSERT_TRUE(q.try_emplace(1, "one"));
    ASSERT_TRUE(q.try_emplace(2, std::string(40, 'x'))); // past SSO, heap-allocated
    EXPECT_FALSE(q.try_emplace(3, "three"));

    auto first = q.try_pop();
    ASSERT_TRUE(first);
    EXPECT_EQ(first->a, 1);
    EXPECT_EQ(first->b, "one");
    Pair second = q.pop();
    EXPECT_EQ(second.b.size(), 40u);
}

// The throwing constructor runs before a cell is claimed (see try_emplace),
// so the exception leaves no hole that would stall the consumers behind it.
TEST(MpmcQueue, ThrowingConstructorLeavesQueueUnchanged) {
    struct Picky {
        explicit Picky(int v) : v(v) {
            if (v < 0) throw std::invalid_argument("negative");
        }
        int v;
    };

    MpmcQueue<Picky, 4> q;
    ASSERT_TRUE(q.try_emplace(7));
    EXPECT_THROW(q.try_emplace(-1), std::invalid_argument);
    EXPECT_THROW(q.emplace(-2), std::invalid_argument);
    EXPECT_EQ(q.size_approx(), 1u);

    ASSERT_TRUE(q.try_emplace(8));
    EXPECT_EQ(q.try_pop()->v, 7);
    EXPECT_EQ(q.try_pop()->v, 8);
    EXPECT_FALSE(q.try_pop());
}

// ---- Object lifetimes ---------------------------------------------------------

using lockfree::test::Tracked;

class MpmcLifetime : public lockfree::test::TrackedTest {};

TEST_F(MpmcLifetime, EmptyQueueConstructsNothing) {
    MpmcQueue<Tracked, 64> q;
    EXPECT_EQ(Tracked::live.load(), 0);
}

TEST_F(MpmcLifetime, OneLiveObjectPerQueuedElement) {
    MpmcQueue<Tracked, 8> q;
    for (int i = 0; i < 5; ++i) ASSERT_TRUE(q.try_emplace(i));
    EXPECT_EQ(Tracked::live.load(), 5);

    Tracked out(-1); // +1 live, held by the test
    ASSERT_TRUE(q.try_pop(out));
    EXPECT_EQ(out.value, 0);
    EXPECT_EQ(Tracked::live.load(), 5); // 4 queued + out

    {
        auto opt = q.try_pop();
        ASSERT_TRUE(opt);
        EXPECT_EQ(opt->value, 1);
        EXPECT_EQ(Tracked::live.load(), 5); // 3 queued + out + opt
    }
    EXPECT_EQ(Tracked::live.load(), 4);
    {
        Tracked p = q.pop();
        EXPECT_EQ(p.value, 2);
        EXPECT_EQ(Tracked::live.load(), 4); // 2 queued + out + p
    }
}

TEST_F(MpmcLifetime, PushCopiesAndMovesWithoutLeaking) {
    MpmcQueue<Tracked, 4> q;
    Tracked a(1);
    ASSERT_TRUE(q.try_push(a));            // copy
    ASSERT_TRUE(q.try_push(Tracked(2)));   // move from a temporary
    q.push(a);                             // blocking copy
    EXPECT_EQ(Tracked::live.load(), 4);
    EXPECT_EQ(a.value, 1);
    EXPECT_EQ(q.try_pop()->value, 1);
    EXPECT_EQ(q.try_pop()->value, 2);
    EXPECT_EQ(q.try_pop()->value, 1);
    EXPECT_EQ(Tracked::live.load(), 1);
}

TEST_F(MpmcLifetime, DestructorDrainsRemainingElements) {
    {
        MpmcQueue<Tracked, 8> q;
        for (int i = 0; i < 8; ++i) ASSERT_TRUE(q.try_emplace(i));
        for (int i = 0; i < 3; ++i) ASSERT_TRUE(q.try_pop());
        EXPECT_EQ(Tracked::live.load(), 5);
    }
    // TearDown checks live == 0 and no double destroys.
}

TEST_F(MpmcLifetime, DestructorDrainsAcrossTheWrapPoint) {
    // Leave elements straddling the end of the buffer so the drain loop has
    // to follow the masked position back around to cell 0.
    {
        MpmcQueue<Tracked, 4> q;
        for (int i = 0; i < 3; ++i) ASSERT_TRUE(q.try_emplace(i));
        for (int i = 0; i < 3; ++i) ASSERT_TRUE(q.try_pop());
        for (int i = 0; i < 4; ++i) ASSERT_TRUE(q.try_emplace(10 + i)); // cells 3,0,1,2
        EXPECT_EQ(Tracked::live.load(), 4);
    }
}

TEST_F(MpmcLifetime, ManyLapsKeepCountsBalanced) {
    MpmcQueue<Tracked, 4> q;
    for (int i = 0; i < 1000; ++i) {
        ASSERT_TRUE(q.try_emplace(i));
        if (i % 3 != 0) ASSERT_TRUE(q.try_pop());
        ASSERT_LE(Tracked::live.load(), 4);
        if (q.size_approx() == 4) {
            while (q.try_pop()) {}
        }
    }
}

// Elements built on one thread and destroyed on another, across many laps.
// The queue is left with whatever the consumers didn't get to, so the
// destructor drain also runs on cells written by several producers.
class MpmcLifetimeThreads : public lockfree::test::TrackedTest {};

TEST_F(MpmcLifetimeThreads, CountsBalanceAcrossThreads) {
    constexpr std::size_t producers = 3, consumers = 3;
    constexpr int left = 2;
    static_assert(consumers * left <= MpmcQueue<Tracked, 8>::capacity());
    const int per_producer = static_cast<int>(lockfree::test::scaled(20000));
    std::atomic<int> popped{0};
    {
        MpmcQueue<Tracked, 8> q;
        lockfree::test::run_stress(producers + consumers, [&](std::size_t index,
                                                              lockfree::test::StressRng& rng) {
            if (index < producers) {
                for (int i = 0; i < per_producer; ++i) {
                    q.emplace(i);
                    lockfree::test::maybe_delay(rng);
                }
                return;
            }
            // Each consumer stops `left` short, so some elements are left
            // for the destructor. consumers * left must fit in the queue, or
            // the producers' last pushes block forever.
            for (int i = 0; i < per_producer - left; ++i) {
                Tracked t = q.pop();
                if (t.canary != Tracked::kAlive) Tracked::bad_destroys.fetch_add(1);
                popped.fetch_add(1, std::memory_order_relaxed);
            }
        });
        EXPECT_EQ(Tracked::live.load(), static_cast<int>(consumers) * left);
    }
    EXPECT_EQ(popped.load(), static_cast<int>(consumers) * (per_producer - left));
}

// Ownership crosses threads: producers allocate, consumers free. ASan
// catches a use-after-free or double free if a cell is handed out twice.
class MpmcMoveOnlyThreads : public lockfree::test::StressTest {};

TEST_F(MpmcMoveOnlyThreads, UniquePtrsChangeHandsExactlyOnce) {
    constexpr std::size_t producers = 4, consumers = 4;
    const std::uint64_t per_producer = lockfree::test::scaled(20000);
    MpmcQueue<std::unique_ptr<std::uint64_t>, 16> q;
    std::atomic<std::uint64_t> sum{0};

    lockfree::test::run_stress(producers + consumers, [&](std::size_t index,
                                                          lockfree::test::StressRng& rng) {
        if (index < producers) {
            for (std::uint64_t i = 1; i <= per_producer; ++i) {
                q.push(std::make_unique<std::uint64_t>(i));
                lockfree::test::maybe_delay(rng);
            }
            return;
        }
        std::uint64_t local = 0;
        for (std::uint64_t i = 0; i < per_producer; ++i) {
            std::unique_ptr<std::uint64_t> p = q.pop();
            local += *p;
        }
        sum.fetch_add(local, std::memory_order_relaxed);
    });

    EXPECT_TRUE(q.empty_approx());
    EXPECT_EQ(sum.load(), producers * per_producer * (per_producer + 1) / 2);
}

// ---- Blocking push / pop -----------------------------------------------------

TEST(MpmcQueue, BlockingCallsReturnRightAwayWhenTheyCan) {
    MpmcQueue<int, 4> q;
    q.push(1);
    const int two = 2;
    q.push(two);
    q.emplace(3);
    EXPECT_EQ(q.size_approx(), 3u);
    EXPECT_EQ(q.pop(), 1);
    int v = 0;
    q.pop(v);
    EXPECT_EQ(v, 2);
    EXPECT_EQ(q.pop(), 3);
    EXPECT_TRUE(q.empty_approx());
}

TEST(MpmcQueue, BlockingPushWaitsForRoom) {
    MpmcQueue<int, 2> q;
    q.push(1);
    q.push(2);
    std::atomic<bool> pushed{false};
    std::thread producer([&] {
        q.push(3);  // full: has to wait for the pop below
        pushed.store(true, std::memory_order_release);
    });
    // Not a proof it's blocked, but a push that ignored "full" would almost
    // always have finished by now.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_FALSE(pushed.load(std::memory_order_acquire));
    EXPECT_EQ(q.pop(), 1);
    producer.join();
    EXPECT_TRUE(pushed.load());
    EXPECT_EQ(q.pop(), 2);
    EXPECT_EQ(q.pop(), 3);
}

TEST(MpmcQueue, BlockingPopWaitsForAnElement) {
    MpmcQueue<std::string, 4> q;
    std::string got;
    std::thread consumer([&] { got = q.pop(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    q.push(std::string("late"));
    consumer.join();
    EXPECT_EQ(got, "late");
    EXPECT_TRUE(q.empty_approx());
}

// The blocking push builds a throwing-constructible element once, then
// retries with the built copy; a failed attempt must not consume it.
TEST(MpmcQueue, BlockingPushOfCopyableTypeKeepsTheValueAcrossRetries) {
    struct Noisy {
        std::string s;
        explicit Noisy(std::string v) : s(std::move(v)) {}
        Noisy(const Noisy& o) : s(o.s) {}  // may throw (allocates)
        Noisy(Noisy&&) noexcept = default;
        Noisy& operator=(Noisy&&) noexcept = default;
    };
    static_assert(!std::is_nothrow_copy_constructible_v<Noisy>);
    MpmcQueue<Noisy, 2> q;
    q.push(Noisy("a"));
    q.push(Noisy("b"));
    const Noisy c("c-with-a-long-enough-string-to-allocate");
    std::thread producer([&] { q.push(c); });
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    EXPECT_EQ(q.pop().s, "a");
    producer.join();
    EXPECT_EQ(q.pop().s, "b");
    EXPECT_EQ(q.pop().s, c.s);
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

// Same shape as above, but everyone uses the blocking calls, so the waiting
// goes through Backoff instead of yield loops in the test. Consumers are
// told up front how many to take, since pop() never gives up.
TEST_P(MpmcProducersConsumers, BlockingCallsExactlyOnce) {
    const auto [producers, consumers] = GetParam();
    const std::uint64_t per_producer = lockfree::test::scaled(10000);
    const std::uint64_t total = producers * per_producer;
    MpmcQueue<std::uint64_t, 16> q;
    std::vector<std::atomic<std::uint8_t>> seen(total);

    lockfree::test::run_stress(producers + consumers, [&](std::size_t index,
                                                          lockfree::test::StressRng& rng) {
        if (index < producers) {
            for (std::uint64_t s = 0; s < per_producer; ++s) {
                q.push(index * per_producer + s);
                lockfree::test::maybe_delay(rng);
            }
            return;
        }
        // Split total across consumers; the first ones take the remainder.
        const std::size_t c = index - producers;
        const std::uint64_t share = total / consumers + (c < total % consumers ? 1 : 0);
        for (std::uint64_t i = 0; i < share; ++i) {
            const std::uint64_t v = q.pop();
            if (v < total) seen[v].fetch_add(1, std::memory_order_relaxed);
        }
    });

    EXPECT_TRUE(q.empty_approx());
    std::uint64_t wrong = 0;
    for (const auto& n : seen) wrong += n.load(std::memory_order_relaxed) != 1;
    EXPECT_EQ(wrong, 0u) << "values missing or popped more than once";
}

INSTANTIATE_TEST_SUITE_P(Shapes, MpmcProducersConsumers, ::testing::ValuesIn(all_shapes()),
                         shape_name);
