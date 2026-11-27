#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "lockfree/hazard_pointer.hpp"
#include "lockfree/michael_scott_queue.hpp"
#include "lockfree/detail/test_hook.hpp"
#include "stress_harness.hpp"
#include "tracked.hpp"

namespace {

using lockfree::HazardPointer;
using lockfree::MichaelScottQueue;
using lockfree::test::maybe_delay;
using lockfree::test::run_stress;
using lockfree::test::scaled;
using lockfree::test::StressRng;
using lockfree::test::StressTest;
using lockfree::test::Tracked;
using lockfree::test::TrackedTest;

TEST(MichaelScottQueue, PopOnNewQueueIsEmpty) {
    MichaelScottQueue<int> q;
    EXPECT_FALSE(q.try_pop().has_value());
    int out = -1;
    EXPECT_FALSE(q.try_pop(out));
    EXPECT_EQ(out, -1);
}

TEST(MichaelScottQueue, FifoOrder) {
    MichaelScottQueue<int> q;
    for (int i = 0; i < 100; ++i) q.push(i);
    for (int i = 0; i < 100; ++i) {
        auto v = q.try_pop();
        ASSERT_TRUE(v.has_value());
        EXPECT_EQ(*v, i);
    }
    EXPECT_FALSE(q.try_pop().has_value());
}

TEST(MichaelScottQueue, InterleavedPushPopNeverReordersOrLoses) {
    MichaelScottQueue<int> q;
    int next_in = 0, next_out = 0;
    for (int round = 0; round < 1000; ++round) {
        for (int k = 0; k < round % 5 + 1; ++k) q.push(next_in++);
        for (int k = 0; k < round % 3 + 1; ++k) {
            int v;
            if (q.try_pop(v)) EXPECT_EQ(v, next_out++);
        }
    }
    while (auto v = q.try_pop()) EXPECT_EQ(*v, next_out++);
    EXPECT_EQ(next_in, next_out);
}

TEST(MichaelScottQueue, MoveOnlyValues) {
    MichaelScottQueue<std::unique_ptr<int>> q;
    q.push(std::make_unique<int>(7));
    q.emplace(new int(8));
    auto a = q.try_pop();
    ASSERT_TRUE(a && *a);
    EXPECT_EQ(**a, 7);
    std::unique_ptr<int> b;
    ASSERT_TRUE(q.try_pop(b));
    EXPECT_EQ(*b, 8);
}

TEST(MichaelScottQueue, BlockingPopReturnsWhatIsThere) {
    MichaelScottQueue<std::string> q;
    q.push("hello");
    EXPECT_EQ(q.pop(), "hello");
}

// Old heads go to the reclaimer, not straight to delete. Scanning after the
// pops (no hazards held any more) frees them.
TEST(MichaelScottQueue, PopsRetireTheOldDummies) {
    HazardPointer::scan();
    const std::size_t before = HazardPointer::retired_count();
    MichaelScottQueue<int> q;
    for (int i = 0; i < 10; ++i) q.push(i);
    for (int i = 0; i < 10; ++i) ASSERT_TRUE(q.try_pop());
    EXPECT_EQ(HazardPointer::retired_count(), before + 10);
    HazardPointer::scan();
    EXPECT_EQ(HazardPointer::retired_count(), 0u);
}

class MichaelScottLifetime : public TrackedTest {};

TEST_F(MichaelScottLifetime, EveryValueDestroyedExactlyOnce) {
    {
        MichaelScottQueue<Tracked> q;
        for (int i = 0; i < 20; ++i) q.emplace(i);
        for (int i = 0; i < 12; ++i) {
            auto v = q.try_pop();
            ASSERT_TRUE(v);
            EXPECT_EQ(v->value, i);
        }
        // 8 left for the destructor, and the dummy must not be destroyed as a T.
    }
    HazardPointer::scan();
}

TEST_F(MichaelScottLifetime, ThrowingConstructorLeavesQueueUnchanged) {
    struct Boom {
        explicit Boom(bool t) {
            if (t) throw std::runtime_error("boom");
        }
    };
    MichaelScottQueue<Boom> q;
    q.emplace(false);
    EXPECT_THROW(q.emplace(true), std::runtime_error);
    EXPECT_TRUE(q.try_pop().has_value());
    EXPECT_FALSE(q.try_pop().has_value());
}

// An element whose move constructor uses another queue: the inner operation
// runs while the outer one holds the thread's cached hazard slots, so it has
// to take its own pair.
struct UsesQueue {
    MichaelScottQueue<int>* other;
    int value;
    UsesQueue(MichaelScottQueue<int>* o, int v) : other(o), value(v) {}
    UsesQueue(UsesQueue&& u) noexcept : other(u.other), value(u.value) {
        other->push(value);
        (void)other->try_pop();
    }
};

TEST(MichaelScottQueue, NestedOperationsGetTheirOwnSlots) {
    MichaelScottQueue<int> other;
    MichaelScottQueue<UsesQueue> q;
    q.emplace(&other, 5);
    auto v = q.try_pop();
    ASSERT_TRUE(v);
    EXPECT_EQ(v->value, 5);
    EXPECT_FALSE(other.try_pop().has_value());
}

// A dequeuer stalls right before its CAS while another thread empties the
// queue and frees what it can. The stalled dequeuer's nodes are still
// protected, so its CAS fails cleanly and it retries on live memory (ASan
// would flag a use-after-free here).
TEST(MichaelScottQueue, StalledDequeuerSurvivesAnotherDrainingTheQueue) {
    MichaelScottQueue<int> q;
    for (int i = 0; i < 4; ++i) q.push(i);
    int calls = 0;
    std::vector<int> stolen;
    lockfree::detail::ScopedTestHook hook([&](std::string_view where) {
        if (where != "ms_dequeue_before_cas" || calls++ != 0) return;
        std::thread([&] {
            while (auto v = q.try_pop()) stolen.push_back(*v);
            HazardPointer::scan();
            q.push(100);
        }).join();
    });
    auto v = q.try_pop();
    ASSERT_TRUE(v);
    EXPECT_EQ(*v, 100);
    EXPECT_EQ(stolen, (std::vector<int>{0, 1, 2, 3}));
}

// A producer that linked its node but hasn't swung tail_ yet. Other threads
// must finish its work, not wait for it.
TEST(MichaelScottQueue, LaggingTailIsHelpedForward) {
    MichaelScottQueue<int> q;
    int calls = 0;
    lockfree::detail::ScopedTestHook hook([&](std::string_view where) {
        if (where != "ms_enqueue_after_link" || calls++ != 0) return;
        std::thread([&] {
            q.push(2);                       // has to help tail_ past 1 first
            EXPECT_EQ(q.try_pop(), 1);
            EXPECT_EQ(q.try_pop(), 2);
            EXPECT_FALSE(q.try_pop().has_value());
        }).join();
    });
    q.push(1);
    EXPECT_FALSE(q.try_pop().has_value());
}

// ---- Threaded stress ------------------------------------------------------

constexpr std::uint64_t make_tagged(std::uint64_t producer, std::uint64_t seq) {
    return (producer << 32) | seq;
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
    for (std::size_t p : {1, 2, 4}) {
        for (std::size_t c : {1, 2, 4}) shapes.push_back({p, c});
    }
    shapes.push_back({8, 8});
    return shapes;
}

class MichaelScottShapes : public StressTest, public ::testing::WithParamInterface<Shape> {};

// Every value comes out exactly once, and each consumer sees each producer's
// values in the order they were pushed (FIFO per producer). Consumers spin on
// an empty queue most of the time, so dequeues constantly race on the dummy,
// help a lagging tail, and retire nodes that other consumers just protected:
// under ASan that's where a reclamation bug turns into a use-after-free.
TEST_P(MichaelScottShapes, ExactlyOnceAndPerProducerOrder) {
    const auto [producers, consumers] = GetParam();
    const std::uint64_t per_producer = scaled(20000);
    const std::uint64_t total = producers * per_producer;
    MichaelScottQueue<std::uint64_t> q;

    std::vector<std::vector<std::atomic<std::uint8_t>>> seen(producers);
    for (auto& row : seen) row = std::vector<std::atomic<std::uint8_t>>(per_producer);
    std::atomic<std::uint64_t> consumed{0};
    std::atomic<std::uint64_t> order_violations{0};
    std::atomic<std::uint64_t> bad_values{0};

    run_stress(producers + consumers, [&](std::size_t index, StressRng& rng) {
        if (index < producers) {
            for (std::uint64_t s = 0; s < per_producer; ++s) {
                q.push(make_tagged(index, s));
                maybe_delay(rng);
            }
            return;
        }
        std::vector<std::uint64_t> last(producers, 0);
        while (consumed.load(std::memory_order_relaxed) < total) {
            auto v = q.try_pop();
            if (!v) continue;
            consumed.fetch_add(1, std::memory_order_relaxed);
            const std::uint64_t p = tag_producer(*v), s = tag_seq(*v);
            if (p >= producers || s >= per_producer) {
                bad_values.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            if (s < last[p]) order_violations.fetch_add(1, std::memory_order_relaxed);
            last[p] = s + 1;
            seen[p][s].fetch_add(1, std::memory_order_relaxed);
            maybe_delay(rng);
        }
    });

    EXPECT_EQ(bad_values.load(), 0u);
    EXPECT_EQ(order_violations.load(), 0u);
    EXPECT_EQ(consumed.load(), total);
    EXPECT_FALSE(q.try_pop().has_value());
    std::uint64_t missing = 0, duplicated = 0;
    for (const auto& row : seen) {
        for (const auto& n : row) {
            const auto c = n.load(std::memory_order_relaxed);
            missing += c == 0;
            duplicated += c > 1;
        }
    }
    EXPECT_EQ(missing, 0u);
    EXPECT_EQ(duplicated, 0u);
}

// Every thread both pushes and pops, so the queue stays short and each node is
// retired moments after it was linked. Blocking pop on the consumer side runs
// the Backoff path too.
TEST_P(MichaelScottShapes, MixedThreadsBlockingPop) {
    const auto [producers, consumers] = GetParam();
    const std::size_t threads = producers + consumers;
    const std::uint64_t per_thread = scaled(10000);
    MichaelScottQueue<std::uint64_t> q;
    std::vector<std::atomic<std::uint8_t>> seen(threads * per_thread);

    run_stress(threads, [&](std::size_t index, StressRng& rng) {
        for (std::uint64_t i = 0; i < per_thread; ++i) {
            q.push(index * per_thread + i);
            // pop() can't hang: this thread's own push is still in the queue
            // or was taken by someone who will push one back.
            const std::uint64_t v = q.pop();
            if (v < seen.size()) seen[v].fetch_add(1, std::memory_order_relaxed);
            maybe_delay(rng);
        }
    });

    EXPECT_FALSE(q.try_pop().has_value());
    std::uint64_t wrong = 0;
    for (const auto& n : seen) wrong += n.load(std::memory_order_relaxed) != 1;
    EXPECT_EQ(wrong, 0u) << "values missing or popped more than once";
}

INSTANTIATE_TEST_SUITE_P(Shapes, MichaelScottShapes, ::testing::ValuesIn(all_shapes()), shape_name);

class MichaelScottTrackedThreads : public TrackedTest {
protected:
    void TearDown() override {
        // Popped values are gone; nodes still waiting in retired lists hold no
        // values, so the live count must already be zero here.
        TrackedTest::TearDown();
        if (HasFailure()) {
            std::fprintf(stderr, "stress seed: LOCKFREE_SEED=%llu\n",
                         static_cast<unsigned long long>(lockfree::test::stress_seed()));
        }
    }
};

// Non-trivial values under contention: each is constructed once, moved out
// once and destroyed once, including the ones left for the queue's destructor.
TEST_F(MichaelScottTrackedThreads, NoLeaksOrDoubleDestroys) {
    constexpr std::size_t threads = 6;
    const std::uint64_t per_thread = scaled(5000);
    std::atomic<std::uint64_t> popped{0};
    {
        MichaelScottQueue<Tracked> q;
        run_stress(threads, [&](std::size_t index, StressRng& rng) {
            for (std::uint64_t i = 0; i < per_thread; ++i) {
                q.emplace(static_cast<int>(index));
                // Pop a bit less than we push, so some are left over.
                if (rng.below(4) != 0) {
                    if (auto v = q.try_pop()) {
                        EXPECT_LT(v->value, static_cast<int>(threads));
                        popped.fetch_add(1, std::memory_order_relaxed);
                    }
                }
                maybe_delay(rng);
            }
        });
        EXPECT_EQ(Tracked::live.load(), static_cast<int>(threads * per_thread - popped.load()));
    }
}

} // namespace
