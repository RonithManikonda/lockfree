#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "lockfree/mpsc_queue.hpp"
#include "stress_harness.hpp"
#include "tracked.hpp"

namespace lockfree {

// Lets a test stop a push between its two steps, which is otherwise a window a
// few instructions wide that a test can't hit on purpose.
struct MpscIntrusiveQueueTestAccess {
    static MpscNode* swing_tail(MpscIntrusiveQueue& q, MpscNode* node) {
        return q.swing_tail(node);
    }
    static void link(MpscNode* prev, MpscNode* node) { MpscIntrusiveQueue::link(prev, node); }
    static void push_stub(MpscIntrusiveQueue& q) { q.push(&q.stub_); }
};

} // namespace lockfree

namespace {

using lockfree::MpscIntrusiveQueue;
using lockfree::MpscNode;
using Access = lockfree::MpscIntrusiveQueueTestAccess;
using lockfree::MpscQueue;
using lockfree::test::Tracked;

struct Item : MpscNode {
    explicit Item(std::uint64_t v = 0) : value(v) {}
    std::uint64_t value;
};

std::uint64_t value_of(MpscNode* n) { return static_cast<Item*>(n)->value; }

TEST(MpscIntrusive, PopOnNewQueueIsEmpty) {
    MpscIntrusiveQueue q;
    EXPECT_EQ(q.pop(), nullptr);
    EXPECT_EQ(q.pop(), nullptr);
}

TEST(MpscIntrusive, SingleItemRoundTrip) {
    MpscIntrusiveQueue q;
    Item a(1);
    q.push(&a);
    EXPECT_EQ(q.pop(), &a);
    EXPECT_EQ(q.pop(), nullptr);
}

TEST(MpscIntrusive, FifoOrder) {
    MpscIntrusiveQueue q;
    std::vector<Item> items(100);
    for (std::uint64_t i = 0; i < 100; ++i) {
        items[i].value = i;
        q.push(&items[i]);
    }
    for (std::uint64_t i = 0; i < 100; ++i) {
        MpscNode* n = q.pop();
        ASSERT_NE(n, nullptr);
        EXPECT_EQ(value_of(n), i);
    }
    EXPECT_EQ(q.pop(), nullptr);
}

// Draining to empty is the only time the stub goes back in, so alternate
// between going empty and having items waiting behind it.
TEST(MpscIntrusive, StubCyclesThroughRepeatedly) {
    MpscIntrusiveQueue q;
    Item a(1), b(2);
    for (int i = 0; i < 1000; ++i) {
        q.push(&a);
        ASSERT_EQ(q.pop(), &a);
        ASSERT_EQ(q.pop(), nullptr);
        q.push(&a);
        q.push(&b);
        ASSERT_EQ(q.pop(), &a);
        ASSERT_EQ(q.pop(), &b);
        ASSERT_EQ(q.pop(), nullptr);
    }
}

TEST(MpscIntrusive, PushWhileConsumerIsPartWayThrough) {
    MpscIntrusiveQueue q;
    Item a(1), b(2), c(3);
    q.push(&a);
    q.push(&b);
    ASSERT_EQ(q.pop(), &a);
    q.push(&c);
    EXPECT_EQ(q.pop(), &b);
    EXPECT_EQ(q.pop(), &c);
    EXPECT_EQ(q.pop(), nullptr);
}

TEST(MpscIntrusive, PoppedNodeCanBePushedAgain) {
    MpscIntrusiveQueue q;
    Item a(1), b(2);
    q.push(&a);
    q.push(&b);
    MpscNode* first = q.pop();
    ASSERT_EQ(first, &a);
    q.push(first);
    EXPECT_EQ(q.pop(), &b);
    EXPECT_EQ(q.pop(), &a);
    EXPECT_EQ(q.pop(), nullptr);
}

// The documented transient: a producer between its exchange and its link makes
// the queue look empty from that point on. Finishing the push reveals it all.
TEST(MpscIntrusive, UnlinkedPushHidesItsItem) {
    MpscIntrusiveQueue q;
    Item a(1);
    MpscNode* prev = Access::swing_tail(q, &a);
    EXPECT_EQ(q.pop(), nullptr);

    Access::link(prev, &a);
    EXPECT_EQ(q.pop(), &a);
    EXPECT_EQ(q.pop(), nullptr);
}

TEST(MpscIntrusive, UnlinkedPushHidesEverythingBehindIt) {
    MpscIntrusiveQueue q;
    Item a(1), b(2), c(3);
    MpscNode* prev = Access::swing_tail(q, &a);

    // Two complete pushes after the stalled one still can't be reached.
    q.push(&b);
    q.push(&c);
    EXPECT_EQ(q.pop(), nullptr);

    Access::link(prev, &a);
    EXPECT_EQ(q.pop(), &a);
    EXPECT_EQ(q.pop(), &b);
    EXPECT_EQ(q.pop(), &c);
    EXPECT_EQ(q.pop(), nullptr);
}

// The stall can start with items already reachable: those still come out, and
// the last one before the gap has to wait, since it has no successor yet.
TEST(MpscIntrusive, UnlinkedPushLeavesItsPredecessorStuck) {
    MpscIntrusiveQueue q;
    Item a(1), b(2), c(3);
    q.push(&a);
    q.push(&b);
    MpscNode* prev = Access::swing_tail(q, &c);

    EXPECT_EQ(q.pop(), &a);
    EXPECT_EQ(q.pop(), nullptr) << "b has no successor, so it can't leave yet";

    Access::link(prev, &c);
    EXPECT_EQ(q.pop(), &b);
    EXPECT_EQ(q.pop(), &c);
    EXPECT_EQ(q.pop(), nullptr);
}

// A producer can swing tail_ right after the consumer checked it and before the
// consumer pushes the stub, so the stub lands behind the stalled node. That race
// sits inside pop, so build the state it leaves directly: pop must skip the stub
// once the link lands.
TEST(MpscIntrusive, StubBehindAStalledProducerIsSkipped) {
    MpscIntrusiveQueue q;
    Item a(1), b(2), c(3);
    q.push(&a);
    MpscNode* prev = Access::swing_tail(q, &b);
    EXPECT_EQ(q.pop(), nullptr);  // head_ now sits on a, which has no successor
    Access::push_stub(q);
    EXPECT_EQ(q.pop(), nullptr);

    Access::link(prev, &b);
    EXPECT_EQ(q.pop(), &a);
    EXPECT_EQ(q.pop(), &b);
    EXPECT_EQ(q.pop(), nullptr);

    q.push(&c);
    EXPECT_EQ(q.pop(), &c);
    EXPECT_EQ(q.pop(), nullptr);
}

class MpscIntrusiveStress : public lockfree::test::StressTest {};

// A quick threaded check of the core; the full producer sweep runs on the
// allocating wrapper.
TEST_F(MpscIntrusiveStress, ProducersDeliverExactlyOnceInOrder) {
    constexpr std::size_t producers = 4;
    const std::uint64_t per_producer = lockfree::test::scaled(50000);

    // value = per_producer * producer + seq
    std::vector<Item> items(producers * per_producer);
    for (std::uint64_t i = 0; i < items.size(); ++i) items[i].value = i;

    MpscIntrusiveQueue q;
    std::vector<std::uint64_t> next(producers, 0);
    std::uint64_t popped = 0, order_violations = 0;

    lockfree::test::run_stress(producers + 1, [&](std::size_t index,
                                                  lockfree::test::StressRng& rng) {
        if (index < producers) {
            for (std::uint64_t s = 0; s < per_producer; ++s) {
                q.push(&items[index * per_producer + s]);
                lockfree::test::maybe_delay(rng);
            }
            return;
        }
        while (popped < items.size()) {
            MpscNode* n = q.pop();
            if (n == nullptr) {
                std::this_thread::yield();
                continue;
            }
            const std::uint64_t v = value_of(n);
            const std::uint64_t p = v / per_producer, s = v % per_producer;
            if (s != next[p]) ++order_violations;
            next[p] = s + 1;
            ++popped;
            lockfree::test::maybe_delay(rng);
        }
    });

    EXPECT_EQ(popped, items.size());
    EXPECT_EQ(order_violations, 0u);
    EXPECT_EQ(q.pop(), nullptr);
}

TEST(MpscQueue, TryPopOnEmptyFails) {
    MpscQueue<int> q;
    int out = 7;
    EXPECT_FALSE(q.try_pop(out));
    EXPECT_EQ(out, 7);
    EXPECT_EQ(q.try_pop(), std::nullopt);
}

TEST(MpscQueue, FifoThroughBothPopForms) {
    MpscQueue<int> q;
    for (int i = 0; i < 6; ++i) q.push(i);
    int out = -1;
    ASSERT_TRUE(q.try_pop(out));
    EXPECT_EQ(out, 0);
    EXPECT_EQ(q.try_pop(), 1);
    EXPECT_EQ(q.pop(), 2);
    q.pop(out);
    EXPECT_EQ(out, 3);
    EXPECT_EQ(q.try_pop(), 4);
    EXPECT_EQ(q.try_pop(), 5);
    EXPECT_EQ(q.try_pop(), std::nullopt);
}

TEST(MpscQueue, EmplaceForwardsConstructorArgs) {
    MpscQueue<std::string> q;
    q.emplace(3u, 'x');
    EXPECT_EQ(q.try_pop(), "xxx");
}

TEST(MpscQueue, PushCopiesAnLvalue) {
    MpscQueue<std::string> q;
    const std::string s = "kept";
    q.push(s);
    EXPECT_EQ(s, "kept");
    EXPECT_EQ(q.try_pop(), "kept");
}

TEST(MpscQueue, MoveOnlyType) {
    MpscQueue<std::unique_ptr<int>> q;
    q.push(std::make_unique<int>(1));
    q.emplace(std::make_unique<int>(2));

    auto first = q.try_pop();
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(**first, 1);

    std::unique_ptr<int> second;
    ASSERT_TRUE(q.try_pop(second));
    EXPECT_EQ(*second, 2);
}

// A type with no default constructor and no assignment still has to work
// through the optional-returning and blocking pops.
TEST(MpscQueue, NonAssignableType) {
    struct Fixed {
        explicit Fixed(int v) : value(v) {}
        Fixed(Fixed&&) noexcept = default;
        Fixed& operator=(Fixed&&) = delete;
        int value;
    };
    MpscQueue<Fixed> q;
    q.emplace(1);
    q.emplace(2);
    EXPECT_EQ(q.try_pop()->value, 1);
    EXPECT_EQ(q.pop().value, 2);
}

TEST(MpscQueue, ThrowingConstructorLeavesQueueUnchanged) {
    struct Picky {
        explicit Picky(int v) : value(v) {
            if (v < 0) throw std::invalid_argument("negative");
        }
        Picky(Picky&&) noexcept = default;
        int value;
    };
    MpscQueue<Picky> q;
    q.emplace(1);
    EXPECT_THROW(q.emplace(-1), std::invalid_argument);
    q.emplace(2);
    EXPECT_EQ(q.try_pop()->value, 1);
    EXPECT_EQ(q.try_pop()->value, 2);
    EXPECT_EQ(q.try_pop(), std::nullopt);
}

class MpscQueueLifetime : public lockfree::test::TrackedTest {};

TEST_F(MpscQueueLifetime, PopDestroysTheNodesCopy) {
    MpscQueue<Tracked> q;
    q.emplace(5);
    EXPECT_EQ(Tracked::live.load(), 1);
    {
        auto v = q.try_pop();
        ASSERT_TRUE(v.has_value());
        EXPECT_EQ(v->value, 5);
        EXPECT_EQ(Tracked::live.load(), 1) << "only the popped value is left";
    }
    EXPECT_EQ(Tracked::live.load(), 0);
}

TEST_F(MpscQueueLifetime, TryPopIntoExistingObject) {
    MpscQueue<Tracked> q;
    q.emplace(5);
    Tracked out(0);
    ASSERT_TRUE(q.try_pop(out));
    EXPECT_EQ(out.value, 5);
    EXPECT_EQ(Tracked::live.load(), 1);
}

TEST_F(MpscQueueLifetime, DestructorDrainsWhatsLeft) {
    {
        MpscQueue<Tracked> q;
        for (int i = 0; i < 100; ++i) q.emplace(i);
        for (int i = 0; i < 40; ++i) q.try_pop();
        EXPECT_EQ(Tracked::live.load(), 60);
    }
    EXPECT_EQ(Tracked::live.load(), 0);
}

// The stub is the last thing in the list after a drain; a destructor that went
// wrong here would try to delete it, which ASan would catch.
TEST_F(MpscQueueLifetime, DestructorAfterFullDrain) {
    MpscQueue<Tracked> q;
    q.emplace(1);
    q.emplace(2);
    q.try_pop();
    q.try_pop();
    EXPECT_FALSE(q.try_pop().has_value());
}

// Producers x one consumer: every value tagged (producer, seq). The consumer
// is the only thread reading the tallies, so they don't need to be atomic.
class MpscProducers : public lockfree::test::StressTest,
                      public ::testing::WithParamInterface<std::size_t> {};

TEST_P(MpscProducers, ExactlyOnceAndPerProducerOrder) {
    const std::size_t producers = GetParam();
    const std::uint64_t per_producer = lockfree::test::scaled(20000);
    const std::uint64_t total = producers * per_producer;

    MpscQueue<std::uint64_t> q;
    std::vector<std::uint8_t> seen(total, 0);
    std::vector<std::uint64_t> next(producers, 0);
    std::uint64_t popped = 0, order_violations = 0, bad_values = 0;

    lockfree::test::run_stress(producers + 1, [&](std::size_t index,
                                                  lockfree::test::StressRng& rng) {
        if (index < producers) {
            for (std::uint64_t s = 0; s < per_producer; ++s) {
                q.push(index * per_producer + s);
                lockfree::test::maybe_delay(rng);
            }
            return;
        }
        while (popped < total) {
            auto v = q.try_pop();
            if (!v) {
                std::this_thread::yield();
                continue;
            }
            ++popped;
            if (*v >= total) {
                ++bad_values;
                continue;
            }
            const std::uint64_t p = *v / per_producer, s = *v % per_producer;
            if (s != next[p]) ++order_violations;
            next[p] = s + 1;
            ++seen[*v];
            lockfree::test::maybe_delay(rng);
        }
    });

    EXPECT_EQ(popped, total);
    EXPECT_EQ(bad_values, 0u);
    EXPECT_EQ(order_violations, 0u);
    EXPECT_EQ(q.try_pop(), std::nullopt);

    std::uint64_t wrong = 0;
    for (auto n : seen) wrong += n != 1;
    EXPECT_EQ(wrong, 0u) << "values missing or popped more than once";
}

// Same, through the blocking pop, with the consumer told how many to expect.
TEST_P(MpscProducers, BlockingPopExactlyOnce) {
    const std::size_t producers = GetParam();
    const std::uint64_t per_producer = lockfree::test::scaled(10000);
    const std::uint64_t total = producers * per_producer;

    MpscQueue<std::uint64_t> q;
    std::vector<std::uint8_t> seen(total, 0);

    lockfree::test::run_stress(producers + 1, [&](std::size_t index,
                                                  lockfree::test::StressRng& rng) {
        if (index < producers) {
            for (std::uint64_t s = 0; s < per_producer; ++s) {
                q.push(index * per_producer + s);
                lockfree::test::maybe_delay(rng);
            }
            return;
        }
        for (std::uint64_t i = 0; i < total; ++i) {
            const std::uint64_t v = q.pop();
            if (v < total) ++seen[v];
        }
    });

    EXPECT_EQ(q.try_pop(), std::nullopt);
    std::uint64_t wrong = 0;
    for (auto n : seen) wrong += n != 1;
    EXPECT_EQ(wrong, 0u) << "values missing or popped more than once";
}

// Slow producers keep the queue near empty, so the consumer keeps pushing the
// stub while producers are pushing too. That's the path with the most ways to
// go wrong, and a fast producer loop rarely gets there.
TEST_P(MpscProducers, NearlyEmptyQueueKeepsCyclingTheStub) {
    const std::size_t producers = GetParam();
    const std::uint64_t per_producer = lockfree::test::scaled(3000);
    const std::uint64_t total = producers * per_producer;

    MpscQueue<std::uint64_t> q;
    std::vector<std::uint64_t> next(producers, 0);
    std::uint64_t popped = 0, order_violations = 0;

    lockfree::test::run_stress(producers + 1, [&](std::size_t index,
                                                  lockfree::test::StressRng& rng) {
        if (index < producers) {
            for (std::uint64_t s = 0; s < per_producer; ++s) {
                q.push(index * per_producer + s);
                if (rng.below(4) == 0) std::this_thread::yield();
            }
            return;
        }
        while (popped < total) {
            auto v = q.try_pop();
            if (!v) continue;  // no yield: stay ahead of the producers
            const std::uint64_t p = *v / per_producer, s = *v % per_producer;
            if (s != next[p]) ++order_violations;
            next[p] = s + 1;
            ++popped;
        }
    });

    EXPECT_EQ(popped, total);
    EXPECT_EQ(order_violations, 0u);
}

INSTANTIATE_TEST_SUITE_P(Shapes, MpscProducers, ::testing::Values(1, 2, 3, 4, 8),
                         [](const auto& info) { return std::to_string(info.param) + "p"; });

class MpscQueueThreadedLifetime : public lockfree::test::TrackedTest {};

// The consumer stops early, so the destructor frees the rest. Tracked's counters
// catch leaks and double destroys, and ASan catches a bad free.
TEST_F(MpscQueueThreadedLifetime, ProducersRaceConsumerThenDestructorDrains) {
    constexpr std::size_t producers = 4;
    const std::uint64_t per_producer = lockfree::test::scaled(5000);
    const std::uint64_t to_pop = producers * per_producer / 2;
    {
        MpscQueue<Tracked> q;
        lockfree::test::run_stress(producers + 1, [&](std::size_t index,
                                                      lockfree::test::StressRng&) {
            if (index < producers) {
                for (std::uint64_t s = 0; s < per_producer; ++s) {
                    q.emplace(static_cast<int>(s));
                }
                return;
            }
            for (std::uint64_t popped = 0; popped < to_pop;) {
                if (q.try_pop()) ++popped;
                else std::this_thread::yield();
            }
        });
        EXPECT_EQ(static_cast<std::uint64_t>(Tracked::live.load()),
                  producers * per_producer - to_pop);
    }
    EXPECT_EQ(Tracked::live.load(), 0);
}

} // namespace
