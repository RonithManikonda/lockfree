#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

#include "lockfree/mpsc_queue.hpp"
#include "stress_harness.hpp"

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

} // namespace
