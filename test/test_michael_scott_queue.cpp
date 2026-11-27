#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "lockfree/hazard_pointer.hpp"
#include "lockfree/michael_scott_queue.hpp"
#include "lockfree/detail/test_hook.hpp"
#include "tracked.hpp"

namespace {

using lockfree::HazardPointer;
using lockfree::MichaelScottQueue;
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

TEST(MichaelScottQueue, ProducersAndConsumersMoveEverythingOnce) {
    constexpr int kThreads = 4;
    constexpr int kPerProducer = 20000;
    MichaelScottQueue<std::uint64_t> q;
    std::vector<std::vector<std::uint64_t>> got(kThreads);
    std::vector<std::thread> ts;
    for (int p = 0; p < kThreads; ++p) {
        ts.emplace_back([&, p] {
            for (int i = 0; i < kPerProducer; ++i) q.push((std::uint64_t(p) << 32) | std::uint64_t(i));
        });
    }
    std::atomic<int> taken{0};
    for (int c = 0; c < kThreads; ++c) {
        ts.emplace_back([&, c] {
            while (taken.load(std::memory_order_relaxed) < kThreads * kPerProducer) {
                if (auto v = q.try_pop()) {
                    got[c].push_back(*v);
                    taken.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    for (auto& t : ts) t.join();

    // Each consumer sees each producer's values in increasing order, and every
    // value shows up exactly once overall.
    std::vector<int> count(kThreads, 0);
    for (const auto& seq : got) {
        std::vector<std::int64_t> last(kThreads, -1);
        for (std::uint64_t v : seq) {
            const auto p = static_cast<int>(v >> 32);
            const auto i = static_cast<std::int64_t>(v & 0xffffffffu);
            EXPECT_GT(i, last[p]);
            last[p] = i;
            ++count[p];
        }
    }
    for (int p = 0; p < kThreads; ++p) EXPECT_EQ(count[p], kPerProducer);
}

} // namespace
