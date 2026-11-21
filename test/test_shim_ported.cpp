#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "lockfree/atomic_stack.hpp"
#include "lockfree/mpmc_queue.hpp"
#include "lockfree/mpsc_queue.hpp"
#include "lockfree/naive/treiber_stack.hpp"
#include "lockfree/spsc_ring_buffer.hpp"

// Only built with LOCKFREE_SCHEDULER. Checks that the structures really go
// through detail::atomic: every operation has to reach the yield hook, or the
// scheduler would have no say over the interleavings inside it. The full unit
// suites also run in this mode (the *_sched tests) to show the wrapper doesn't
// change behaviour; this file checks the yields themselves.

#if !defined(LOCKFREE_SCHEDULER)
#error "test_shim_ported is meant for the LOCKFREE_SCHEDULER build"
#endif

namespace {

using lockfree::detail::AtomicOp;
using lockfree::detail::YieldPoint;

// Single-threaded tests, so plain globals are fine here.
std::vector<YieldPoint> g_points;

void record(const YieldPoint& p) noexcept { g_points.push_back(p); }

class ShimPorted : public ::testing::Test {
protected:
    void SetUp() override {
        g_points.clear();
        g_points.reserve(1024);
        lockfree::detail::yield_hook() = &record;
    }
    void TearDown() override { lockfree::detail::yield_hook() = nullptr; }

    // Yields recorded since the last call, then forgets them.
    std::vector<YieldPoint> take() { return std::exchange(g_points, {}); }

    // True if at least one yield point touched memory inside `obj`.
    template <typename Obj>
    static bool any_inside(const std::vector<YieldPoint>& pts, const Obj& obj) {
        auto lo = reinterpret_cast<std::uintptr_t>(&obj);
        auto hi = lo + sizeof(Obj);
        for (const auto& p : pts) {
            auto a = reinterpret_cast<std::uintptr_t>(p.address);
            if (a >= lo && a < hi) return true;
        }
        return false;
    }

    static std::set<AtomicOp> ops(const std::vector<YieldPoint>& pts) {
        std::set<AtomicOp> s;
        for (const auto& p : pts) s.insert(p.op);
        return s;
    }

    // The source location should name the structure's header, not the shim.
    static bool all_from(const std::vector<YieldPoint>& pts, const std::string& header) {
        for (const auto& p : pts)
            if (std::string(p.where.file_name()).find(header) == std::string::npos) return false;
        return !pts.empty();
    }
};

TEST_F(ShimPorted, SpscYieldsOnPushAndPop) {
    lockfree::SpscRingBuffer<int, 4> q;
    take(); // construction

    ASSERT_TRUE(q.try_push(1));
    auto push = take();
    EXPECT_TRUE(any_inside(push, q));
    EXPECT_TRUE(ops(push).count(AtomicOp::store));
    EXPECT_TRUE(all_from(push, "spsc_ring_buffer.hpp"));

    EXPECT_EQ(q.try_pop(), std::optional<int>(1));
    auto pop = take();
    EXPECT_TRUE(any_inside(pop, q));
    EXPECT_TRUE(ops(pop).count(AtomicOp::load));
    EXPECT_TRUE(all_from(pop, "spsc_ring_buffer.hpp"));
}

TEST_F(ShimPorted, MpmcYieldsOnPositionCasAndCellSequence) {
    lockfree::MpmcQueue<int, 4> q;
    take();

    ASSERT_TRUE(q.try_push(7));
    auto push = take();
    EXPECT_TRUE(any_inside(push, q));
    EXPECT_TRUE(ops(push).count(AtomicOp::compare_exchange));
    EXPECT_TRUE(all_from(push, "mpmc_queue.hpp"));

    int out = 0;
    ASSERT_TRUE(q.try_pop(out));
    EXPECT_EQ(out, 7);
    auto pop = take();
    EXPECT_TRUE(ops(pop).count(AtomicOp::compare_exchange));
    EXPECT_TRUE(ops(pop).count(AtomicOp::store)); // releasing the cell for the next lap
}

TEST_F(ShimPorted, MpscYieldsOnTailExchangeAndLink) {
    lockfree::MpscQueue<int> q;
    take();

    q.push(3);
    auto push = take();
    EXPECT_TRUE(any_inside(push, q));
    EXPECT_TRUE(ops(push).count(AtomicOp::exchange));
    EXPECT_TRUE(ops(push).count(AtomicOp::store)); // prev->next = node
    EXPECT_TRUE(all_from(push, "mpsc_queue.hpp"));

    EXPECT_EQ(q.try_pop(), std::optional<int>(3));
    EXPECT_FALSE(take().empty());
}

TEST_F(ShimPorted, AtomicStackYieldsThroughTaggedHead) {
    lockfree::AtomicStack<int> s;
    take();

    s.push(5);
    auto push = take();
    EXPECT_TRUE(any_inside(push, s));
    EXPECT_TRUE(ops(push).count(AtomicOp::compare_exchange));
    // The head CAS lives in tagged_ptr.hpp; the node's next store in the stack.
    std::set<std::string> files;
    for (const auto& p : push) {
        std::string f = p.where.file_name();
        files.insert(f.substr(f.find_last_of('/') + 1));
    }
    EXPECT_TRUE(files.count("tagged_ptr.hpp"));

    EXPECT_EQ(s.try_pop(), std::optional<int>(5));
    EXPECT_TRUE(ops(take()).count(AtomicOp::compare_exchange));
}

TEST_F(ShimPorted, NaiveTreiberStackYieldsOnHeadCas) {
    lockfree::naive::TreiberStack<int> s;
    take();

    s.push(9);
    auto push = take();
    EXPECT_TRUE(any_inside(push, s));
    EXPECT_TRUE(ops(push).count(AtomicOp::compare_exchange));
    EXPECT_TRUE(all_from(push, "treiber_stack.hpp"));

    EXPECT_EQ(s.try_pop(), std::optional<int>(9));
    auto pop = take();
    // pop reads head, then head->next, then CASes: the load of next is the
    // spot the ABA test preempts at, so it has to be a yield point too.
    EXPECT_GE(pop.size(), 3u);
    EXPECT_TRUE(ops(pop).count(AtomicOp::compare_exchange));
}

TEST_F(ShimPorted, NoHookMeansNoCallbacks) {
    lockfree::detail::yield_hook() = nullptr;
    lockfree::MpmcQueue<int, 2> q;
    ASSERT_TRUE(q.try_push(1));
    int out = 0;
    ASSERT_TRUE(q.try_pop(out));
    EXPECT_TRUE(g_points.empty());
}

} // namespace
