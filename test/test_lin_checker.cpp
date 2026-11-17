#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <deque>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

#include "linearizability/checker.hpp"
#include "linearizability/history.hpp"
#include "linearizability/spec.hpp"

using lockfree::lin::check;
using lockfree::lin::CheckOptions;
using lockfree::lin::FifoQueueSpec;
using lockfree::lin::first_illegal;
using lockfree::lin::happens_before;
using lockfree::lin::History;
using lockfree::lin::LifoStackSpec;
using lockfree::lin::Operation;
using lockfree::lin::OpKind;
using lockfree::lin::split_at_quiescent_points;
using lockfree::lin::Verdict;

namespace {

Operation push(std::int64_t v, std::uint64_t inv, std::uint64_t resp, std::uint32_t thread = 0) {
    return {thread, OpKind::push, v, std::nullopt, inv, resp};
}

Operation pop(std::optional<std::int64_t> r, std::uint64_t inv, std::uint64_t resp,
              std::uint32_t thread = 0) {
    return {thread, OpKind::pop, 0, r, inv, resp};
}

// The definition, checked the slow way: try every permutation that respects
// real time. Only usable for a handful of ops, which is the point: it shares
// no code with the real search.
template <typename Spec>
bool brute_force_linearizable(const std::vector<Operation>& ops) {
    std::vector<std::size_t> order(ops.size());
    std::iota(order.begin(), order.end(), 0);
    do {
        bool respects_time = true;
        for (std::size_t a = 0; a < order.size() && respects_time; ++a) {
            for (std::size_t b = a + 1; b < order.size(); ++b) {
                if (happens_before(ops[order[b]], ops[order[a]])) {
                    respects_time = false;
                    break;
                }
            }
        }
        if (!respects_time) continue;
        std::vector<Operation> seq;
        for (std::size_t i : order) seq.push_back(ops[i]);
        if (!first_illegal<Spec>(seq)) return true;
    } while (std::next_permutation(order.begin(), order.end()));
    return false;
}

} // namespace

TEST(LinChecker, EmptyHistoryIsLinearizable) {
    auto r = check<FifoQueueSpec>(History{});
    EXPECT_EQ(r.verdict, Verdict::linearizable);
    EXPECT_EQ(r.segments, 0u);
}

TEST(LinChecker, SequentialHistoryFollowsTheSpec) {
    History ok({push(1, 0, 1), push(2, 2, 3), pop(1, 4, 5), pop(2, 6, 7), pop(std::nullopt, 8, 9)});
    EXPECT_TRUE(check<FifoQueueSpec>(ok));
    EXPECT_FALSE(check<LifoStackSpec>(ok));

    History bad({push(1, 0, 1), push(2, 2, 3), pop(2, 4, 5)});
    EXPECT_FALSE(check<FifoQueueSpec>(bad));
    EXPECT_TRUE(check<LifoStackSpec>(bad));
}

TEST(LinChecker, OverlappingOpsMayTakeEffectInEitherOrder) {
    // Two pushes overlap, so a later pop may see either value first.
    for (std::int64_t first : {1, 2}) {
        const std::int64_t second = 3 - first;
        History h({push(1, 0, 10, 0), push(2, 5, 15, 1), pop(first, 20, 21, 0),
                   pop(second, 22, 23, 0)});
        EXPECT_TRUE(check<FifoQueueSpec>(h)) << "first pop " << first;
    }
}

TEST(LinChecker, RealTimeOrderIsBinding) {
    // push 1 finished before push 2 started: a FIFO must hand 1 out first,
    // even though a reordering would make the pops legal.
    History h({push(1, 0, 1, 0), push(2, 2, 3, 1), pop(2, 4, 5, 0), pop(1, 6, 7, 0)});
    auto r = check<FifoQueueSpec>(h);
    EXPECT_EQ(r.verdict, Verdict::not_linearizable);
}

TEST(LinChecker, EqualStampsCountAsOverlapping) {
    // response == invoke can't be ordered by the clock, so this is fine.
    History h({push(1, 0, 5, 0), push(2, 5, 6, 1), pop(2, 7, 8, 0), pop(1, 9, 10, 0)});
    EXPECT_TRUE(check<FifoQueueSpec>(h));
}

TEST(LinChecker, PopMayTakeEffectBeforeAnOverlappingPush) {
    // The empty pop overlaps the push, so it can be ordered before it.
    History h({push(1, 0, 10, 0), pop(std::nullopt, 2, 4, 1), pop(1, 11, 12, 1)});
    EXPECT_TRUE(check<FifoQueueSpec>(h));

    // But not once the push has finished.
    History late({push(1, 0, 1, 0), pop(std::nullopt, 2, 4, 1), pop(1, 11, 12, 1)});
    EXPECT_FALSE(check<FifoQueueSpec>(late));
}

TEST(LinChecker, SplitsAtQuiescentPoints) {
    History h({push(1, 0, 10, 0), push(2, 5, 15, 1),  // segment 1
               pop(1, 20, 30, 0), pop(2, 25, 26, 1),  // segment 2
               pop(std::nullopt, 31, 32, 0)});        // segment 3
    h.sort_by_invoke();
    auto segments = split_at_quiescent_points(h.ops());
    ASSERT_EQ(segments.size(), 3u);
    EXPECT_EQ(segments[0].size(), 2u);
    EXPECT_EQ(segments[1].size(), 2u);
    EXPECT_EQ(segments[2].size(), 1u);
    EXPECT_EQ(check<FifoQueueSpec>(h).segments, 3u);
}

TEST(LinChecker, BacksUpAcrossQuiescentPoints) {
    // Segment 1 leaves the queue as [1,2] or [2,1]. The search tries push 1
    // first, finds that segment 2 can't follow from [1,2], and has to back up
    // into segment 1 to try the other order.
    History h({push(1, 0, 10, 0), push(2, 5, 15, 1), pop(2, 20, 21, 0), pop(1, 22, 23, 0)});
    auto r = check<FifoQueueSpec>(h);
    EXPECT_EQ(r.segments, 3u);
    EXPECT_TRUE(r) << r.message;
}

TEST(LinChecker, ReportsTheFailingSegmentAndLongestLegalPrefix) {
    History h({push(1, 0, 1), push(2, 2, 3), pop(1, 4, 5), pop(1, 6, 7)});
    auto r = check<FifoQueueSpec>(h);
    ASSERT_EQ(r.verdict, Verdict::not_linearizable);
    // Fully sequential, so each op is its own segment and the duplicate pop
    // is the one that fails.
    ASSERT_EQ(r.failing_segment.size(), 1u);
    EXPECT_EQ(r.failing_segment[0].invoke_ts, 6u);
    EXPECT_TRUE(r.best_order.empty());
    EXPECT_NE(r.message.find("segment 4 of 4"), std::string::npos) << r.message;

    // One segment: three concurrent ops, the pop can't match anything.
    History c({push(1, 0, 10, 0), push(2, 1, 10, 1), pop(3, 2, 10, 2)});
    auto rc = check<FifoQueueSpec>(c);
    ASSERT_EQ(rc.verdict, Verdict::not_linearizable);
    EXPECT_EQ(rc.failing_segment.size(), 3u);
    EXPECT_EQ(rc.best_order.size(), 2u);
}

TEST(LinChecker, RejectsIllFormedHistories) {
    History h({push(1, 10, 5)});
    auto r = check<FifoQueueSpec>(h);
    EXPECT_EQ(r.verdict, Verdict::ill_formed);
    EXPECT_FALSE(r.message.empty());
}

TEST(LinChecker, MemoizationCollapsesEquivalentOrders) {
    // Ten fully overlapping pushes of the same value and a pop of a value
    // nobody pushed. Proving that's not linearizable means ruling out every
    // order: 11! of them for a naive search. But all orders of the same
    // subset of pushes leave the same queue, so memoized there are only
    // 2^10 (subset, state) configurations to visit.
    std::vector<Operation> ops;
    for (std::uint32_t t = 0; t < 10; ++t) ops.push_back(push(5, 0, 100, t));
    ops.push_back(pop(6, 0, 100, 10));
    auto r = check<FifoQueueSpec>(History(ops));
    EXPECT_EQ(r.verdict, Verdict::not_linearizable);
    EXPECT_LE(r.configurations, 1u << 10);
    EXPECT_EQ(r.best_order.size(), 10u);
}

TEST(LinChecker, GivesUpWhenTheBudgetRunsOut) {
    std::vector<Operation> ops;
    for (std::uint32_t t = 0; t < 10; ++t) ops.push_back(push(t, 0, 100, t));
    ops.push_back(pop(42, 0, 100, 10));
    CheckOptions tight;
    tight.max_configurations = 50;
    auto r = check<FifoQueueSpec>(History(ops), tight);
    EXPECT_EQ(r.verdict, Verdict::gave_up);
    EXPECT_FALSE(r.message.empty());

}

TEST(LinChecker, AgreesWithBruteForceOnRandomSmallHistories) {
    // Random little histories over a few threads with random overlaps and
    // results, so plenty of each verdict. Every one is also decided by trying
    // all permutations.
    std::mt19937_64 rng(20261117);
    int linearizable = 0;
    int not_linearizable = 0;
    for (int round = 0; round < 1500; ++round) {
        const int n = std::uniform_int_distribution<int>(1, 7)(rng);
        std::vector<Operation> ops;
        std::vector<std::uint64_t> thread_clock(3, 0);
        std::int64_t next_value = 1;
        for (int i = 0; i < n; ++i) {
            const auto t = std::uniform_int_distribution<std::uint32_t>(0, 2)(rng);
            const std::uint64_t inv = thread_clock[t] + std::uniform_int_distribution<int>(0, 3)(rng);
            const std::uint64_t resp = inv + std::uniform_int_distribution<int>(0, 6)(rng);
            thread_clock[t] = resp + 1;
            if (rng() % 2) {
                ops.push_back(push(next_value++, inv, resp, t));
            } else {
                std::optional<std::int64_t> r;
                if (next_value > 1 && rng() % 4) {
                    r = std::uniform_int_distribution<std::int64_t>(1, next_value - 1)(rng);
                }
                ops.push_back(pop(r, inv, resp, t));
            }
        }
        History h(ops);
        h.sort_by_invoke();
        const auto& sorted = h.ops();

        const bool fifo = brute_force_linearizable<FifoQueueSpec>(sorted);
        const bool lifo = brute_force_linearizable<LifoStackSpec>(sorted);
        ASSERT_EQ(bool(check<FifoQueueSpec>(h)), fifo) << "round " << round;
        ASSERT_EQ(bool(check<LifoStackSpec>(h)), lifo) << "round " << round;
        (fifo ? linearizable : not_linearizable)++;
    }
    // Make sure the generator isn't degenerate.
    EXPECT_GT(linearizable, 200);
    EXPECT_GT(not_linearizable, 200);
}

namespace {

// A long history that's linearizable by construction: ops take effect on a
// real queue at increasing instants, and each op's interval is stretched
// around its instant. Threads never overlap themselves. `max_depth` keeps the
// queue short, the way a stress run with live consumers does.
std::vector<Operation> synthetic_queue_history(std::uint64_t seed, int threads, int ops_count,
                                               int width, int max_depth) {
    std::mt19937_64 rng(seed);
    std::deque<std::int64_t> q;
    std::int64_t next_value = 0;
    std::vector<std::uint64_t> busy_until(threads, 0);
    std::vector<Operation> ops;
    std::uint64_t now = 1;
    for (int i = 0; i < ops_count; ++i) {
        now += rng() % 3;
        const auto t = static_cast<std::uint32_t>(rng() % threads);
        if (busy_until[t] >= now) now = busy_until[t] + 1;
        const std::uint64_t inv = std::max(busy_until[t] + 1, now - std::min<std::uint64_t>(now - 1, rng() % width));
        const std::uint64_t resp = now + rng() % width;
        busy_until[t] = resp;
        if (static_cast<int>(q.size()) <= static_cast<int>(rng() % max_depth)) {
            q.push_back(++next_value);
            ops.push_back(push(next_value, inv, resp, t));
        } else {
            std::optional<std::int64_t> r;
            if (!q.empty()) {
                r = q.front();
                q.pop_front();
            }
            ops.push_back(pop(r, inv, resp, t));
        }
    }
    return ops;
}

} // namespace

TEST(LinChecker, LongConcurrentHistoryChecksQuickly) {
    // 8 threads with up to ~10 ops in flight: the search stays close to one
    // configuration per op instead of exploring the window's permutations.
    auto ops = synthetic_queue_history(7, 8, 5000, 10, 4);
    auto r = check<FifoQueueSpec>(History(ops));
    ASSERT_TRUE(r) << r.message;
    EXPECT_LT(r.configurations, 20u * ops.size());

    // Swap two pops' results, picking b so that the value it got was pushed
    // only after a had already returned. After the swap a "returns" a value
    // nobody had pushed yet, which no reordering can explain.
    auto pushed_at = [&](std::int64_t v) {
        return *std::find_if(ops.begin(), ops.end(), [&](const Operation& op) {
            return op.kind == OpKind::push && op.value == v;
        });
    };
    std::size_t a = 0;
    while (ops[a].kind != OpKind::pop || !ops[a].result) ++a;
    std::size_t b = a + 1;
    while (ops[b].kind != OpKind::pop || !ops[b].result ||
           !happens_before(ops[a], pushed_at(*ops[b].result))) {
        ++b;
    }
    std::swap(ops[a].result, ops[b].result);
    auto bad = check<FifoQueueSpec>(History(ops));
    EXPECT_EQ(bad.verdict, Verdict::not_linearizable);
    EXPECT_FALSE(bad.failing_segment.empty());
}
