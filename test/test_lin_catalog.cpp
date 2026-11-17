#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "linearizability/checker.hpp"
#include "linearizability/history.hpp"
#include "linearizability/spec.hpp"

// Hand-written histories with known answers, one per behaviour worth naming.
// Each case is checked against a FIFO queue, a LIFO stack, or both. Times are
// small integers; a thread's ops never overlap each other.
//
// Reading a case: push(v, t, invoke, response) / pop(result, t, invoke,
// response), where result `none` means the pop reported empty.

using lockfree::lin::check;
using lockfree::lin::CheckResult;
using lockfree::lin::FifoQueueSpec;
using lockfree::lin::History;
using lockfree::lin::LifoStackSpec;
using lockfree::lin::Operation;
using lockfree::lin::OpKind;
using lockfree::lin::Verdict;

namespace {

constexpr std::optional<std::int64_t> none = std::nullopt;

Operation push(std::int64_t v, std::uint32_t t, std::uint64_t inv, std::uint64_t resp) {
    return {t, OpKind::push, v, std::nullopt, inv, resp};
}

Operation pop(std::optional<std::int64_t> r, std::uint32_t t, std::uint64_t inv,
              std::uint64_t resp) {
    return {t, OpKind::pop, 0, r, inv, resp};
}

struct Case {
    std::string name;
    std::vector<Operation> ops;
    // Expected answer per spec; nullopt = this case isn't about that spec.
    std::optional<bool> fifo;
    std::optional<bool> lifo;
};

// GTest prints this in failure messages and test names.
void PrintTo(const Case& c, std::ostream* os) { *os << c.name; }

std::vector<Case> catalog() {
    return {
        // ---- Both ----------------------------------------------------------
        {"empty_history", {}, true, true},
        {"pop_on_empty_structure", {pop(none, 0, 0, 1)}, true, true},
        {"pop_of_a_value_never_pushed",
         {push(1, 0, 0, 1), pop(2, 1, 2, 3)}, false, false},
        {"same_value_popped_twice",
         {push(1, 0, 0, 1), pop(1, 1, 2, 3), pop(1, 2, 4, 5)}, false, false},
        {"pop_finishes_before_its_push_starts",
         {pop(1, 1, 0, 1), push(1, 0, 2, 3)}, false, false},
        {"pop_overlapping_its_push",
         // The pop may take effect just after the push, inside both intervals.
         {push(1, 0, 0, 10), pop(1, 1, 2, 4)}, true, true},
        {"empty_pop_after_a_completed_push",
         // 1 is in the structure for the whole pop; nobody else took it.
         {push(1, 0, 0, 1), pop(none, 1, 2, 3)}, false, false},
        {"empty_pop_while_another_pop_takes_the_last_item",
         {push(1, 0, 0, 1), pop(1, 1, 2, 10), pop(none, 2, 3, 5)}, true, true},
        {"empty_pop_overlapping_the_only_push",
         {push(1, 0, 0, 10), pop(none, 1, 2, 4), pop(1, 1, 11, 12)}, true, true},
        {"two_sequential_empty_pops_then_the_item_appears",
         // The first empty pop could precede the push, the second can't.
         {push(1, 0, 0, 6), pop(none, 1, 1, 2), pop(none, 1, 7, 8), pop(1, 1, 9, 10)},
         false, false},

        // ---- Queue ---------------------------------------------------------
        {"fifo_sequential",
         {push(1, 0, 0, 1), push(2, 0, 2, 3), pop(1, 1, 4, 5), pop(2, 1, 6, 7)}, true, false},
        {"fifo_overlapping_pushes_dequeue_in_either_order",
         // push 2 overlaps push 1, so 2 may be first in the queue.
         {push(1, 0, 0, 10), push(2, 1, 5, 15), pop(2, 2, 20, 21), pop(1, 2, 22, 23)},
         true, true},
        {"fifo_ordered_pushes_dequeued_backwards",
         // push 1 finished before push 2 started: 2 can't come out first.
         {push(1, 0, 0, 1), push(2, 1, 2, 3), pop(2, 2, 4, 5), pop(1, 2, 6, 7)}, false, true},
        {"fifo_ordered_pushes_dequeued_backwards_in_one_segment",
         // Same as above, but a slow push of 99 overlaps everything, so there's
         // no quiescent point to lean on: the search itself has to respect
         // that push 1 finished before push 2 started.
         {push(99, 3, 0, 100), push(1, 0, 1, 2), push(2, 1, 3, 4), pop(2, 2, 5, 6),
          pop(1, 2, 7, 8), pop(99, 2, 101, 102)},
         false, true},
        {"fifo_pop_overtakes_a_pending_older_value",
         // 1 was enqueued before 2 started, yet the pop returned 2 and a later
         // pop still found 1: 1 should have come out first.
         {push(1, 0, 0, 1), push(2, 0, 2, 3), pop(2, 1, 4, 20), pop(1, 2, 21, 22)},
         false, true},
        {"fifo_concurrent_pops_may_return_in_either_order",
         // Pops overlap each other, so whichever got 1 may have gone first.
         {push(1, 0, 0, 1), push(2, 0, 2, 3), pop(2, 1, 4, 10), pop(1, 2, 5, 11)}, true, true},
        {"fifo_stalled_producer_hides_a_completed_push",
         // What an MPSC consumer can see: producer 0 is stalled mid-push of 1
         // (between its tail exchange and its link store), producer 1's push of
         // 2 has *returned*, yet the consumer reports empty. 2 was in the queue
         // the whole time, so as a FIFO history this is not linearizable,
         // which is exactly the docs/mpsc_queue.md caveat.
         {push(1, 0, 0, 100), push(2, 1, 1, 3), pop(none, 2, 5, 6), pop(1, 2, 101, 102),
          pop(2, 2, 103, 104)},
         false, std::nullopt},
        {"fifo_needs_backtracking_into_an_earlier_quiescent_segment",
         // Segment 1 leaves [1,2] or [2,1]; the later pops only fit [2,1].
         {push(1, 0, 0, 10), push(2, 1, 5, 15), pop(2, 0, 20, 21), pop(1, 0, 22, 23),
          pop(none, 0, 24, 25)},
         true, std::nullopt},
        {"fifo_three_producers_one_consumer",
         // Three overlapping producers, each pushing twice. The consumer's
         // order interleaves them but keeps each producer's own order.
         {push(10, 0, 0, 4), push(20, 1, 1, 3), push(30, 2, 2, 6),
          push(11, 0, 5, 9), push(21, 1, 7, 8), push(31, 2, 10, 12),
          pop(20, 3, 13, 14), pop(10, 3, 15, 16), pop(30, 3, 17, 18),
          pop(21, 3, 19, 20), pop(11, 3, 21, 22), pop(31, 3, 23, 24)},
         true, std::nullopt},
        {"fifo_three_producers_one_consumer_breaks_producer_order",
         // Same, but 11 comes out before 10: producer 0 pushed 10 first.
         {push(10, 0, 0, 4), push(20, 1, 1, 3), push(30, 2, 2, 6),
          push(11, 0, 5, 9), push(21, 1, 7, 8), push(31, 2, 10, 12),
          pop(20, 3, 13, 14), pop(11, 3, 15, 16), pop(30, 3, 17, 18),
          pop(21, 3, 19, 20), pop(10, 3, 21, 22), pop(31, 3, 23, 24)},
         false, std::nullopt},

        // ---- Stack ---------------------------------------------------------
        {"lifo_sequential",
         {push(1, 0, 0, 1), push(2, 0, 2, 3), pop(2, 1, 4, 5), pop(1, 1, 6, 7)}, false, true},
        {"lifo_pop_under_the_top",
         // 2 was pushed after 1 and is still there, so a pop can't return 1.
         {push(1, 0, 0, 1), push(2, 0, 2, 3), pop(1, 1, 4, 5), pop(2, 1, 6, 7)},
         std::nullopt, false},
        {"lifo_pop_under_the_top_in_one_segment",
         // A slow push of 99 keeps this one segment; 99 may land anywhere, but
         // 2 is still above 1 whatever it does.
         {push(99, 3, 0, 100), push(1, 0, 1, 2), push(2, 1, 3, 4), pop(1, 2, 5, 6),
          pop(2, 2, 7, 8), pop(99, 2, 101, 102)},
         true, false},
        {"lifo_pop_before_an_overlapping_push_takes_effect",
         // push 2 is in flight, so the pop may run before it and return 1.
         {push(1, 0, 0, 1), push(2, 1, 2, 10), pop(1, 2, 3, 5), pop(2, 2, 11, 12)},
         std::nullopt, true},
        {"lifo_aba_lost_item",
         // The symptom of the naive Treiber stack's ABA bug: after 3 is
         // popped, the next pop skips straight past 2 to 1, and 2 is never
         // seen again.
         {push(1, 0, 0, 1), push(2, 0, 2, 3), push(3, 0, 4, 5), pop(3, 1, 6, 7),
          pop(1, 1, 8, 9), pop(none, 1, 10, 11)},
         false, false},
        {"lifo_aba_resurrected_item",
         // The other ABA symptom: a popped node comes back.
         {push(1, 0, 0, 1), push(2, 0, 2, 3), pop(2, 1, 4, 5), pop(1, 1, 6, 7),
          pop(2, 1, 8, 9)},
         false, false},
        {"lifo_concurrent_pushes_and_pops",
         // Two threads each push then pop with everything overlapping; every
         // pop returning the other thread's value is a legal outcome.
         {push(1, 0, 0, 4), push(2, 1, 1, 5), pop(2, 0, 6, 9), pop(1, 1, 7, 10)},
         true, true},
    };
}

void expect_verdict(const CheckResult& r, bool expected, const char* spec) {
    if (expected) {
        EXPECT_EQ(r.verdict, Verdict::linearizable) << spec << ": " << r.message;
        return;
    }
    EXPECT_EQ(r.verdict, Verdict::not_linearizable) << spec << ": " << r.message;
    // A rejection always points at something: the stuck segment, and an
    // order through it that's strictly shorter than the segment.
    EXPECT_FALSE(r.failing_segment.empty()) << spec;
    EXPECT_LT(r.best_order.size(), r.failing_segment.size()) << spec;
}

class LinCatalog : public ::testing::TestWithParam<Case> {};

} // namespace

TEST_P(LinCatalog, CheckerGivesTheExpectedAnswer) {
    const Case& c = GetParam();
    History h(c.ops);
    ASSERT_EQ(h.check_well_formed(), std::nullopt) << "the case itself is malformed";
    ASSERT_TRUE(c.fifo || c.lifo) << "case checks nothing";
    if (c.fifo) expect_verdict(check<FifoQueueSpec>(h), *c.fifo, "fifo");
    if (c.lifo) expect_verdict(check<LifoStackSpec>(h), *c.lifo, "lifo");
}

TEST_P(LinCatalog, AnswerDoesNotDependOnInputOrderOrThreadNames) {
    // The checker sorts internally; shuffling the input and renaming threads
    // must not change anything.
    const Case& c = GetParam();
    std::mt19937 rng(1234);
    for (int round = 0; round < 10; ++round) {
        std::vector<Operation> ops = c.ops;
        std::shuffle(ops.begin(), ops.end(), rng);
        for (Operation& op : ops) op.thread = 100 - op.thread;
        History h(ops);
        if (c.fifo) EXPECT_EQ(bool(check<FifoQueueSpec>(h)), *c.fifo) << "round " << round;
        if (c.lifo) EXPECT_EQ(bool(check<LifoStackSpec>(h)), *c.lifo) << "round " << round;
    }
}

INSTANTIATE_TEST_SUITE_P(Histories, LinCatalog, ::testing::ValuesIn(catalog()),
                         [](const ::testing::TestParamInfo<Case>& info) { return info.param.name; });
