#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "linearizability/checker.hpp"
#include "linearizability/history.hpp"
#include "linearizability/log_format.hpp"
#include "linearizability/minimize.hpp"
#include "linearizability/recorder.hpp"
#include "linearizability/spec.hpp"
#include "lockfree/mpsc_queue.hpp"
#include "stress_harness.hpp"

using lockfree::lin::check;
using lockfree::lin::FifoQueueSpec;
using lockfree::lin::History;
using lockfree::lin::LifoStackSpec;
using lockfree::lin::minimize;
using lockfree::lin::Operation;
using lockfree::lin::OpKind;
using lockfree::lin::read_log;
using lockfree::lin::Recorder;
using lockfree::lin::Verdict;
using lockfree::lin::write_log;

namespace {

constexpr std::optional<std::int64_t> none = std::nullopt;

Operation push(std::int64_t v, std::uint32_t t, std::uint64_t inv, std::uint64_t resp) {
    return {t, OpKind::push, v, std::nullopt, inv, resp};
}

Operation pop(std::optional<std::int64_t> r, std::uint32_t t, std::uint64_t inv,
              std::uint64_t resp) {
    return {t, OpKind::pop, 0, r, inv, resp};
}

bool same_op(const Operation& a, const Operation& b) {
    return a.thread == b.thread && a.kind == b.kind && a.value == b.value &&
           a.result == b.result && a.invoke_ts == b.invoke_ts && a.response_ts == b.response_ts;
}

History parse(const std::string& text) {
    std::istringstream in(text);
    auto r = read_log(in);
    EXPECT_FALSE(r.error) << r.error->line << ": " << r.error->message;
    return r.history;
}

std::string parse_error(const std::string& text) {
    std::istringstream in(text);
    auto r = read_log(in);
    if (!r.error) return "";
    return std::to_string(r.error->line) + ": " + r.error->message;
}

// Every value or empty pop left in a minimized window must be needed:
// dropping it makes the window pass.
template <typename Spec>
void expect_one_minimal(const History& window) {
    std::set<std::int64_t> values;
    for (const Operation& op : window.ops()) {
        if (op.kind == OpKind::push) values.insert(op.value);
        if (op.kind == OpKind::pop && op.result) values.insert(*op.result);
    }
    for (std::int64_t v : values) {
        History rest;
        for (const Operation& op : window.ops()) {
            const bool is_v = (op.kind == OpKind::push && op.value == v) ||
                              (op.kind == OpKind::pop && op.result == v);
            if (!is_v) rest.add(op);
        }
        EXPECT_EQ(check<Spec>(rest).verdict, Verdict::linearizable) << "value " << v << " isn't needed";
    }
}

TEST(LinLog, ParsesEveryFieldAndSkipsCommentsAndBlankLines) {
    const History h = parse(
        "# header\n"
        "\n"
        "0 push 7 100 140   # trailing comment\n"
        "  1\tpop 7 120 180\n"
        "3 pop empty 190 200\n"
        "2 push -5 1 2\n");
    ASSERT_EQ(h.size(), 4u);
    EXPECT_TRUE(same_op(h[0], push(7, 0, 100, 140)));
    EXPECT_TRUE(same_op(h[1], pop(7, 1, 120, 180)));
    EXPECT_TRUE(same_op(h[2], pop(none, 3, 190, 200)));
    EXPECT_TRUE(same_op(h[3], push(-5, 2, 1, 2)));
}

TEST(LinLog, WriteThenReadRoundTrips) {
    const History h({push(1, 0, 10, 20), pop(1, 1, 15, 30), pop(none, 1, 31, 40),
                     push(-9'000'000'000'000, 2, 0, 18'000'000'000'000'000'000ull)});
    std::ostringstream out;
    write_log(out, h);
    const History back = parse(out.str());
    ASSERT_EQ(back.size(), h.size());
    for (std::size_t i = 0; i < h.size(); ++i) EXPECT_TRUE(same_op(back[i], h[i])) << i;
}

TEST(LinLog, ReportsTheFirstBadLineByNumber) {
    EXPECT_EQ(parse_error("0 push 1 1 2\n0 push 1 2\n"), "2: expected 5 fields (thread kind value invoke response), got 4");
    EXPECT_EQ(parse_error("0 peek 1 1 2\n"), "1: unknown op 'peek' (push or pop)");
    EXPECT_EQ(parse_error("x push 1 1 2\n"), "1: bad thread id 'x'");
    EXPECT_EQ(parse_error("0 push empty 1 2\n"), "1: bad push value 'empty'");
    EXPECT_EQ(parse_error("0 pop 12x 1 2\n"), "1: bad pop result '12x' (a number or 'empty')");
    EXPECT_EQ(parse_error("0 pop 1 -1 2\n"), "1: bad invoke time '-1'");
    EXPECT_EQ(parse_error("# ok\n0 pop 1 1 2.5\n"), "2: bad response time '2.5'");
    EXPECT_EQ(parse_error("0 push 99999999999999999999 1 2\n"),
              "1: bad push value '99999999999999999999'");
}

TEST(LinMinimize, LinearizableHistoryHasNoWindow) {
    const History h({push(1, 0, 0, 10), push(2, 1, 5, 15), pop(2, 2, 20, 30), pop(1, 2, 31, 40)});
    EXPECT_FALSE(minimize<FifoQueueSpec>(h));
}

TEST(LinMinimize, DropsUnrelatedValuesAroundAQueueReorder) {
    // 3 is pushed strictly before 4 but popped after it. 1, 2, 5, 6 are noise.
    const History h({push(1, 0, 100, 110), push(2, 1, 105, 115), pop(1, 2, 120, 130),
                     pop(2, 2, 135, 140), push(3, 0, 150, 160), push(4, 0, 170, 180),
                     push(5, 1, 175, 185), pop(4, 2, 190, 200), pop(3, 2, 205, 210),
                     pop(5, 2, 215, 220), push(6, 1, 230, 240), pop(6, 2, 250, 260)});
    const auto m = minimize<FifoQueueSpec>(h);
    ASSERT_TRUE(m);
    const History expected({push(3, 0, 150, 160), push(4, 0, 170, 180), pop(4, 2, 190, 200),
                            pop(3, 2, 205, 210)});
    ASSERT_EQ(m->window.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_TRUE(same_op(m->window[i], expected[i])) << to_string(m->window[i]);
    }
}

TEST(LinMinimize, KeepsAnEmptyPopThatCausesTheFailure) {
    // 1 was pushed and is still there, so the empty pop is a lie. Value 2
    // comes and goes before any of that and is noise. (Noise has to be
    // harmless on its own: a 2 pushed after 1 and popped first would be a
    // second, smaller violation, and the minimizer may return that one.)
    const History h({push(2, 1, 0, 2), pop(2, 2, 3, 4), push(1, 0, 5, 10),
                     pop(none, 2, 20, 30), pop(1, 2, 40, 50)});
    const auto m = minimize<FifoQueueSpec>(h);
    ASSERT_TRUE(m);
    const History expected({push(1, 0, 5, 10), pop(none, 2, 20, 30), pop(1, 2, 40, 50)});
    ASSERT_EQ(m->window.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_TRUE(same_op(m->window[i], expected[i])) << to_string(m->window[i]);
    }
}

TEST(LinMinimize, NeedsContextFromBeforeTheFailingSegment) {
    // Value 1 is pushed in an earlier quiescent segment and popped late, so
    // the elements of the failing segment alone pass; the window must pull 1
    // back in from outside it.
    const History h({push(1, 0, 0, 10), push(2, 0, 20, 30), pop(2, 1, 40, 50),
                     pop(1, 1, 60, 70)});
    const auto m = minimize<FifoQueueSpec>(h);
    ASSERT_TRUE(m);
    EXPECT_EQ(m->window.size(), 4u);
    expect_one_minimal<FifoQueueSpec>(m->window);
}

TEST(LinMinimize, StackWindowIsOneMinimal) {
    // 1 then 2 pushed in sequence on a stack, then popped 1 first: LIFO says 2.
    const History h({push(10, 1, 0, 5), push(1, 0, 10, 20), push(2, 0, 30, 40),
                     pop(10, 1, 6, 9), pop(1, 2, 50, 60), pop(2, 2, 70, 80),
                     pop(none, 1, 90, 95)});
    const auto m = minimize<LifoStackSpec>(h);
    ASSERT_TRUE(m);
    EXPECT_EQ(check<LifoStackSpec>(m->window).verdict, Verdict::not_linearizable);
    expect_one_minimal<LifoStackSpec>(m->window);
    EXPECT_EQ(m->window.size(), 4u);
}

// The end-to-end case lin_check exists for: a long recorded run with one
// real problem in it. The MPSC queue's empty pops behind a stalled producer
// aren't linearizable (see test_lin_stress), so record them and let the
// minimizer pull out the ops that show it: a push that returned, an empty
// pop after it, and the pop that finally got the value.
//
// Producers take credits so the queue stays shallow. Without that the queue
// gets deep and *proving* a failure (the search has to run out of options)
// blows past any sane budget; the budget below keeps a bad run from hanging.
class LinMinimizeStress : public lockfree::test::StressTest {};

TEST_F(LinMinimizeStress, ShrinksARecordedMpscFailureToThreeOps) {
    constexpr std::size_t producers = 3;
    constexpr std::uint64_t per_producer = 400;
    lockfree::lin::CheckOptions options;
    options.max_configurations = 1'000'000;
    for (int attempt = 0; attempt < 20; ++attempt) {
        lockfree::MpscQueue<std::int64_t> q;
        std::atomic<int> in_queue{0};
        Recorder rec(producers + 1, 4 * per_producer);
        lockfree::test::run_stress(producers + 1, [&](std::size_t t, lockfree::test::StressRng& rng) {
            auto& log = rec.log(t);
            if (t < producers) {
                for (std::uint64_t i = 0; i < per_producer; ++i) {
                    while (in_queue.load(std::memory_order_relaxed) >= 4) std::this_thread::yield();
                    in_queue.fetch_add(1, std::memory_order_relaxed);
                    const auto v = static_cast<std::int64_t>(t * 1'000'000 + i);
                    log.push(v, [&] {
                        q.push(v);
                        return true;
                    });
                    lockfree::test::maybe_delay(rng);
                }
            } else {
                for (std::uint64_t got = 0; got < producers * per_producer;) {
                    if (log.pop([&] { return q.try_pop(); })) {
                        ++got;
                        in_queue.fetch_sub(1, std::memory_order_relaxed);
                    } else {
                        std::this_thread::yield();
                    }
                }
            }
        });
        const History h = rec.merge();
        if (check<FifoQueueSpec>(h, options).verdict != Verdict::not_linearizable) continue;

        const auto m = minimize<FifoQueueSpec>(h, options);
        ASSERT_TRUE(m);
        ASSERT_EQ(m->window.size(), 3u);
        const History& w = m->window;
        EXPECT_EQ(w[0].kind, OpKind::push);
        EXPECT_TRUE(w[1].kind == OpKind::pop && !w[1].result);
        EXPECT_TRUE(w[2].kind == OpKind::pop && w[2].result == w[0].value);
        EXPECT_TRUE(lockfree::lin::happens_before(w[0], w[1]));
        return;
    }
    GTEST_SKIP() << "no run hit the stalled-producer case in 20 attempts";
}

} // namespace
