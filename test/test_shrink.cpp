#include <gtest/gtest.h>

#include <cstddef>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "scheduler/explore.hpp"
#include "scheduler/shrink.hpp"

using lockfree::detail::atomic;
using lockfree::sched::Case;
using lockfree::sched::Decision;
using lockfree::sched::Outcome;
using lockfree::sched::require;
using lockfree::sched::ScriptedStrategy;
using lockfree::sched::Switch;
using lockfree::sched::ThreadId;
using lockfree::sched::TraceStep;

namespace {

// A lost update buried in noise: each thread does unrelated atomic work before
// and after its increment, so a random failing schedule switches threads many
// times besides the one preemption that matters.
Case noisy_lost_update() {
    struct Shared {
        atomic<int> x{0};
        atomic<int> noise[3]{};
    };
    auto s = std::make_shared<Shared>();
    auto body = [s](int id) {
        return [s, id] {
            for (int i = 0; i < 6; ++i) s->noise[id].fetch_add(1);
            s->x.store(s->x.load() + 1);
            for (int i = 0; i < 6; ++i) s->noise[id].fetch_add(1);
        };
    };
    return Case{{body(0), body(1), body(2)}, [s] { require(s->x.load() == 3, "lost an increment"); }};
}

lockfree::sched::ExploreResult find_noisy_failure() {
    return lockfree::sched::explore(noisy_lost_update, 2000, 4);
}

TEST(Shrink, SwitchesOfKeepsOnlyChanges) {
    // Start T0, T0 continues twice, T1 preempts at T0's 2nd yield, T1 finishes
    // and T0 is picked again (a nobody-current decision, always a switch).
    const std::vector<TraceStep> trace{
        {0, std::nullopt, 0, 0, std::nullopt},
        {1, 0, 0, 0, std::nullopt},
        {2, 0, 1, 1, std::nullopt},
        {3, 1, 0, 1, std::nullopt},
        {4, std::nullopt, 1, 0, std::nullopt},
    };
    const std::vector<Switch> want{{std::nullopt, 0, 0}, {0, 1, 1}, {std::nullopt, 1, 0}};
    EXPECT_EQ(lockfree::sched::switches_of(trace), want);
    EXPECT_TRUE(lockfree::sched::switches_of({}).empty());
}

TEST(Shrink, ScriptedStrategyFollowsTheScriptAndOtherwiseStays) {
    ScriptedStrategy s({{std::nullopt, 0, 2}, {2, 2, 1}, {1, 0, 7}});
    s.begin_run(3);
    const std::vector<ThreadId> all{0, 1, 2};
    const lockfree::detail::YieldPoint p{lockfree::detail::AtomicOp::load, nullptr, {}};
    EXPECT_EQ(s.pick(Decision{std::nullopt, nullptr, all, 0}), 2u);
    EXPECT_EQ(s.pick(Decision{2, &p, all, 1}), 2u); // T2's yield 0
    EXPECT_EQ(s.pick(Decision{2, &p, all, 2}), 2u); // yield 1
    EXPECT_EQ(s.pick(Decision{2, &p, all, 3}), 1u); // yield 2: scripted switch
    // Thread 7 isn't runnable: keep running the current one.
    EXPECT_EQ(s.pick(Decision{1, &p, all, 4}), 1u);
    // Nobody current, nothing scripted: lowest runnable.
    const std::vector<ThreadId> rest{0, 2};
    EXPECT_EQ(s.pick(Decision{std::nullopt, nullptr, rest, 5}), 0u);
}

TEST(Shrink, ReplayingTheFullSwitchListReproducesTheRun) {
    auto found = find_noisy_failure();
    ASSERT_FALSE(found.ok());

    std::vector<TraceStep> trace;
    auto exact = lockfree::sched::replay_choices(noisy_lost_update, found.failure.choices, {}, &trace);
    EXPECT_EQ(exact.choices, found.failure.choices);
    ASSERT_EQ(trace.size(), exact.choices.size());

    auto r = lockfree::sched::replay_switches(noisy_lost_update, lockfree::sched::switches_of(trace));
    EXPECT_EQ(r.choices, found.failure.choices);
    EXPECT_EQ(r.failure, found.failure.failure);
}

TEST(Shrink, CutsANoisyLostUpdateDownToItsPreemption) {
    auto found = find_noisy_failure();
    ASSERT_FALSE(found.ok());

    auto s = lockfree::sched::shrink(noisy_lost_update, found.failure.choices);
    ASSERT_TRUE(s.reproduced);
    EXPECT_EQ(s.run.outcome, Outcome::failed);
    EXPECT_EQ(s.run.failure, "check: lost an increment");

    // One preemption between a load and its store is enough: the preempting
    // thread runs to the end, the lowest-numbered thread is picked next by
    // default, and its stale store loses an increment. Allow one extra switch
    // in case ddmin lands on an equivalent two-switch form.
    const std::string trace = lockfree::sched::format_trace(s.trace);
    EXPECT_LE(s.script.size(), 2u) << trace;
    EXPECT_GT(s.original_switches, s.script.size());

    // The shrunk list is a real schedule: replaying it fails the same way.
    auto again = lockfree::sched::replay_switches(noisy_lost_update, s.script);
    EXPECT_EQ(again.failure, s.run.failure);
    EXPECT_EQ(again.choices, s.run.choices);

    // And it reads as one: the trace shows the preemption and the stale store.
    EXPECT_NE(trace.find("<- preempts"), std::string::npos) << trace;
    EXPECT_NE(trace.find("store"), std::string::npos) << trace;
    EXPECT_NE(trace.find("test_shrink.cpp"), std::string::npos) << trace;
    std::printf("minimal interleaving (%zu of %zu switches, %zu replays):\n%s", s.script.size(),
                s.original_switches, s.attempts, trace.c_str());
}

TEST(Shrink, PassingScheduleIsReportedAsNotReproduced) {
    // With no switches each thread runs to completion before the next starts.
    ScriptedStrategy none({});
    const auto serial = lockfree::sched::run_case(noisy_lost_update, none).choices;

    auto s = lockfree::sched::shrink(noisy_lost_update, serial);
    EXPECT_FALSE(s.reproduced);
    EXPECT_TRUE(s.run.ok());
    EXPECT_EQ(s.attempts, 1u);
}

TEST(Shrink, AttemptBudgetIsRespected) {
    auto found = find_noisy_failure();
    ASSERT_FALSE(found.ok());

    // The final replay of the result comes on top of the budget.
    auto s = lockfree::sched::shrink(noisy_lost_update, found.failure.choices, {}, 5);
    ASSERT_TRUE(s.reproduced);
    EXPECT_LE(s.attempts, 6u);
    EXPECT_EQ(s.run.failure, "check: lost an increment");
}

} // namespace
