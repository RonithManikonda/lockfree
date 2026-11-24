#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <lockfree/atomic_stack.hpp>
#include <lockfree/naive/treiber_stack.hpp>

#include "scheduler/coverage.hpp"
#include "scheduler/explore.hpp"
#include "scheduler/pct.hpp"
#include "scheduler/shrink.hpp"

// The scheduler pointed at a real bug: the ABA problem in the textbook
// Treiber stack. test_atomic_stack.cpp reproduces it with a hand-placed test
// hook; here nothing says where the preemption goes. The strategy has to find
// on its own that T0 must stop between reading head->next and its CAS, and
// stay stopped while T1 runs all of its operations. PCT does, in a few hundred
// runs; a plain random walk practically never does (see RandomWalkFor-
// Comparison), which is the case for PCT in one test.
//
// The same scenario then runs against the tagged AtomicStack for the same
// budget, and every schedule has to pass.

using lockfree::sched::Case;
using lockfree::sched::CaseFactory;
using lockfree::sched::CoverageStrategy;
using lockfree::sched::ExploreResult;
using lockfree::sched::PctStrategy;
using lockfree::sched::RandomStrategy;
using lockfree::sched::require;
using lockfree::sched::SeededStrategy;

namespace {

constexpr std::size_t kBudget = 5000;
constexpr std::uint64_t kSeed = 0x5eed'aba;

// Drains can't trust a corrupted stack to end, so they stop after a few more
// pops than there could ever be values.
constexpr int kMaxDrain = 16;

// Two stacks, s and other, plus one result slot per pop. For the naive stack
// they share one NodePool, the way two structures share malloc; with a single
// stack and a LIFO pool, nodes come back in mirror order and the stale next
// happens to be right (see NaiveTreiberAba.SameInterleavingOnOneStackHappens-
// ToBeHarmless). AtomicStack keeps its free list to itself, so its two stacks
// can't share nodes; the operations are the same.
struct NaiveStacks {
    lockfree::naive::NodePool<int> pool;
    lockfree::naive::TreiberStack<int> s{pool};
    lockfree::naive::TreiberStack<int> other{pool};
};

struct TaggedStacks {
    lockfree::AtomicStack<int> s;
    lockfree::AtomicStack<int> other;
};

template <typename Stacks>
struct Shared {
    Stacks stacks;
    std::optional<int> t0_pop;
    std::optional<int> t1_pops[2];
};

// s starts as 1 -> 2. T0 pops once. T1 pops twice, pushes 30 to other and 40
// to s. The pushes recycle the nodes T1 just freed, so if T0 sleeps through
// all of that between its read of next and its CAS, the naive CAS still finds
// its old head at the top and installs a next that now belongs to `other`.
//
// Check: every value pushed comes out exactly once, counting what the threads
// popped and what is left in both stacks. Any schedule of a correct stack
// passes it; the ABA run hands 30 out twice (once from each stack).
template <typename Stacks>
Case aba_scenario() {
    auto sh = std::make_shared<Shared<Stacks>>();
    // Built before the run, so these don't yield.
    sh->stacks.s.push(2);
    sh->stacks.s.push(1);

    auto t0 = [sh] { sh->t0_pop = sh->stacks.s.try_pop(); };
    auto t1 = [sh] {
        sh->t1_pops[0] = sh->stacks.s.try_pop();
        sh->t1_pops[1] = sh->stacks.s.try_pop();
        sh->stacks.other.push(30);
        sh->stacks.s.push(40);
    };
    auto check = [sh] {
        std::vector<int> seen;
        auto note = [&](const std::optional<int>& v) {
            if (v) seen.push_back(*v);
        };
        note(sh->t0_pop);
        for (const auto& v : sh->t1_pops) note(v);
        for (auto* stack : {&sh->stacks.s, &sh->stacks.other}) {
            for (int i = 0; i < kMaxDrain; ++i) {
                auto v = stack->try_pop();
                if (!v) break;
                seen.push_back(*v);
            }
        }
        std::sort(seen.begin(), seen.end());
        require(seen == std::vector<int>{1, 2, 30, 40},
                "values not conserved: some pushed value came out twice or not at all");
    };
    return Case{{t0, t1}, check};
}

ExploreResult explore_with(const CaseFactory& factory, SeededStrategy& strategy) {
    CoverageStrategy covered(strategy);
    auto result = lockfree::sched::explore(factory, covered, kBudget, kSeed);
    std::printf("[%s] %zu run%s, %s\n", strategy.name(), result.runs, result.runs == 1 ? "" : "s",
                result.ok() ? "no failure" : "found a failure");
    std::fputs(covered.coverage().summary(3).c_str(), stdout);
    return result;
}

TEST(InterleavingAba, PctFindsTheNaiveStackBug) {
    PctStrategy pct(0);
    auto result = explore_with(aba_scenario<NaiveStacks>, pct);
    ASSERT_FALSE(result.ok()) << "no ABA within " << kBudget << " PCT schedules";
    EXPECT_LT(result.runs, kBudget);
    EXPECT_NE(result.failure.failure.find("not conserved"), std::string::npos)
        << result.failure.failure;

    // The seed alone brings the failure back.
    auto again = lockfree::sched::replay(aba_scenario<NaiveStacks>, pct, *result.failing_seed);
    EXPECT_EQ(again.outcome, lockfree::sched::Outcome::failed);
    EXPECT_EQ(again.choices, result.failure.choices);
}

TEST(InterleavingAba, RandomWalkForComparison) {
    // Printed next to PCT's count, not asserted. The bug needs T0 to stay
    // preempted through the ~25 yield points of T1's four operations, and a
    // coin flip at each of them makes that roughly a 2^-25 event: 200k random
    // runs found nothing when this was written. PCT needs a single change
    // point in the right window, which is exactly what its depth bound buys.
    RandomStrategy random(0);
    auto result = explore_with(aba_scenario<NaiveStacks>, random);
    if (!result.ok()) EXPECT_EQ(result.failure.outcome, lockfree::sched::Outcome::failed);
}

TEST(InterleavingAba, ShrunkScheduleIsTheTextbookInterleaving) {
    PctStrategy pct(0);
    auto found = lockfree::sched::explore(aba_scenario<NaiveStacks>, pct, kBudget, kSeed);
    ASSERT_FALSE(found.ok());

    auto shrunk = lockfree::sched::shrink(aba_scenario<NaiveStacks>, found.failure.choices);
    ASSERT_TRUE(shrunk.reproduced);
    EXPECT_EQ(shrunk.run.outcome, lockfree::sched::Outcome::failed);
    std::printf("shrunk %zu switches to %zu in %zu replays:\n%s", shrunk.original_switches,
                shrunk.script.size(), shrunk.attempts,
                lockfree::sched::format_trace(shrunk.trace).c_str());

    // What's left is the bug itself: start a thread, preempt T0 once inside
    // its pop, and hand back to T0 when T1 is done. Allow the odd extra switch
    // ddmin can't prove redundant, but nothing like the original run.
    EXPECT_LE(shrunk.script.size(), 4u);
    EXPECT_LE(shrunk.script.size(), shrunk.original_switches);
    const bool t0_preempted_by_t1 =
        std::any_of(shrunk.script.begin(), shrunk.script.end(),
                    [](const auto& sw) { return sw.from == 0u && sw.to == 1u; });
    EXPECT_TRUE(t0_preempted_by_t1);
}

TEST(InterleavingAba, TaggedStackSurvivesRandomBudget) {
    RandomStrategy random(0);
    auto result = explore_with(aba_scenario<TaggedStacks>, random);
    EXPECT_TRUE(result.ok()) << result.failure.failure << " (replay seed " << result.failing_seed.value_or(0)
                             << ")";
    EXPECT_EQ(result.runs, kBudget);
}

TEST(InterleavingAba, TaggedStackSurvivesPctBudget) {
    PctStrategy pct(0);
    auto result = explore_with(aba_scenario<TaggedStacks>, pct);
    EXPECT_TRUE(result.ok()) << result.failure.failure << " (replay seed " << result.failing_seed.value_or(0)
                             << ")";
    EXPECT_EQ(result.runs, kBudget);
}

} // namespace
