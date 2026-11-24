#pragma once

// Bounded exhaustive exploration: every schedule of a scenario with at most
// `preemption_bound` preemptions, enumerated depth-first.
//
// Random and PCT runs sample the schedule space; for a scenario small enough
// (two threads, two or three operations each) it can simply be enumerated, and
// then "passed" means "passed every schedule", not "passed the ones we drew".
// Unbounded enumeration blows up fast, so it's bounded the way CHESS does it
// (Musuvathi & Qadeer, PLDI 2007): count only preemptions, i.e. switches away
// from a thread that could have kept running. Switches when nobody is current
// (the start of the run, or right after a thread finished) are free. Most real
// bugs need one or two preemptions, and with a fixed bound the number of
// schedules grows polynomially in the run length instead of exponentially.
//
// The search is stateless: each schedule is a fresh run (the scenario is
// rebuilt by its factory), steered down a recorded prefix of choices and then
// left to its default, "keep running the current thread". After the run, the
// deepest decision that still has an untried alternative within the bound is
// advanced, and everything after it is thrown away. That needs the scenario to
// be deterministic given the choices, which the scheduler guarantees as long
// as the threads' code is (no clocks, no randomness, no real-thread races).
//
// Spin loops don't mix well with a preemption bound: a thread spinning on a
// condition only another thread can change keeps the baton under the default
// choice, so once the bound is spent the run ends at the step limit. Scenarios
// here should use the try_ operations and accept failure as a result.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "scheduler/explore.hpp"

namespace lockfree::sched {

struct ExhaustiveOptions {
    // Preemptions allowed per schedule.
    std::size_t preemption_bound = 2;
    // Give up (complete = false) after this many runs.
    std::size_t max_runs = 1'000'000;
};

struct ExhaustiveResult {
    std::size_t runs = 0;
    // Every schedule within the bound was run (and none failed).
    bool complete = false;
    // A replayed prefix met a different decision than the one recorded: the
    // scenario isn't a function of the schedule, so the search is meaningless.
    bool nondeterministic = false;
    // The first failing run's result; its choices replay through
    // replay_choices() (shrink.hpp) or shrink().
    std::optional<RunResult> failure;
    // Largest number of preemptions any explored schedule used.
    std::size_t max_preemptions_seen = 0;

    bool ok() const noexcept { return complete && !failure && !nondeterministic; }
};

// Runs every schedule of the case with at most options.preemption_bound
// preemptions, stopping at the first failure.
ExhaustiveResult explore_exhaustive(const CaseFactory& factory, ExhaustiveOptions options = {},
                                    Options run_options = {});

} // namespace lockfree::sched
