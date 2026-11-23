#pragma once

// Shrinking a failing schedule down to the few context switches that matter.
//
// A failing run from explore() usually switches threads dozens of times, and
// almost all of those switches are noise. shrink() takes the run's recorded
// choices and looks for a much smaller set of switches that still produces the
// same failure, then prints that run as a step-by-step trace.
//
// The schedule is represented as a list of switches: "when thread u reaches
// its k-th yield point, run thread t instead". Keying a switch to u's own
// progress rather than to a global step number matters: deleting an earlier
// switch shifts every later global step, but u's k-th yield point is still
// the same line of u's code. Replaying a list (ScriptedStrategy) follows each
// switch whose key comes up, and otherwise keeps running whichever thread is
// running. So deleting a switch means "don't preempt here", and any subset of
// the switches is still a valid schedule. That makes delta debugging (ddmin,
// Zeller & Hildebrandt 2002) a natural fit: try deleting chunks of switches,
// keep each deletion that still fails, refine until no single switch can go.
//
// "Still fails" means the same outcome and the same failure message, so the
// shrinker can't wander off from the bug it was given to a different one.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "scheduler/explore.hpp"

namespace lockfree::sched {

struct Switch {
    // The thread at its yield point, or nullopt for a decision with nobody
    // current (the start of the run, or right after a thread finished).
    std::optional<ThreadId> from;
    // How many decisions `from` had already been current for (0-based); for
    // nullopt, how many nobody-current decisions came before this one.
    std::uint64_t at;
    ThreadId to;

    friend bool operator==(const Switch&, const Switch&) = default;
};

// Follows a switch list. When a decision's (current thread, count) matches a
// switch whose target is runnable, runs that target; otherwise keeps the
// current thread, or (when nobody is current) takes the lowest-numbered
// runnable one.
class ScriptedStrategy : public Strategy {
public:
    explicit ScriptedStrategy(std::vector<Switch> script) : script_(std::move(script)) {}

    void begin_run(std::size_t n) override {
        count_.assign(n, 0);
        none_count_ = 0;
    }
    ThreadId pick(const Decision& d) override;

private:
    std::vector<Switch> script_;
    std::vector<std::uint64_t> count_;
    std::uint64_t none_count_ = 0;
};

// One line of a trace: at decision `step`, `thread` ran and performed
// `access` (nullopt when the thread was only starting up).
struct TraceStep {
    std::uint64_t step;
    std::optional<ThreadId> from; // Decision::current
    std::uint64_t from_count;     // as in Switch::at
    ThreadId thread;
    std::optional<detail::YieldPoint> access;

    bool preempted() const noexcept { return from && *from != thread; }
};

// The switches a run made: every decision where the picked thread isn't the
// current one, including every nobody-current decision.
std::vector<Switch> switches_of(const std::vector<TraceStep>& trace);

// Runs the case once with exactly these choices (falling back like
// ScriptedStrategy where one doesn't fit), recording what each step did.
RunResult replay_choices(const CaseFactory& factory, const std::vector<ThreadId>& choices,
                         Options options = {}, std::vector<TraceStep>* trace = nullptr);

// Replays a switch list once, recording what each step did.
RunResult replay_switches(const CaseFactory& factory, const std::vector<Switch>& script,
                          Options options = {}, std::vector<TraceStep>* trace = nullptr);

// Human-readable trace: one line per step (the access the picked thread goes
// on to make), plus a marker wherever a running thread was preempted:
//
//   T0  load      test_shrink.cpp:31
//   T1  <- preempts T0
//   T1  load      test_shrink.cpp:31
//   ...
std::string format_trace(const std::vector<TraceStep>& trace);

struct ShrinkResult {
    std::vector<Switch> script;   // the minimal switch list
    RunResult run;                // the run it produces (still failing)
    std::vector<TraceStep> trace; // that run, step by step
    std::size_t original_switches = 0;
    std::size_t attempts = 0;     // replays spent shrinking
    bool reproduced = false;      // false if the input didn't fail to begin with
};

// Shrinks a failing choice sequence (RunResult::choices of a failed run).
// Stops early after `max_attempts` replays and returns the best list so far.
ShrinkResult shrink(const CaseFactory& factory, const std::vector<ThreadId>& failing_choices,
                    Options options = {}, std::size_t max_attempts = 2000);

} // namespace lockfree::sched
