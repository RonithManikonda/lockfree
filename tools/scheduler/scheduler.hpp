#pragma once

// Adversarial scheduler core: runs K logical threads on real std::threads,
// exactly one at a time, and lets a Strategy decide who runs at every yield
// point.
//
// Yield points come from lockfree::detail::atomic (detail/atomic_ref.hpp):
// with LOCKFREE_SCHEDULER defined, every atomic load/store/RMW calls the
// installed hook first. While a run is in progress that hook belongs to the
// scheduler: the calling thread stops, the strategy picks the next thread,
// and the baton passes to it. Each thread waits on its own condition
// variable, so a handoff wakes exactly the thread that was picked.
//
// Because only one thread moves at a time and every switch happens at an
// atomic access, a run is a deterministic function of the strategy's choices
// (as long as the threads' code is deterministic itself). That's what makes
// seeds replayable and failing schedules shrinkable later on.
//
// Rules for code run under the scheduler:
//   - Don't block outside yield points (no mutexes, condvars, sleeps). The
//     scheduler can't see those, so a thread blocked on one hangs the run.
//   - Spinning is fine as long as the spin touches a detail::atomic: each
//     iteration is a yield point, so the strategy can run someone else.
//   - Report failures by throwing (see require()). Strategies must not throw;
//     they're called from the noexcept hook.
//   - Only one run at a time per process: the hook is a process-wide pointer.

#if !defined(LOCKFREE_SCHEDULER)
#error "the scheduler needs LOCKFREE_SCHEDULER; link lockfree_scheduler, which defines it"
#endif

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "lockfree/detail/atomic_ref.hpp"

namespace lockfree::sched {

// Logical thread id: the index of the thread's function in run()'s argument.
using ThreadId = std::size_t;

// Everything a strategy sees when it has to choose.
struct Decision {
    // The thread that just reached a yield point and is about to perform
    // `point`'s access. nullopt at the start of a run and right after a thread
    // finishes, when somebody new has to be picked.
    std::optional<ThreadId> current;
    // The access `current` is about to make; nullptr exactly when `current` is
    // nullopt.
    const detail::YieldPoint* point = nullptr;
    // Threads that haven't finished, in ascending order. Never empty, and
    // contains *current when that's set.
    std::span<const ThreadId> runnable;
    // Decisions already made in this run.
    std::uint64_t step = 0;
};

class Strategy {
public:
    virtual ~Strategy() = default;

    // Called once before each run, from the thread that calls run().
    virtual void begin_run(std::size_t thread_count) { (void)thread_count; }

    // Must return a member of d.runnable. Returning *d.current lets the
    // current thread carry on; anything else is a context switch.
    virtual ThreadId pick(const Decision& d) = 0;
};

struct Options {
    // Decisions allowed per run. A run that needs more is reported as
    // Outcome::step_limit: with one thread moving at a time, a livelock or a
    // wait that can never be satisfied shows up as a run that never ends.
    std::uint64_t max_steps = 100'000;
};

enum class Outcome {
    completed,  // every thread returned normally
    failed,     // a thread threw; the other threads still ran to the end
    step_limit, // the run was abandoned after max_steps decisions
};

struct RunResult {
    Outcome outcome = Outcome::completed;
    std::uint64_t steps = 0;
    // The thread picked at each decision, in order (choices.size() == steps).
    std::vector<ThreadId> choices;
    // The first exception's message for `failed`; a description for
    // `step_limit`.
    std::string failure;

    bool ok() const noexcept { return outcome == Outcome::completed; }
};

// What require() throws. Any exception escaping a thread fails the run; this
// one just gives tests a recognisable type.
struct Failure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

inline void require(bool condition, const char* what) {
    if (!condition) throw Failure(what);
}

class Scheduler {
public:
    explicit Scheduler(Strategy& strategy, Options options = {})
        : strategy_(strategy), options_(options) {}

    // Runs threads[i] as logical thread i and returns once they've all
    // finished, or once the run hits the step limit.
    //
    // A run that hits the limit can't be unwound: its threads are stopped
    // inside noexcept atomic operations, so there's no way to make them
    // return. They're left parked on a condition variable for the rest of the
    // process and never run again, and their functions (with whatever they
    // captured) are kept alive with them. That's a deliberate leak, fine for a
    // test that's about to report a failure anyway.
    RunResult run(std::vector<std::function<void()>> threads);

private:
    Strategy& strategy_;
    Options options_;
};

} // namespace lockfree::sched
