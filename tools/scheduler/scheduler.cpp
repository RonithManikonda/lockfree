#include "scheduler/scheduler.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

namespace lockfree::sched {
namespace {

// State of one run, shared by run() and the run's threads. Held through a
// shared_ptr because threads abandoned at the step limit outlive run().
//
// Everything here is guarded by `m`. Only one thread is ever doing real work,
// so the lock is never contended for long; it's there so each handoff is a
// proper happens-before edge (the next thread sees everything the previous
// one wrote, plain memory included, and TSan agrees).
struct RunState {
    RunState(Strategy& s, Options o, std::vector<std::function<void()>> fns)
        : strategy(s), options(o), bodies(std::move(fns)), cv(bodies.size()),
          finished(bodies.size(), false), remaining(bodies.size()) {}

    Strategy& strategy;
    const Options options;
    const std::vector<std::function<void()>> bodies;

    std::mutex m;
    std::vector<std::condition_variable> cv; // one per thread, so a handoff wakes only the target
    std::condition_variable done;            // run() waits here

    std::optional<ThreadId> active; // who holds the baton
    std::vector<bool> finished;
    std::size_t remaining;
    bool abandoned = false;

    std::vector<ThreadId> runnable; // scratch for choose()
    RunResult result;
};

// Which run (if any) the calling thread belongs to. Threads that aren't part
// of a run, like the one that called run(), pass straight through the hook.
struct Context {
    RunState* state = nullptr;
    ThreadId id = 0;
};
thread_local Context tls_context;

// Asks the strategy who runs next and records the answer. Returns nullopt
// once the step budget is spent. Called with `m` held.
std::optional<ThreadId> choose(RunState& s, std::optional<ThreadId> current,
                               const detail::YieldPoint* point) {
    if (s.result.steps >= s.options.max_steps) return std::nullopt;

    s.runnable.clear();
    for (ThreadId t = 0; t < s.finished.size(); ++t)
        if (!s.finished[t]) s.runnable.push_back(t);

    ThreadId next = s.strategy.pick(Decision{current, point, s.runnable, s.result.steps});
    if (!std::binary_search(s.runnable.begin(), s.runnable.end(), next)) {
        // A broken strategy, and we're inside a noexcept hook: no clean way out.
        std::fprintf(stderr, "lockfree::sched: strategy picked thread %zu, which isn't runnable\n",
                     next);
        std::abort();
    }
    ++s.result.steps;
    s.result.choices.push_back(next);
    return next;
}

void hand_to(RunState& s, ThreadId next) {
    s.active = next;
    s.cv[next].notify_one();
}

// Blocks until `id` holds the baton. After the run is abandoned nobody is
// handed the baton again, so for those threads this never returns.
void wait_turn(RunState& s, std::unique_lock<std::mutex>& lk, ThreadId id) {
    s.cv[id].wait(lk, [&] { return s.active == id; });
}

void give_up(RunState& s) {
    s.abandoned = true;
    s.active.reset();
    s.result.outcome = Outcome::step_limit;
    std::string msg = "step limit (" + std::to_string(s.options.max_steps) +
                      " decisions) reached: livelock, or a wait nothing can satisfy";
    if (!s.result.failure.empty()) msg += "; earlier failure: " + s.result.failure;
    s.result.failure = std::move(msg);
    s.done.notify_one();
}

void on_yield(const detail::YieldPoint& point) noexcept {
    Context ctx = tls_context;
    if (ctx.state == nullptr) return;
    RunState& s = *ctx.state;

    std::unique_lock lk(s.m);
    std::optional<ThreadId> next = choose(s, ctx.id, &point);
    if (!next) {
        give_up(s);
        wait_turn(s, lk, ctx.id); // parks this thread for good
    } else if (*next != ctx.id) {
        hand_to(s, *next);
        wait_turn(s, lk, ctx.id);
    }
}

void record_failure(RunState& s, std::string what) {
    std::lock_guard lk(s.m);
    if (s.result.failure.empty()) s.result.failure = std::move(what);
    if (s.result.outcome == Outcome::completed) s.result.outcome = Outcome::failed;
}

void thread_main(std::shared_ptr<RunState> state, ThreadId id) {
    RunState& s = *state;
    tls_context = Context{&s, id};
    {
        std::unique_lock lk(s.m);
        wait_turn(s, lk, id);
    }

    try {
        s.bodies[id]();
    } catch (const std::exception& e) {
        record_failure(s, e.what());
    } catch (...) {
        record_failure(s, "non-standard exception");
    }

    tls_context = Context{};
    std::unique_lock lk(s.m);
    s.finished[id] = true;
    if (--s.remaining == 0) {
        s.active.reset();
        s.done.notify_one();
        return;
    }
    if (std::optional<ThreadId> next = choose(s, std::nullopt, nullptr))
        hand_to(s, *next);
    else
        give_up(s); // this thread is done, so it can still exit normally
}

// The hook is process-wide, so two concurrent runs would steal each other's
// yield points.
std::atomic<bool> g_run_in_progress{false};

} // namespace

RunResult Scheduler::run(std::vector<std::function<void()>> threads) {
    if (threads.empty()) return {};
    if (g_run_in_progress.exchange(true))
        throw std::logic_error("lockfree::sched: only one Scheduler::run at a time");

    const std::size_t n = threads.size();
    auto state = std::make_shared<RunState>(strategy_, options_, std::move(threads));
    strategy_.begin_run(n);

    detail::YieldHook& hook = detail::yield_hook();
    const detail::YieldHook previous = hook;
    hook = &on_yield;

    std::vector<std::thread> workers;
    workers.reserve(n);
    for (ThreadId id = 0; id < n; ++id) workers.emplace_back(thread_main, state, id);

    bool abandoned;
    {
        std::unique_lock lk(state->m);
        if (std::optional<ThreadId> first = choose(*state, std::nullopt, nullptr))
            hand_to(*state, *first);
        else
            give_up(*state); // max_steps == 0
        state->done.wait(lk, [&] { return state->remaining == 0 || state->abandoned; });
        abandoned = state->abandoned;
    }

    for (auto& w : workers) {
        if (abandoned)
            w.detach(); // parked forever; `state` stays alive through their copies
        else
            w.join();
    }

    hook = previous;
    RunResult result;
    {
        std::lock_guard lk(state->m);
        result = state->result;
    }
    g_run_in_progress.store(false);
    return result;
}

} // namespace lockfree::sched
