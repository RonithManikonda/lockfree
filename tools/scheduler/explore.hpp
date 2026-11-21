#pragma once

// Random exploration on top of the scheduler core: run a scenario many times
// under RandomStrategy, stop at the first schedule that breaks it, and print a
// seed that replays that exact schedule.
//
// Every run gets its own 64-bit seed, derived from the exploration seed and
// the run's index. Given the same scenario, that one number fixes every
// choice the strategy makes, and so (see scheduler.hpp) the whole run. That's
// why a failing run is reported as a single seed rather than a list of
// choices: `replay(factory, seed)` is all it takes to get it back.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include "scheduler/scheduler.hpp"

namespace lockfree::sched {

// SplitMix64: tiny, fast, and fine for picking threads. Also used to turn
// (exploration seed, run index) into well-spread per-run seeds.
class SplitMix64 {
public:
    explicit SplitMix64(std::uint64_t seed) noexcept : state_(seed) {}

    std::uint64_t next() noexcept {
        std::uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    // Uniform in [0, n) up to a bias of n / 2^64, which is nothing for thread
    // counts.
    std::uint64_t below(std::uint64_t n) noexcept { return next() % n; }

private:
    std::uint64_t state_;
};

// At every decision, picks uniformly among the runnable threads. Simple, and
// good at shallow bugs (a preemption at one or two specific points); PCT does
// better on deeper ones.
class RandomStrategy : public Strategy {
public:
    explicit RandomStrategy(std::uint64_t seed) noexcept : seed_(seed), rng_(seed) {}

    // Takes effect at the next begin_run.
    void reseed(std::uint64_t seed) noexcept { seed_ = seed; }

    void begin_run(std::size_t) override { rng_ = SplitMix64(seed_); }

    ThreadId pick(const Decision& d) override { return d.runnable[rng_.below(d.runnable.size())]; }

private:
    std::uint64_t seed_;
    SplitMix64 rng_;
};

// One scenario instance. The factory builds fresh state for every run (a
// failed run can leave a structure in any state, and an abandoned one keeps
// its threads' captures alive), so state the threads share should be owned
// through what they capture, typically a shared_ptr.
struct Case {
    std::vector<std::function<void()>> threads;
    // Optional. Runs on the calling thread after every thread has returned,
    // outside the scheduler; throw (or require()) to fail the run.
    std::function<void()> check;
};

using CaseFactory = std::function<Case()>;

// The seed run `index` of an exploration seeded with `seed` uses.
std::uint64_t run_seed(std::uint64_t seed, std::size_t index) noexcept;

struct ExploreResult {
    std::size_t runs = 0;                    // runs made, including the failing one
    std::optional<std::uint64_t> failing_seed; // set when some run failed
    RunResult failure;                       // that run's result

    bool ok() const noexcept { return !failing_seed.has_value(); }
};

// Builds a case and runs it under `strategy`: the threads, then the check if
// they all completed.
RunResult run_case(const CaseFactory& factory, Strategy& strategy, Options options = {});

// Up to `iterations` runs under RandomStrategy. Stops at the first failure and
// prints its message and replay seed to stderr.
ExploreResult explore(const CaseFactory& factory, std::size_t iterations, std::uint64_t seed,
                      Options options = {});

// Runs the case once with the schedule a run seed picks.
RunResult replay(const CaseFactory& factory, std::uint64_t run_seed, Options options = {});

} // namespace lockfree::sched
