#pragma once

// PCT (Probabilistic Concurrency Testing, Burckhardt et al., ASPLOS 2010).
//
// Instead of flipping a coin at every step, PCT gives each thread a random
// priority and always runs the highest-priority runnable thread. On top of
// that it picks d - 1 random "change points" (step numbers); when a run
// reaches one, the thread that's running drops below everybody else. So a
// whole run contains only d - 1 forced preemptions, placed at random.
//
// Why that helps: most concurrency bugs need only a couple of preemptions at
// the right places (the bug's "depth"). For a bug of depth d among k threads
// and n steps, one PCT run hits it with probability at least 1 / (k * n^(d-1)).
// A random walk spreads its switches everywhere, and the odds that it also
// runs every *other* step in the right order shrink exponentially with n.
//
// Two additions for the structures here:
//   - Spin loops. Strict priorities let a high-priority thread spin forever
//     waiting on a low-priority one. A thread that runs `max_run` decisions
//     in a row is demoted to the bottom, which is the "yield after a while"
//     a real OS scheduler gives you anyway.
//   - CAS bias. The interesting window in a CAS loop is right after a
//     successful compare_exchange, before the follow-up writes (ABA, a
//     claimed-but-unpublished cell). When a thread reaches the yield after a
//     compare_exchange, it's demoted with probability `cas_preempt`.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "scheduler/explore.hpp"

namespace lockfree::sched {

struct PctOptions {
    // Bug depth to aim for: d - 1 change points per run. depth 1 means no
    // change points at all (just a random fixed priority order).
    std::size_t depth = 3;
    // Change points are spread over [1, expected_steps]. Roughly the length
    // of a run; a run that ends before a change point simply skips it.
    std::uint64_t expected_steps = 200;
    // Consecutive decisions one thread may win before it's demoted. 0
    // disables the starvation guard.
    std::uint64_t max_run = 64;
    // Probability of demoting a thread at the yield right after its
    // compare_exchange. 0 disables the bias.
    double cas_preempt = 0.25;
};

class PctStrategy : public SeededStrategy {
public:
    explicit PctStrategy(std::uint64_t seed, PctOptions options = {}) noexcept
        : seed_(seed), options_(options), rng_(seed) {}

    void reseed(std::uint64_t seed) noexcept override { seed_ = seed; }
    const char* name() const noexcept override { return "pct"; }

    void begin_run(std::size_t thread_count) override;
    ThreadId pick(const Decision& d) override;

    const PctOptions& options() const noexcept { return options_; }

    // This run's change points, ascending (for tests and trace output).
    const std::vector<std::uint64_t>& change_points() const noexcept { return change_points_; }

private:
    void demote(ThreadId t) noexcept { priority_[t] = --floor_; }
    bool coin(double p) noexcept;

    std::uint64_t seed_;
    PctOptions options_;
    SplitMix64 rng_;

    // Higher runs first. Initial priorities are a random permutation of
    // [0, k); every demotion goes below the lowest priority handed out so
    // far, so demotions stack up in the order they happened.
    std::vector<std::int64_t> priority_;
    std::int64_t floor_ = 0;
    std::vector<std::uint64_t> change_points_;
    std::size_t next_change_ = 0;
    // Whether each thread's previous yield point was a compare_exchange, i.e.
    // whether the access it's about to make is the first one after a CAS.
    std::vector<bool> after_cas_;
    // Starvation guard bookkeeping.
    std::optional<ThreadId> last_;
    std::uint64_t run_length_ = 0;
};

} // namespace lockfree::sched
