#include "scheduler/exhaustive.hpp"

#include <algorithm>
#include <cstdio>
#include <utility>

namespace lockfree::sched {

namespace {

// One decision of the current schedule.
struct Frame {
    std::optional<ThreadId> current;
    // The alternatives, in the order they're tried: the current thread first
    // (the free, default choice), then the other runnable threads ascending.
    std::vector<ThreadId> candidates;
    std::size_t idx = 0;              // the one this schedule takes
    std::size_t preemptions_before = 0;

    // Taking anything but the current thread while there is one costs a
    // preemption; with nobody current every choice is free.
    std::size_t cost(std::size_t i) const noexcept { return current && i > 0 ? 1 : 0; }
    std::size_t preemptions_after() const noexcept { return preemptions_before + cost(idx); }
};

std::vector<ThreadId> candidates_of(const Decision& d) {
    std::vector<ThreadId> out;
    out.reserve(d.runnable.size());
    if (d.current) out.push_back(*d.current);
    for (ThreadId t : d.runnable)
        if (!d.current || t != *d.current) out.push_back(t);
    return out;
}

// Replays the frames recorded so far, then extends the schedule with default
// choices, recording a frame for each new decision.
class DfsStrategy : public Strategy {
public:
    explicit DfsStrategy(std::vector<Frame>& frames) : frames_(frames) {}

    bool nondeterministic() const noexcept { return nondeterministic_; }

    ThreadId pick(const Decision& d) override {
        const auto step = static_cast<std::size_t>(d.step);
        if (step < frames_.size()) {
            const Frame& f = frames_[step];
            if (f.current == d.current && f.candidates == candidates_of(d)) return f.candidates[f.idx];
            // Strategies can't throw (we're inside the scheduler's noexcept
            // hook), so note it and finish the run somehow.
            nondeterministic_ = true;
            return d.current ? *d.current : d.runnable.front();
        }
        Frame f;
        f.current = d.current;
        f.candidates = candidates_of(d);
        f.preemptions_before = frames_.empty() ? 0 : frames_.back().preemptions_after();
        frames_.push_back(std::move(f));
        return frames_.back().candidates[0];
    }

private:
    std::vector<Frame>& frames_;
    bool nondeterministic_ = false;
};

// Moves to the next schedule in depth-first order: the deepest decision with
// an untried alternative the bound still allows. False when there's none.
bool advance(std::vector<Frame>& frames, std::size_t bound) {
    while (!frames.empty()) {
        Frame& f = frames.back();
        const std::size_t next = f.idx + 1;
        if (next < f.candidates.size() && f.preemptions_before + f.cost(next) <= bound) {
            f.idx = next;
            return true;
        }
        frames.pop_back();
    }
    return false;
}

} // namespace

ExhaustiveResult explore_exhaustive(const CaseFactory& factory, ExhaustiveOptions options,
                                    Options run_options) {
    ExhaustiveResult result;
    std::vector<Frame> frames;
    for (;;) {
        if (result.runs == options.max_runs) return result;

        // Only the prefix up to the decision advance() changed is kept; the
        // run re-records everything after it.
        const std::size_t prefix = frames.size();
        DfsStrategy strategy(frames);
        RunResult r = run_case(factory, strategy, run_options);
        ++result.runs;

        if (strategy.nondeterministic() || r.steps < prefix) {
            std::fprintf(stderr, "lockfree::sched: exhaustive run %zu didn't follow its recorded prefix; "
                                 "the scenario isn't deterministic\n",
                         result.runs);
            result.nondeterministic = true;
            return result;
        }
        // A run that stopped early (step limit) may have recorded fewer frames
        // than it made decisions; that doesn't matter, it ends the search.
        if (!frames.empty())
            result.max_preemptions_seen =
                std::max(result.max_preemptions_seen, frames.back().preemptions_after());

        if (!r.ok()) {
            std::fprintf(stderr,
                         "lockfree::sched: exhaustive run %zu (preemption bound %zu) failed: %s\n",
                         result.runs, options.preemption_bound, r.failure.c_str());
            result.failure = std::move(r);
            return result;
        }
        if (!advance(frames, options.preemption_bound)) {
            result.complete = true;
            return result;
        }
    }
}

} // namespace lockfree::sched
