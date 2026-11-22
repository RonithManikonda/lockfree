#include "scheduler/pct.hpp"

#include <algorithm>
#include <numeric>

namespace lockfree::sched {

void PctStrategy::begin_run(std::size_t thread_count) {
    rng_ = SplitMix64(seed_);

    priority_.resize(thread_count);
    std::iota(priority_.begin(), priority_.end(), std::int64_t{0});
    for (std::size_t i = thread_count; i > 1; --i) // Fisher-Yates
        std::swap(priority_[i - 1], priority_[rng_.below(i)]);
    floor_ = 0;

    // d - 1 distinct steps in [1, expected_steps]. Step 0 would just be a
    // different initial order, which the permutation already randomizes.
    change_points_.clear();
    const std::uint64_t wanted = options_.depth > 0 ? options_.depth - 1 : 0;
    const std::uint64_t span = options_.expected_steps;
    while (change_points_.size() < std::min(wanted, span)) {
        const std::uint64_t c = 1 + rng_.below(span);
        if (std::find(change_points_.begin(), change_points_.end(), c) == change_points_.end())
            change_points_.push_back(c);
    }
    std::sort(change_points_.begin(), change_points_.end());
    next_change_ = 0;

    after_cas_.assign(thread_count, false);
    last_.reset();
    run_length_ = 0;
}

bool PctStrategy::coin(double p) noexcept {
    if (p <= 0.0) return false;
    if (p >= 1.0) return true;
    // 53 random bits -> [0, 1).
    return static_cast<double>(rng_.next() >> 11) * 0x1.0p-53 < p;
}

ThreadId PctStrategy::pick(const Decision& d) {
    auto top = [&] {
        return *std::max_element(d.runnable.begin(), d.runnable.end(),
                                 [&](ThreadId a, ThreadId b) { return priority_[a] < priority_[b]; });
    };

    if (d.current) {
        const ThreadId t = *d.current;
        const bool was_cas = after_cas_[t];
        after_cas_[t] = d.point->op == detail::AtomicOp::compare_exchange;
        // Only a thread that's actually running can be caught right after its
        // CAS; d.current is exactly that thread.
        if (was_cas && coin(options_.cas_preempt)) demote(t);
    }

    // A change point demotes whoever would run now. At the start of a run or
    // right after a thread finishes there's no current thread, so that's the
    // one the priorities were about to pick.
    while (next_change_ < change_points_.size() && change_points_[next_change_] <= d.step) {
        ++next_change_;
        demote(d.current ? *d.current : top());
    }

    if (options_.max_run > 0 && last_ && run_length_ >= options_.max_run &&
        std::binary_search(d.runnable.begin(), d.runnable.end(), *last_))
        demote(*last_);

    const ThreadId next = top();
    if (last_ == next) {
        ++run_length_;
    } else {
        last_ = next;
        run_length_ = 1;
    }
    return next;
}

} // namespace lockfree::sched
