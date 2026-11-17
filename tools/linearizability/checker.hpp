#pragma once
#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "linearizability/history.hpp"

// Linearizability checker.
//
// A history is linearizable if the operations can be put in one sequential
// order such that
//   1. the order respects real time: if a finished before b started, a comes
//      first (overlapping operations may go either way), and
//   2. running that order against the sequential spec reproduces every
//      recorded result.
//
// The search is Wing & Gong's, with Lowe's "just-in-time" refinement and
// memoization:
//
//   - At each step only *minimal* operations are candidates: unlinearized ops
//     that no other unlinearized op happens-before. In practice that's the
//     ops invoked before the earliest pending response, so the branching
//     factor is the number of ops in flight at once, not the history length.
//   - A candidate is taken only if the spec accepts its recorded result in
//     the current state; otherwise that whole branch is cut immediately.
//   - (set of linearized ops, spec state) pairs already explored are
//     remembered. Two branches that linearized the same ops and ended in the
//     same state have identical futures, so the second one is skipped. This
//     is what turns the worst case from "every permutation" into "every
//     reachable configuration".
//
// Long histories are cut at quiescent points: places where every op so far
// has responded before the next one is invoked. Every op before such a point
// happens-before every op after it, so the search finishes one segment
// completely before it can touch the next, and the memo key only needs the
// current segment's bits instead of one bit per op in the whole history.
//
// It's still one search, not one per segment. A segment can end in several
// spec states (overlapping pushes of 1 and 2 leave [1,2] or [2,1]) and only
// some may fit what the next segment saw, so the search can back up across a
// quiescent point and try another. A start state that already failed for a
// segment is in the memo too (segment k, nothing done yet, that state), so it
// isn't retried. Precomputing every end state of every segment instead would
// be exponential in the number of unpopped overlapping pushes, which stress
// runs produce all the time.
namespace lockfree::lin {

enum class Verdict : std::uint8_t {
    linearizable,
    not_linearizable,
    ill_formed, // the history itself is broken (recorder bug), see message
    gave_up,    // hit a CheckOptions limit before reaching an answer
};

inline const char* to_string(Verdict v) noexcept {
    switch (v) {
    case Verdict::linearizable: return "linearizable";
    case Verdict::not_linearizable: return "not linearizable";
    case Verdict::ill_formed: return "ill-formed history";
    case Verdict::gave_up: return "gave up";
    }
    return "?";
}

struct CheckOptions {
    // Distinct (segment, linearized ops, state) configurations visited.
    // Bounds time and memory on a pathological history.
    std::size_t max_configurations = 20'000'000;
};

struct CheckResult {
    Verdict verdict = Verdict::linearizable;
    std::string message;
    std::size_t segments = 0;
    std::size_t configurations = 0;
    // On not_linearizable: the segment where the search got furthest before
    // getting stuck, in invocation order, and the longest legal order it found
    // inside that segment. The ops of the segment missing from best_order are
    // the ones to look at.
    std::vector<Operation> failing_segment;
    std::vector<Operation> best_order;

    explicit operator bool() const noexcept { return verdict == Verdict::linearizable; }
};

// Splits ops (sorted by invocation) at quiescent points. Each returned span
// is one segment; their concatenation is `ops`.
inline std::vector<std::span<const Operation>> split_at_quiescent_points(
    std::span<const Operation> ops) {
    std::vector<std::span<const Operation>> segments;
    std::size_t begin = 0;
    std::uint64_t latest_response = 0;
    for (std::size_t i = 0; i < ops.size(); ++i) {
        // Strict, like happens_before(): a tie means the clock can't order
        // them, so they stay in one segment.
        if (i > begin && latest_response < ops[i].invoke_ts) {
            segments.push_back(ops.subspan(begin, i - begin));
            begin = i;
            latest_response = 0;
        }
        latest_response = std::max(latest_response, ops[i].response_ts);
    }
    if (begin < ops.size()) segments.push_back(ops.subspan(begin));
    return segments;
}

namespace detail {

// Which ops of a segment are already linearized.
class OpSet {
public:
    explicit OpSet(std::size_t n = 0) : words_((n + 63) / 64, 0) {}

    bool contains(std::size_t i) const noexcept { return (words_[i / 64] >> (i % 64)) & 1u; }
    void insert(std::size_t i) noexcept { words_[i / 64] |= std::uint64_t{1} << (i % 64); }
    void erase(std::size_t i) noexcept { words_[i / 64] &= ~(std::uint64_t{1} << (i % 64)); }

    // Lowest index not in the set (may be >= the segment size). Skips full
    // words, so it stays cheap on long segments where the front is all done.
    std::size_t first_missing() const noexcept {
        for (std::size_t w = 0; w < words_.size(); ++w) {
            if (words_[w] != ~std::uint64_t{0}) {
                return w * 64 + static_cast<std::size_t>(std::countr_one(words_[w]));
            }
        }
        return words_.size() * 64;
    }

    std::size_t hash() const noexcept {
        std::uint64_t h = 0x9E3779B97F4A7C15ull;
        for (std::uint64_t w : words_) h = (h ^ w) * 0x100000001B3ull + (h >> 29);
        return static_cast<std::size_t>(h);
    }
    friend bool operator==(const OpSet&, const OpSet&) = default;

private:
    std::vector<std::uint64_t> words_;
};

template <typename Spec>
struct Config {
    std::size_t segment;
    OpSet done;
    Spec state;
    friend bool operator==(const Config&, const Config&) = default;
};

template <typename Spec>
struct ConfigHash {
    std::size_t operator()(const Config<Spec>& c) const noexcept {
        return (c.done.hash() + c.segment) ^ (c.state.hash() * 0x9E3779B97F4A7C15ull);
    }
};

// The search itself: an explicit DFS (a history can be tens of thousands of
// ops deep, too deep to recurse on). One frame per linearized op, holding the
// state *before* that op, so backtracking is a pop.
template <typename Spec>
class Search {
public:
    enum class Status { found, exhausted, budget };

    Search(std::vector<std::span<const Operation>> segments, std::size_t budget)
        : segments_(std::move(segments)), budget_(budget) {
        done_.reserve(segments_.size());
        for (auto seg : segments_) done_.emplace_back(seg.size());
        count_.assign(segments_.size(), 0);
    }

    Status run() {
        if (segments_.empty()) return Status::found;
        Spec state;
        if (!visit(state)) return budget_hit_ ? Status::budget : Status::exhausted;

        struct Frame {
            Spec before;
            std::size_t segment;
            std::size_t op;   // index inside the segment
            std::size_t rank; // its position in that depth's candidate list
        };
        std::vector<Frame> stack;
        std::size_t scan = 0; // rank of the next candidate to try at this depth

        for (;;) {
            const auto ops = segments_[seg_];
            const std::vector<std::size_t> candidates = minimal_ops();
            bool advanced = false;
            for (std::size_t rank = scan; rank < candidates.size(); ++rank) {
                const std::size_t i = candidates[rank];
                Spec next = state;
                if (!next.apply(ops[i])) continue;
                take(i);
                if (!visit(next)) {
                    untake(seg_of_last_take_, i);
                    if (budget_hit_) return Status::budget;
                    continue;
                }
                stack.push_back({std::move(state), seg_of_last_take_, i, rank});
                state = std::move(next);
                advanced = true;
                break;
            }

            if (advanced) {
                if (seg_ == segments_.size()) return Status::found;
                scan = 0;
                continue;
            }

            // Dead end. Every path that doesn't succeed ends in one, so
            // remembering the deepest dead end gives the deepest path without
            // copying the stack on every step forward.
            if (stack.size() > best_.size()) {
                best_.clear();
                for (const Frame& f : stack) best_.emplace_back(f.segment, f.op);
            }

            // Undo the last op and try its next sibling.
            if (stack.empty()) return Status::exhausted;
            Frame top = std::move(stack.back());
            stack.pop_back();
            untake(top.segment, top.op);
            state = std::move(top.before);
            scan = top.rank + 1;
        }
    }

    std::size_t configurations() const noexcept { return configurations_; }

    // The deepest legal order the search reached, as (segment, index) pairs.
    const std::vector<std::pair<std::size_t, std::size_t>>& best_order() const noexcept {
        return best_;
    }

private:
    // Marks op i of the current segment linearized, moving on to the next
    // segment once this one is complete.
    void take(std::size_t i) noexcept {
        seg_of_last_take_ = seg_;
        done_[seg_].insert(i);
        if (++count_[seg_] == segments_[seg_].size()) ++seg_;
    }

    void untake(std::size_t segment, std::size_t i) noexcept {
        done_[segment].erase(i);
        --count_[segment];
        seg_ = segment;
    }

    // The minimal unlinearized ops of the current segment, earliest
    // response first.
    //
    // An op is minimal iff no pending op responded strictly before it was
    // invoked, i.e. its invoke_ts <= the earliest response among unlinearized
    // ops. Ops are sorted by invocation, so that earliest response can be
    // found by walking forward from the first pending op until one is invoked
    // after the running minimum: every later op responds later still and
    // can't lower it. Earlier segments are all done and later ones all happen
    // after this one, so only the current segment matters.
    //
    // The order is a heuristic, not a correctness matter: every candidate is
    // eventually tried. Earliest response first is "linearize what's about to
    // become overdue": a push that's still in flight for a while can wait
    // until a pop needs its value, instead of being guessed into the queue
    // early and found out of order thousands of ops later.
    std::vector<std::size_t> minimal_ops() const {
        const auto ops = segments_[seg_];
        const OpSet& done = done_[seg_];
        const std::size_t first = done.first_missing();
        std::uint64_t min_response = UINT64_MAX;
        std::size_t end = first;
        for (; end < ops.size() && ops[end].invoke_ts <= min_response; ++end) {
            if (!done.contains(end)) min_response = std::min(min_response, ops[end].response_ts);
        }
        std::vector<std::size_t> out;
        for (std::size_t i = first; i < end; ++i) {
            if (!done.contains(i) && ops[i].invoke_ts <= min_response) out.push_back(i);
        }
        std::stable_sort(out.begin(), out.end(), [&](std::size_t a, std::size_t b) {
            return ops[a].response_ts < ops[b].response_ts;
        });
        return out;
    }

    // Records the current (segment, done ops, state) as explored. False if it
    // already was, or the budget ran out. A finished history is never
    // "already explored": reaching it ends the search.
    bool visit(const Spec& s) {
        if (seg_ == segments_.size()) return true;
        if (configurations_ >= budget_) {
            budget_hit_ = true;
            return false;
        }
        if (!memo_.insert(Config<Spec>{seg_, done_[seg_], s}).second) return false;
        ++configurations_;
        return true;
    }

    std::vector<std::span<const Operation>> segments_;
    std::vector<OpSet> done_;
    std::vector<std::size_t> count_;
    std::size_t seg_ = 0;
    std::size_t seg_of_last_take_ = 0;
    std::unordered_set<Config<Spec>, ConfigHash<Spec>> memo_;
    std::vector<std::pair<std::size_t, std::size_t>> best_;
    std::size_t configurations_ = 0;
    std::size_t budget_;
    bool budget_hit_ = false;
};

} // namespace detail

// Checks `history` against the sequential spec `Spec` (FifoQueueSpec,
// LifoStackSpec, ...). The history doesn't need to be sorted.
template <typename Spec>
CheckResult check(const History& history, const CheckOptions& options = {}) {
    CheckResult result;
    if (auto problem = history.check_well_formed()) {
        result.verdict = Verdict::ill_formed;
        result.message = *problem;
        return result;
    }

    History sorted = history;
    sorted.sort_by_invoke();
    auto segments = split_at_quiescent_points(sorted.ops());
    result.segments = segments.size();

    detail::Search<Spec> search(segments, options.max_configurations);
    const auto status = search.run();
    result.configurations = search.configurations();

    using Status = typename detail::Search<Spec>::Status;
    if (status == Status::budget) {
        result.verdict = Verdict::gave_up;
        result.message = "gave up after " + std::to_string(result.configurations) +
                         " configurations (" + std::to_string(segments.size()) + " segments)";
    } else if (status == Status::exhausted) {
        // The deepest path finished some number of whole segments; the one
        // after those is where it got stuck.
        const auto& best = search.best_order();
        std::size_t stuck = 0;
        for (const auto& [seg, op] : best) {
            if (seg > stuck) stuck = seg;
        }
        std::size_t in_stuck = 0;
        for (const auto& [seg, op] : best) in_stuck += seg == stuck;
        if (!best.empty() && in_stuck == segments[stuck].size()) {
            ++stuck;
            in_stuck = 0;
        }
        result.verdict = Verdict::not_linearizable;
        result.failing_segment.assign(segments[stuck].begin(), segments[stuck].end());
        for (const auto& [seg, op] : best) {
            if (seg == stuck) result.best_order.push_back(segments[seg][op]);
        }
        result.message = "no legal order for segment " + std::to_string(stuck + 1) + " of " +
                         std::to_string(segments.size()) + " (" +
                         std::to_string(segments[stuck].size()) +
                         " ops); longest legal prefix has " + std::to_string(in_stuck);
    }
    return result;
}

} // namespace lockfree::lin
