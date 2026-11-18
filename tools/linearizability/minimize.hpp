#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "linearizability/checker.hpp"
#include "linearizability/history.hpp"

// Shrinks a non-linearizable history to a small piece that still fails.
//
// A stress run that fails the checker has thousands of ops, and the failing
// segment alone can't be checked on its own: it starts with whatever the
// earlier segments left in the structure. So instead of cutting the history
// by time, this cuts it by *value*. An element is either one value (every
// push of it and every pop that returned it) or one empty pop.
//
// Dropping a whole element is safe in one direction: if the full history has
// a legal order, deleting one value's pushes and pops from that order leaves a
// legal order for the rest (the other values keep their relative order in the
// queue or stack, and an empty pop stays empty). So if what's left still
// fails, the original failed for a reason that's entirely inside what's left.
// That makes the result a real witness, not just a smaller history that
// happens to fail differently.
//
// The search drops elements while the rest still fails, big chunks first,
// then single elements until a full pass drops nothing. The result is
// 1-minimal: removing any single remaining element makes it pass. It starts
// from the elements that touch the failing segment and falls back to all of
// them only if those alone pass.
namespace lockfree::lin {

struct MinimizeResult {
    History window;           // sorted by invocation
    std::size_t checks = 0;   // checker calls spent
};

namespace detail {

// Element key: a value, or the index of an empty pop (kept apart from values
// by the bool).
using ElementKey = std::pair<bool, std::int64_t>;

inline ElementKey element_of(const Operation& op, std::size_t index) {
    if (op.kind == OpKind::push) return {false, op.value};
    if (op.result) return {false, *op.result};
    return {true, static_cast<std::int64_t>(index)};
}

} // namespace detail

// nullopt if `history` doesn't fail the check (linearizable, ill-formed, or
// the checker gave up).
template <typename Spec>
std::optional<MinimizeResult> minimize(const History& history, const CheckOptions& options = {}) {
    using detail::ElementKey;

    History sorted = history;
    sorted.sort_by_invoke();
    const auto& ops = sorted.ops();

    MinimizeResult out;
    auto fails = [&](const std::set<ElementKey>& keep) {
        History h;
        for (std::size_t i = 0; i < ops.size(); ++i) {
            if (keep.count(detail::element_of(ops[i], i))) h.add(ops[i]);
        }
        ++out.checks;
        return check<Spec>(h, options).verdict == Verdict::not_linearizable;
    };

    std::set<ElementKey> all;
    for (std::size_t i = 0; i < ops.size(); ++i) all.insert(detail::element_of(ops[i], i));
    ++out.checks;
    const CheckResult full = check<Spec>(sorted, options);
    if (full.verdict != Verdict::not_linearizable) return std::nullopt;

    // The failing segment's ops, located by identity in the sorted history
    // so empty pops get the right index.
    std::set<ElementKey> keep;
    for (const Operation& f : full.failing_segment) {
        for (std::size_t i = 0; i < ops.size(); ++i) {
            const Operation& o = ops[i];
            if (o.thread == f.thread && o.kind == f.kind && o.value == f.value &&
                o.result == f.result && o.invoke_ts == f.invoke_ts &&
                o.response_ts == f.response_ts) {
                keep.insert(detail::element_of(o, i));
                break;
            }
        }
    }
    if (keep.empty() || !fails(keep)) keep = all;

    // Delta-debugging style: try dropping big chunks first (halves, then
    // quarters, ...), so a long history whose problem is in a few values costs
    // O(k log n) checks rather than one check per element. At chunk size 1,
    // repeat until a pass drops nothing; that's what makes it 1-minimal.
    std::vector<ElementKey> elements(keep.begin(), keep.end());
    std::size_t chunk = std::max<std::size_t>(elements.size() / 2, 1);
    for (;;) {
        bool shrunk = false;
        for (std::size_t begin = 0; begin < elements.size();) {
            const std::size_t end = std::min(begin + chunk, elements.size());
            std::set<ElementKey> rest(elements.begin(), elements.begin() + begin);
            rest.insert(elements.begin() + end, elements.end());
            if (!rest.empty() && fails(rest)) {
                elements.erase(elements.begin() + begin, elements.begin() + end);
                shrunk = true;
            } else {
                begin = end;
            }
        }
        if (chunk > 1) {
            chunk /= 2;
        } else if (!shrunk) {
            break;
        }
    }
    keep = std::set<ElementKey>(elements.begin(), elements.end());

    for (std::size_t i = 0; i < ops.size(); ++i) {
        if (keep.count(detail::element_of(ops[i], i))) out.window.add(ops[i]);
    }
    return out;
}

} // namespace lockfree::lin
