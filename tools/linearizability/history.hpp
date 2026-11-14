#pragma once
#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// What a concurrent run looked like from the outside: every operation each
// thread called, what it returned, and when it started and finished. The
// linearizability checker takes a History and a sequential spec and asks
// whether some order of the operations, consistent with real time, would
// have produced exactly these results on a single-threaded structure.
namespace lockfree::lin {

enum class OpKind : std::uint8_t { push, pop };

struct Operation {
    std::uint32_t thread = 0;
    OpKind kind = OpKind::push;
    // The argument of a push. Unused for pop.
    std::int64_t value = 0;
    // What a pop returned: the value, or nullopt for "empty". Unused for push.
    // Only successful pushes are recorded: a try_push that found a bounded
    // queue full changed nothing, and the specs don't model capacity.
    std::optional<std::int64_t> result;
    // Timestamps from one clock shared by all threads (steady_clock in ns).
    // The operation took effect at some instant inside [invoke_ts, response_ts].
    std::uint64_t invoke_ts = 0;
    std::uint64_t response_ts = 0;
};

// a finished before b started, so every valid order must put a first. Ties
// count as overlapping: with a coarse clock (24 MHz on Apple Silicon) equal
// stamps don't prove anything about order.
inline bool happens_before(const Operation& a, const Operation& b) noexcept {
    return a.response_ts < b.invoke_ts;
}

inline bool overlaps(const Operation& a, const Operation& b) noexcept {
    return !happens_before(a, b) && !happens_before(b, a);
}

// "t2 pop -> 7 [100, 140]", "t0 push 3 [90, 95]", "t1 pop -> empty [...]".
inline std::string to_string(const Operation& op) {
    std::string s = "t" + std::to_string(op.thread);
    if (op.kind == OpKind::push) {
        s += " push " + std::to_string(op.value);
    } else {
        s += " pop -> ";
        s += op.result ? std::to_string(*op.result) : std::string("empty");
    }
    s += " [" + std::to_string(op.invoke_ts) + ", " + std::to_string(op.response_ts) + "]";
    return s;
}

class History {
public:
    History() = default;
    explicit History(std::vector<Operation> ops) : ops_(std::move(ops)) {}

    void add(const Operation& op) { ops_.push_back(op); }

    const std::vector<Operation>& ops() const noexcept { return ops_; }
    std::size_t size() const noexcept { return ops_.size(); }
    bool empty() const noexcept { return ops_.empty(); }
    const Operation& operator[](std::size_t i) const noexcept { return ops_[i]; }

    // Invocation order, ties broken by thread so the result is deterministic.
    // The checker walks candidates in this order.
    void sort_by_invoke() {
        std::stable_sort(ops_.begin(), ops_.end(), [](const Operation& a, const Operation& b) {
            if (a.invoke_ts != b.invoke_ts) return a.invoke_ts < b.invoke_ts;
            return a.thread < b.thread;
        });
    }

    // A recorder bug, not a structure bug, if this fails: every op must end
    // after it starts, and one thread's ops must not overlap each other since
    // a thread does one call at a time. Returns a description of the first
    // problem, or nullopt.
    std::optional<std::string> check_well_formed() const {
        std::unordered_map<std::uint32_t, const Operation*> last_by_thread;
        std::vector<const Operation*> sorted;
        sorted.reserve(ops_.size());
        for (const Operation& op : ops_) {
            if (op.response_ts < op.invoke_ts) return "responds before invoked: " + to_string(op);
            if (op.kind == OpKind::push && op.result) return "push with a result: " + to_string(op);
            sorted.push_back(&op);
        }
        std::stable_sort(sorted.begin(), sorted.end(), [](const Operation* a, const Operation* b) {
            return a->invoke_ts < b->invoke_ts;
        });
        for (const Operation* op : sorted) {
            auto [it, first] = last_by_thread.try_emplace(op->thread, op);
            if (first) continue;
            // Equal stamps are allowed: two quick calls can land in one tick.
            if (it->second->response_ts > op->invoke_ts) {
                return "overlapping ops on one thread: " + to_string(*it->second) + " and " +
                       to_string(*op);
            }
            it->second = op;
        }
        return std::nullopt;
    }

private:
    std::vector<Operation> ops_;
};

} // namespace lockfree::lin
