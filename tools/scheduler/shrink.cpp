#include "scheduler/shrink.hpp"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <string_view>

namespace lockfree::sched {

std::vector<Switch> switches_of(const std::vector<TraceStep>& trace) {
    std::vector<Switch> out;
    for (const TraceStep& t : trace)
        if (!t.from || *t.from != t.thread) out.push_back({t.from, t.from_count, t.thread});
    return out;
}

ThreadId ScriptedStrategy::pick(const Decision& d) {
    const std::uint64_t at = d.current ? count_[*d.current]++ : none_count_++;
    for (const Switch& s : script_) {
        if (s.from == d.current && s.at == at) {
            if (std::binary_search(d.runnable.begin(), d.runnable.end(), s.to)) return s.to;
            break;
        }
    }
    if (d.current) return *d.current;
    return d.runnable.front();
}

namespace {

// Wraps another strategy and writes down what every step did. A thread that's
// picked performs the access it was stopped at (its last yield point), or
// starts up if it hasn't run yet.
class Recorder : public Strategy {
public:
    Recorder(Strategy& inner, std::vector<TraceStep>& out) : inner_(inner), out_(out) {}

    void begin_run(std::size_t n) override {
        inner_.begin_run(n);
        pending_.assign(n, std::nullopt);
        count_.assign(n, 0);
        none_count_ = 0;
        out_.clear();
    }

    ThreadId pick(const Decision& d) override {
        if (d.current) pending_[*d.current] = *d.point;
        const std::uint64_t at = d.current ? count_[*d.current]++ : none_count_++;
        const ThreadId t = inner_.pick(d);
        out_.push_back({d.step, d.current, at, t, pending_[t]});
        return t;
    }

private:
    Strategy& inner_;
    std::vector<TraceStep>& out_;
    std::vector<std::optional<detail::YieldPoint>> pending_;
    std::vector<std::uint64_t> count_;
    std::uint64_t none_count_ = 0;
};

// Replays a recorded choice sequence verbatim while it fits.
class ChoiceReplay : public Strategy {
public:
    explicit ChoiceReplay(const std::vector<ThreadId>& choices) : choices_(choices) {}

    ThreadId pick(const Decision& d) override {
        if (d.step < choices_.size() &&
            std::binary_search(d.runnable.begin(), d.runnable.end(), choices_[d.step]))
            return choices_[d.step];
        return d.current ? *d.current : d.runnable.front();
    }

private:
    const std::vector<ThreadId>& choices_;
};

const char* op_name(detail::AtomicOp op) {
    switch (op) {
    case detail::AtomicOp::load: return "load";
    case detail::AtomicOp::store: return "store";
    case detail::AtomicOp::exchange: return "exchange";
    case detail::AtomicOp::compare_exchange: return "cas";
    case detail::AtomicOp::fetch_add: return "fetch_add";
    case detail::AtomicOp::fetch_sub: return "fetch_sub";
    }
    return "?";
}

std::string_view basename(const char* path) {
    std::string_view p(path);
    const auto slash = p.find_last_of("/\\");
    return slash == std::string_view::npos ? p : p.substr(slash + 1);
}

bool same_failure(const RunResult& a, const RunResult& b) {
    return a.outcome == b.outcome && a.failure == b.failure;
}

} // namespace

RunResult replay_switches(const CaseFactory& factory, const std::vector<Switch>& script,
                          Options options, std::vector<TraceStep>* trace) {
    ScriptedStrategy scripted(script);
    if (trace == nullptr) return run_case(factory, scripted, options);
    Recorder recorder(scripted, *trace);
    return run_case(factory, recorder, options);
}

RunResult replay_choices(const CaseFactory& factory, const std::vector<ThreadId>& choices,
                         Options options, std::vector<TraceStep>* trace) {
    ChoiceReplay exact(choices);
    if (trace == nullptr) return run_case(factory, exact, options);
    Recorder recorder(exact, *trace);
    return run_case(factory, recorder, options);
}

std::string format_trace(const std::vector<TraceStep>& trace) {
    std::string out;
    char line[256];
    std::optional<ThreadId> last;
    for (const TraceStep& s : trace) {
        if (last && s.preempted()) {
            std::snprintf(line, sizeof line, "  T%zu  <- preempts T%zu\n", s.thread, *last);
            out += line;
        }
        if (s.access) {
            const std::string_view file = basename(s.access->where.file_name());
            std::snprintf(line, sizeof line, "  T%zu  %-10s %.*s:%" PRIuLEAST32 "\n", s.thread,
                          op_name(s.access->op), static_cast<int>(file.size()), file.data(),
                          s.access->where.line());
        } else {
            std::snprintf(line, sizeof line, "  T%zu  start\n", s.thread);
        }
        out += line;
        last = s.thread;
    }
    return out;
}

ShrinkResult shrink(const CaseFactory& factory, const std::vector<ThreadId>& failing_choices,
                    Options options, std::size_t max_attempts) {
    ShrinkResult result;
    std::vector<TraceStep> original;
    const RunResult target = replay_choices(factory, failing_choices, options, &original);
    std::vector<Switch> best = switches_of(original);
    result.original_switches = best.size();
    result.attempts = 1;
    if (target.ok()) {
        result.script = std::move(best);
        result.run = target;
        return result;
    }
    result.reproduced = true;

    auto still_fails = [&](const std::vector<Switch>& candidate) {
        ++result.attempts;
        return same_failure(replay_switches(factory, candidate, options), target);
    };

    // ddmin over the switch list, removing complements only: every accepted
    // step makes the list strictly shorter.
    std::size_t chunks = 2;
    while (best.size() >= 1 && result.attempts < max_attempts) {
        chunks = std::min(chunks, best.size());
        const std::size_t size = (best.size() + chunks - 1) / chunks;
        bool removed = false;
        for (std::size_t start = 0; start < best.size() && result.attempts < max_attempts;
             start += size) {
            std::vector<Switch> candidate(best.begin(), best.begin() + start);
            candidate.insert(candidate.end(),
                             best.begin() + std::min(start + size, best.size()), best.end());
            if (still_fails(candidate)) {
                best = std::move(candidate);
                chunks = std::max<std::size_t>(chunks - 1, 2);
                removed = true;
                break;
            }
        }
        if (removed) continue;
        if (chunks >= best.size()) break; // single switches, none removable: 1-minimal
        chunks = std::min(chunks * 2, best.size());
    }

    result.script = std::move(best);
    result.run = replay_switches(factory, result.script, options, &result.trace);
    ++result.attempts;
    return result;
}

} // namespace lockfree::sched
