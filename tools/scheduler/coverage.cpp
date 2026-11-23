#include "scheduler/coverage.hpp"

#include <algorithm>
#include <cinttypes>
#include <cstdio>

namespace lockfree::sched {
namespace {

constexpr std::uint64_t fnv_offset = 0xcbf29ce484222325ULL;
constexpr std::uint64_t fnv_prime = 0x100000001b3ULL;

std::string_view basename(const char* path) {
    std::string_view p(path);
    const auto slash = p.find_last_of("/\\");
    return slash == std::string_view::npos ? p : p.substr(slash + 1);
}

std::string pair_line(const SitePair& p, std::uint64_t count) {
    char line[256];
    std::snprintf(line, sizeof line, "  %8" PRIu64 "  %s -> %s\n", count, p.first.to_string().c_str(),
                  p.second.to_string().c_str());
    return line;
}

} // namespace

Site Site::of(const detail::YieldPoint& p) noexcept {
    return Site{basename(p.where.file_name()), static_cast<std::uint32_t>(p.where.line())};
}

std::string Site::to_string() const {
    if (is_start()) return "start";
    return std::string(file) + ":" + std::to_string(line);
}

std::string Coverage::summary(std::size_t top) const {
    char head[192];
    std::snprintf(head, sizeof head,
                  "coverage: %zu runs, %zu distinct schedules, %zu distinct switch pairs\n", runs_,
                  schedules_.size(), pairs_.size());
    std::string out = head;
    if (pairs_.empty() || top == 0) return out;

    std::vector<std::pair<SitePair, std::uint64_t>> sorted(pairs_.begin(), pairs_.end());
    // Most frequent first; ties keep the map's (site) order, so the output is
    // stable from one run of the program to the next.
    std::stable_sort(sorted.begin(), sorted.end(),
                     [](const auto& a, const auto& b) { return a.second > b.second; });

    const std::size_t n = std::min(top, sorted.size());
    out += "most frequent:\n";
    for (std::size_t i = 0; i < n; ++i) out += pair_line(sorted[i].first, sorted[i].second);
    if (sorted.size() > n) {
        out += "rarest:\n";
        const std::size_t from = std::max(n, sorted.size() - n);
        for (std::size_t i = sorted.size(); i-- > from;)
            out += pair_line(sorted[i].first, sorted[i].second);
    }
    return out;
}

void CoverageStrategy::flush() {
    if (in_run_) coverage_.add_schedule(hash_);
    in_run_ = false;
}

void CoverageStrategy::begin_run(std::size_t thread_count) {
    flush();
    inner_.begin_run(thread_count);
    pending_.assign(thread_count, Site{});
    hash_ = fnv_offset;
    in_run_ = true;
}

ThreadId CoverageStrategy::pick(const Decision& d) {
    if (d.current) pending_[*d.current] = Site::of(*d.point);
    const ThreadId next = inner_.pick(d);

    // Only preemptions count: a pick with nobody current (start of the run,
    // or after a thread finished) isn't a race between two sites.
    if (d.current && next != *d.current) coverage_.add_switch({pending_[*d.current], pending_[next]});

    hash_ = (hash_ ^ static_cast<std::uint64_t>(next)) * fnv_prime;
    return next;
}

const Coverage& CoverageStrategy::coverage() {
    flush();
    return coverage_;
}

} // namespace lockfree::sched
