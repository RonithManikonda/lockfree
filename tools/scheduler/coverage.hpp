#pragma once

// Coverage for schedule exploration: how much of the interleaving space did a
// batch of runs actually touch?
//
// "N runs passed" says little on its own. A strategy can easily produce the
// same handful of schedules over and over, so two numbers are kept:
//
//   - Context-switch pairs. At every preemption, the pair (site the running
//     thread was stopped at, site the next thread resumes at), where a site is
//     a source line of an atomic access. Most interleaving bugs come down to
//     one specific pair ("T0 between its load and its store, T1 at its
//     store"), so the set of pairs seen is a rough map of which races were
//     tried. It's also bounded by (number of sites)^2, so it saturates, and
//     watching it stop growing tells you more runs won't add much.
//   - Distinct schedules: a hash of each run's choice sequence. Counts how
//     many genuinely different runs there were.
//
// Coverage is collected by wrapping the exploring strategy, so explore()
// needs no changes:
//
//   CoverageStrategy covered(pct);
//   auto result = explore(factory, covered, 1000, seed);
//   std::fputs(covered.coverage().summary().c_str(), stderr);

#include <compare>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "scheduler/explore.hpp"

namespace lockfree::sched {

// A source line of an atomic access. The file is the basename of
// std::source_location::file_name(), which has static storage. A
// default-constructed Site stands for "thread hasn't started yet".
struct Site {
    std::string_view file;
    std::uint32_t line = 0;

    static Site of(const detail::YieldPoint& p) noexcept;
    bool is_start() const noexcept { return line == 0 && file.empty(); }
    std::string to_string() const;

    friend auto operator<=>(const Site&, const Site&) = default;
    friend bool operator==(const Site&, const Site&) = default;
};

using SitePair = std::pair<Site, Site>; // (preempted at, resumed at)

class Coverage {
public:
    void add_switch(const SitePair& pair) { ++pairs_[pair]; }
    void add_schedule(std::uint64_t hash) {
        ++runs_;
        schedules_.insert(hash);
    }

    std::size_t runs() const noexcept { return runs_; }
    std::size_t distinct_schedules() const noexcept { return schedules_.size(); }
    std::size_t distinct_pairs() const noexcept { return pairs_.size(); }
    const std::map<SitePair, std::uint64_t>& pairs() const noexcept { return pairs_; }

    // Totals, then the `top` most frequent pairs and the `top` rarest ones
    // (the rare end is where an extra few thousand runs would help).
    std::string summary(std::size_t top = 5) const;

private:
    std::size_t runs_ = 0;
    std::set<std::uint64_t> schedules_;
    std::map<SitePair, std::uint64_t> pairs_;
};

// Forwards every call to `inner` and records what each run did. Usable
// wherever the inner strategy is, including explore() and replay().
class CoverageStrategy : public SeededStrategy {
public:
    explicit CoverageStrategy(SeededStrategy& inner) noexcept : inner_(inner) {}

    void reseed(std::uint64_t seed) noexcept override { inner_.reseed(seed); }
    const char* name() const noexcept override { return inner_.name(); }

    void begin_run(std::size_t thread_count) override;
    ThreadId pick(const Decision& d) override;

    // Includes the run in progress (or the last one), so it's complete as
    // soon as explore() returns.
    const Coverage& coverage();

private:
    void flush();

    SeededStrategy& inner_;
    Coverage coverage_;
    // Per run: where each thread is parked, and the choice hash so far.
    std::vector<Site> pending_;
    std::uint64_t hash_ = 0;
    bool in_run_ = false;
};

} // namespace lockfree::sched
