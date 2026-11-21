#include "scheduler/explore.hpp"

#include <cinttypes>
#include <cstdio>
#include <exception>
#include <string>
#include <utility>

namespace lockfree::sched {

std::uint64_t run_seed(std::uint64_t seed, std::size_t index) noexcept {
    // Mix the index in before hashing, so neighbouring runs (and neighbouring
    // exploration seeds) don't get correlated streams.
    return SplitMix64(seed ^ (0x6a09e667f3bcc909ULL * (index + 1))).next();
}

RunResult run_case(const CaseFactory& factory, Strategy& strategy, Options options) {
    Case c = factory();
    RunResult r = Scheduler(strategy, options).run(std::move(c.threads));
    if (r.ok() && c.check) {
        try {
            c.check();
        } catch (const std::exception& e) {
            r.outcome = Outcome::failed;
            r.failure = std::string("check: ") + e.what();
        } catch (...) {
            r.outcome = Outcome::failed;
            r.failure = "check: non-standard exception";
        }
    }
    return r;
}

ExploreResult explore(const CaseFactory& factory, std::size_t iterations, std::uint64_t seed,
                      Options options) {
    ExploreResult result;
    RandomStrategy strategy(seed);
    for (std::size_t i = 0; i < iterations; ++i) {
        const std::uint64_t s = run_seed(seed, i);
        strategy.reseed(s);
        RunResult r = run_case(factory, strategy, options);
        ++result.runs;
        if (!r.ok()) {
            std::fprintf(stderr,
                         "lockfree::sched: run %zu of %zu failed after %" PRIu64
                         " steps: %s\n  replay seed: 0x%016" PRIx64 "\n",
                         i + 1, iterations, r.steps, r.failure.c_str(), s);
            result.failing_seed = s;
            result.failure = std::move(r);
            break;
        }
    }
    return result;
}

RunResult replay(const CaseFactory& factory, std::uint64_t seed, Options options) {
    RandomStrategy strategy(seed);
    return run_case(factory, strategy, options);
}

} // namespace lockfree::sched
