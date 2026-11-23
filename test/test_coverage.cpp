#include <gtest/gtest.h>

#include <cstddef>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "scheduler/coverage.hpp"
#include "scheduler/explore.hpp"
#include "scheduler/pct.hpp"

using lockfree::detail::atomic;
using lockfree::sched::Case;
using lockfree::sched::Coverage;
using lockfree::sched::CoverageStrategy;
using lockfree::sched::PctOptions;
using lockfree::sched::PctStrategy;
using lockfree::sched::RandomStrategy;
using lockfree::sched::require;
using lockfree::sched::Site;
using lockfree::sched::SitePair;

namespace {

// Two threads, each a load then a store on line kIncLine. The lost update is
// still possible, but the check accepts it (x is 1 or 2), so every run passes
// and the tests here only look at what coverage recorded.
constexpr std::uint32_t kIncLine = __LINE__ + 4;
Case racy_increments() {
    auto x = std::make_shared<atomic<int>>(0);
    auto inc = [x] {
        x->store(x->load() + 1);
    };
    return Case{{inc, inc}, [x] {
                    const int v = x->load();
                    require(v == 1 || v == 2, "impossible value");
                }};
}

TEST(Coverage, SiteOrderingAndFormatting) {
    const Site start{};
    const Site a{"a.cpp", 3};
    const Site b{"a.cpp", 7};
    EXPECT_TRUE(start.is_start());
    EXPECT_FALSE(a.is_start());
    EXPECT_LT(start, a);
    EXPECT_LT(a, b);
    EXPECT_EQ(start.to_string(), "start");
    EXPECT_EQ(b.to_string(), "a.cpp:7");
}

TEST(Coverage, SummaryListsFrequentAndRarePairs) {
    Coverage c;
    const Site s1{"q.hpp", 10}, s2{"q.hpp", 20}, s3{"q.hpp", 30};
    for (int i = 0; i < 5; ++i) c.add_switch({s1, s2});
    c.add_switch({s2, s3});
    c.add_switch({s3, s1});
    c.add_schedule(1);
    c.add_schedule(1);
    c.add_schedule(2);
    EXPECT_EQ(c.runs(), 3u);
    EXPECT_EQ(c.distinct_schedules(), 2u);
    EXPECT_EQ(c.distinct_pairs(), 3u);

    const std::string s = c.summary(1);
    EXPECT_NE(s.find("3 runs, 2 distinct schedules, 3 distinct switch pairs"), std::string::npos) << s;
    // Top pair first, then the rarest ones.
    const auto most = s.find("q.hpp:10 -> q.hpp:20");
    const auto rare = s.find("rarest:");
    ASSERT_NE(most, std::string::npos) << s;
    ASSERT_NE(rare, std::string::npos) << s;
    EXPECT_LT(most, rare);
}

TEST(Coverage, NoPreemptionsMeansNoPairs) {
    // PCT at depth 1 with no CAS bias and nothing to spin on never preempts:
    // every pick with somebody current keeps that thread.
    PctOptions o;
    o.depth = 1;
    o.cas_preempt = 0.0;
    PctStrategy pct(1, o);
    CoverageStrategy covered(pct);
    auto result = lockfree::sched::explore(racy_increments, covered, 50, 7);
    ASSERT_TRUE(result.ok()) << result.failure.failure;

    const Coverage& c = covered.coverage();
    EXPECT_EQ(c.runs(), 50u);
    EXPECT_EQ(c.distinct_pairs(), 0u);
    // Only the order the two threads run in varies.
    EXPECT_LE(c.distinct_schedules(), 2u);
}

TEST(Coverage, RandomExplorationFindsTheRaceSites) {
    RandomStrategy random(0);
    CoverageStrategy covered(random);
    auto result = lockfree::sched::explore(racy_increments, covered, 300, 11);
    ASSERT_TRUE(result.ok()) << result.failure.failure;

    const Coverage& c = covered.coverage();
    EXPECT_EQ(c.runs(), 300u);
    EXPECT_GT(c.distinct_schedules(), 5u);

    // Every access is on kIncLine, so every pair is either (inc, inc) or
    // (inc, start) for a thread that hasn't run yet.
    const Site inc{"test_coverage.cpp", kIncLine};
    const Site start{};
    for (const auto& [pair, count] : c.pairs()) {
        EXPECT_EQ(pair.first, inc) << pair.first.to_string();
        EXPECT_TRUE(pair.second == inc || pair.second == start) << pair.second.to_string();
        EXPECT_GT(count, 0u);
    }
    EXPECT_EQ(c.pairs().count(SitePair{inc, inc}), 1u);
    EXPECT_EQ(c.pairs().count(SitePair{inc, start}), 1u);
    std::fputs(covered.coverage().summary(3).c_str(), stdout);
}

TEST(Coverage, SameSeedSameCoverage) {
    auto collect = [] {
        RandomStrategy random(0);
        CoverageStrategy covered(random);
        lockfree::sched::explore(racy_increments, covered, 100, 42);
        return covered.coverage();
    };
    const Coverage a = collect();
    const Coverage b = collect();
    EXPECT_EQ(a.distinct_schedules(), b.distinct_schedules());
    EXPECT_EQ(a.pairs(), b.pairs());
}

TEST(Coverage, ReplayingOneSeedIsOneSchedule) {
    RandomStrategy random(0);
    CoverageStrategy covered(random);
    const std::uint64_t seed = lockfree::sched::run_seed(9, 0);
    for (int i = 0; i < 4; ++i) lockfree::sched::replay(racy_increments, covered, seed);
    EXPECT_EQ(covered.coverage().runs(), 4u);
    EXPECT_EQ(covered.coverage().distinct_schedules(), 1u);
}

} // namespace
