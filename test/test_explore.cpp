#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <vector>

#include "lockfree/mpmc_queue.hpp"
#include "scheduler/explore.hpp"

using lockfree::detail::atomic;
using lockfree::sched::Case;
using lockfree::sched::Decision;
using lockfree::sched::Outcome;
using lockfree::sched::RandomStrategy;
using lockfree::sched::require;
using lockfree::sched::ThreadId;

namespace {

// Two threads increment a counter with a separate load and store: the classic
// lost update. Fails only if one thread is preempted between its load and its
// store and the other gets its whole increment in.
Case lost_update() {
    auto x = std::make_shared<atomic<int>>(0);
    auto inc = [x] { x->store(x->load() + 1); };
    return Case{{inc, inc}, [x] { require(x->load() == 2, "lost an increment"); }};
}

// The same, done right.
Case fetch_add_counter() {
    auto x = std::make_shared<atomic<int>>(0);
    auto inc = [x] { x->fetch_add(1); };
    return Case{{inc, inc, inc}, [x] { require(x->load() == 3, "lost an increment"); }};
}

TEST(Explore, FindsALostUpdateAndReplaysIt) {
    auto result = lockfree::sched::explore(lost_update, 1000, 1);
    ASSERT_FALSE(result.ok());
    EXPECT_LT(result.runs, 1000u);
    EXPECT_EQ(result.failure.outcome, Outcome::failed);
    EXPECT_EQ(result.failure.failure, "check: lost an increment");

    // The seed alone brings back the same schedule, every time.
    for (int i = 0; i < 5; ++i) {
        auto again = lockfree::sched::replay(lost_update, *result.failing_seed);
        EXPECT_EQ(again.outcome, Outcome::failed);
        EXPECT_EQ(again.choices, result.failure.choices);
    }
}

TEST(Explore, CorrectCounterSurvivesTheBudget) {
    auto result = lockfree::sched::explore(fetch_add_counter, 500, 7);
    EXPECT_TRUE(result.ok()) << result.failure.failure;
    EXPECT_EQ(result.runs, 500u);
}

TEST(Explore, SameSeedSameExploration) {
    auto a = lockfree::sched::explore(lost_update, 1000, 99);
    auto b = lockfree::sched::explore(lost_update, 1000, 99);
    ASSERT_FALSE(a.ok());
    EXPECT_EQ(a.runs, b.runs);
    EXPECT_EQ(a.failing_seed, b.failing_seed);
}

TEST(Explore, RunsCoverDifferentSchedules) {
    // A random strategy that kept producing the same schedule would be
    // useless. Three threads x three increments have plenty of interleavings.
    std::set<std::vector<ThreadId>> distinct;
    for (std::size_t i = 0; i < 50; ++i) {
        RandomStrategy strategy(lockfree::sched::run_seed(3, i));
        auto r = lockfree::sched::run_case(
            [] {
                auto x = std::make_shared<atomic<int>>(0);
                auto inc = [x] {
                    for (int k = 0; k < 3; ++k) x->fetch_add(1);
                };
                return Case{{inc, inc, inc}, {}};
            },
            strategy);
        ASSERT_TRUE(r.ok());
        distinct.insert(r.choices);
    }
    EXPECT_GT(distinct.size(), 40u);
}

TEST(Explore, RunSeedsDiffer) {
    std::set<std::uint64_t> seeds;
    for (std::uint64_t base : {0ULL, 1ULL})
        for (std::size_t i = 0; i < 100; ++i) seeds.insert(lockfree::sched::run_seed(base, i));
    EXPECT_EQ(seeds.size(), 200u);
}

TEST(Explore, RandomStrategyOnlyPicksRunnable) {
    RandomStrategy strategy(5);
    strategy.begin_run(4);
    const std::vector<ThreadId> runnable{1, 3};
    for (int i = 0; i < 1000; ++i) {
        ThreadId t = strategy.pick(Decision{std::nullopt, nullptr, runnable, 0});
        EXPECT_TRUE(t == 1 || t == 3);
    }
}

TEST(Explore, ThreadFailureIsReportedWithoutACheck) {
    auto result = lockfree::sched::explore(
        [] {
            auto x = std::make_shared<atomic<int>>(0);
            return Case{{[x] { x->store(1); },
                         [x] { require(x->load() == 0, "saw the other thread's store"); }},
                        {}};
        },
        1000, 11);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.failure.failure, "saw the other thread's store");
}

TEST(Explore, MpmcKeepsEveryValueUnderRandomSchedules) {
    // Two producers, two consumers, a queue smaller than the traffic, so runs
    // hit both full and empty. Every value comes out exactly once.
    constexpr int per_producer = 4;
    auto result = lockfree::sched::explore(
        [] {
            struct Shared {
                lockfree::MpmcQueue<int, 2> q;
                atomic<int> consumed{0};
                std::vector<int> seen[2];
            };
            auto s = std::make_shared<Shared>();
            auto producer = [s](int base) {
                return [s, base] {
                    for (int i = 0; i < per_producer; ++i)
                        while (!s->q.try_push(base + i)) {
                        }
                };
            };
            auto consumer = [s](int idx) {
                return [s, idx] {
                    while (s->consumed.load() < 2 * per_producer)
                        if (auto v = s->q.try_pop()) {
                            s->seen[idx].push_back(*v);
                            s->consumed.fetch_add(1);
                        }
                };
            };
            return Case{{producer(0), producer(100), consumer(0), consumer(1)}, [s] {
                            std::multiset<int> all(s->seen[0].begin(), s->seen[0].end());
                            all.insert(s->seen[1].begin(), s->seen[1].end());
                            std::multiset<int> want;
                            for (int i = 0; i < per_producer; ++i) want.insert({i, 100 + i});
                            require(all == want, "values lost or duplicated");
                        }};
        },
        300, 2026);
    EXPECT_TRUE(result.ok()) << result.failure.failure;
}

} // namespace
