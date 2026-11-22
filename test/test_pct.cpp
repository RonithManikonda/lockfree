#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <vector>

#include "lockfree/mpmc_queue.hpp"
#include "scheduler/explore.hpp"
#include "scheduler/pct.hpp"

using lockfree::detail::atomic;
using lockfree::sched::Case;
using lockfree::sched::Decision;
using lockfree::sched::Outcome;
using lockfree::sched::PctOptions;
using lockfree::sched::PctStrategy;
using lockfree::sched::require;
using lockfree::sched::ThreadId;

namespace {

PctOptions opts(std::size_t depth, double cas_preempt, std::uint64_t max_run = 64) {
    PctOptions o;
    o.depth = depth;
    o.expected_steps = 20;
    o.max_run = max_run;
    o.cas_preempt = cas_preempt;
    return o;
}

// Load, then store: lost if a thread is preempted between the two. That's a
// depth-2 bug (one preemption in the right place).
Case lost_update() {
    auto x = std::make_shared<atomic<int>>(0);
    auto inc = [x] { x->store(x->load() + 1); };
    return Case{{inc, inc}, [x] { require(x->load() == 2, "lost an increment"); }};
}

// A claims a flag with a CAS and only then publishes the data the flag
// promises. B trusts the flag. Broken exactly when A is preempted right after
// its CAS, which is what the CAS bias aims for.
Case flag_before_data() {
    struct Shared {
        atomic<int> flag{0};
        atomic<int> data{0};
    };
    auto s = std::make_shared<Shared>();
    return Case{{[s] {
                     int expected = 0;
                     if (s->flag.compare_exchange_strong(expected, 1)) s->data.store(42);
                 },
                 [s] {
                     if (s->flag.load() == 1) require(s->data.load() == 42, "flag set before data");
                 }},
                {}};
}

TEST(Pct, DepthOneNeverPreempts) {
    // No change points, no CAS bias, no spinning: every thread runs to the end
    // before the next one starts, so the lost update can't happen.
    PctStrategy pct(1, opts(1, 0.0));
    auto result = lockfree::sched::explore(lost_update, pct, 300, 1);
    EXPECT_TRUE(result.ok()) << result.failure.failure;
}

TEST(Pct, DepthTwoFindsTheLostUpdate) {
    PctStrategy pct(1, opts(2, 0.0));
    auto result = lockfree::sched::explore(lost_update, pct, 300, 1);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.failure.failure, "check: lost an increment");

    // Replay with the same options gives the same schedule.
    PctStrategy again(0, opts(2, 0.0));
    auto r = lockfree::sched::replay(lost_update, again, *result.failing_seed);
    EXPECT_EQ(r.outcome, Outcome::failed);
    EXPECT_EQ(r.choices, result.failure.choices);
}

TEST(Pct, CasBiasCatchesTheWindowAfterACas) {
    PctStrategy without(1, opts(1, 0.0));
    EXPECT_TRUE(lockfree::sched::explore(flag_before_data, without, 200, 5).ok());

    // With p = 1 every run where A goes first fails, so it's quick.
    PctStrategy with(1, opts(1, 1.0));
    auto result = lockfree::sched::explore(flag_before_data, with, 200, 5);
    ASSERT_FALSE(result.ok());
    EXPECT_LT(result.runs, 10u);
    EXPECT_EQ(result.failure.failure, "flag set before data");
}

Case spin_on_flag() {
    auto flag = std::make_shared<atomic<int>>(0);
    return Case{{[flag] {
                     while (flag->load() == 0) {
                     }
                 },
                 [flag] { flag->store(1); }},
                {}};
}

TEST(Pct, StarvationGuardLetsSpinnersFinish) {
    lockfree::sched::Options small;
    small.max_steps = 5'000;

    // Without the guard, whichever seeds give the spinner the higher priority
    // spin until the step limit.
    PctStrategy strict(3, opts(1, 0.0, 0));
    auto stuck = lockfree::sched::explore(spin_on_flag, strict, 50, 3, small);
    ASSERT_FALSE(stuck.ok());
    EXPECT_EQ(stuck.failure.outcome, Outcome::step_limit);

    PctStrategy guarded(3, opts(1, 0.0, 16));
    auto result = lockfree::sched::explore(spin_on_flag, guarded, 50, 3, small);
    EXPECT_TRUE(result.ok()) << result.failure.failure;
}

TEST(Pct, OnlyPicksRunnableAndSticksToPriorities) {
    PctStrategy pct(9, opts(1, 0.0, 0));
    pct.begin_run(4);
    const std::vector<ThreadId> runnable{1, 3};
    ThreadId first = pct.pick(Decision{std::nullopt, nullptr, runnable, 0});
    ASSERT_TRUE(first == 1 || first == 3);
    // No change points and no guard: the same thread wins every time.
    lockfree::detail::YieldPoint load{lockfree::detail::AtomicOp::load, nullptr, {}};
    for (std::uint64_t step = 1; step < 200; ++step)
        EXPECT_EQ(pct.pick(Decision{first, &load, runnable, step}), first);
}

TEST(Pct, ChangePointsAreDistinctAndInRange) {
    PctOptions o = opts(5, 0.0);
    o.expected_steps = 6;
    for (std::uint64_t seed = 0; seed < 100; ++seed) {
        PctStrategy pct(seed, o);
        pct.begin_run(3);
        const auto& cps = pct.change_points();
        ASSERT_EQ(cps.size(), 4u);
        EXPECT_TRUE(std::set<std::uint64_t>(cps.begin(), cps.end()).size() == cps.size());
        for (auto c : cps) EXPECT_TRUE(c >= 1 && c <= 6);
    }
    // More change points wanted than steps to put them on: take what fits.
    o.depth = 10;
    o.expected_steps = 3;
    PctStrategy pct(0, o);
    pct.begin_run(2);
    EXPECT_EQ(pct.change_points().size(), 3u);
}

TEST(Pct, MpmcKeepsEveryValue) {
    constexpr int per_producer = 4;
    PctStrategy pct(0, PctOptions{});
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
        pct, 300, 2026);
    EXPECT_TRUE(result.ok()) << result.failure.failure;
}

} // namespace
