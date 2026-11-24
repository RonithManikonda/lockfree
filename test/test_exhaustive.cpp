#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <lockfree/mpmc_queue.hpp>
#include <lockfree/spsc_ring_buffer.hpp>

#include "scheduler/exhaustive.hpp"
#include "scheduler/shrink.hpp"

using lockfree::detail::atomic;
using lockfree::sched::Case;
using lockfree::sched::ExhaustiveOptions;
using lockfree::sched::ExhaustiveResult;
using lockfree::sched::require;

namespace {

ExhaustiveResult run_bounded(const lockfree::sched::CaseFactory& factory, std::size_t bound,
                             const char* label) {
    ExhaustiveOptions o;
    o.preemption_bound = bound;
    auto r = lockfree::sched::explore_exhaustive(factory, o);
    std::printf("[%s] bound %zu: %zu schedules, %s (max preemptions used: %zu)\n", label, bound, r.runs,
                r.ok() ? "all passed" : "stopped early", r.max_preemptions_seen);
    return r;
}

// Two threads, each making `k` atomic increments of a shared clock and noting
// which thread got which tick. The check collects the resulting access order
// ("AABB", "ABAB", ...) into `orders`, so a test can see exactly which
// interleavings the search reached.
lockfree::sched::CaseFactory access_orders(int k, std::shared_ptr<std::set<std::string>> orders) {
    return [k, orders] {
        struct Shared {
            atomic<int> clock{0};
            std::string order;
        };
        auto sh = std::make_shared<Shared>();
        sh->order.assign(static_cast<std::size_t>(2 * k), '?');
        auto body = [sh, k](char name) {
            return [sh, k, name] {
                // Only one thread runs at a time, so the plain string is fine.
                for (int i = 0; i < k; ++i) sh->order[static_cast<std::size_t>(sh->clock.fetch_add(1))] = name;
            };
        };
        return Case{{body('A'), body('B')}, [sh, orders] { orders->insert(sh->order); }};
    };
}

std::set<std::string> orders_within(int k, std::size_t bound) {
    auto orders = std::make_shared<std::set<std::string>>();
    ExhaustiveOptions o;
    o.preemption_bound = bound;
    auto r = lockfree::sched::explore_exhaustive(access_orders(k, orders), o);
    EXPECT_TRUE(r.ok());
    return *orders;
}

TEST(Exhaustive, BoundZeroRunsEachThreadToCompletion) {
    EXPECT_EQ(orders_within(2, 0), (std::set<std::string>{"AABB", "BBAA"}));
}

TEST(Exhaustive, EachPreemptionAddsTheOrdersItCanReach) {
    // ABBA needs one preemption (A after its first op); the switch back to A
    // comes when B has finished, which is free. ABAB needs two.
    EXPECT_EQ(orders_within(2, 1), (std::set<std::string>{"AABB", "ABBA", "BAAB", "BBAA"}));
    EXPECT_EQ(orders_within(2, 2).size(), 6u); // all of C(4, 2)
}

TEST(Exhaustive, LargeBoundReachesEveryInterleaving) {
    EXPECT_EQ(orders_within(3, 6).size(), 20u); // C(6, 3)
}

TEST(Exhaustive, FindsALostUpdateWithOnePreemption) {
    auto lost_update = [] {
        auto x = std::make_shared<atomic<int>>(0);
        auto inc = [x] { x->store(x->load() + 1); };
        return Case{{inc, inc}, [x] { require(x->load() == 2, "lost an increment"); }};
    };
    auto none = run_bounded(lost_update, 0, "lost update");
    EXPECT_TRUE(none.ok()) << "without preemptions the increments can't interleave";

    auto one = run_bounded(lost_update, 1, "lost update");
    ASSERT_TRUE(one.failure.has_value());
    EXPECT_FALSE(one.complete);
    EXPECT_NE(one.failure->failure.find("lost an increment"), std::string::npos);

    // The recorded choices replay the failure.
    auto again = lockfree::sched::replay_choices(lost_update, one.failure->choices);
    EXPECT_EQ(again.outcome, lockfree::sched::Outcome::failed);
}

TEST(Exhaustive, StopsAtMaxRuns) {
    auto orders = std::make_shared<std::set<std::string>>();
    ExhaustiveOptions o;
    o.preemption_bound = 6;
    o.max_runs = 3;
    auto r = lockfree::sched::explore_exhaustive(access_orders(3, orders), o);
    EXPECT_EQ(r.runs, 3u);
    EXPECT_FALSE(r.complete);
    EXPECT_FALSE(r.ok());
}

TEST(Exhaustive, NoticesANondeterministicScenario) {
    // Every other build of the case brings a third thread, so the second run
    // meets a different set of runnable threads at its very first decision.
    auto builds = std::make_shared<int>(0);
    auto flaky = [builds] {
        const bool extra = (*builds)++ % 2 == 1;
        auto x = std::make_shared<atomic<int>>(0);
        auto inc = [x] { x->fetch_add(1); };
        Case c{{inc, inc}, {}};
        if (extra) c.threads.push_back(inc);
        return c;
    };
    ExhaustiveOptions o;
    o.preemption_bound = 2;
    auto r = lockfree::sched::explore_exhaustive(flaky, o);
    EXPECT_TRUE(r.nondeterministic);
    EXPECT_FALSE(r.ok());
}

// ---- The real structures, every schedule within two preemptions ------------

// SPSC, capacity 2: the producer tries to push 1, 2, 3 (the third can only
// fit if the consumer got there first), the consumer tries three pops. Every
// value pushed must come out once, in order, from the pops or the drain.
TEST(ExhaustiveSpsc, ThreeTryPushesAgainstThreeTryPops) {
    auto factory = [] {
        struct Shared {
            lockfree::SpscRingBuffer<int, 2> q;
            std::vector<int> pushed, popped;
        };
        auto sh = std::make_shared<Shared>();
        auto producer = [sh] {
            for (int v = 1; v <= 3; ++v)
                if (sh->q.try_push(v)) sh->pushed.push_back(v);
        };
        auto consumer = [sh] {
            for (int i = 0; i < 3; ++i)
                if (auto v = sh->q.try_pop()) sh->popped.push_back(*v);
        };
        auto check = [sh] {
            std::vector<int> out = sh->popped;
            while (auto v = sh->q.try_pop()) out.push_back(*v);
            require(out == sh->pushed, "SPSC lost, duplicated or reordered a value");
        };
        return Case{{producer, consumer}, check};
    };
    auto r = run_bounded(factory, 2, "spsc 3x3");
    EXPECT_TRUE(r.ok()) << (r.failure ? r.failure->failure : "");
    EXPECT_EQ(r.max_preemptions_seen, 2u);
}

// MPMC, capacity 2: both threads push then pop, so pops race pushes and each
// other. Pops may come back empty, but what's pushed comes out exactly once.
template <int OpsPerThread>
Case mpmc_push_pop() {
    struct Shared {
        lockfree::MpmcQueue<int, 2> q;
        std::vector<int> pushed, popped;
    };
    auto sh = std::make_shared<Shared>();
    auto body = [sh](int base) {
        return [sh, base] {
            // OpsPerThread - 1 pushes, then one pop.
            for (int i = 0; i < OpsPerThread - 1; ++i)
                if (sh->q.try_push(base + i)) sh->pushed.push_back(base + i);
            if (auto v = sh->q.try_pop()) sh->popped.push_back(*v);
        };
    };
    auto check = [sh] {
        std::vector<int> out = sh->popped;
        while (auto v = sh->q.try_pop()) out.push_back(*v);
        std::vector<int> in = sh->pushed;
        std::sort(out.begin(), out.end());
        std::sort(in.begin(), in.end());
        require(out == in, "MPMC lost or duplicated a value");
    };
    return Case{{body(10), body(20)}, check};
}

TEST(ExhaustiveMpmc, PushThenPopOnEachSide) {
    auto r = run_bounded(mpmc_push_pop<2>, 2, "mpmc 2x2");
    EXPECT_TRUE(r.ok()) << (r.failure ? r.failure->failure : "");
}

TEST(ExhaustiveMpmc, TwoPushesThenPopFillsTheQueue) {
    // Four pushes into capacity 2: some have to fail, depending on the pops.
    auto r = run_bounded(mpmc_push_pop<3>, 2, "mpmc 2x3");
    EXPECT_TRUE(r.ok()) << (r.failure ? r.failure->failure : "");
}

} // namespace
