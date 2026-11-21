#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "lockfree/spsc_ring_buffer.hpp"
#include "scheduler/scheduler.hpp"

using lockfree::detail::atomic;
using lockfree::sched::Decision;
using lockfree::sched::Options;
using lockfree::sched::Outcome;
using lockfree::sched::Scheduler;
using lockfree::sched::Strategy;
using lockfree::sched::ThreadId;

namespace {

// Switches at every yield point, to the next runnable id after the current
// one (wrapping around). As fair as it gets, and as many switches as possible.
class RoundRobin : public Strategy {
public:
    ThreadId pick(const Decision& d) override {
        if (!d.current) return d.runnable.front();
        for (ThreadId t : d.runnable)
            if (t > *d.current) return t;
        return d.runnable.front();
    }
};

// Never preempts: whoever runs keeps running until it finishes.
class RunToCompletion : public Strategy {
public:
    ThreadId pick(const Decision& d) override { return d.current.value_or(d.runnable.front()); }
};

// Picks pseudo-randomly from a fixed seed and records what it saw, so two runs
// can be compared decision by decision.
class RecordingLcg : public Strategy {
public:
    explicit RecordingLcg(std::uint64_t seed) : seed_(seed) {}

    struct Seen {
        std::optional<ThreadId> current;
        unsigned line;
        ThreadId picked;
        bool operator==(const Seen&) const = default;
    };
    std::vector<Seen> seen;

    void begin_run(std::size_t) override {
        state_ = seed_;
        seen.clear();
    }
    ThreadId pick(const Decision& d) override {
        state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
        ThreadId t = d.runnable[(state_ >> 33) % d.runnable.size()];
        seen.push_back({d.current, d.point ? d.point->where.line() : 0u, t});
        return t;
    }

private:
    std::uint64_t seed_;
    std::uint64_t state_ = 0;
};

TEST(Scheduler, SingleThreadRunsToCompletion) {
    RoundRobin rr;
    Scheduler s(rr);
    atomic<int> x{0};
    auto r = s.run({[&] {
        x.store(1);
        x.fetch_add(2);
        (void)x.load();
    }});
    EXPECT_TRUE(r.ok()) << r.failure;
    // One decision to start the thread, then one per atomic access.
    EXPECT_EQ(r.steps, 4u);
    EXPECT_EQ(r.choices, (std::vector<ThreadId>{0, 0, 0, 0}));
    EXPECT_EQ(x.load(), 3);
}

TEST(Scheduler, EmptyRunIsANoOp) {
    RoundRobin rr;
    auto r = Scheduler(rr).run({});
    EXPECT_TRUE(r.ok());
    EXPECT_EQ(r.steps, 0u);
}

// The log is a plain vector: the baton handoff has to order the appends, and
// TSan checks that it does.
std::vector<int> three_stores_each(Strategy& strategy) {
    std::vector<int> log;
    atomic<int> x{0};
    auto body = [&](int id) {
        return [&, id] {
            for (int i = 0; i < 3; ++i) {
                x.store(id);
                log.push_back(id);
            }
        };
    };
    auto r = Scheduler(strategy).run({body(0), body(1)});
    EXPECT_TRUE(r.ok()) << r.failure;
    return log;
}

TEST(Scheduler, RunToCompletionSerializesThreads) {
    RunToCompletion rtc;
    EXPECT_EQ(three_stores_each(rtc), (std::vector<int>{0, 0, 0, 1, 1, 1}));
}

TEST(Scheduler, RoundRobinSwitchesAtEveryYield) {
    RoundRobin rr;
    EXPECT_EQ(three_stores_each(rr), (std::vector<int>{0, 1, 0, 1, 0, 1}));
}

TEST(Scheduler, OnlyOneThreadRunsAtATime) {
    // A deliberately unprotected read-modify-write with an OS yield in the
    // middle. If two threads could ever run between the same pair of yield
    // points, increments would get lost (and TSan would call it a race).
    constexpr int threads = 4;
    constexpr int iterations = 200;
    int counter = 0;
    atomic<int> tick{0};
    auto body = [&] {
        for (int i = 0; i < iterations; ++i) {
            int v = counter;
            std::this_thread::yield();
            counter = v + 1;
            tick.fetch_add(1, std::memory_order_relaxed);
        }
    };
    RoundRobin rr;
    auto r = Scheduler(rr).run({body, body, body, body});
    EXPECT_TRUE(r.ok()) << r.failure;
    EXPECT_EQ(counter, threads * iterations);
    EXPECT_EQ(tick.load(), threads * iterations);
}

TEST(Scheduler, SameChoicesGiveTheSameRun) {
    auto scenario = [](RecordingLcg& strategy) {
        auto q = std::make_shared<lockfree::SpscRingBuffer<int, 4>>();
        std::vector<int> popped;
        auto r = Scheduler(strategy).run({
            [q] {
                for (int i = 0; i < 6; ++i)
                    while (!q->try_push(i)) {
                    }
            },
            [q, &popped] {
                while (popped.size() < 6)
                    if (auto v = q->try_pop()) popped.push_back(*v);
            },
        });
        EXPECT_TRUE(r.ok()) << r.failure;
        EXPECT_EQ(popped, (std::vector<int>{0, 1, 2, 3, 4, 5}));
        return std::make_pair(r.choices, strategy.seen);
    };

    RecordingLcg a(42), b(42);
    auto [choices_a, seen_a] = scenario(a);
    auto [choices_b, seen_b] = scenario(b);
    EXPECT_EQ(choices_a, choices_b);
    EXPECT_EQ(seen_a, seen_b);
    EXPECT_GT(seen_a.size(), 20u);
}

TEST(Scheduler, StepLimitStopsASpinThatNeverEnds) {
    // Shared, not on the stack: the spinning thread is parked for good when
    // the run is abandoned, and it holds a reference to the flag.
    auto flag = std::make_shared<atomic<bool>>(false);
    RoundRobin rr;
    auto r = Scheduler(rr, Options{.max_steps = 1000}).run({[flag] {
        while (!flag->load()) {
        }
    }});
    EXPECT_EQ(r.outcome, Outcome::step_limit);
    EXPECT_EQ(r.steps, 1000u);
    EXPECT_NE(r.failure.find("step limit"), std::string::npos);
}

TEST(Scheduler, UnfairStrategyLivelocksWhereAFairOneFinishes) {
    // Thread 0 waits for thread 1's flag. Never preempting means thread 1
    // never gets to set it.
    auto run_with = [](Strategy& strategy) {
        auto flag = std::make_shared<atomic<bool>>(false);
        return Scheduler(strategy, Options{.max_steps = 5000})
            .run({[flag] {
                      while (!flag->load()) {
                      }
                  },
                  [flag] { flag->store(true); }});
    };
    RunToCompletion rtc;
    EXPECT_EQ(run_with(rtc).outcome, Outcome::step_limit);
    RoundRobin rr;
    EXPECT_TRUE(run_with(rr).ok());
}

TEST(Scheduler, ExceptionFailsTheRunButOthersFinish) {
    atomic<int> x{0};
    bool other_finished = false;
    RoundRobin rr;
    auto r = Scheduler(rr).run({
        [&] {
            x.fetch_add(1);
            lockfree::sched::require(false, "boom");
        },
        [&] {
            for (int i = 0; i < 3; ++i) x.fetch_add(1);
            other_finished = true;
        },
    });
    EXPECT_EQ(r.outcome, Outcome::failed);
    EXPECT_EQ(r.failure, "boom");
    EXPECT_TRUE(other_finished);
    EXPECT_EQ(x.load(), 4);
}

TEST(Scheduler, HookIsRestoredAndOutsideThreadsPassThrough) {
    struct Counting : RoundRobin {
        int picks = 0;
        ThreadId pick(const Decision& d) override {
            ++picks;
            return RoundRobin::pick(d);
        }
    } strategy;
    atomic<int> x{0};
    auto r = Scheduler(strategy).run({[&] { x.store(1); }});
    ASSERT_TRUE(r.ok());
    EXPECT_EQ(lockfree::detail::yield_hook(), nullptr);

    // Atomics touched by a thread that isn't part of a run never reach the
    // strategy.
    int before = strategy.picks;
    x.store(2);
    (void)x.load();
    EXPECT_EQ(strategy.picks, before);
}

TEST(Scheduler, SpscExactSequenceUnderRoundRobin) {
    constexpr int count = 100;
    lockfree::SpscRingBuffer<int, 4> q;
    std::vector<int> got;
    RoundRobin rr;
    auto r = Scheduler(rr).run({
        [&] {
            for (int i = 0; i < count; ++i)
                while (!q.try_push(i)) {
                }
        },
        [&] {
            while (got.size() < count)
                if (auto v = q.try_pop()) got.push_back(*v);
        },
    });
    ASSERT_TRUE(r.ok()) << r.failure;
    ASSERT_EQ(got.size(), static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) EXPECT_EQ(got[i], i);
}

} // namespace
