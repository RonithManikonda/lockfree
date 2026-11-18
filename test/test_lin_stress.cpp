#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include "linearizability/checker.hpp"
#include "linearizability/history.hpp"
#include "linearizability/recorder.hpp"
#include "linearizability/spec.hpp"
#include "lockfree/atomic_stack.hpp"
#include "lockfree/mpmc_queue.hpp"
#include "lockfree/mpsc_queue.hpp"
#include "lockfree/spsc_ring_buffer.hpp"
#include "stress_harness.hpp"

// Short stress runs of the real structures, recorded and then checked for
// linearizability. The other stress tests check conservation (every value
// comes out exactly once) and per-producer order; this checks the stronger
// claim that the whole run looks like *some* sequential execution that
// respects real time.
//
// The checker's cost grows with how many elements sit in the structure, not
// with how long the run is (see NOTES on the checker), so every run here
// keeps the structure nearly empty: tiny capacities for the bounded queues,
// a credit limit for MPSC, and stack threads that pop what they push.
//
// Each round is short and fixed-size; LOCKFREE_STRESS_SCALE changes how many
// rounds run, not how long each one is. Threads that never all pause at once
// leave few quiescent points, so a round is mostly one long segment, and the
// checker's cost grows faster than linearly with segment length. Ten short
// rounds cost ten times one round; one round ten times longer cost ~100x.
//
// Pushed values are unique (thread * kStride + i) so a popped value names
// exactly one push.

using lockfree::lin::check;
using lockfree::lin::CheckResult;
using lockfree::lin::FifoQueueSpec;
using lockfree::lin::History;
using lockfree::lin::LifoStackSpec;
using lockfree::lin::Recorder;
using lockfree::lin::Verdict;
using lockfree::test::maybe_delay;
using lockfree::test::run_stress;
using lockfree::test::scaled;
using lockfree::test::StressRng;
using lockfree::test::StressTest;

namespace {

constexpr std::int64_t kStride = 1'000'000'000;

std::uint64_t rounds() { return scaled(3); }

std::int64_t value_for(std::size_t thread, std::uint64_t i) {
    return static_cast<std::int64_t>(thread) * kStride + static_cast<std::int64_t>(i);
}

// Everything worth seeing when a run fails: the verdict, the stuck segment,
// and how far the search got through it.
std::string describe(const CheckResult& r) {
    std::string s = std::string(to_string(r.verdict)) + ": " + r.message + "\n";
    s += "segment (" + std::to_string(r.failing_segment.size()) + " ops):\n";
    for (const auto& op : r.failing_segment) s += "  " + to_string(op) + "\n";
    s += "longest legal order:\n";
    for (const auto& op : r.best_order) s += "  " + to_string(op) + "\n";
    return s;
}

template <typename Spec>
void expect_linearizable(const History& h) {
    const CheckResult r = check<Spec>(h);
    EXPECT_EQ(r.verdict, Verdict::linearizable) << describe(r);
}

// Consumers stop once they've popped everything; with multiple consumers
// that's a shared count.
bool all_popped(const std::atomic<std::uint64_t>& popped, std::uint64_t total) {
    return popped.load(std::memory_order_relaxed) >= total;
}

class LinStress : public StressTest {};

TEST_F(LinStress, SpscRingBuffer) {
    const std::uint64_t ops = 3000;
    for (std::uint64_t round = 0; round < rounds(); ++round) {
        // Capacity 4: full and empty both come up constantly, which is where
        // the cached indices have to be refreshed.
        lockfree::SpscRingBuffer<std::int64_t, 4> q;
        Recorder rec(2, 2 * ops);
        run_stress(2, [&](std::size_t t, StressRng& rng) {
            auto& log = rec.log(t);
            if (t == 0) {
                for (std::uint64_t i = 0; i < ops;) {
                    const std::int64_t v = value_for(0, i);
                    if (log.push(v, [&] { return q.try_push(v); })) ++i;
                    maybe_delay(rng);
                }
            } else {
                // One consumer, so its empty answers are real claims and are
                // recorded: once a push has returned, the consumer must see it.
                for (std::uint64_t got = 0; got < ops;) {
                    if (log.pop([&] { return q.try_pop(); })) ++got;
                    maybe_delay(rng);
                }
            }
        });
        const History h = rec.merge();
        expect_linearizable<FifoQueueSpec>(h);
        if (HasFailure()) return;
    }
}

TEST_F(LinStress, MpmcQueueTwoByTwo) {
    constexpr std::size_t producers = 2;
    constexpr std::size_t consumers = 2;
    const std::uint64_t per_producer = 1500;
    const std::uint64_t total = producers * per_producer;
    for (std::uint64_t round = 0; round < rounds(); ++round) {
        lockfree::MpmcQueue<std::int64_t, 4> q;
        std::atomic<std::uint64_t> popped{0};
        Recorder rec(producers + consumers, 2 * per_producer);
        run_stress(producers + consumers, [&](std::size_t t, StressRng& rng) {
            auto& log = rec.log(t);
            if (t < producers) {
                for (std::uint64_t i = 0; i < per_producer;) {
                    const std::int64_t v = value_for(t, i);
                    if (log.push(v, [&] { return q.try_push(v); })) {
                        ++i;
                    } else {
                        std::this_thread::yield();
                    }
                    maybe_delay(rng);
                }
            } else {
                // Empty pops are dropped: a producer that claimed a cell and
                // stalled hides every later cell, even ones whose pushes
                // already returned (docs/mpmc_queue.md).
                while (!all_popped(popped, total)) {
                    if (log.pop_nonempty([&] { return q.try_pop(); })) {
                        popped.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        std::this_thread::yield();
                    }
                    maybe_delay(rng);
                }
            }
        });
        const History h = rec.merge();
        ASSERT_EQ(h.size(), 2 * total);
        expect_linearizable<FifoQueueSpec>(h);
        if (HasFailure()) return;
    }
}

TEST_F(LinStress, MpscQueueThreeProducers) {
    constexpr std::size_t producers = 3;
    constexpr std::int64_t credit_limit = 4;
    const std::uint64_t per_producer = 1000;
    const std::uint64_t total = producers * per_producer;
    for (std::uint64_t round = 0; round < rounds(); ++round) {
        lockfree::MpscQueue<std::int64_t> q;
        // Unbounded queue, so producers take a credit before pushing and the
        // consumer hands it back after popping. That keeps the depth near
        // credit_limit (it can overshoot by one per producer, which is fine).
        // The credit traffic happens outside the recorded calls.
        std::atomic<std::int64_t> in_queue{0};
        Recorder rec(producers + 1, 2 * per_producer);
        run_stress(producers + 1, [&](std::size_t t, StressRng& rng) {
            auto& log = rec.log(t);
            if (t < producers) {
                for (std::uint64_t i = 0; i < per_producer; ++i) {
                    while (in_queue.load(std::memory_order_relaxed) >= credit_limit) {
                        std::this_thread::yield();
                    }
                    in_queue.fetch_add(1, std::memory_order_relaxed);
                    const std::int64_t v = value_for(t, i);
                    log.push(v, [&] {
                        q.push(v);
                        return true;
                    });
                    maybe_delay(rng);
                }
            } else {
                // Same reason as MPMC for dropping empties: a push stalled
                // between its tail exchange and its link hides the pushes
                // behind it (docs/mpsc_queue.md).
                for (std::uint64_t got = 0; got < total;) {
                    if (log.pop_nonempty([&] { return q.try_pop(); })) {
                        ++got;
                        in_queue.fetch_sub(1, std::memory_order_relaxed);
                    } else {
                        std::this_thread::yield();
                    }
                    maybe_delay(rng);
                }
            }
        });
        const History h = rec.merge();
        ASSERT_EQ(h.size(), 2 * total);
        expect_linearizable<FifoQueueSpec>(h);
        if (HasFailure()) return;
    }
}

TEST_F(LinStress, AtomicStackFourThreads) {
    constexpr std::size_t threads = 4;
    const std::uint64_t per_thread = 1500;
    for (std::uint64_t round = 0; round < rounds(); ++round) {
        lockfree::AtomicStack<std::int64_t> s;
        Recorder rec(threads, per_thread);
        run_stress(threads, [&](std::size_t t, StressRng& rng) {
            auto& log = rec.log(t);
            // Each thread holds at most two of its own pushes "unpaid", so
            // the stack never gets deeper than 2 * threads. Pops can still
            // take anyone's value or come back empty, and empty is recorded:
            // a Treiber stack's empty is a real claim (head was null at the
            // load).
            int held = 0;
            std::uint64_t next = 0;
            for (std::uint64_t i = 0; i < per_thread; ++i) {
                const bool do_push = held == 0 || (held < 2 && rng.below(2) == 0);
                if (do_push) {
                    const std::int64_t v = value_for(t, next++);
                    log.push(v, [&] {
                        s.push(v);
                        return true;
                    });
                    ++held;
                } else {
                    log.pop([&] { return s.try_pop(); });
                    --held;
                }
                maybe_delay(rng);
            }
        });
        const History h = rec.merge();
        expect_linearizable<LifoStackSpec>(h);
        if (HasFailure()) return;
    }
}

// The pipeline has to be able to say no. A "queue" that hands out its newest
// element whenever it holds more than one is a fine conservation-test citizen
// (nothing lost, nothing duplicated) but not FIFO, and the same record-and-
// check path must catch it.
class NewestFirstQueue {
public:
    void push(std::int64_t v) {
        std::lock_guard lock(mu_);
        items_.push_back(v);
    }

    std::optional<std::int64_t> try_pop() {
        std::lock_guard lock(mu_);
        if (items_.empty()) return std::nullopt;
        std::int64_t v = items_.size() > 1 ? items_.back() : items_.front();
        if (items_.size() > 1) {
            items_.pop_back();
        } else {
            items_.pop_front();
        }
        return v;
    }

    std::size_t size() {
        std::lock_guard lock(mu_);
        return items_.size();
    }

private:
    std::mutex mu_;
    std::deque<std::int64_t> items_;
};

TEST_F(LinStress, CatchesANonFifoQueue) {
    NewestFirstQueue q;
    const std::uint64_t ops = 200;
    Recorder rec(2, ops);
    run_stress(2, [&](std::size_t t, StressRng&) {
        auto& log = rec.log(t);
        if (t == 0) {
            for (std::uint64_t i = 0; i < ops; ++i) {
                const std::int64_t v = value_for(0, i);
                log.push(v, [&] {
                    q.push(v);
                    return true;
                });
            }
        } else {
            // Wait (unrecorded) until three pushes have finished, so the very
            // first pop takes the newest of three non-overlapping pushes: a
            // guaranteed violation rather than a timing-dependent one.
            while (q.size() < 3) std::this_thread::yield();
            for (std::uint64_t got = 0; got < ops;) {
                if (log.pop([&] { return q.try_pop(); })) ++got;
            }
        }
    });
    const CheckResult r = check<FifoQueueSpec>(rec.merge());
    EXPECT_EQ(r.verdict, Verdict::not_linearizable) << r.message;
    EXPECT_FALSE(r.failing_segment.empty());
}

} // namespace
