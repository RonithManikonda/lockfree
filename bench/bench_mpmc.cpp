// MPMC throughput at 1p1c, 2p2c, 4p4c and 8p8c against the mutex baseline.
//
// Google Benchmark runs the body on P + C threads (->Threads). The first P
// are producers and push one item per iteration; the rest are consumers and
// pop one item per iteration. Every thread runs the same number of
// iterations, which is why only P == C shapes are registered: pushes and
// pops then balance exactly and nobody waits on an item that never comes.
//
// The framework puts a barrier at the start and end of the timed loop, so
// thread 0 can build the queue before it and tear it down after it.
// items_per_second is the whole group's transfer rate: pops per second of
// wall time, summed over the consumers (see the note in run_mpmc).
//
// Both queues wait the same way when full or empty (spin with Backoff), so
// the difference is the queue, not the waiting policy. 8p8c is 16 threads on
// a 14-core machine; it's there to show what oversubscription does.

#include <benchmark/benchmark.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "affinity.hpp"
#include "baseline_queues.hpp"
#include "lockfree/atomic_backoff.hpp"
#include "lockfree/mpmc_queue.hpp"

namespace {

constexpr std::size_t queue_capacity = 1024;

// Values carry (producer, seq) so consumers can check per-producer order
// for the price of a shift and a compare.
constexpr std::uint64_t tag(std::uint64_t producer, std::uint64_t seq) {
    return producer << 40 | seq;
}

template <typename Queue>
void run_mpmc(benchmark::State& state, std::unique_ptr<Queue>& q) {
    const auto threads = static_cast<std::size_t>(state.threads());
    const std::size_t producers = threads / 2;
    const auto index = static_cast<std::size_t>(state.thread_index());
    const bool pinned = lockfree::bench::pin_current_thread(static_cast<int>(index));

    if (index == 0) q = std::make_unique<Queue>();

    bool out_of_order = false;
    if (index < producers) {
        std::uint64_t seq = 0;
        for (auto _ : state) {
            lockfree::Backoff backoff;
            while (!q->try_push(tag(index, seq))) backoff.pause();
            ++seq;
        }
    } else {
        std::vector<std::uint64_t> next(producers, 0);
        std::uint64_t value = 0;
        for (auto _ : state) {
            lockfree::Backoff backoff;
            while (!q->try_pop(value)) backoff.pause();
            const std::uint64_t p = value >> 40, seq = value & ((std::uint64_t{1} << 40) - 1);
            if (p >= producers || seq < next[p]) out_of_order = true;
            else next[p] = seq + 1;
        }
        // The framework sums items over threads but divides by the *summed*
        // real time of all threads, which turns the rate into a per-thread
        // average. Scaling by the thread count undoes that, so the counter
        // reads as total pops per second of wall time.
        state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) * state.threads());
    }

    if (out_of_order) state.SkipWithError("consumer saw a producer's items out of order");
    state.counters["pinned"] = benchmark::Counter(pinned ? 1 : 0, benchmark::Counter::kAvgThreads);
    if (index == 0) {
        // Pushes and pops balanced, so it's empty; a leftover would mean a
        // lost pop somewhere.
        std::uint64_t leftover;
        if (q->try_pop(leftover)) state.SkipWithError("queue not empty after the run");
        q.reset();
    }
}

using Mpmc = lockfree::MpmcQueue<std::uint64_t, queue_capacity>;

// MutexQueue takes its capacity at runtime; give it the same bound.
struct BoundedMutexQueue : lockfree::bench::MutexQueue<std::uint64_t> {
    BoundedMutexQueue() : MutexQueue(queue_capacity) {}
};

void BM_MpmcQueue(benchmark::State& state) {
    static std::unique_ptr<Mpmc> q;
    run_mpmc(state, q);
}

void BM_MutexQueue(benchmark::State& state) {
    static std::unique_ptr<BoundedMutexQueue> q;
    run_mpmc(state, q);
}

} // namespace

BENCHMARK(BM_MpmcQueue)->Threads(2)->Threads(4)->Threads(8)->Threads(16)->UseRealTime();
BENCHMARK(BM_MutexQueue)->Threads(2)->Threads(4)->Threads(8)->Threads(16)->UseRealTime();
