// N:1 throughput: MpscQueue against MpmcQueue and the mutex baseline, with 1, 2,
// 4 and 8 producers feeding one consumer.
//
// Google Benchmark runs the body on P + 1 threads. Every thread runs the same
// number of iterations, so a producer pushes one item per iteration and the
// consumer pops P. That balances exactly and nobody waits on an item that
// never comes. items_per_second is pops per second of wall time.
//
// MpscQueue is unbounded, while the other two stop at queue_capacity. Left alone
// its producers could run far ahead and turn this into an allocator test on a
// huge backlog, so it gets a credit window of the same total size: a producer
// waits once it's window items ahead of what the consumer has reported. The
// consumer only reports every report_every pops from that producer, to keep
// this bookkeeping from dominating what's being measured.
//
// BM_MpscIntrusive is the same queue core without the allocator: each producer
// recycles a fixed ring of nodes. The gap between it and BM_MpscQueue is what
// new/delete costs, with the consumer freeing memory other threads allocated.

#include <benchmark/benchmark.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include "affinity.hpp"
#include "baseline_queues.hpp"
#include "lockfree/atomic_backoff.hpp"
#include "lockfree/detail/cache_line.hpp"
#include "lockfree/mpmc_queue.hpp"
#include "lockfree/mpsc_queue.hpp"

namespace {

constexpr std::size_t queue_capacity = 1024;
constexpr std::uint64_t report_every = 64;

constexpr std::uint64_t tag(std::uint64_t producer, std::uint64_t seq) {
    return producer << 40 | seq;
}

struct alignas(lockfree::detail::cache_line_size) Credit {
    std::atomic<std::uint64_t> consumed{0};
};

using Mpmc = lockfree::MpmcQueue<std::uint64_t, queue_capacity>;

struct BoundedMutexQueue : lockfree::bench::MutexQueue<std::uint64_t> {
    BoundedMutexQueue() : MutexQueue(queue_capacity) {}
};

struct WindowedMpsc : lockfree::MpscQueue<std::uint64_t> {
    bool try_push(std::uint64_t v) {
        push(v);
        return true;
    }
};

template <typename Queue>
struct Shared {
    std::unique_ptr<Queue> q;
    std::vector<Credit> credits;
};

template <typename Queue>
void run_n_to_1(benchmark::State& state, Shared<Queue>& shared) {
    constexpr bool windowed = std::is_same_v<Queue, WindowedMpsc>;
    const auto threads = static_cast<std::size_t>(state.threads());
    const std::size_t producers = threads - 1;
    const auto index = static_cast<std::size_t>(state.thread_index());
    const bool pinned = lockfree::bench::pin_current_thread(static_cast<int>(index));

    if (index == 0) {
        shared.q = std::make_unique<Queue>();
        shared.credits = std::vector<Credit>(producers);
    }

    const std::uint64_t window = queue_capacity / producers;
    bool out_of_order = false;
    if (index < producers) {
        std::uint64_t seq = 0, limit = window;
        for (auto _ : state) {
            if constexpr (windowed) {
                lockfree::Backoff backoff;
                while (seq >= limit) {
                    limit = shared.credits[index].consumed.load(std::memory_order_acquire) + window;
                    if (seq >= limit) backoff.pause();
                }
            }
            lockfree::Backoff backoff;
            while (!shared.q->try_push(tag(index, seq))) backoff.pause();
            ++seq;
        }
    } else {
        std::vector<std::uint64_t> next(producers, 0);
        std::uint64_t value = 0;
        for (auto _ : state) {
            for (std::size_t i = 0; i < producers; ++i) {
                lockfree::Backoff backoff;
                while (!shared.q->try_pop(value)) backoff.pause();
                const std::uint64_t p = value >> 40, seq = value & ((std::uint64_t{1} << 40) - 1);
                if (p >= producers || seq != next[p]) {
                    out_of_order = true;
                    continue;
                }
                next[p] = seq + 1;
                if constexpr (windowed) {
                    if (next[p] % report_every == 0) {
                        shared.credits[p].consumed.store(next[p], std::memory_order_release);
                    }
                }
            }
        }
        // The framework sums items over threads but divides by the summed real
        // time of all threads. Scaling by the thread count undoes that, so this
        // reads as pops per second of wall time. See the same note in bench_mpmc.
        state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) *
                                static_cast<std::int64_t>(producers) * state.threads());
    }

    if (out_of_order) state.SkipWithError("consumer saw a producer's items out of order");
    state.counters["pinned"] = benchmark::Counter(pinned ? 1 : 0, benchmark::Counter::kAvgThreads);
    if (index == 0) {
        std::uint64_t leftover;
        if (shared.q->try_pop(leftover)) state.SkipWithError("queue not empty after the run");
        shared.q.reset();
    }
}

struct RingItem : lockfree::MpscNode {
    std::uint64_t value = 0;
};

struct IntrusiveShared {
    std::unique_ptr<lockfree::MpscIntrusiveQueue> q;
    std::vector<Credit> credits;
    std::vector<std::vector<RingItem>> rings;
};

void BM_MpscIntrusive(benchmark::State& state) {
    static IntrusiveShared shared;
    const auto threads = static_cast<std::size_t>(state.threads());
    const std::size_t producers = threads - 1;
    const auto index = static_cast<std::size_t>(state.thread_index());
    const bool pinned = lockfree::bench::pin_current_thread(static_cast<int>(index));
    const std::uint64_t window = queue_capacity / producers;

    // Thread 0 sets this up before the framework's start barrier, so the other
    // threads can't look at any of it until they're inside the timed loop.
    if (index == 0) {
        shared.q = std::make_unique<lockfree::MpscIntrusiveQueue>();
        shared.credits = std::vector<Credit>(producers);
        shared.rings.clear();
        shared.rings.resize(producers);
        for (auto& ring : shared.rings) ring = std::vector<RingItem>(window);
    }

    bool out_of_order = false;
    if (index < producers) {
        std::uint64_t seq = 0, limit = window;
        for (auto _ : state) {
            lockfree::Backoff backoff;
            while (seq >= limit) {
                limit = shared.credits[index].consumed.load(std::memory_order_acquire) + window;
                if (seq >= limit) backoff.pause();
            }
            // Staying within the window means the consumer has reported this
            // slot's previous use, so nobody else is touching the node. The
            // acquire above pairs with the consumer's release when it reports.
            RingItem& item = shared.rings[index][seq % window];
            item.value = tag(index, seq);
            shared.q->push(&item);
            ++seq;
        }
    } else {
        std::vector<std::uint64_t> next(producers, 0);
        for (auto _ : state) {
            for (std::size_t i = 0; i < producers; ++i) {
                lockfree::Backoff backoff;
                lockfree::MpscNode* n;
                while ((n = shared.q->pop()) == nullptr) backoff.pause();
                const std::uint64_t value = static_cast<RingItem*>(n)->value;
                const std::uint64_t p = value >> 40, seq = value & ((std::uint64_t{1} << 40) - 1);
                if (p >= producers || seq != next[p]) {
                    out_of_order = true;
                    continue;
                }
                next[p] = seq + 1;
                if (next[p] % report_every == 0) {
                    shared.credits[p].consumed.store(next[p], std::memory_order_release);
                }
            }
        }
        state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) *
                                static_cast<std::int64_t>(producers) * state.threads());
    }

    if (out_of_order) state.SkipWithError("consumer saw a producer's items out of order");
    state.counters["pinned"] = benchmark::Counter(pinned ? 1 : 0, benchmark::Counter::kAvgThreads);
    if (index == 0) {
        if (shared.q->pop() != nullptr) state.SkipWithError("queue not empty after the run");
        shared.q.reset();
    }
}

void BM_MpscQueue(benchmark::State& state) {
    static Shared<WindowedMpsc> shared;
    run_n_to_1(state, shared);
}

void BM_MpmcQueue(benchmark::State& state) {
    static Shared<Mpmc> shared;
    run_n_to_1(state, shared);
}

void BM_MutexQueue(benchmark::State& state) {
    static Shared<BoundedMutexQueue> shared;
    run_n_to_1(state, shared);
}

} // namespace

BENCHMARK(BM_MpscIntrusive)->Threads(2)->Threads(3)->Threads(5)->Threads(9)->UseRealTime();
BENCHMARK(BM_MpscQueue)->Threads(2)->Threads(3)->Threads(5)->Threads(9)->UseRealTime();
BENCHMARK(BM_MpmcQueue)->Threads(2)->Threads(3)->Threads(5)->Threads(9)->UseRealTime();
BENCHMARK(BM_MutexQueue)->Threads(2)->Threads(3)->Threads(5)->Threads(9)->UseRealTime();
