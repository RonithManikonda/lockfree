// Michael-Scott queue throughput (hazard pointers and epochs) against the
// bounded MPMC queue and a mutex queue, from 1p1c to 8p8c.
//
// Same shape as bench_mpmc: P producers and P consumers under ->Threads(2P),
// one push or pop per iteration, every value tagged (producer, seq) so the
// consumers can check per-producer FIFO order.
//
// The MS queue is unbounded, so producers that get ahead would just grow the
// list for the length of the run (hundreds of MB at these rates) and the
// numbers would end up measuring malloc. Every queue here therefore runs
// behind the same credit window: a producer takes a credit before pushing and
// a consumer returns one after popping, so at most `window` items are ever in
// flight. The MPMC queue's capacity equals the window, so it never fills
// either. That costs every queue one extra contended counter per operation,
// and it's not cheap (compare BM_Mpmc with BM_MpmcNoWindow, the same queue
// without credits), but it's the same counter for all of them.
//
// What to expect: the MS queue allocates a node per push and retires one per
// pop, so it starts behind MPMC (preallocated cells). The interesting part is
// the gap between the two reclaimers: hazard pointers pay a seq_cst fence per
// protect (two per dequeue, one per enqueue), epochs one pin per operation.

#include <benchmark/benchmark.h>

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "affinity.hpp"
#include "baseline_queues.hpp"
#include "lockfree/atomic_backoff.hpp"
#include "lockfree/detail/cache_line.hpp"
#include "lockfree/michael_scott_queue.hpp"
#include "lockfree/mpmc_queue.hpp"

namespace {

constexpr std::int64_t window = 1024;

constexpr std::uint64_t tag(std::uint64_t producer, std::uint64_t seq) {
    return producer << 40 | seq;
}

// The credit window. Producers spin while it's spent; consumers only ever
// give credits back.
struct alignas(lockfree::detail::cache_line_size) Credits {
    std::atomic<std::int64_t> in_flight{0};

    void take() {
        lockfree::Backoff backoff;
        for (;;) {
            std::int64_t n = in_flight.load(std::memory_order_relaxed);
            if (n < window && in_flight.compare_exchange_weak(n, n + 1, std::memory_order_relaxed))
                return;
            backoff.pause();
        }
    }
    void give() { in_flight.fetch_sub(1, std::memory_order_relaxed); }
};

// try_push for the unbounded queue, so one body drives all four.
template <typename Reclaimer>
struct Msq : lockfree::MichaelScottQueue<std::uint64_t, Reclaimer> {
    bool try_push(std::uint64_t v) {
        this->push(v);
        return true;
    }
};

using MsqHazard = Msq<lockfree::HazardPointerReclaim>;
using MsqEpoch = Msq<lockfree::EpochReclaim>;
using Mpmc = lockfree::MpmcQueue<std::uint64_t, static_cast<std::size_t>(window)>;

struct UnboundedMutexQueue : lockfree::bench::MutexQueue<std::uint64_t> {
    UnboundedMutexQueue() : MutexQueue(std::numeric_limits<std::size_t>::max()) {}
};

template <typename Queue>
struct Shared {
    Queue q;
    Credits credits;
};

// Windowed = false skips the credits. Only safe for a bounded queue; it's
// there so the window's own cost shows up next to MPMC's normal numbers.
template <typename Queue, bool Windowed = true>
void run_msq(benchmark::State& state, std::unique_ptr<Shared<Queue>>& sh) {
    const auto threads = static_cast<std::size_t>(state.threads());
    const std::size_t producers = threads / 2;
    const auto index = static_cast<std::size_t>(state.thread_index());
    const bool pinned = lockfree::bench::pin_current_thread(static_cast<int>(index));

    // Thread 0 builds the shared state before the start barrier; nobody else
    // touches it until the timed loop (see NOTES on bench setup).
    if (index == 0) sh = std::make_unique<Shared<Queue>>();

    bool out_of_order = false;
    if (index < producers) {
        std::uint64_t seq = 0;
        for (auto _ : state) {
            if constexpr (Windowed) sh->credits.take();
            lockfree::Backoff backoff;
            while (!sh->q.try_push(tag(index, seq))) backoff.pause();
            ++seq;
        }
    } else {
        std::vector<std::uint64_t> next(producers, 0);
        std::uint64_t value = 0;
        for (auto _ : state) {
            lockfree::Backoff backoff;
            while (!sh->q.try_pop(value)) backoff.pause();
            if constexpr (Windowed) sh->credits.give();
            const std::uint64_t p = value >> 40, seq = value & ((std::uint64_t{1} << 40) - 1);
            if (p >= producers || seq < next[p]) out_of_order = true;
            else next[p] = seq + 1;
        }
        // Group rate, as in bench_mpmc: undo the framework's per-thread average.
        state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) * state.threads());
    }

    if (out_of_order) state.SkipWithError("consumer saw a producer's items out of order");
    state.counters["pinned"] = benchmark::Counter(pinned ? 1 : 0, benchmark::Counter::kAvgThreads);
    if (index == 0) {
        std::uint64_t leftover;
        if (sh->q.try_pop(leftover)) state.SkipWithError("queue not empty after the run");
        sh.reset();
    }
}

void BM_MsqHazard(benchmark::State& state) {
    static std::unique_ptr<Shared<MsqHazard>> sh;
    run_msq(state, sh);
}

void BM_MsqEpoch(benchmark::State& state) {
    static std::unique_ptr<Shared<MsqEpoch>> sh;
    run_msq(state, sh);
}

void BM_Mpmc(benchmark::State& state) {
    static std::unique_ptr<Shared<Mpmc>> sh;
    run_msq(state, sh);
}

void BM_MpmcNoWindow(benchmark::State& state) {
    static std::unique_ptr<Shared<Mpmc>> sh;
    run_msq<Mpmc, false>(state, sh);
}

void BM_MutexQueue(benchmark::State& state) {
    static std::unique_ptr<Shared<UnboundedMutexQueue>> sh;
    run_msq(state, sh);
}

} // namespace

BENCHMARK(BM_MsqHazard)->Threads(2)->Threads(4)->Threads(8)->Threads(16)->UseRealTime();
BENCHMARK(BM_MsqEpoch)->Threads(2)->Threads(4)->Threads(8)->Threads(16)->UseRealTime();
BENCHMARK(BM_Mpmc)->Threads(2)->Threads(4)->Threads(8)->Threads(16)->UseRealTime();
BENCHMARK(BM_MpmcNoWindow)->Threads(2)->Threads(4)->Threads(8)->Threads(16)->UseRealTime();
BENCHMARK(BM_MutexQueue)->Threads(2)->Threads(4)->Threads(8)->Threads(16)->UseRealTime();
