// AtomicStack throughput at 1-8 threads against std::mutex + std::vector.
//
// Every thread pushes one value and then pops one per iteration. A thread has
// always pushed at least as many as it popped, so its pop always finds
// something (maybe another thread's value); a failed pop is reported as an
// error rather than retried. That keeps the benchmark about contention on the
// head, with no waiting policy mixed in.
//
// items_per_second counts push+pop pairs per second of wall time, summed over
// all threads (see the note in bench_mpmc.cpp about scaling by the thread
// count). 1 thread is the uncontended cost; past that every operation fights
// over one cache line. Each push+pop pair is four CASes in AtomicStack (take a
// node from the free list, push it, pop it, give it back), against two
// lock/unlock pairs for the mutex.
//
// First numbers (M4 Pro, unpinned, two runs, pairs/s): AtomicStack 72M at 1
// thread, 9-16M at 2, 7-14M at 4, 2-4M at 8; MutexStack 120M, 6-89M, 28M, 46M.
// Backoff on lost CASes roughly doubled the lock-free numbers at 2-4 threads,
// but under this much contention the mutex still wins from 4 threads on.

#include <benchmark/benchmark.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "affinity.hpp"
#include "lockfree/atomic_stack.hpp"

namespace {

// The lock-based stack being measured against.
class MutexStack {
public:
    void push(std::uint64_t v) {
        std::lock_guard lock(mutex_);
        items_.push_back(v);
    }

    bool try_pop(std::uint64_t& out) {
        std::lock_guard lock(mutex_);
        if (items_.empty()) return false;
        out = items_.back();
        items_.pop_back();
        return true;
    }

private:
    std::mutex mutex_;
    std::vector<std::uint64_t> items_;
};

template <typename Stack>
void run_stack(benchmark::State& state, std::unique_ptr<Stack>& s) {
    const auto index = static_cast<std::size_t>(state.thread_index());
    const bool pinned = lockfree::bench::pin_current_thread(static_cast<int>(index));

    if (index == 0) s = std::make_unique<Stack>();

    bool empty_pop = false;
    std::uint64_t value = 0, sum = 0;
    for (auto _ : state) {
        s->push(index);
        if (!s->try_pop(value)) empty_pop = true;
        sum += value;
    }
    benchmark::DoNotOptimize(sum);
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) * state.threads());

    if (empty_pop) state.SkipWithError("pop found the stack empty");
    state.counters["pinned"] = benchmark::Counter(pinned ? 1 : 0, benchmark::Counter::kAvgThreads);
    if (index == 0) {
        if (s->try_pop(value)) state.SkipWithError("stack not empty after the run");
        s.reset();
    }
}

void BM_AtomicStack(benchmark::State& state) {
    static std::unique_ptr<lockfree::AtomicStack<std::uint64_t>> s;
    run_stack(state, s);
}

void BM_MutexStack(benchmark::State& state) {
    static std::unique_ptr<MutexStack> s;
    run_stack(state, s);
}

} // namespace

BENCHMARK(BM_AtomicStack)->ThreadRange(1, 8)->UseRealTime();
BENCHMARK(BM_MutexStack)->ThreadRange(1, 8)->UseRealTime();
