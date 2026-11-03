// SPSC throughput: the benchmark thread is the producer, a second thread is
// the consumer. Each iteration moves one item through the queue, so
// items_per_second is end-to-end transfer rate. Real time, not CPU time:
// the producer's CPU time says nothing about how long the consumer took.

#include <benchmark/benchmark.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>

#include "baseline_queues.hpp"
#include "lockfree/spsc_ring_buffer.hpp"

namespace {

template <typename Queue>
void run_throughput(benchmark::State& state, Queue& q) {
    std::atomic<bool> ready{false};
    std::atomic<bool> done{false};
    std::atomic<std::uint64_t> received{0};
    bool out_of_order = false;

    std::thread consumer([&] {
        std::uint64_t expected = 0;
        std::uint64_t value = 0;
        ready.store(true, std::memory_order_release);
        for (;;) {
            if (q.try_pop(value)) {
                // Checking order costs one compare and catches a broken queue
                // that would otherwise just post a great number.
                if (value != expected) out_of_order = true;
                ++expected;
            } else if (done.load(std::memory_order_acquire)) {
                // done is set after the last push, so one more empty pop
                // after seeing it means everything has been drained.
                if (!q.try_pop(value)) break;
                if (value != expected) out_of_order = true;
                ++expected;
            }
        }
        received.store(expected, std::memory_order_relaxed);
    });

    // Don't time thread startup.
    while (!ready.load(std::memory_order_acquire)) std::this_thread::yield();

    std::uint64_t next = 0;
    for (auto _ : state) {
        while (!q.try_push(next)) {
        }
        ++next;
    }
    done.store(true, std::memory_order_release);
    consumer.join();

    if (out_of_order) {
        state.SkipWithError("consumer saw items out of order");
    } else if (received.load(std::memory_order_relaxed) != next) {
        state.SkipWithError("consumer lost items");
    }
    state.SetItemsProcessed(static_cast<std::int64_t>(next));
}

template <std::size_t N>
void BM_SpscRingBuffer(benchmark::State& state) {
    // Heap: the 65536-slot queue is 512 KiB, too much for a thread stack.
    auto q = std::make_unique<lockfree::SpscRingBuffer<std::uint64_t, N>>();
    run_throughput(state, *q);
}

void BM_MutexQueue(benchmark::State& state) {
    lockfree::bench::MutexQueue<std::uint64_t> q(static_cast<std::size_t>(state.range(0)));
    run_throughput(state, q);
}

} // namespace

BENCHMARK_TEMPLATE(BM_SpscRingBuffer, 64)->UseRealTime();
BENCHMARK_TEMPLATE(BM_SpscRingBuffer, 1024)->UseRealTime();
BENCHMARK_TEMPLATE(BM_SpscRingBuffer, 65536)->UseRealTime();
BENCHMARK(BM_MutexQueue)->Arg(64)->Arg(1024)->Arg(65536)->UseRealTime();
