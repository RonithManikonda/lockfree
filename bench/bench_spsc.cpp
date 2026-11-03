// SPSC throughput: the benchmark thread is the producer, a second thread is
// the consumer. Each iteration moves one item through the queue, so
// items_per_second is end-to-end transfer rate. Real time, not CPU time:
// the producer's CPU time says nothing about how long the consumer took.

#include <benchmark/benchmark.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <thread>

#include "baseline_queues.hpp"
#include "latency.hpp"
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

// Ping-pong latency: the benchmark thread pushes a token into `ping`, an
// echo thread pops it and pushes it back on `pong`, and we time each full
// round trip. Only one token is ever in flight, so this measures how long a
// handoff takes when the other side is waiting for it, not how many items
// fit through when both sides stay busy (that's the throughput bench).
//
// Each sample is one round trip, i.e. two handoffs. steady_clock on Apple
// Silicon ticks at 24 MHz (~42 ns), so the low percentiles are quantized to
// that; the shape of the tail is still what matters.
template <typename Ping, typename Pong>
void run_ping_pong(benchmark::State& state, Ping& ping, Pong& pong) {
    std::atomic<bool> ready{false};
    std::atomic<bool> done{false};

    std::thread echo([&] {
        std::uint64_t token = 0;
        ready.store(true, std::memory_order_release);
        while (!done.load(std::memory_order_acquire)) {
            if (ping.try_pop(token)) {
                while (!pong.try_push(token)) {
                }
            }
        }
    });
    while (!ready.load(std::memory_order_acquire)) std::this_thread::yield();

    lockfree::bench::LatencyHistogram histogram;
    std::uint64_t token = 0;
    std::uint64_t reply = 0;
    bool mismatch = false;
    for (auto _ : state) {
        const auto start = std::chrono::steady_clock::now();
        while (!ping.try_push(token)) {
        }
        while (!pong.try_pop(reply)) {
        }
        const auto stop = std::chrono::steady_clock::now();
        histogram.record(static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count()));
        if (reply != token) mismatch = true;
        ++token;
    }
    // The last token has already come back, so the echo thread is idle and
    // will see done on its next spin.
    done.store(true, std::memory_order_release);
    echo.join();

    if (mismatch) state.SkipWithError("echo returned the wrong token");
    state.counters["p50_ns"] = static_cast<double>(histogram.percentile(0.50));
    state.counters["p99_ns"] = static_cast<double>(histogram.percentile(0.99));
    state.counters["p999_ns"] = static_cast<double>(histogram.percentile(0.999));
    state.counters["max_ns"] = static_cast<double>(histogram.max());
}

void BM_SpscRoundTrip(benchmark::State& state) {
    lockfree::SpscRingBuffer<std::uint64_t, 64> ping;
    lockfree::SpscRingBuffer<std::uint64_t, 64> pong;
    run_ping_pong(state, ping, pong);
}

void BM_MutexRoundTrip(benchmark::State& state) {
    lockfree::bench::MutexQueue<std::uint64_t> ping(64);
    lockfree::bench::MutexQueue<std::uint64_t> pong(64);
    run_ping_pong(state, ping, pong);
}

} // namespace

BENCHMARK_TEMPLATE(BM_SpscRingBuffer, 64)->UseRealTime();
BENCHMARK_TEMPLATE(BM_SpscRingBuffer, 1024)->UseRealTime();
BENCHMARK_TEMPLATE(BM_SpscRingBuffer, 65536)->UseRealTime();
BENCHMARK(BM_MutexQueue)->Arg(64)->Arg(1024)->Arg(65536)->UseRealTime();
BENCHMARK(BM_SpscRoundTrip)->UseRealTime();
BENCHMARK(BM_MutexRoundTrip)->UseRealTime();
