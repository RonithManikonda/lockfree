# lockfree

[![ci](https://github.com/RonithManikonda/lockfree/actions/workflows/ci.yml/badge.svg)](https://github.com/RonithManikonda/lockfree/actions/workflows/ci.yml)

A from-scratch C++20 **concurrent queue library** built to develop deep understanding
of lock-free programming — paired with a **verification harness** (sanitizers,
stress tests, linearizability checking, a deterministic scheduler) and a
**benchmark suite**.

## Status

| Piece | State |
|-------|-------|
| `SpscRingBuffer<T, N>` | done: unit + two-thread stress tests, [design notes](docs/spsc_ring_buffer.md) |
| Stress harness (`test/stress_harness.hpp`) | done: start barrier, seeded per-thread RNG, `LOCKFREE_SEED` / `LOCKFREE_STRESS_SCALE` |
| Benchmarks (`bench/bench_spsc.cpp`) | SPSC throughput vs a mutex queue, ping-pong latency p50/p99/p99.9 |
| Verification | `scripts/check.sh` (Debug, TSan, ASan, UBSan); CI on Linux gcc/clang and macOS arm64 |
| MPMC, MPSC, atomic stack, reclamation, primitives | not started (headers are stubs) |
| Linearizability checker, adversarial scheduler | not started |

## Usage

Header-only: add `include/` to your include path (or link the `lockfree` CMake
target) and include what you need.

```cpp
#include <lockfree/spsc_ring_buffer.hpp>

#include <cstdio>
#include <string>
#include <thread>

int main() {
    // Capacity must be a power of two; all N slots are usable.
    lockfree::SpscRingBuffer<std::string, 1024> q;

    // Exactly one thread pushes and exactly one pops.
    std::thread producer([&] {
        for (int i = 0; i < 100; ++i) {
            while (!q.try_push("msg " + std::to_string(i))) std::this_thread::yield();
        }
    });

    for (int received = 0; received < 100;) {
        if (auto s = q.try_pop()) {  // std::optional<std::string>
            std::puts(s->c_str());
            ++received;
        } else {
            std::this_thread::yield();
        }
    }
    producer.join();
}
```

`try_push` / `try_pop` never block: they return `false` (or an empty
`optional`) when the queue is full or empty, and the caller decides whether to
spin, yield, or do something else. `try_emplace(args...)` constructs in place.

## Structures, by tier

**Tier 1 — foundations**
- `SpscRingBuffer<T, N>` — cache-line-padded head/tail, acquire/release, cached-index optimization
- Baseline benchmark vs `std::mutex` + `std::deque`

**Tier 2 — contention**
- `MpscQueue<T>` — intrusive, Vyukov-style, single producer-side CAS
- `MpmcQueue<T, N>` — bounded, per-slot sequence numbers
- `AtomicStack<T>` — reproduce ABA first, then fix with tagged pointers

**Tier 3 — reclamation**
- Hazard pointers, then epoch-based reclamation
- Unbounded Michael–Scott queue on top of one of them

**Tier 4 — primitives**
- `SeqLock` — single-writer snapshots
- Ticket spinlock and MCS lock (for contrast)
- `AtomicSharedPtr` (stretch goal)

## Verification (non-negotiable)

- TSan + ASan in CI
- Stress tests with randomized thread counts and delays
- Linearizability checking via timestamped operation logs (`tools/linearizability`)
- Exhaustive small-interleaving / model checking for SPSC and MPMC
- **Adversarial scheduler** (`tools/scheduler`) — a mini-Loom for C++: an atomic-op
  shim yields to a seed-driven controller so failing runs replay exactly; heuristics
  preempt right after a CAS, delay inside the reclamation window, widen the ABA gap;
  failure schedules are shrunk to a minimal breaking interleaving, with coverage output
- Run the suite on **ARM** (Apple Silicon / Graviton) so x86 TSO doesn't hide a
  missing acquire/release

## Benchmarks

- Throughput vs thread count
- Latency percentiles (p50 / p99 / p99.9 — not means)
- Thread pinning, cross-socket vs same-socket
- `perf stat` for cache misses and false sharing
- Reference comparison vs `moodycamel::ConcurrentQueue` and `boost::lockfree`

## Layout

```
include/lockfree/   header-only data structures
  detail/atomic_ref.hpp   atomic shim (real std::atomic, or scheduler-controlled in test mode)
test/               GTest unit tests + stress harness
bench/              Google Benchmark suite
tools/
  scheduler/          deterministic interleaving controller (mini-Loom)
  linearizability/    offline linearizability checker
docs/               per-structure writeup: invariants + memory-ordering justification
```

## Build

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

Sanitizers:

```bash
cmake -S . -B build-tsan -DLOCKFREE_SANITIZER=thread && cmake --build build-tsan
cmake -S . -B build-asan -DLOCKFREE_SANITIZER=address && cmake --build build-asan
```

Or run everything (Debug, TSan, ASan, UBSan) and get one pass/fail:

```bash
scripts/check.sh            # or e.g. scripts/check.sh tsan
```

## Requirements

- C++20 compiler (Clang or GCC)
- CMake >= 3.20
- GTest and Google Benchmark are fetched automatically if not installed
