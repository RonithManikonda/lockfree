#pragma once
#include <atomic>
#include <thread>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace lockfree {

// Tell the CPU we're in a spin-wait loop. It doesn't order memory; it only
// changes how the core spends the wait.
//
// x86: `pause` keeps the core from flooding the pipeline with speculative
// loads of the contended line (and the memory-order machine clear when the
// line finally changes), and hands resources to the sibling hyperthread.
// ARM: `yield` is the architectural hint for the same thing. Some cores treat
// it as a near no-op (Rust's spin_loop uses `isb` on aarch64 for a longer
// stall); the exponential growth in Backoff matters more than the exact
// instruction.
inline void cpu_relax() noexcept {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    _mm_pause();
#elif defined(__aarch64__) || defined(__arm__)
    asm volatile("yield" ::: "memory");
#else
    // Unknown target: at least keep the loop from being optimized away.
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

// Exponential backoff for contended CAS loops and spin-waits.
//
//   Backoff backoff;
//   while (!head.compare_exchange_weak(expected, desired)) backoff.pause();
//
// The first spin_limit + 1 pauses spin 1, 2, 4, ... 2^spin_limit times with
// cpu_relax(). Spreading retries out in time is what breaks up a convoy of
// threads that all failed the same CAS and would otherwise retry in lock
// step. After that it falls back to std::this_thread::yield(): if we've been
// waiting this long, the thread we're waiting on is probably descheduled
// (oversubscription), and spinning only delays it further.
//
// Not thread-safe; each waiting thread owns its own Backoff.
class Backoff {
public:
    // 2^6 = 64 relax instructions per pause at the top of the spin phase,
    // 127 in total before the first yield: a few hundred ns to ~1 µs.
    static constexpr unsigned spin_limit = 6;

    void pause() noexcept {
        if (step_ <= spin_limit) {
            for (unsigned i = 0, n = 1u << step_; i < n; ++i) cpu_relax();
            ++step_;
        } else {
            std::this_thread::yield();
        }
    }

    // True once pause() has moved on to yielding.
    bool is_yielding() const noexcept { return step_ > spin_limit; }

    // Call after making progress so the next wait starts with short spins.
    void reset() noexcept { step_ = 0; }

private:
    unsigned step_ = 0;
};

} // namespace lockfree
