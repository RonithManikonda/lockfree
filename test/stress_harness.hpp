#pragma once

// Small harness for multi-threaded stress tests.
//
//   run_stress(threads, [&](std::size_t index, StressRng& rng) { ... });
//
// All threads wait on a latch and start together, so the interesting part
// actually overlaps instead of thread 0 finishing before thread 3 exists.
// Each thread gets its own RNG derived from one base seed. The seed comes
// from LOCKFREE_SEED if set, otherwise it's random, and StressTest prints it
// when a test fails so the run can be repeated with the same delays.
//
// LOCKFREE_STRESS_SCALE multiplies iteration counts (see scaled()), so CI can
// run short and a local soak can run long without editing the tests.

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <latch>
#include <limits>
#include <random>
#include <thread>
#include <vector>

namespace lockfree::test {

// splitmix64: tiny, fast, and good enough to decorrelate per-thread seeds
// that differ only by the thread index.
inline std::uint64_t mix_seed(std::uint64_t x) noexcept {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

// xorshift64*. Satisfies UniformRandomBitGenerator so it also works with
// <random> distributions.
class StressRng {
public:
    using result_type = std::uint64_t;

    explicit StressRng(std::uint64_t seed) noexcept : state_(mix_seed(seed) | 1) {}

    result_type operator()() noexcept {
        state_ ^= state_ >> 12;
        state_ ^= state_ << 25;
        state_ ^= state_ >> 27;
        return state_ * 0x2545f4914f6cdd1dull;
    }

    // Uniform-ish in [0, bound); the modulo bias is irrelevant here.
    std::uint64_t below(std::uint64_t bound) noexcept { return (*this)() % bound; }

    static constexpr result_type min() noexcept { return 0; }
    static constexpr result_type max() noexcept {
        return std::numeric_limits<result_type>::max();
    }

private:
    std::uint64_t state_;
};

namespace detail {

inline std::uint64_t read_seed_from_env() {
    if (const char* s = std::getenv("LOCKFREE_SEED"); s && *s) {
        return std::strtoull(s, nullptr, 0);
    }
    std::random_device rd;
    return (std::uint64_t{rd()} << 32) ^ rd();
}

inline double read_scale_from_env() {
    if (const char* s = std::getenv("LOCKFREE_STRESS_SCALE"); s && *s) {
        const double v = std::strtod(s, nullptr);
        if (v > 0) return v;
    }
    return 1.0;
}

} // namespace detail

// One base seed per process, so every test in a binary replays together.
inline std::uint64_t stress_seed() {
    static const std::uint64_t seed = detail::read_seed_from_env();
    return seed;
}

inline double stress_scale() {
    static const double scale = detail::read_scale_from_env();
    return scale;
}

// Scale an iteration count by LOCKFREE_STRESS_SCALE, never below 1.
inline std::uint64_t scaled(std::uint64_t base) {
    const double n = static_cast<double>(base) * stress_scale();
    return n < 1.0 ? 1 : static_cast<std::uint64_t>(n);
}

// Occasionally burn a random number of cycles. Shifts the relative timing of
// the threads so runs explore more interleavings than lock-step loops would,
// e.g. the consumer catching up exactly as the producer wraps.
inline void maybe_delay(StressRng& rng) noexcept {
    if (rng.below(64) != 0) return;
    const std::uint64_t spins = rng.below(256);
    for (std::uint64_t i = 0; i < spins; ++i) {
        // Compiler-only barrier: keeps the loop from being optimized away
        // without adding a hardware fence that would itself order things.
        std::atomic_signal_fence(std::memory_order_seq_cst);
    }
}

// Runs fn(thread_index, rng) on `threads` threads that all start together.
// fn must not throw. Use EXPECT_* (not ASSERT_*) inside fn, or record
// results and check them after run_stress returns; an ASSERT that returns
// early from one side can leave the other side spinning forever.
template <typename Fn>
void run_stress(std::size_t threads, Fn&& fn) {
    std::latch start(static_cast<std::ptrdiff_t>(threads));
    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (std::size_t i = 0; i < threads; ++i) {
        pool.emplace_back([&, i] {
            StressRng rng(stress_seed() ^ mix_seed(i));
            start.arrive_and_wait();
            fn(i, rng);
        });
    }
    for (auto& t : pool) t.join();
}

// Fixture for stress tests: prints the seed when the test fails.
class StressTest : public ::testing::Test {
protected:
    void TearDown() override {
        if (HasFailure()) {
            std::fprintf(stderr, "stress seed: LOCKFREE_SEED=%llu (scale %g)\n",
                         static_cast<unsigned long long>(stress_seed()), stress_scale());
        }
    }
};

} // namespace lockfree::test
