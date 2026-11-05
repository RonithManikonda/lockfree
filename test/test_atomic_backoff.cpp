#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>

#include "lockfree/atomic_backoff.hpp"
#include "stress_harness.hpp"

using lockfree::Backoff;

TEST(Backoff, SpinsBeforeYielding) {
    Backoff b;
    for (unsigned i = 0; i <= Backoff::spin_limit; ++i) {
        EXPECT_FALSE(b.is_yielding()) << "pause " << i;
        b.pause();
    }
    EXPECT_TRUE(b.is_yielding());
    b.pause();  // stays in the yield phase
    EXPECT_TRUE(b.is_yielding());
}

TEST(Backoff, ResetStartsOver) {
    Backoff b;
    for (unsigned i = 0; i <= Backoff::spin_limit; ++i) b.pause();
    ASSERT_TRUE(b.is_yielding());
    b.reset();
    EXPECT_FALSE(b.is_yielding());
}

TEST(Backoff, CpuRelaxIsCallable) {
    // Mostly a compile check that the per-arch hint assembles here.
    for (int i = 0; i < 16; ++i) lockfree::cpu_relax();
}

class BackoffStress : public lockfree::test::StressTest {};

// Every thread hammers one counter with a CAS loop that backs off on
// failure. Lost updates would show up as a short total.
TEST_F(BackoffStress, ContendedCasCounterLosesNothing) {
    constexpr std::size_t threads = 8;
    const std::uint64_t per_thread = lockfree::test::scaled(20000);
    std::atomic<std::uint64_t> counter{0};

    lockfree::test::run_stress(threads, [&](std::size_t, lockfree::test::StressRng& rng) {
        for (std::uint64_t i = 0; i < per_thread; ++i) {
            Backoff backoff;
            std::uint64_t seen = counter.load(std::memory_order_relaxed);
            while (!counter.compare_exchange_weak(seen, seen + 1, std::memory_order_relaxed)) {
                backoff.pause();
            }
            lockfree::test::maybe_delay(rng);
        }
    });

    EXPECT_EQ(counter.load(), threads * per_thread);
}
