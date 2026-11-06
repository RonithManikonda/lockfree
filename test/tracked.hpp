#pragma once

// An element type that counts its live instances and catches double
// destruction, for checking that a container constructs and destroys
// exactly what it should.
//
// Each object carries a canary that its destructor clears, so destroying it
// twice (or destroying storage that never held an object) trips the check.
// The counters are atomic so threaded tests can use the type too; relaxed is
// enough because they're only read after the threads have been joined.

#include <gtest/gtest.h>

#include <atomic>

namespace lockfree::test {

struct Tracked {
    static inline std::atomic<int> live{0};
    static inline std::atomic<int> bad_destroys{0};
    static constexpr unsigned kAlive = 0xA11CE;

    explicit Tracked(int v) : value(v) { live.fetch_add(1, std::memory_order_relaxed); }
    Tracked(const Tracked& o) : value(o.value) { live.fetch_add(1, std::memory_order_relaxed); }
    Tracked(Tracked&& o) noexcept : value(o.value) { live.fetch_add(1, std::memory_order_relaxed); }
    Tracked& operator=(const Tracked&) = default;
    Tracked& operator=(Tracked&&) noexcept = default;
    ~Tracked() {
        if (canary != kAlive) bad_destroys.fetch_add(1, std::memory_order_relaxed);
        canary = 0;
        live.fetch_sub(1, std::memory_order_relaxed);
    }

    int value;
    unsigned canary = kAlive;
};

// Resets the counters before each test and checks them after it.
class TrackedTest : public ::testing::Test {
protected:
    void SetUp() override {
        Tracked::live = 0;
        Tracked::bad_destroys = 0;
    }
    void TearDown() override {
        EXPECT_EQ(Tracked::live.load(), 0) << "leaked or over-destroyed elements";
        EXPECT_EQ(Tracked::bad_destroys.load(), 0);
    }
};

} // namespace lockfree::test
