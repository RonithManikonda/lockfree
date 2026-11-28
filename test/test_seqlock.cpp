#include <gtest/gtest.h>

#include <lockfree/seqlock.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "stress_harness.hpp"

using lockfree::SeqLock;

namespace {

// Bigger than a word and with an odd tail, so the word copy has to handle a
// partial last word.
struct Snapshot {
    std::uint64_t a;
    std::uint64_t b;
    std::uint32_t c;
    bool operator==(const Snapshot&) const = default;
};
static_assert(sizeof(Snapshot) == 24);

struct Odd {
    std::uint8_t bytes[5];
};

TEST(SeqLock, DefaultConstructsTheValue) {
    SeqLock<Snapshot> lock;
    EXPECT_EQ(lock.load(), (Snapshot{0, 0, 0}));
    EXPECT_EQ(lock.version(), 0u);
}

TEST(SeqLock, LoadReturnsTheLastStore) {
    SeqLock<Snapshot> lock(Snapshot{1, 2, 3});
    EXPECT_EQ(lock.load(), (Snapshot{1, 2, 3}));
    lock.store(Snapshot{4, 5, 6});
    lock.store(Snapshot{7, 8, 9});
    EXPECT_EQ(lock.load(), (Snapshot{7, 8, 9}));
    EXPECT_EQ(lock.version(), 2u);

    Snapshot out{};
    EXPECT_TRUE(lock.try_load(out));
    EXPECT_EQ(out, (Snapshot{7, 8, 9}));
}

TEST(SeqLock, ValueSmallerThanAWord) {
    SeqLock<Odd> lock(Odd{{1, 2, 3, 4, 5}});
    Odd v = lock.load();
    EXPECT_EQ(v.bytes[0], 1);
    EXPECT_EQ(v.bytes[4], 5);
}

// The reader side of an overlapping write, played on one thread through the
// hook between "counter is odd" and "data written".
TEST(SeqLock, ReadDuringAWriteFails) {
    SeqLock<Snapshot> lock(Snapshot{1, 1, 1});
    bool tried = false;
    lockfree::detail::ScopedTestHook hook([&](std::string_view where) {
        if (where != "seqlock_store_mid_write") return;
        tried = true;
        Snapshot out{};
        EXPECT_FALSE(lock.try_load(out)) << "counter is odd, the read must not succeed";
    });
    lock.store(Snapshot{2, 2, 2});
    EXPECT_TRUE(tried);
    EXPECT_EQ(lock.load(), (Snapshot{2, 2, 2}));
}

// A write that lands between the reader's copy and its re-check: the copy is
// stale, and the re-check is what throws it away.
TEST(SeqLock, WriteAfterTheCopyFailsTheRecheck) {
    SeqLock<Snapshot> lock(Snapshot{1, 1, 1});
    bool wrote = false;
    lockfree::detail::ScopedTestHook hook([&](std::string_view where) {
        if (where != "seqlock_load_after_copy" || wrote) return;
        wrote = true;
        lock.store(Snapshot{2, 2, 2});
    });
    Snapshot out{};
    EXPECT_FALSE(lock.try_load(out));
    EXPECT_TRUE(wrote);
    EXPECT_TRUE(lock.try_load(out));
    EXPECT_EQ(out, (Snapshot{2, 2, 2}));
}

// --- Torn-read stress ------------------------------------------------------

// Every field of a written value is the same number, so a torn read (some
// words from one store, some from another) shows up as fields that differ.
// Wide enough to span two cache lines' worth of words, which makes tearing
// likely if the re-check were missing.
struct Wide {
    std::array<std::uint64_t, 16> f;
};

Wide wide(std::uint64_t v) {
    Wide w;
    w.f.fill(v);
    return w;
}

bool consistent(const Wide& w) {
    for (auto x : w.f)
        if (x != w.f[0]) return false;
    return true;
}

using SeqLockStress = lockfree::test::StressTest;

TEST_F(SeqLockStress, OneWriterReadersNeverSeeATornOrOlderValue) {
    constexpr std::size_t readers = 4;
    const std::uint64_t writes = lockfree::test::scaled(200'000);
    SeqLock<Wide> lock(wide(0));
    std::atomic<bool> done{false};
    std::array<std::uint64_t, readers> torn{}, backwards{}, reads{};

    lockfree::test::run_stress(readers + 1, [&](std::size_t index, lockfree::test::StressRng& rng) {
        if (index == 0) {
            for (std::uint64_t v = 1; v <= writes; ++v) {
                lock.store(wide(v));
                lockfree::test::maybe_delay(rng);
            }
            done.store(true, std::memory_order_release);
            return;
        }
        const std::size_t r = index - 1;
        std::uint64_t last = 0;
        while (!done.load(std::memory_order_acquire)) {
            const Wide w = lock.load();
            ++reads[r];
            if (!consistent(w)) ++torn[r];
            // One writer, so values only grow; a reader that sees one must
            // never see an older one after it.
            if (w.f[0] < last) ++backwards[r];
            last = w.f[0];
            lockfree::test::maybe_delay(rng);
        }
    });

    for (std::size_t r = 0; r < readers; ++r) {
        EXPECT_EQ(torn[r], 0u) << "reader " << r << " of " << reads[r] << " reads";
        EXPECT_EQ(backwards[r], 0u) << "reader " << r;
    }
    EXPECT_EQ(lock.load().f[0], writes);
    EXPECT_EQ(lock.version(), writes);
}

TEST_F(SeqLockStress, ConcurrentWritersNeverInterleaveTheirWords) {
    constexpr std::size_t writers = 3, readers = 3;
    const std::uint64_t per_writer = lockfree::test::scaled(50'000);
    SeqLock<Wide> lock(wide(0));
    std::atomic<std::size_t> writers_left{writers};
    std::array<std::uint64_t, readers> torn{};

    lockfree::test::run_stress(writers + readers, [&](std::size_t index, lockfree::test::StressRng& rng) {
        if (index < writers) {
            // Distinct values per writer: writer i writes i + 1, i + 1 + writers, ...
            for (std::uint64_t k = 0; k < per_writer; ++k) {
                lock.store(wide(index + 1 + k * writers));
                lockfree::test::maybe_delay(rng);
            }
            writers_left.fetch_sub(1, std::memory_order_release);
            return;
        }
        const std::size_t r = index - writers;
        while (writers_left.load(std::memory_order_acquire) != 0) {
            if (!consistent(lock.load())) ++torn[r];
            lockfree::test::maybe_delay(rng);
        }
    });

    for (std::size_t r = 0; r < readers; ++r) EXPECT_EQ(torn[r], 0u) << "reader " << r;
    EXPECT_TRUE(consistent(lock.load()));
    EXPECT_EQ(lock.version(), writers * per_writer);
}

} // namespace
