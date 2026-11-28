#include <gtest/gtest.h>

#include <lockfree/ticket_spinlock.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include "stress_harness.hpp"

// Contract and mutual-exclusion tests run for every lock type through a
// typed suite (McsLock joins later); fairness tests are per lock, since only
// the queue locks promise FIFO order.

using lockfree::TicketSpinlock;

namespace {

template <typename Lock>
class LockTest : public ::testing::Test {};

using Locks = ::testing::Types<TicketSpinlock>;
TYPED_TEST_SUITE(LockTest, Locks);

TYPED_TEST(LockTest, TryLockOnlyWhenFree) {
    TypeParam lock;
    ASSERT_TRUE(lock.try_lock());
    EXPECT_FALSE(lock.try_lock());
    lock.unlock();
    EXPECT_TRUE(lock.try_lock());
    lock.unlock();
}

TYPED_TEST(LockTest, WorksWithStdGuards) {
    TypeParam lock;
    {
        std::lock_guard guard(lock);
        EXPECT_FALSE(lock.try_lock());
    }
    std::unique_lock ul(lock, std::try_to_lock);
    EXPECT_TRUE(ul.owns_lock());
}

TYPED_TEST(LockTest, ManyLockUnlockCyclesOnOneThread) {
    // Long enough to wrap nothing, but exercises the counters well past the
    // first few values.
    TypeParam lock;
    for (int i = 0; i < 100'000; ++i) {
        lock.lock();
        lock.unlock();
    }
    EXPECT_TRUE(lock.try_lock());
    lock.unlock();
}

// Every thread increments a plain (non-atomic) counter under the lock. A
// lost update shows up in the total; TSan reports the race directly if
// acquire/release were missing. `inside` catches two holders at once even
// when their increments happen not to collide.
template <typename Lock>
void mutual_exclusion(std::size_t threads, std::uint64_t per_thread) {
    Lock lock;
    std::uint64_t counter = 0;
    std::atomic<int> inside{0};
    std::atomic<std::uint64_t> overlaps{0};

    lockfree::test::run_stress(threads, [&](std::size_t, lockfree::test::StressRng& rng) {
        for (std::uint64_t i = 0; i < per_thread; ++i) {
            lock.lock();
            if (inside.fetch_add(1, std::memory_order_relaxed) != 0) overlaps.fetch_add(1);
            ++counter;
            lockfree::test::maybe_delay(rng);
            inside.fetch_sub(1, std::memory_order_relaxed);
            lock.unlock();
        }
    });

    EXPECT_EQ(overlaps.load(), 0u);
    EXPECT_EQ(counter, threads * per_thread);
}

template <typename Lock>
class LockStress : public lockfree::test::StressTest {};
TYPED_TEST_SUITE(LockStress, Locks);

TYPED_TEST(LockStress, MutualExclusionTwoThreads) {
    mutual_exclusion<TypeParam>(2, lockfree::test::scaled(100'000));
}

TYPED_TEST(LockStress, MutualExclusionEightThreads) {
    mutual_exclusion<TypeParam>(8, lockfree::test::scaled(20'000));
}

TYPED_TEST(LockStress, TryLockAndLockMixed) {
    // try_lock must never succeed while someone holds the lock or queues for
    // it; half the threads use it, half block.
    TypeParam lock;
    std::uint64_t counter = 0;
    std::atomic<std::uint64_t> taken{0};
    const std::uint64_t per_thread = lockfree::test::scaled(20'000);

    lockfree::test::run_stress(4, [&](std::size_t index, lockfree::test::StressRng&) {
        for (std::uint64_t i = 0; i < per_thread; ++i) {
            if (index % 2 == 0) {
                lock.lock();
            } else if (!lock.try_lock()) {
                continue;
            }
            ++counter;
            lock.unlock();
            taken.fetch_add(1, std::memory_order_relaxed);
        }
    });
    EXPECT_EQ(counter, taken.load());
    EXPECT_GE(taken.load(), 2 * per_thread); // the blocking half always gets in
}

// FIFO: with the lock held, start waiters one at a time and let each take
// its ticket before the next starts. Arrival order is then known, and the
// lock has to hand out the critical section in exactly that order.
TEST(TicketSpinlock, GrantsInArrivalOrder) {
    constexpr std::uint32_t waiters = 6;
    TicketSpinlock lock;
    std::vector<std::uint32_t> order;
    std::vector<std::thread> pool;

    lock.lock();
    for (std::uint32_t i = 0; i < waiters; ++i) {
        pool.emplace_back([&, i] {
            lock.lock();
            order.push_back(i); // under the lock
            lock.unlock();
        });
        // Holder + the i + 1 waiters so far. The thread has taken its ticket
        // once the count moves, even if it's still spinning.
        while (lock.queue_length_approx() != i + 2) std::this_thread::yield();
    }
    lock.unlock();
    for (auto& t : pool) t.join();

    std::vector<std::uint32_t> expected(waiters);
    for (std::uint32_t i = 0; i < waiters; ++i) expected[i] = i;
    EXPECT_EQ(order, expected);
    EXPECT_EQ(lock.queue_length_approx(), 0u);
}

TEST(TicketSpinlock, TryLockDoesNotJumpTheQueue) {
    TicketSpinlock lock;
    lock.lock();
    std::thread waiter([&] {
        lock.lock();
        lock.unlock();
    });
    while (lock.queue_length_approx() != 2) std::this_thread::yield();
    lock.unlock();
    // The waiter may or may not have run yet. Either way, try_lock may only
    // succeed once the queue has drained, never in front of it.
    while (!lock.try_lock()) std::this_thread::yield();
    EXPECT_EQ(lock.queue_length_approx(), 1u);
    lock.unlock();
    waiter.join();
}

} // namespace
