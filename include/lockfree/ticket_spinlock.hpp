#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>

#include "lockfree/atomic_backoff.hpp"
#include "lockfree/detail/atomic_ref.hpp"
#include "lockfree/detail/cache_line.hpp"

namespace lockfree {

// FIFO spinlock: take a number, wait until it's called.
//
//   lock:   my = next_.fetch_add(1);  wait until serving_ == my
//   unlock: serving_ += 1             (only the holder writes it)
//
// Unlike a test-and-set lock, the order threads arrive in (the order of the
// fetch_adds) is the order they get the lock, so nobody starves. The price is
// that every waiter spins on the same `serving_` line, and each unlock
// invalidates it in every waiter's cache at once. McsLock fixes that with a
// line per waiter; docs/locks.md has the comparison.
//
// Proportional backoff: a waiter knows how many tickets are ahead of it
// (my - serving_), so it pauses roughly that many hand-off times before
// looking again, instead of hammering the line on every iteration. The
// first in line still polls tightly.
//
// Meets Lockable (lock / try_lock / unlock), so std::lock_guard and
// std::unique_lock work. Not recursive. 2^32 threads queued at once would
// wrap the tickets; that's not a real limit.
class TicketSpinlock {
public:
    TicketSpinlock() = default;
    TicketSpinlock(const TicketSpinlock&) = delete;
    TicketSpinlock& operator=(const TicketSpinlock&) = delete;

    void lock() noexcept {
        // Relaxed: the ticket only fixes our place in line. What we must see
        // from the previous holder comes through serving_ below.
        const std::uint32_t my = next_.fetch_add(1, std::memory_order_relaxed);
        std::uint32_t polls = 0;
        for (;;) {
            // Acquire pairs with unlock's release: the previous holder's
            // critical section happens-before ours.
            const std::uint32_t now = serving_.load(std::memory_order_acquire);
            if (now == my) return;
            // Unsigned subtraction, so this is right across wraparound.
            const std::uint32_t ahead = my - now;
            for (std::uint32_t i = 0; i < ahead * spins_per_waiter; ++i) cpu_relax();
            // With more threads than cores, the holder may not even be
            // running. Spinning then only steals its time slice.
            if (++polls > yield_after) std::this_thread::yield();
        }
    }

    // Takes the lock only if nobody holds it or waits for it, so it never
    // jumps the queue.
    bool try_lock() noexcept {
        // Acquire pairs with the last unlock, like the load in lock().
        std::uint32_t now = serving_.load(std::memory_order_acquire);
        // Free means next_ == serving_. If the CAS sees next_ == now, no
        // ticket was taken past `now`, and serving_ never passes next_, so
        // the `now` we loaded was still current: it came from the latest
        // unlock, the one we need to synchronize with. The CAS itself only
        // claims the ticket, hence relaxed.
        return next_.compare_exchange_strong(now, now + 1, std::memory_order_relaxed,
                                             std::memory_order_relaxed);
    }

    void unlock() noexcept {
        // Only the holder writes serving_, so a load + store is enough; no
        // read-modify-write needed.
        const std::uint32_t now = serving_.load(std::memory_order_relaxed);
        serving_.store(now + 1, std::memory_order_release);
    }

    // Holder plus waiters at some recent moment. For tests and stats.
    std::uint32_t queue_length_approx() const noexcept {
        return next_.load(std::memory_order_relaxed) - serving_.load(std::memory_order_relaxed);
    }

private:
    // Roughly the cost of one hand-off in cpu_relax() units on the machines
    // this was tuned on (M4 Pro); only the order of magnitude matters.
    static constexpr std::uint32_t spins_per_waiter = 32;
    static constexpr std::uint32_t yield_after = 1024;

    // Arrivals hit next_, waiters poll serving_: separate lines, so a new
    // arrival doesn't knock the line out from under every spinner.
    alignas(detail::cache_line_size) detail::atomic<std::uint32_t> next_{0};
    alignas(detail::cache_line_size) detail::atomic<std::uint32_t> serving_{0};
};

} // namespace lockfree
