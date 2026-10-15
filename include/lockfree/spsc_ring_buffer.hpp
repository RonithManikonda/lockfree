#pragma once
#include <atomic>
#include <cstddef>

#include "lockfree/detail/cache_line.hpp"

namespace lockfree {

// Bounded single-producer / single-consumer queue.
//
// read_ and write_ are free-running counters: they only ever increase and are
// mapped to a slot with `& (N - 1)`. Because they never wrap back to the slot
// range, "empty" (write_ == read_) and "full" (write_ - read_ == N) are
// distinguishable without sacrificing a slot. Unsigned overflow at 2^64 is
// harmless: N divides 2^64, so the masked index and the difference stay right.
//
// Invariant: 0 <= write_ - read_ <= N. Only the producer stores write_, only
// the consumer stores read_.
template <typename T, std::size_t N>
class SpscRingBuffer {
    static_assert(N >= 2, "capacity must be at least 2");
    static_assert((N & (N - 1)) == 0, "capacity must be a power of two");

public:
    // Producer only.
    bool try_push(const T& item) {
        const std::size_t w = write_.load(std::memory_order_relaxed);
        // acquire pairs with the consumer's release store of read_: once we
        // see the slot as free, the consumer's read of it has finished.
        if (w - read_.load(std::memory_order_acquire) == N) return false;
        buffer_[w & mask] = item;
        // release publishes the slot contents before the new count.
        write_.store(w + 1, std::memory_order_release);
        return true;
    }

    // Consumer only.
    bool try_pop(T& out) {
        const std::size_t r = read_.load(std::memory_order_relaxed);
        // acquire pairs with the producer's release store of write_, so the
        // slot write is visible before we read it.
        if (r == write_.load(std::memory_order_acquire)) return false;
        out = buffer_[r & mask];
        // release: our read of the slot happens-before the producer reuses it.
        read_.store(r + 1, std::memory_order_release);
        return true;
    }

    static constexpr std::size_t capacity() noexcept { return N; }

    // Only a snapshot: the other side may change it immediately. Exact when
    // called from the producer or consumer while the other side is idle.
    std::size_t size_approx() const noexcept {
        // Load read_ first: write_ can only have grown since, so the
        // difference is never negative. It can overshoot N if the consumer
        // and producer both advanced in between, hence the clamp.
        const std::size_t r = read_.load(std::memory_order_acquire);
        const std::size_t w = write_.load(std::memory_order_acquire);
        const std::size_t n = w - r;
        return n > N ? N : n;
    }

    bool empty_approx() const noexcept { return size_approx() == 0; }

private:
    static constexpr std::size_t mask = N - 1;

    // Each index on its own line so the producer and consumer don't
    // invalidate each other's cache line on every operation.
    alignas(detail::cache_line_size) std::atomic<std::size_t> read_{0};
    alignas(detail::cache_line_size) std::atomic<std::size_t> write_{0};
    alignas(detail::cache_line_size) T buffer_[N];
};

} // namespace lockfree
