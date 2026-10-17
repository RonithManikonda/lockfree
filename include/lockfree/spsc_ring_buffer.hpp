#pragma once
#include <atomic>
#include <cstddef>
#include <new>
#include <optional>
#include <type_traits>
#include <utility>

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
//
// Slots are raw storage, not T objects. A slot holds a live T exactly when its
// counter value is in [read_, write_): push constructs into it before
// publishing write_, pop destroys it before publishing read_. So T needs no
// default constructor, and nothing is constructed for empty slots.
//
// Each side also keeps a private copy of the other side's counter
// (read_cache_ for the producer, write_cache_ for the consumer) and only
// reloads the real atomic when the queue looks full or empty from that copy.
// The copy can only be stale in the safe direction: the real read_ is never
// behind read_cache_, so the producer may think the queue is fuller than it
// is, never emptier (and the mirror image for the consumer).
template <typename T, std::size_t N>
class SpscRingBuffer {
    static_assert(N >= 2, "capacity must be at least 2");
    static_assert((N & (N - 1)) == 0, "capacity must be a power of two");
    static_assert(std::is_nothrow_destructible_v<T>,
                  "pop and the destructor can't recover from a throwing ~T");

public:
    SpscRingBuffer() = default;
    SpscRingBuffer(const SpscRingBuffer&) = delete;
    SpscRingBuffer& operator=(const SpscRingBuffer&) = delete;

    // Destroys whatever was pushed and never popped. Must not race with
    // either side, like any destructor.
    ~SpscRingBuffer() {
        const std::size_t w = write_.load(std::memory_order_relaxed);
        for (std::size_t r = read_.load(std::memory_order_relaxed); r != w; ++r) {
            slot(r)->~T();
        }
    }

    // Producer only. Constructs the element in place. If T's constructor
    // throws, nothing was published and the queue is unchanged.
    template <typename... Args>
    bool try_emplace(Args&&... args) noexcept(
        std::is_nothrow_constructible_v<T, Args&&...>) {
        const std::size_t w = write_.load(std::memory_order_relaxed);
        if (w - read_cache_ == N) {
            // acquire pairs with the consumer's release store of read_: once
            // we see the slot as free, the consumer's destruction of the old
            // element has finished and we can construct over it. Every value
            // in read_cache_ came from such a load, so trusting the cache on
            // the fast path is covered by the same happens-before edge.
            read_cache_ = read_.load(std::memory_order_acquire);
            if (w - read_cache_ == N) return false;
        }
        ::new (static_cast<void*>(storage(w))) T(std::forward<Args>(args)...);
        // release publishes the constructed element before the new count.
        write_.store(w + 1, std::memory_order_release);
        return true;
    }

    bool try_push(const T& item) noexcept(std::is_nothrow_copy_constructible_v<T>) {
        return try_emplace(item);
    }

    bool try_push(T&& item) noexcept(std::is_nothrow_move_constructible_v<T>) {
        return try_emplace(std::move(item));
    }

    // Consumer only. Move-assigns into `out`. If that throws, the element
    // stays in the queue (the slot isn't released).
    bool try_pop(T& out) noexcept(std::is_nothrow_move_assignable_v<T>) {
        const std::size_t r = read_.load(std::memory_order_relaxed);
        if (!has_item(r)) return false;
        T* p = slot(r);
        out = std::move(*p);
        p->~T();
        // release: our read and destruction of the slot happen-before the
        // producer constructs into it again.
        read_.store(r + 1, std::memory_order_release);
        return true;
    }

    // Consumer only. Works for types that aren't default-constructible or
    // assignable, at the cost of an optional.
    std::optional<T> try_pop() noexcept(std::is_nothrow_move_constructible_v<T>) {
        const std::size_t r = read_.load(std::memory_order_relaxed);
        if (!has_item(r)) return std::nullopt;
        T* p = slot(r);
        std::optional<T> out(std::move(*p));
        p->~T();
        read_.store(r + 1, std::memory_order_release);
        return out;
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

    struct Slot {
        alignas(T) std::byte bytes[sizeof(T)];
    };

    // Consumer only: is there an element at counter value r?
    bool has_item(std::size_t r) noexcept {
        if (r != write_cache_) return true;
        // acquire pairs with the producer's release store of write_, so the
        // constructed element is visible before we touch it. As on the
        // producer side, cached values all came from this acquire load.
        write_cache_ = write_.load(std::memory_order_acquire);
        return r != write_cache_;
    }

    void* storage(std::size_t index) noexcept { return buffer_[index & mask].bytes; }

    // launder: the bytes are reused for a new T each lap, so a plain cast
    // would let the compiler assume it still points at the old object.
    T* slot(std::size_t index) noexcept {
        return std::launder(static_cast<T*>(storage(index)));
    }

    // Each side's line holds its own counter plus its cached copy of the
    // other side's, so the hot path only touches lines the thread already
    // owns. Without the caches, every push read read_ and every pop read
    // write_, so each operation pulled the other core's line over (and the
    // next store on that core had to invalidate our copy again). With them,
    // a line only crosses cores when a side actually runs out of room or
    // items according to its copy, which in a steady stream is about once
    // per lap instead of once per element.
    //
    // Consumer's line: read_ is shared (the producer reloads it when the
    // queue looks full), write_cache_ is touched by the consumer only.
    alignas(detail::cache_line_size) std::atomic<std::size_t> read_{0};
    std::size_t write_cache_{0};
    // Producer's line.
    alignas(detail::cache_line_size) std::atomic<std::size_t> write_{0};
    std::size_t read_cache_{0};
    alignas(detail::cache_line_size) Slot buffer_[N];
};

} // namespace lockfree
