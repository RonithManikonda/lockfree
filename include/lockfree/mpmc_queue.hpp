#pragma once
#include <atomic>
#include <cstddef>
#include <new>
#include <optional>
#include <type_traits>
#include <utility>

#include "lockfree/detail/cache_line.hpp"

namespace lockfree {

// Bounded multi-producer / multi-consumer queue (Dmitry Vyukov's design).
//
// enqueue_pos_ and dequeue_pos_ are free-running counters, like in
// SpscRingBuffer, but now several threads race to advance each one with a
// CAS. Winning the CAS for position p gives a thread exclusive ownership of
// cell p & (N - 1) for that lap.
//
// What makes it work is the per-cell sequence number. For the cell that
// position p maps to:
//   seq == p          the cell is free for the producer that claims p
//   seq == p + 1      it holds the element pushed at p, ready for the
//                     consumer that claims p
//   seq == p + N      the consumer is done; free for the producer of p + N
// A producer only claims p once it has seen seq == p, so it can never write
// into a cell whose previous element hasn't been consumed yet; that shows up
// as seq == p - N + 1 ("full"). Because seq encodes the lap, not just a
// full/empty bit, a producer that read a stale position can't mistake a cell
// from an earlier or later lap for its own (no ABA on the cell state).
//
// The position CASes are relaxed: they only hand out tickets. All the
// publishing goes through seq (release store after writing the cell,
// acquire load before touching it), so only seq needs ordering.
//
// Lock-free, not wait-free: a thread that claimed a position and then got
// descheduled before storing seq blocks that one cell, and anyone who wants
// it reports full / empty until it resumes.
template <typename T, std::size_t N>
class MpmcQueue {
    static_assert(N >= 2, "capacity must be at least 2");
    static_assert((N & (N - 1)) == 0, "capacity must be a power of two");
    static_assert(std::is_nothrow_destructible_v<T>,
                  "pop and the destructor can't recover from a throwing ~T");
    static_assert(std::is_nothrow_move_constructible_v<T>,
                  "a claimed cell can't be given back, so filling it must not throw");

public:
    MpmcQueue() noexcept {
        for (std::size_t i = 0; i < N; ++i) cells_[i].seq.store(i, std::memory_order_relaxed);
    }
    MpmcQueue(const MpmcQueue&) = delete;
    MpmcQueue& operator=(const MpmcQueue&) = delete;

    // Destroys whatever was pushed and never popped. Must not race with any
    // push or pop.
    ~MpmcQueue() {
        const std::size_t end = enqueue_pos_.load(std::memory_order_relaxed);
        for (std::size_t p = dequeue_pos_.load(std::memory_order_relaxed); p != end; ++p) {
            slot(p)->~T();
        }
    }

    // Any thread. If T's constructor throws, nothing was claimed and the
    // queue is unchanged.
    template <typename... Args>
    bool try_emplace(Args&&... args) noexcept(
        std::is_nothrow_constructible_v<T, Args&&...>) {
        if constexpr (std::is_nothrow_constructible_v<T, Args&&...>) {
            return emplace_claimed(std::forward<Args>(args)...);
        } else {
            // Build it before claiming a cell: once we own a position, every
            // consumer behind us waits for it, so we can't bail out.
            T tmp(std::forward<Args>(args)...);
            return emplace_claimed(std::move(tmp));
        }
    }

    bool try_push(const T& item) noexcept(std::is_nothrow_copy_constructible_v<T>) {
        return try_emplace(item);
    }

    bool try_push(T&& item) noexcept { return try_emplace(std::move(item)); }

    // Any thread. Move-assigns into `out`.
    bool try_pop(T& out) noexcept
        requires std::is_nothrow_move_assignable_v<T>
    {
        std::size_t pos;
        Cell* c = claim_for_pop(pos);
        if (c == nullptr) return false;
        out = std::move(*c->get());
        release_after_pop(c, pos);
        return true;
    }

    // Any thread. Works for types that aren't default-constructible or
    // assignable.
    std::optional<T> try_pop() noexcept {
        std::size_t pos;
        Cell* c = claim_for_pop(pos);
        if (c == nullptr) return std::nullopt;
        std::optional<T> out(std::move(*c->get()));
        release_after_pop(c, pos);
        return out;
    }

    static constexpr std::size_t capacity() noexcept { return N; }

    // A snapshot that may be stale by the time it returns. Counts claimed
    // positions, so a push that is mid-way through filling its cell counts.
    std::size_t size_approx() const noexcept {
        // dequeue first: enqueue can only have grown since, so the
        // difference isn't negative, but it can overshoot N.
        const std::size_t d = dequeue_pos_.load(std::memory_order_relaxed);
        const std::size_t e = enqueue_pos_.load(std::memory_order_relaxed);
        const std::size_t n = e - d;
        return n > N ? N : n;
    }

    bool empty_approx() const noexcept { return size_approx() == 0; }

private:
    static constexpr std::size_t mask = N - 1;

    struct Cell {
        std::atomic<std::size_t> seq;
        alignas(T) std::byte bytes[sizeof(T)];

        // launder: each lap constructs a new T in the same bytes.
        T* get() noexcept { return std::launder(reinterpret_cast<T*>(bytes)); }
    };

    // Distance between the cell's sequence and what we want it to be, as a
    // signed number so "behind" and "ahead" survive counter wraparound.
    static std::ptrdiff_t lag(std::size_t seq, std::size_t want) noexcept {
        return static_cast<std::ptrdiff_t>(seq - want);
    }

    template <typename... Args>
    bool emplace_claimed(Args&&... args) noexcept {
        std::size_t pos = enqueue_pos_.load(std::memory_order_relaxed);
        Cell* c;
        for (;;) {
            c = &cells_[pos & mask];
            // acquire pairs with the consumer's release in release_after_pop:
            // seeing seq == pos means the previous element in this cell has
            // been moved out and destroyed, so we can construct over it.
            const std::size_t seq = c->seq.load(std::memory_order_acquire);
            const std::ptrdiff_t d = lag(seq, pos);
            if (d == 0) {
                // On failure the CAS reloads pos and we look at the new cell.
                if (enqueue_pos_.compare_exchange_weak(pos, pos + 1,
                                                       std::memory_order_relaxed)) {
                    break;
                }
            } else if (d < 0) {
                // The cell is still a lap behind: its last element hasn't
                // been consumed. The queue is full (as of this load).
                return false;
            } else {
                // Another producer already took pos; catch up.
                pos = enqueue_pos_.load(std::memory_order_relaxed);
            }
        }
        ::new (static_cast<void*>(c->bytes)) T(std::forward<Args>(args)...);
        // release publishes the constructed element to the consumer that
        // will acquire-load seq == pos + 1.
        c->seq.store(pos + 1, std::memory_order_release);
        return true;
    }

    // Returns the cell this thread now owns (and its position in `pos`), or
    // nullptr if the queue looked empty.
    Cell* claim_for_pop(std::size_t& pos) noexcept {
        pos = dequeue_pos_.load(std::memory_order_relaxed);
        for (;;) {
            Cell* c = &cells_[pos & mask];
            // acquire pairs with the producer's release: the element is fully
            // constructed before we read it.
            const std::size_t seq = c->seq.load(std::memory_order_acquire);
            const std::ptrdiff_t d = lag(seq, pos + 1);
            if (d == 0) {
                if (dequeue_pos_.compare_exchange_weak(pos, pos + 1,
                                                       std::memory_order_relaxed)) {
                    return c;
                }
            } else if (d < 0) {
                // Nothing pushed at pos yet (or the push hasn't finished).
                return nullptr;
            } else {
                pos = dequeue_pos_.load(std::memory_order_relaxed);
            }
        }
    }

    void release_after_pop(Cell* c, std::size_t pos) noexcept {
        c->get()->~T();
        // release: our read and destruction happen-before the producer of
        // the next lap constructs into this cell.
        c->seq.store(pos + N, std::memory_order_release);
    }

    T* slot(std::size_t pos) noexcept { return cells_[pos & mask].get(); }

    // Producers hammer enqueue_pos_, consumers hammer dequeue_pos_; keep
    // them off each other's line and off the cells.
    alignas(detail::cache_line_size) std::atomic<std::size_t> enqueue_pos_{0};
    alignas(detail::cache_line_size) std::atomic<std::size_t> dequeue_pos_{0};
    alignas(detail::cache_line_size) Cell cells_[N];
};

} // namespace lockfree
