#pragma once
#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "lockfree/atomic_backoff.hpp"
#include "lockfree/detail/atomic_ref.hpp"
#include "lockfree/detail/cache_line.hpp"
#include "lockfree/detail/test_hook.hpp"

namespace lockfree {

// Sequence lock for a small trivially copyable value: readers never block the
// writer and never write shared memory; they just retry when a write overlaps.
// Good for data that's read far more often than it's written and is cheap to
// copy (a timestamp pair, a config snapshot, market data).
//
// The sequence counter is even when the value is stable and odd while a write
// is in progress:
//
//   writer: seq = odd  -> write the data -> seq = next even
//   reader: s1 = seq (retry if odd) -> copy the data -> s2 = seq (retry if s1 != s2)
//
// If s1 == s2 and it's even, no write overlapped the copy, so the copy is one
// consistent value.
//
// The data race problem. In the textbook version the reader copies the data
// with plain loads while the writer may be storing to it. Even though the
// torn copy is thrown away, that's a data race, which is undefined behaviour
// in C++ (and TSan reports it). So the value is kept as an array of 64-bit
// words, each a relaxed atomic. Concurrent relaxed accesses aren't a race,
// and on arm64/x86 a relaxed 8-byte load or store is the same instruction as
// a plain one, so this costs nothing. (Boehm, "Can seqlocks get along with
// programming language memory models?", MSPC 2012, which is also where the
// fence placement below comes from.)
//
// Writers are serialized by the counter itself: a writer CASes an even value
// to odd, so any number of threads may call store(). Readers are wait-free
// as long as writes are rare; a reader can starve under a constant stream of
// writes.
template <typename T>
class SeqLock {
    static_assert(std::is_trivially_copyable_v<T>, "the value is copied word by word");

    static constexpr std::size_t word_count = (sizeof(T) + sizeof(std::uint64_t) - 1) / sizeof(std::uint64_t);
    using Words = std::array<std::uint64_t, word_count>;

public:
    // T needs to be default constructible only for this constructor.
    SeqLock() : SeqLock(T{}) {}

    explicit SeqLock(const T& initial) noexcept { write_words(to_words(initial)); }

    SeqLock(const SeqLock&) = delete;
    SeqLock& operator=(const SeqLock&) = delete;

    // Any thread. Waits (with Backoff) only for another store in progress.
    void store(const T& value) noexcept {
        const Words words = to_words(value);
        Backoff backoff;
        std::uint64_t s = seq_.load(std::memory_order_relaxed);
        for (;;) {
            // Acquire: if we take over from another writer, its data stores
            // happen-before ours, so each word's latest value is ours, never
            // a mix of the two writers.
            if ((s & 1) == 0 &&
                seq_.compare_exchange_weak(s, s + 1, std::memory_order_acquire, std::memory_order_relaxed))
                break;
            backoff.pause();
            s = seq_.load(std::memory_order_relaxed);
        }
        // The odd value must be visible before any new data is. A release
        // fence orders the CAS before the relaxed data stores below: a reader
        // whose acquire fence follows a load that saw one of our data stores
        // will also see the odd counter (or later) in its second load.
        std::atomic_thread_fence(std::memory_order_release);
        LOCKFREE_TEST_HOOK("seqlock_store_mid_write");
        write_words(words);
        // Publishes the data: a reader that acquires s + 2 sees all of it.
        seq_.store(s + 2, std::memory_order_release);
    }

    // One attempt: false if a write overlapped. Never blocks.
    bool try_load(T& out) const noexcept {
        Words words;
        if (!try_read(words)) return false;
        out = from_words(words);
        return true;
    }

    // Retries until a read doesn't overlap a write.
    T load() const noexcept {
        Words words;
        Backoff backoff;
        while (!try_read(words)) backoff.pause();
        return from_words(words);
    }

    // Completed stores so far (the counter is 2 per store). Racy by nature;
    // for tests and stats.
    std::uint64_t version() const noexcept { return seq_.load(std::memory_order_acquire) / 2; }

private:
    bool try_read(Words& words) const noexcept {
        const std::uint64_t s1 = seq_.load(std::memory_order_acquire);
        if (s1 & 1) return false;
        for (std::size_t i = 0; i < word_count; ++i) words[i] = data_[i].load(std::memory_order_relaxed);
        LOCKFREE_TEST_HOOK("seqlock_load_after_copy");
        // Keeps the data loads above from moving below the re-read of seq_.
        // A plain acquire load of seq_ wouldn't do it: acquire stops later
        // accesses moving up, not earlier ones moving down.
        std::atomic_thread_fence(std::memory_order_acquire);
        return seq_.load(std::memory_order_relaxed) == s1;
    }

    static Words to_words(const T& value) noexcept {
        Words words{}; // zero padding bytes and the tail of the last word
        std::memcpy(words.data(), &value, sizeof(T));
        return words;
    }

    static T from_words(const Words& words) noexcept {
        std::array<std::byte, sizeof(T)> bytes;
        std::memcpy(bytes.data(), words.data(), sizeof(T));
        return std::bit_cast<T>(bytes);
    }

    void write_words(const Words& words) noexcept {
        for (std::size_t i = 0; i < word_count; ++i) data_[i].store(words[i], std::memory_order_relaxed);
    }

    // Counter and data on one line on purpose: a reader needs both, and
    // only the (rare) writer dirties it.
    alignas(detail::cache_line_size) detail::atomic<std::uint64_t> seq_{0};
    detail::atomic<std::uint64_t> data_[word_count];
};

} // namespace lockfree
