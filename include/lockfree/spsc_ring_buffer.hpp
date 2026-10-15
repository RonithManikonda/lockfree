#pragma once
#include <atomic>
#include <cstddef>

#include "lockfree/detail/cache_line.hpp"

template <typename T, size_t N>
class SpscRingBuffer {
private:
    T buffer_[N];
    alignas(lockfree::detail::cache_line_size) std::atomic<size_t> read_{0};
    alignas(lockfree::detail::cache_line_size) std::atomic<size_t> write_{0};

public:
    bool push(const T& item) {
        size_t w = write_.load(std::memory_order_relaxed);
        size_t next = (w + 1) % N;
        if (next == read_.load(std::memory_order_acquire)) return false;
        buffer_[w] = item;
        write_.store(next, std::memory_order_release);
        return true;
    }

    bool pop(T& out) {
        size_t r = read_.load(std::memory_order_relaxed);
        size_t next = (r + 1) % N;
        if (r == write_.load(std::memory_order_acquire)) return false;
        out = buffer_[r];
        read_.store(next, std::memory_order_release);
        return true;
    }
};
