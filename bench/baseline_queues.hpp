#pragma once

// Lock-based queues the lock-free structures are measured against. They have
// the same try_push / try_pop interface so one benchmark body can drive
// either kind.

#include <cstddef>
#include <deque>
#include <mutex>
#include <utility>

namespace lockfree::bench {

// std::mutex + std::deque with a capacity limit, so "full" behaves like it
// does for the bounded lock-free queues and the producer can't just run away
// and turn the benchmark into a deque-growth test.
template <typename T>
class MutexQueue {
public:
    explicit MutexQueue(std::size_t capacity) : capacity_(capacity) {}

    bool try_push(T item) {
        std::lock_guard lock(mutex_);
        if (items_.size() == capacity_) return false;
        items_.push_back(std::move(item));
        return true;
    }

    bool try_pop(T& out) {
        std::lock_guard lock(mutex_);
        if (items_.empty()) return false;
        out = std::move(items_.front());
        items_.pop_front();
        return true;
    }

    std::size_t capacity() const noexcept { return capacity_; }

private:
    std::mutex mutex_;
    std::deque<T> items_;
    const std::size_t capacity_;
};

} // namespace lockfree::bench
