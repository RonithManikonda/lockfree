#pragma once
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <span>
#include <vector>

#include "linearizability/history.hpp"

// Sequential specifications: what a single-threaded queue or stack would
// return. The checker keeps one spec state per search branch, so specs are
// small value types with three things:
//
//   bool apply(const Operation&)  - if op's recorded result is what this
//                                   state would produce, perform it and return
//                                   true; otherwise return false and leave the
//                                   state untouched, so the caller can try a
//                                   different op without copying first.
//   operator==, hash()            - the checker memoizes (linearized ops,
//                                   state) pairs it already ruled out.
namespace lockfree::lin {

namespace detail {

// FNV-1a over the elements. Good enough for a memo table key; equality is
// still checked on collisions.
template <typename Range>
std::size_t hash_values(const Range& values) noexcept {
    std::uint64_t h = 14695981039346656037ull;
    for (std::int64_t v : values) {
        auto bits = static_cast<std::uint64_t>(v);
        for (int i = 0; i < 8; ++i) {
            h ^= (bits >> (8 * i)) & 0xFF;
            h *= 1099511628211ull;
        }
    }
    return static_cast<std::size_t>(h);
}

} // namespace detail

class FifoQueueSpec {
public:
    bool apply(const Operation& op) {
        if (op.kind == OpKind::push) {
            items_.push_back(op.value);
            return true;
        }
        if (items_.empty()) return !op.result.has_value();
        if (op.result != items_.front()) return false;
        items_.pop_front();
        return true;
    }

    std::size_t size() const noexcept { return items_.size(); }
    std::size_t hash() const noexcept { return detail::hash_values(items_); }
    friend bool operator==(const FifoQueueSpec&, const FifoQueueSpec&) = default;

private:
    std::deque<std::int64_t> items_;
};

class LifoStackSpec {
public:
    bool apply(const Operation& op) {
        if (op.kind == OpKind::push) {
            items_.push_back(op.value);
            return true;
        }
        if (items_.empty()) return !op.result.has_value();
        if (op.result != items_.back()) return false;
        items_.pop_back();
        return true;
    }

    std::size_t size() const noexcept { return items_.size(); }
    std::size_t hash() const noexcept { return detail::hash_values(items_); }
    friend bool operator==(const LifoStackSpec&, const LifoStackSpec&) = default;

private:
    std::vector<std::int64_t> items_;
};

// Runs ops in the given order against a fresh Spec. Returns the index of the
// first op whose result the spec disagrees with, or nullopt if the whole
// sequence is legal. The checker's job is to find an order that makes this
// return nullopt; tests use it directly to pin down the specs.
template <typename Spec>
std::optional<std::size_t> first_illegal(std::span<const Operation> ops) {
    Spec spec;
    for (std::size_t i = 0; i < ops.size(); ++i) {
        if (!spec.apply(ops[i])) return i;
    }
    return std::nullopt;
}

} // namespace lockfree::lin
