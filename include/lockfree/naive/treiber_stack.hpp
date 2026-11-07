#pragma once
#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <type_traits>
#include <vector>

#include "lockfree/detail/test_hook.hpp"

// A Treiber stack with plain pointers, written the way it looks in a textbook
// and kept deliberately broken. It exists to reproduce the ABA problem in tests
// (and later to be found by the scheduler); use lockfree::AtomicStack for real.
//
// Nodes are recycled through a NodePool instead of being freed, which is what
// makes ABA reachable: a popped node's address comes back for a later push, so
// "head is still X" no longer means "nothing happened". Several stacks can
// share one pool, the way every structure in a program shares malloc.
namespace lockfree::naive {

template <typename T>
struct TreiberNode {
    // Atomic only so a racing reader of a recycled node isn't a C++ data race;
    // the ordering that matters is on the stack's head.
    std::atomic<TreiberNode*> next{nullptr};
    T value{};
};

// A free list of nodes (itself a naive Treiber stack). Every node it hands out
// stays allocated until the pool is destroyed, so a stale pointer always points
// at some node, just maybe not the one the reader thinks.
template <typename T>
class NodePool {
public:
    using Node = TreiberNode<T>;

    NodePool() = default;
    NodePool(const NodePool&) = delete;
    NodePool& operator=(const NodePool&) = delete;

    Node* acquire() {
        Node* node = free_.load(std::memory_order_acquire);
        while (node != nullptr &&
               !free_.compare_exchange_weak(node, node->next.load(std::memory_order_relaxed),
                                            std::memory_order_acquire)) {
        }
        if (node != nullptr) return node;

        // Fresh allocations are rare (the pool only grows to the peak number of
        // live nodes), so a mutex here doesn't matter. Keeping every node in
        // all_ means teardown never has to walk a list ABA may have corrupted.
        std::lock_guard lock(mutex_);
        return all_.emplace_back(std::make_unique<Node>()).get();
    }

    void release(Node* node) noexcept {
        Node* head = free_.load(std::memory_order_relaxed);
        do {
            node->next.store(head, std::memory_order_relaxed);
        } while (!free_.compare_exchange_weak(head, node, std::memory_order_release,
                                              std::memory_order_relaxed));
    }

    // Nodes ever allocated; reuse keeps this at the peak live count.
    std::size_t allocated() const {
        std::lock_guard lock(mutex_);
        return all_.size();
    }

private:
    std::atomic<Node*> free_{nullptr};
    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<Node>> all_;
};

// Values are copied into pooled nodes and never destroyed individually, so T
// has to be trivial. That keeps the broken states the ABA tests create from
// turning into double destruction on teardown.
template <typename T>
class TreiberStack {
    static_assert(std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T>,
                  "naive::TreiberStack only holds trivial values");

public:
    using Node = TreiberNode<T>;

    TreiberStack() : own_pool_(std::make_unique<NodePool<T>>()), pool_(own_pool_.get()) {}
    // The pool must outlive the stack.
    explicit TreiberStack(NodePool<T>& pool) : pool_(&pool) {}

    TreiberStack(const TreiberStack&) = delete;
    TreiberStack& operator=(const TreiberStack&) = delete;

    void push(T value) {
        Node* node = pool_->acquire();
        node->value = value;
        Node* head = head_.load(std::memory_order_relaxed);
        do {
            node->next.store(head, std::memory_order_relaxed);
        } while (!head_.compare_exchange_weak(head, node, std::memory_order_release,
                                              std::memory_order_relaxed));
    }

    std::optional<T> try_pop() {
        Node* head = head_.load(std::memory_order_acquire);
        for (;;) {
            if (head == nullptr) return std::nullopt;
            Node* next = head->next.load(std::memory_order_relaxed);

            // The bug lives here. If head is popped, recycled and pushed again
            // before the CAS, the CAS still sees the same pointer and succeeds,
            // installing a `next` that may no longer belong to this stack.
            LOCKFREE_TEST_HOOK("treiber_pop_before_cas");

            if (head_.compare_exchange_weak(head, next, std::memory_order_acquire,
                                            std::memory_order_acquire)) {
                break;
            }
        }
        T value = head->value;
        pool_->release(head);
        return value;
    }

    bool empty_approx() const noexcept { return head_.load(std::memory_order_relaxed) == nullptr; }

private:
    std::unique_ptr<NodePool<T>> own_pool_;
    NodePool<T>* pool_;
    std::atomic<Node*> head_{nullptr};
};

} // namespace lockfree::naive
