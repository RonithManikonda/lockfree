#pragma once
#include <atomic>
#include <cstddef>
#include <memory>
#include <new>
#include <optional>
#include <type_traits>
#include <utility>

#include "lockfree/atomic_backoff.hpp"
#include "lockfree/detail/cache_line.hpp"
#include "lockfree/detail/tagged_ptr.hpp"
#include "lockfree/detail/test_hook.hpp"

namespace lockfree {

// Unbounded lock-free stack (Treiber) that is safe from ABA.
//
// The head is a tagged pointer: every successful CAS bumps the tag, so a pop
// that read head = {X, 7} can't succeed against {X, 9}, even though X is back
// on top. That's the fix for the interleaving naive::TreiberStack falls for.
//
// Tags only work because nodes are never freed while the stack exists. A pop
// that lost the race may still read `next` from a node someone else already
// popped; that read has to land on valid memory, even if what it reads is
// garbage that the failing CAS then throws away. So popped nodes go to a free
// list (another tagged stack) and pushes take from it, and memory only goes
// back to the OS in the destructor. Structures that do need to free nodes
// while running need hazard pointers or epochs instead.
template <typename T>
class AtomicStack {
    static_assert(std::is_nothrow_destructible_v<T>,
                  "pop and the destructor can't recover from a throwing ~T");
    static_assert(std::is_nothrow_move_constructible_v<T>,
                  "a popped node has already left the stack, so moving out must not throw");

    struct Node {
        // Atomic because a pop that is about to lose its CAS can read next
        // while the node's new owner writes it.
        std::atomic<Node*> next{nullptr};
        alignas(T) std::byte storage[sizeof(T)];

        T* value() noexcept { return std::launder(reinterpret_cast<T*>(storage)); }
    };

    using Head = detail::TaggedAtomic<Node>;

public:
    // Which representation this platform got; see tagged_ptr.hpp.
    static constexpr bool uses_wide_tag = detail::wide_cas_is_lock_free;

    AtomicStack() = default;
    AtomicStack(const AtomicStack&) = delete;
    AtomicStack& operator=(const AtomicStack&) = delete;

    // Destroys what's left and frees every node. Must not race with anything.
    ~AtomicStack() {
        while (Node* n = pop_node(head_, kItemsHook)) {
            n->value()->~T();
            delete n;
        }
        while (Node* n = pop_node(free_, kFreeListHook)) delete n;
    }

    // Any thread. If T's constructor throws, the stack is unchanged (the node
    // goes back to the free list).
    template <typename... Args>
    void emplace(Args&&... args) {
        Node* n = pop_node(free_, kFreeListHook);
        if (n == nullptr) n = new Node;
        try {
            ::new (n->storage) T(std::forward<Args>(args)...);
        } catch (...) {
            push_node(free_, n);
            throw;
        }
        push_node(head_, n);
    }

    void push(const T& item) { emplace(item); }

    void push(T&& item) { emplace(std::move(item)); }

    std::optional<T> try_pop() noexcept {
        Node* n = pop_node(head_, kItemsHook);
        if (n == nullptr) return std::nullopt;
        std::optional<T> out(std::move(*n->value()));
        release(n);
        return out;
    }

    bool try_pop(T& out) noexcept
        requires std::is_nothrow_move_assignable_v<T>
    {
        Node* n = pop_node(head_, kItemsHook);
        if (n == nullptr) return false;
        out = std::move(*n->value());
        release(n);
        return true;
    }

    // A snapshot; may be stale by the time the caller looks at it.
    bool empty_approx() const noexcept {
        return head_.load(std::memory_order_relaxed).ptr == nullptr;
    }

private:
    // Names of the test hook points in pop_node, one per list.
    static constexpr const char* kItemsHook = "atomic_stack_pop_before_cas";
    static constexpr const char* kFreeListHook = "atomic_stack_free_list_pop_before_cas";

    void release(Node* n) noexcept {
        n->value()->~T();
        push_node(free_, n);
    }

    static void push_node(Head& head, Node* n) noexcept {
        auto expected = head.load(std::memory_order_relaxed);
        Backoff backoff;
        for (;;) {
            n->next.store(expected.ptr, std::memory_order_relaxed);
            // release publishes the node: its value for the item stack, and
            // the destroyed-and-done state for the free list.
            if (head.compare_exchange_weak(expected, n, std::memory_order_release,
                                           std::memory_order_relaxed)) {
                return;
            }
            // Everyone is after the same line. Retrying at once mostly just
            // steals it back from whoever is about to succeed.
            backoff.pause();
        }
    }

    static Node* pop_node(Head& head, [[maybe_unused]] const char* hook) noexcept {
        // acquire pairs with push_node's release, so the node's value and next
        // are visible before we use them.
        auto expected = head.load(std::memory_order_acquire);
        Backoff backoff;
        for (;;) {
            if (expected.ptr == nullptr) return nullptr;
            // If another thread pops this node first, next may be stale or
            // belong to some later use of the node. The tag makes the CAS
            // below fail in that case, so whatever we read is discarded.
            Node* next = expected.ptr->next.load(std::memory_order_relaxed);

            LOCKFREE_TEST_HOOK(hook);

            if (head.compare_exchange_weak(expected, next, std::memory_order_acquire,
                                           std::memory_order_acquire)) {
                return expected.ptr;
            }
            backoff.pause();
        }
    }

    alignas(detail::cache_line_size) Head head_;
    alignas(detail::cache_line_size) Head free_;
};

} // namespace lockfree
