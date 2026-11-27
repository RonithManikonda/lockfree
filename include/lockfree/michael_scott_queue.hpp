#pragma once
#include <atomic>
#include <cstddef>
#include <memory>
#include <new>
#include <optional>
#include <type_traits>
#include <utility>

#include "lockfree/atomic_backoff.hpp"
#include "lockfree/detail/atomic_ref.hpp"
#include "lockfree/detail/cache_line.hpp"
#include "lockfree/detail/test_hook.hpp"
#include "lockfree/epoch.hpp"
#include "lockfree/hazard_pointer.hpp"

namespace lockfree {

// Reclamation policies for MichaelScottQueue. A policy provides
//
//   class Guard {                       // one per queue operation
//       T* protect(std::size_t slot, const detail::atomic<T*>& src);
//       // slot is 0 or 1; the returned pointer stays dereferenceable
//       // until the guard dies or that slot is protected again
//   };
//   static void retire(T* p);          // p is unlinked; free it when safe
//
// so the queue's code is the same whichever scheme frees its nodes.

// Hazard pointers: two slots per operation (head and head->next on dequeue,
// tail on enqueue).
//
// Claiming a HazardPointer walks the global record list, which is too slow to
// do twice per operation. So each thread keeps one pair of slots for all
// queues and reuses it. If a guard is created while the pair is in use (an
// element's move constructor that touches another queue, say), that inner
// guard claims a fresh pair of its own.
struct HazardPointerReclaim {
    class Guard {
    public:
        Guard() {
            Cache& c = cache();
            if (!c.busy) {
                c.busy = true;
                slots_ = &c.slots;
            } else {
                owned_ = std::make_unique<Slots>();
                slots_ = owned_.get();
            }
        }

        ~Guard() {
            slots_->hp[0].reset();
            slots_->hp[1].reset();
            if (!owned_) cache().busy = false;
        }

        Guard(const Guard&) = delete;
        Guard& operator=(const Guard&) = delete;

        template <typename T>
        T* protect(std::size_t slot, const detail::atomic<T*>& src) noexcept {
            return slots_->hp[slot].protect(src);
        }

    private:
        struct Slots {
            HazardPointer hp[2];
        };
        struct Cache {
            Slots slots;
            bool busy = false;
        };
        static Cache& cache() {
            thread_local Cache c;
            return c;
        }

        Slots* slots_ = nullptr;
        std::unique_ptr<Slots> owned_;
    };

    template <typename T>
    static void retire(T* p) {
        HazardPointer::retire(p);
    }
};

// Epochs: the guard pins the thread for the whole operation, so protect() is
// just an acquire load; nothing loaded while pinned can be freed until the
// guard goes away. Cheaper per pointer, but a thread stalled inside an
// operation holds back every node retired after it (docs/reclamation.md).
//
// retire() runs inside the operation's own guard. Epoch::retire nests its pin
// under ours, and the occasional collect() it triggers is still correct while
// pinned; our pin only delays what that collect can free by one operation.
struct EpochReclaim {
    class Guard {
    public:
        Guard() = default;
        Guard(const Guard&) = delete;
        Guard& operator=(const Guard&) = delete;

        template <typename T>
        T* protect(std::size_t /*slot*/, const detail::atomic<T*>& src) noexcept {
            // Acquire, for the same reason protect() above ends on one: the
            // caller reads the node's next and value through this pointer.
            return src.load(std::memory_order_acquire);
        }

    private:
        EpochGuard pin_;
    };

    template <typename T>
    static void retire(T* p) {
        Epoch::retire(p);
    }
};

// Unbounded lock-free multi-producer / multi-consumer FIFO queue (Michael and
// Scott, 1996). A singly linked list with a dummy node at the front:
//
//   head_ -> dummy -> n1 -> n2 -> n3 <- tail_
//
// The front element lives in head_->next; head_ itself is always a node whose
// value was already taken (or the initial dummy). Dequeue swings head_ one
// node forward and takes the value out of the new head, which becomes the
// next dummy. That way head_ and tail_ never point at "no node", and an empty
// queue is just head_ == tail_ with next == nullptr.
//
// Enqueue is two CASes: link the new node after the last one
// (tail->next: nullptr -> node), then swing tail_ to it. Between the two,
// tail_ lags one node behind the real end. Any thread that notices
// (tail->next != nullptr) swings tail_ forward itself before going on, so a
// producer preempted after the first CAS can't block anyone. That "helping" is
// what makes the queue lock-free rather than just obstruction-free.
//
// Dequeued nodes are freed while other threads may still be reading them, so
// the queue goes through a Reclaimer (see the policies above): every node it
// dereferences is protected first, and old heads are retired, not deleted.
template <typename T, typename Reclaimer = HazardPointerReclaim>
class MichaelScottQueue {
    static_assert(std::is_nothrow_destructible_v<T>,
                  "pop and the destructor can't recover from a throwing ~T");
    static_assert(std::is_nothrow_move_constructible_v<T>,
                  "the value is moved out after the CAS, when the node is already ours");

public:
    MichaelScottQueue() {
        Node* dummy = new Node;
        head_.store(dummy, std::memory_order_relaxed);
        tail_.store(dummy, std::memory_order_relaxed);
    }

    MichaelScottQueue(const MichaelScottQueue&) = delete;
    MichaelScottQueue& operator=(const MichaelScottQueue&) = delete;

    // Must not race with any other operation. Destroys the values that were
    // never popped; nodes retired earlier belong to the reclaimer.
    ~MichaelScottQueue() {
        Node* n = head_.load(std::memory_order_relaxed);
        Node* next = n->next.load(std::memory_order_relaxed);
        delete n; // the dummy: no value
        for (n = next; n != nullptr; n = next) {
            next = n->next.load(std::memory_order_relaxed);
            n->destroy_value();
            delete n;
        }
    }

    // Any thread. Allocates one node. If that or T's constructor throws, the
    // queue is unchanged.
    template <typename... Args>
    void emplace(Args&&... args) {
        Node* node = new Node;
        try {
            node->construct_value(std::forward<Args>(args)...);
        } catch (...) {
            delete node;
            throw;
        }
        try {
            enqueue(node);
        } catch (...) {
            // Only the guard can throw (a thread's first hazard slots are
            // allocated), and that happens before the node is linked.
            node->destroy_value();
            delete node;
            throw;
        }
    }

    void push(const T& item) { emplace(item); }

    void push(T&& item) { emplace(std::move(item)); }

    // Any thread. False only when the queue was empty at some point during the
    // call (it's linearizable, unlike MpscQueue's pop).
    //
    // Not noexcept: a thread's first operation may allocate its hazard slots
    // (bad_alloc). After that nothing in push or pop allocates except the node.
    bool try_pop(T& out)
        requires std::is_nothrow_move_assignable_v<T>
    {
        typename Reclaimer::Guard guard;
        Node* n = dequeue(guard);
        if (n == nullptr) return false;
        out = std::move(*n->value());
        n->destroy_value();
        return true;
    }

    std::optional<T> try_pop() {
        typename Reclaimer::Guard guard;
        Node* n = dequeue(guard);
        if (n == nullptr) return std::nullopt;
        std::optional<T> out(std::move(*n->value()));
        n->destroy_value();
        return out;
    }

    // Spins with Backoff until an item arrives; waits forever if none does.
    T pop() {
        Backoff backoff;
        for (;;) {
            if (std::optional<T> v = try_pop()) return std::move(*v);
            backoff.pause();
        }
    }

private:
    friend struct MichaelScottQueueTestAccess;

    struct Node {
        detail::atomic<Node*> next{nullptr};
        // Raw storage: the dummy carries no value, and a value is destroyed
        // when it's taken, while the node lives on as the next dummy.
        alignas(T) std::byte storage[sizeof(T)];

        template <typename... Args>
        void construct_value(Args&&... args) {
            ::new (static_cast<void*>(storage)) T(std::forward<Args>(args)...);
        }
        T* value() noexcept { return std::launder(reinterpret_cast<T*>(storage)); }
        void destroy_value() noexcept { value()->~T(); }
    };

    void enqueue(Node* node) {
        typename Reclaimer::Guard guard;
        Backoff backoff;
        for (;;) {
            // Protected, so even if tail_ moves on and this node is dequeued
            // and retired, it isn't freed under us, and the CAS below can't
            // hit a recycled address (no ABA).
            Node* tail = guard.protect(0, tail_);
            Node* next = tail->next.load(std::memory_order_acquire);
            if (next != nullptr) {
                // tail_ is lagging: finish the other producer's second step.
                // A failed CAS means someone else already did.
                tail_.compare_exchange_strong(tail, next, std::memory_order_release,
                                              std::memory_order_relaxed);
                continue;
            }
            // The linearization point. Release publishes the node's value to
            // the dequeuer that acquires this link.
            Node* expected = nullptr;
            if (tail->next.compare_exchange_strong(expected, node, std::memory_order_release,
                                                   std::memory_order_relaxed)) {
                LOCKFREE_TEST_HOOK("ms_enqueue_after_link");
                // Best effort: if this fails, someone helped already.
                tail_.compare_exchange_strong(tail, node, std::memory_order_release,
                                              std::memory_order_relaxed);
                return;
            }
            // Lost to another producer; its node is now tail->next.
            backoff.pause();
        }
    }

    // Unlinks the front node and returns the node holding the front value
    // (the new dummy), still protected by `guard`. The caller moves the value
    // out; no other thread touches it, since only the CAS winner owns it.
    Node* dequeue(typename Reclaimer::Guard& guard) noexcept {
        Backoff backoff;
        for (;;) {
            Node* head = guard.protect(0, head_);
            Node* tail = tail_.load(std::memory_order_acquire);
            Node* next = guard.protect(1, head->next);
            // protect(1) only proves head->next == next, which stays true
            // forever once set. What makes next safe is that head was still
            // the head after next's hazard was published: then next was
            // reachable, so it can't have been retired yet.
            if (head != head_.load(std::memory_order_acquire)) continue;
            if (next == nullptr) return nullptr; // empty: head == tail, no successor
            if (head == tail) {
                // Non-empty but tail_ lags at the dummy. Help it forward
                // before moving head_: head_ must never pass tail_, or the
                // retired head would still be reachable through tail_.
                tail_.compare_exchange_strong(tail, next, std::memory_order_release,
                                              std::memory_order_relaxed);
                continue;
            }
            LOCKFREE_TEST_HOOK("ms_dequeue_before_cas");
            // The value isn't read before this CAS (the 1996 paper does, for
            // plain values): a loser would copy it for nothing, and moving a
            // non-trivial T out of a node another thread is about to own
            // isn't allowed. Hazard slot 1 keeps next alive after the CAS.
            if (head_.compare_exchange_strong(head, next, std::memory_order_acquire,
                                              std::memory_order_relaxed)) {
                Reclaimer::retire(head);
                return next;
            }
            backoff.pause();
        }
    }

    // Producers hit tail_, consumers head_: separate lines.
    alignas(detail::cache_line_size) detail::atomic<Node*> head_{nullptr};
    alignas(detail::cache_line_size) detail::atomic<Node*> tail_{nullptr};
};

} // namespace lockfree
