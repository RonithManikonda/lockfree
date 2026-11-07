#pragma once
#include <atomic>

#include "lockfree/detail/cache_line.hpp"

namespace lockfree {

// Embed (or derive from) this to make an object queueable. The queue never
// allocates: a node is owned by the queue from push until pop returns it, and
// can only be in one queue at a time.
struct MpscNode {
    std::atomic<MpscNode*> next{nullptr};
};

// Unbounded intrusive multi-producer / single-consumer queue (Dmitry Vyukov's
// design). A singly linked list where nodes go in at tail_ and come out at
// head_:
//
//   head_ -> n1 -> n2 -> n3 <- tail_
//
// A push is two steps: swing tail_ to the new node with an exchange, then link
// the old tail to it. The exchange is what orders the producers and it can't
// fail, so there's no CAS loop; each producer does a fixed number of steps.
//
// The catch is the gap between those two steps. A producer that has exchanged
// but not linked yet has cut the list in two:
//
//   head_ -> n1 -> n2    n3 -> n4 <- tail_      (n2->next not set yet)
//
// The consumer can't reach n3 or n4 until that producer runs again, so pop
// reports empty even though the queue isn't. Nothing is lost, the items just
// stay hidden. This means pop returning nullptr is only a definite "empty" if
// no push is in progress, and it's why the queue isn't strictly lock-free: one
// preempted producer can hold up the consumer. docs/mpsc_queue.md has more.
//
// stub_ exists so the list always has a node for head_ to sit on. A node can
// only be handed back once it has a successor, so when the list is down to
// its last node, pop pushes stub_ behind it to free it.
class MpscIntrusiveQueue {
public:
    MpscIntrusiveQueue() noexcept = default;
    MpscIntrusiveQueue(const MpscIntrusiveQueue&) = delete;
    MpscIntrusiveQueue& operator=(const MpscIntrusiveQueue&) = delete;

    // Any thread. The node must not be in a queue.
    void push(MpscNode* node) noexcept { link(swing_tail(node), node); }

    // Consumer thread only. Returns the oldest reachable node, or nullptr if
    // there is none. See the class comment for what nullptr does and doesn't
    // mean. The caller owns the returned node, and no producer touches it
    // again: a node only comes back after its successor was linked, and that
    // link is the last thing a producer does to it.
    MpscNode* pop() noexcept {
        MpscNode* head = head_;
        // acquire pairs with the release in link(): everything the producer
        // wrote to the node before pushing it is visible to us.
        MpscNode* next = head->next.load(std::memory_order_acquire);
        if (head == &stub_) {
            if (next == nullptr) return nullptr;
            // Step off the stub; it's not an element.
            head_ = next;
            head = next;
            next = next->next.load(std::memory_order_acquire);
        }
        if (next != nullptr) {
            head_ = next;
            return head;
        }

        // head has no successor yet. Either it's the last node, or a producer
        // is between swing_tail and link and a successor is on its way. We
        // only compare this pointer and never dereference it, so relaxed is
        // enough. It can't read older than head: head's own swing_tail
        // happened before the link we followed to get here.
        if (head != tail_.load(std::memory_order_relaxed)) return nullptr;

        // head was the last node. Push the stub behind it so head can leave.
        push(&stub_);
        next = head->next.load(std::memory_order_acquire);
        if (next != nullptr) {
            head_ = next;
            return head;
        }
        // Another producer got in after the tail_ load above and hasn't linked
        // yet. The stub stays in the list behind it and gets skipped later.
        return nullptr;
    }

private:
    friend struct MpscIntrusiveQueueTestAccess;

    // Returns the previous tail; the caller must link() it to node.
    MpscNode* swing_tail(MpscNode* node) noexcept {
        // Relaxed is fine: nobody can see the node until the exchange, and the
        // exchange's release covers this store.
        node->next.store(nullptr, std::memory_order_relaxed);
        // release publishes that nullptr to the next producer, which will link
        // through this node. acquire is the other half: our store to
        // prev->next must come after the store of nullptr by prev's producer,
        // or that nullptr could overwrite our link.
        return tail_.exchange(node, std::memory_order_acq_rel);
    }

    static void link(MpscNode* prev, MpscNode* node) noexcept {
        // release publishes the node's contents to the consumer. After this
        // store the consumer may pop prev, so prev can't be touched again.
        prev->next.store(node, std::memory_order_release);
    }

    // Producers hammer tail_; only the consumer touches head_, so it's a plain
    // pointer. stub_ gets its own line because producers write to its next.
    alignas(detail::cache_line_size) std::atomic<MpscNode*> tail_{&stub_};
    alignas(detail::cache_line_size) MpscNode* head_{&stub_};
    alignas(detail::cache_line_size) MpscNode stub_;
};

} // namespace lockfree
