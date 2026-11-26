#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "lockfree/detail/atomic_ref.hpp"
#include "lockfree/detail/cache_line.hpp"

// Epoch-based reclamation (Keir Fraser, 2004): the other answer to "when is it
// safe to free a node that other threads might still be reading?".
//
// Hazard pointers track each pointer a reader holds. EBR tracks something
// coarser: *when* each thread started reading. Readers pin themselves for the
// duration of an operation (EpochGuard), and pinning records the current
// global epoch. A thread that unlinks a node retires it, tagged with the
// epoch at the time. The global epoch only moves from e to e + 1 once every
// pinned thread has been seen in e. So once the global epoch is two past a
// node's tag, every thread that was pinned when the node was unlinked has
// unpinned since, and nobody can still hold it.
//
//   {
//       EpochGuard guard;                   // pin
//       Node* h = head_.load(acquire);      // safe to use until the guard ends
//       ... CAS head_ to h->next ...
//       Epoch::retire(h);                   // after unlinking it
//   }                                       // unpin
//
// Compared with hazard pointers: reads are cheaper (one store and one fence
// per operation, not per pointer), and there's no protect-and-validate loop.
// The price is that one thread that stays pinned (descheduled, stuck in a
// long operation) stops the epoch for everyone, and retired memory piles up
// without bound until it moves on.
//
// Each thread keeps three limbo bags, one per epoch modulo 3: a bag tagged e
// can be emptied once the global epoch reaches e + 2, which is also exactly
// when its slot is about to be needed again.

namespace lockfree {

namespace detail::epoch {

struct alignas(cache_line_size) Record {
    // 0 when the owner isn't pinned, otherwise (epoch << 1) | 1. One word, so
    // a collector reads "pinned, and where" in a single load.
    detail::atomic<std::uint64_t> state{0};
    // Whether some thread owns this record.
    detail::atomic<bool> in_use{false};
    // Written once before the record is published, never again.
    Record* next = nullptr;
};

inline constexpr std::uint64_t pinned_state(std::uint64_t epoch) noexcept { return (epoch << 1) | 1; }

struct Retired {
    void* ptr;
    void (*reclaim)(void*);
};

struct Bag {
    std::uint64_t epoch = 0;
    std::vector<Retired> items;
};

// Bags left by threads that exited before their contents could be freed.
struct OrphanBag {
    Bag bag;
    OrphanBag* next = nullptr;
};

inline void free_all(std::vector<Retired>& items) {
    // Reclaim functions may retire more; those go to the caller's bags, not
    // into the vector being walked.
    std::vector<Retired> doomed = std::exchange(items, {});
    for (const Retired& r : doomed) r.reclaim(r.ptr);
}

class Domain {
public:
    Domain() = default;
    Domain(const Domain&) = delete;
    Domain& operator=(const Domain&) = delete;

    // Static destruction: every thread_local is gone, nobody is pinned.
    ~Domain() {
        for (OrphanBag* o = orphans_.exchange(nullptr, std::memory_order_acquire); o != nullptr;) {
            free_all(o->bag.items);
            delete std::exchange(o, o->next);
        }
        for (Record* r = head_.load(std::memory_order_acquire); r != nullptr;) {
            delete std::exchange(r, r->next);
        }
    }

    Record* acquire() {
        for (Record* r = head_.load(std::memory_order_acquire); r != nullptr; r = r->next) {
            bool expected = false;
            if (!r->in_use.load(std::memory_order_relaxed) &&
                r->in_use.compare_exchange_strong(expected, true, std::memory_order_acquire,
                                                  std::memory_order_relaxed)) {
                return r;
            }
        }
        auto* r = new Record;
        r->in_use.store(true, std::memory_order_relaxed);
        Record* old = head_.load(std::memory_order_relaxed);
        do {
            r->next = old;
        } while (!head_.compare_exchange_weak(old, r, std::memory_order_release,
                                              std::memory_order_relaxed));
        return r;
    }

    void release(Record* r) noexcept {
        r->state.store(0, std::memory_order_release);
        r->in_use.store(false, std::memory_order_release);
    }

    std::uint64_t epoch() const noexcept { return global_.load(std::memory_order_acquire); }

    void pin(Record* r) noexcept {
        // The epoch read here may already be stale; that only makes this
        // thread hold the epoch back a little longer, never less.
        const std::uint64_t e = global_.load(std::memory_order_relaxed);
        r->state.store(pinned_state(e), std::memory_order_relaxed);
        // The pin has to be visible to collectors before this thread loads any
        // shared pointer, which is a store-load ordering: only a seq_cst fence
        // gives it. It pairs with the fence in try_advance(): either the
        // collector sees us pinned, or our later loads see every unlink that
        // came before its fence.
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    void unpin(Record* r) noexcept {
        // Release: everything read while pinned happens-before a collector
        // that sees the 0 (acquire in try_advance) and later frees.
        r->state.store(0, std::memory_order_release);
    }

    // Moves the global epoch forward by one if every pinned thread has caught
    // up with it. Returns whether the epoch moved (by us or by someone else).
    bool try_advance() noexcept {
        std::uint64_t g = global_.load(std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        for (Record* r = head_.load(std::memory_order_acquire); r != nullptr; r = r->next) {
            const std::uint64_t s = r->state.load(std::memory_order_acquire);
            if ((s & 1) != 0 && (s >> 1) != g) return false;
        }
        // Release so a thread that sees the new epoch (acquire in epoch())
        // also sees the unpins we just read: that chain is what makes a free
        // after the epoch moved race-free.
        // A failed CAS means somebody else advanced it first, which is just
        // as good.
        global_.compare_exchange_strong(g, g + 1, std::memory_order_release, std::memory_order_relaxed);
        return true;
    }

    // The epoch a retirement made now must be tagged with. Called after the
    // object was unlinked; the fence orders that unlink before the read, so
    // the tag is at least the epoch of any thread that could still see it.
    std::uint64_t retire_epoch() noexcept {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        return global_.load(std::memory_order_relaxed);
    }

    void orphan(Bag bag) {
        auto* o = new OrphanBag{std::move(bag)};
        OrphanBag* old = orphans_.load(std::memory_order_relaxed);
        do {
            o->next = old;
        } while (!orphans_.compare_exchange_weak(old, o, std::memory_order_release,
                                                 std::memory_order_relaxed));
    }

    // Frees orphaned bags that have expired by `now`, puts the rest back.
    void collect_orphans(std::uint64_t now) {
        OrphanBag* o = orphans_.exchange(nullptr, std::memory_order_acquire);
        while (o != nullptr) {
            OrphanBag* next = std::exchange(o->next, nullptr);
            if (o->bag.epoch + 2 <= now) {
                free_all(o->bag.items);
                delete o;
            } else {
                orphan(std::move(o->bag));
                delete o;
            }
            o = next;
        }
    }

private:
    detail::atomic<std::uint64_t> global_{0};
    detail::atomic<Record*> head_{nullptr};
    detail::atomic<OrphanBag*> orphans_{nullptr};
};

inline Domain& domain() {
    static Domain d;
    return d;
}

struct ThreadState {
    Record* record = nullptr;
    std::size_t depth = 0;      // nested EpochGuards
    Bag bags[3];                // indexed by epoch % 3
    std::size_t since_collect = 0;
    bool collecting = false;

    ThreadState() { (void)domain(); } // destroyed after every ThreadState

    ~ThreadState() {
        for (Bag& b : bags) {
            if (!b.items.empty()) domain().orphan(std::move(b));
        }
        if (record != nullptr) domain().release(record);
    }

    std::size_t retired() const noexcept {
        return bags[0].items.size() + bags[1].items.size() + bags[2].items.size();
    }
};

inline ThreadState& thread_state() {
    thread_local ThreadState s;
    return s;
}

// Retires between automatic collection attempts.
inline constexpr std::size_t kCollectEvery = 64;

} // namespace detail::epoch

// Pins the calling thread for its lifetime. Pointers loaded from a structure
// that retires through Epoch stay valid until the guard is destroyed. Guards
// nest; only the outermost one pins and unpins.
class EpochGuard {
public:
    EpochGuard() {
        auto& ts = detail::epoch::thread_state();
        if (ts.depth++ > 0) return;
        if (ts.record == nullptr) ts.record = detail::epoch::domain().acquire();
        detail::epoch::domain().pin(ts.record);
    }

    ~EpochGuard() {
        auto& ts = detail::epoch::thread_state();
        if (--ts.depth == 0) detail::epoch::domain().unpin(ts.record);
    }

    EpochGuard(const EpochGuard&) = delete;
    EpochGuard& operator=(const EpochGuard&) = delete;
};

class Epoch {
public:
    // Defers `p`'s deletion until every thread that could have seen it has
    // unpinned. Call only after p is unreachable from the structure. Same
    // deleter rules as HazardPointer::retire.
    template <typename T, typename Deleter = std::default_delete<T>>
        requires std::is_empty_v<Deleter> && std::is_default_constructible_v<Deleter>
    static void retire(T* p, Deleter = {}) {
        retire_erased(const_cast<void*>(static_cast<const void*>(p)),
                      [](void* q) { Deleter{}(static_cast<T*>(q)); });
    }

    static void retire(void* p, void (*reclaim)(void*)) { retire_erased(p, reclaim); }

    // Tries to advance the epoch, then frees whatever the calling thread (or
    // an exited thread) retired at least two epochs ago. Calling it while
    // pinned is fine, but the caller's own pin can hold the epoch back.
    static void collect() {
        auto& ts = detail::epoch::thread_state();
        if (ts.collecting) return;
        ts.collecting = true;
        struct Done {
            bool& flag;
            ~Done() { flag = false; }
        } done{ts.collecting};

        auto& d = detail::epoch::domain();
        d.try_advance();
        const std::uint64_t now = d.epoch();
        for (auto& bag : ts.bags) {
            if (!bag.items.empty() && bag.epoch + 2 <= now) detail::epoch::free_all(bag.items);
        }
        d.collect_orphans(now);
        ts.since_collect = 0;
    }

    // Objects the calling thread retired that haven't been freed yet.
    static std::size_t retired_count() noexcept { return detail::epoch::thread_state().retired(); }

    static std::uint64_t current() noexcept { return detail::epoch::domain().epoch(); }

    // One advance attempt; false if some pinned thread is behind.
    static bool try_advance() noexcept { return detail::epoch::domain().try_advance(); }

private:
    static void retire_erased(void* p, void (*reclaim)(void*)) {
        auto& ts = detail::epoch::thread_state();
        {
            // Tag while pinned: then the global epoch can't run ahead of
            // this thread by more than one while the tag is being taken.
            EpochGuard guard;
            const std::uint64_t e = detail::epoch::domain().retire_epoch();
            auto& bag = ts.bags[e % 3];
            if (bag.epoch != e && !bag.items.empty()) {
                // This thread's tags never go backwards, so an older tag in
                // this slot is at most e - 3, which expired at e - 1.
                detail::epoch::free_all(bag.items);
            }
            bag.epoch = e;
            bag.items.push_back({p, reclaim});
        }
        // Outside the guard, so our own pin doesn't hold the epoch back.
        if (++ts.since_collect >= detail::epoch::kCollectEvery) collect();
    }
};

} // namespace lockfree
