#pragma once
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "lockfree/detail/atomic_ref.hpp"
#include "lockfree/detail/cache_line.hpp"
#include "lockfree/detail/test_hook.hpp"

// Hazard pointers (Maged Michael, 2004): safe memory reclamation for lock-free
// structures that free nodes while other threads may still be reading them.
//
// The problem: a thread loads `head`, gets node X, and is about to read
// X->next. Meanwhile another thread unlinks X and deletes it. The first
// thread's read is now a use-after-free, and if X's memory was reused it can
// also be an ABA. AtomicStack dodges this by never freeing nodes; the
// Michael-Scott queue can't.
//
// The fix: before dereferencing a shared pointer, a reader publishes it in a
// hazard slot ("I'm using X"). A thread that unlinks X doesn't delete it; it
// retires it onto a thread-local list. Every so often it scans all hazard
// slots and deletes only the retired nodes nobody has published.
//
//   HazardPointer hp;                       // claims a slot
//   Node* h = hp.protect(head_);            // h is safe to dereference
//   ... read h->next, CAS head_ ...
//   hp.reset();                             // done with h
//   HazardPointer::retire(h);               // after unlinking it
//
// The one subtle part is protect(): publishing and then re-reading the source
// (see the comment there).
//
// All slots live in one process-wide list. A slot is never freed, only
// released and reused, so a scan can walk the list without any reclamation
// problem of its own.

namespace lockfree {

namespace detail::hazard {

struct alignas(cache_line_size) Record {
    // What the owner is protecting, or nullptr.
    detail::atomic<const void*> hazard{nullptr};
    // Whether some HazardPointer owns this record right now.
    detail::atomic<bool> active{false};
    // Written once before the record is published, never again.
    Record* next = nullptr;
};

struct Retired {
    void* ptr;
    void (*reclaim)(void*);
};

// A batch of retired objects left behind by a thread that exited while some
// of them were still protected. The next scan on any thread adopts it.
struct OrphanBatch {
    std::vector<Retired> items;
    OrphanBatch* next = nullptr;
};

class Domain {
public:
    Domain() = default;
    Domain(const Domain&) = delete;
    Domain& operator=(const Domain&) = delete;

    // Runs during static destruction, after every thread_local (including the
    // main thread's) is gone, so nobody can hold a record or a hazard.
    ~Domain() {
        for (OrphanBatch* b = orphans_.exchange(nullptr, std::memory_order_acquire); b != nullptr;) {
            for (const Retired& r : b->items) r.reclaim(r.ptr);
            delete std::exchange(b, b->next);
        }
        for (Record* r = head_.load(std::memory_order_acquire); r != nullptr;) {
            delete std::exchange(r, r->next);
        }
    }

    Record* acquire() {
        // Reuse a released record if there is one. The acquire on success
        // pairs with release(): the previous owner's last hazard write (nullptr)
        // is visible before we publish over it.
        for (Record* r = head_.load(std::memory_order_acquire); r != nullptr; r = r->next) {
            bool expected = false;
            if (!r->active.load(std::memory_order_relaxed) &&
                r->active.compare_exchange_strong(expected, true, std::memory_order_acquire,
                                                  std::memory_order_relaxed)) {
                return r;
            }
        }
        // None free: add one. Records are only ever pushed, never popped, so
        // this push can't suffer ABA.
        auto* r = new Record;
        r->active.store(true, std::memory_order_relaxed);
        record_count_.fetch_add(1, std::memory_order_relaxed);
        Record* old = head_.load(std::memory_order_relaxed);
        do {
            r->next = old;
        } while (!head_.compare_exchange_weak(old, r, std::memory_order_release,
                                              std::memory_order_relaxed));
        return r;
    }

    void release(Record* r) noexcept {
        // Release: whatever the owner read through its hazard happens-before a
        // scanner that sees nullptr here and frees the object.
        r->hazard.store(nullptr, std::memory_order_release);
        r->active.store(false, std::memory_order_release);
    }

    std::size_t record_count() const noexcept {
        return record_count_.load(std::memory_order_relaxed);
    }

    // Frees every entry of `list` that no hazard slot points at, keeps the
    // rest. Also adopts orphaned batches into `list` first.
    void scan(std::vector<Retired>& list) {
        adopt_orphans(list);
        if (list.empty()) return;

        // Pairs with the fence in HazardPointer::protect(). Every entry of
        // `list` was unlinked before it was retired, so before this fence.
        // For a reader that published one of them: either its fence comes
        // first in the single total order of seq_cst fences, and then the
        // loads below see its hazard; or ours comes first, and then its
        // re-validating load sees the unlink, so it never uses the pointer.
        std::atomic_thread_fence(std::memory_order_seq_cst);

        std::vector<const void*> hazards;
        hazards.reserve(record_count());
        for (Record* r = head_.load(std::memory_order_acquire); r != nullptr; r = r->next) {
            // Acquire pairs with release(): if a reader is done with an
            // object, its reads of it happen-before our delete.
            if (const void* p = r->hazard.load(std::memory_order_acquire)) hazards.push_back(p);
        }
        std::sort(hazards.begin(), hazards.end());

        // A reclaim function may retire more objects (a node owning another
        // node); those land in `list` again, not in the vector being walked.
        std::vector<Retired> pending = std::exchange(list, {});
        for (const Retired& r : pending) {
            if (std::binary_search(hazards.begin(), hazards.end(), static_cast<const void*>(r.ptr))) {
                list.push_back(r);
            } else {
                r.reclaim(r.ptr);
            }
        }
    }

    void orphan(std::vector<Retired> items) {
        auto* b = new OrphanBatch{std::move(items)};
        OrphanBatch* old = orphans_.load(std::memory_order_relaxed);
        do {
            b->next = old;
        } while (!orphans_.compare_exchange_weak(old, b, std::memory_order_release,
                                                 std::memory_order_relaxed));
    }

private:
    void adopt_orphans(std::vector<Retired>& list) {
        // Take the whole list at once, so batches can't be popped twice.
        OrphanBatch* b = orphans_.exchange(nullptr, std::memory_order_acquire);
        while (b != nullptr) {
            list.insert(list.end(), b->items.begin(), b->items.end());
            delete std::exchange(b, b->next);
        }
    }

    detail::atomic<Record*> head_{nullptr};
    detail::atomic<std::size_t> record_count_{0};
    detail::atomic<OrphanBatch*> orphans_{nullptr};
};

inline Domain& domain() {
    static Domain d;
    return d;
}

// Retired objects of the calling thread, waiting for a scan.
struct ThreadState {
    std::vector<Retired> retired;
    bool scanning = false;

    ThreadState() {
        // Make sure the domain is constructed first, so it's destroyed after
        // every ThreadState, including the main thread's.
        (void)domain();
    }

    ~ThreadState() {
        if (retired.empty()) return;
        domain().scan(retired);
        // Still protected by somebody: hand them to whoever scans next.
        if (!retired.empty()) domain().orphan(std::move(retired));
    }
};

inline ThreadState& thread_state() {
    thread_local ThreadState s;
    return s;
}

// Scan once the list is about twice the number of slots. With H slots, at
// most H entries survive a scan, so each scan frees at least half of what it
// looks at and the cost per retire stays constant. The base keeps a lone
// thread with one slot from scanning every other retire.
inline constexpr std::size_t kScanBase = 64;

} // namespace detail::hazard

// One hazard slot, owned by the calling thread for this object's lifetime.
// Cheap to create after the first few (records are reused), but a hot loop
// should keep one around rather than make one per operation.
class HazardPointer {
public:
    HazardPointer() : record_(detail::hazard::domain().acquire()) {}
    ~HazardPointer() { detail::hazard::domain().release(record_); }

    HazardPointer(const HazardPointer&) = delete;
    HazardPointer& operator=(const HazardPointer&) = delete;

    // Loads `src` and protects the result: until reset() (or another
    // protect), the returned object won't be freed by any scan. May return
    // nullptr, which protects nothing.
    template <typename T>
    T* protect(const detail::atomic<T*>& src) noexcept {
        T* p = src.load(std::memory_order_relaxed);
        for (;;) {
            // Release for the same reason reset() is: re-protecting replaces
            // the previous hazard, and a scanner that sees the new value may
            // free the old object. Our reads of it must happen-before that.
            record_->hazard.store(p, std::memory_order_release);
            LOCKFREE_TEST_HOOK("hazard_protect_before_validate");
            // Publishing isn't enough: between our load of src and the store
            // above, p may have been unlinked, retired, and scanned by a
            // thread that didn't see our hazard yet. Re-reading src after the
            // publish closes that gap. If src still holds p, p was reachable
            // after our hazard was visible (the seq_cst fences here and in
            // scan() order the two), so no scan can free it now. If src
            // changed, try again with the new value.
            //
            // The fence is what stops the hazard store from being reordered
            // after the re-read, which is a plain store-load reordering that
            // ARM and even x86 allow. Acquire on the re-read so the caller
            // also sees the object's contents.
            std::atomic_thread_fence(std::memory_order_seq_cst);
            T* again = src.load(std::memory_order_acquire);
            if (again == p) return p;
            p = again;
        }
    }

    // Stop protecting anything.
    void reset() noexcept { record_->hazard.store(nullptr, std::memory_order_release); }

    // Defers `p`'s deletion until no hazard slot points at it. Call only after
    // p is unreachable from the structure (no new reader can find it); readers
    // that already protected it keep it alive until they move on.
    //
    // Deleter must be an empty, default-constructible function object (the
    // default deletes with `delete`); a stateless type keeps every retired
    // entry at two words.
    template <typename T, typename Deleter = std::default_delete<T>>
        requires std::is_empty_v<Deleter> && std::is_default_constructible_v<Deleter>
    static void retire(T* p, Deleter = {}) {
        retire_erased(const_cast<void*>(static_cast<const void*>(p)),
                      [](void* q) { Deleter{}(static_cast<T*>(q)); });
    }

    // Same, with a plain function as the deleter.
    static void retire(void* p, void (*reclaim)(void*)) { retire_erased(p, reclaim); }

    // Scans the calling thread's retired list now, regardless of its length.
    static void scan() {
        auto& ts = detail::hazard::thread_state();
        // A reclaim function that retires something can end up back here.
        // The outer scan will see that entry next time; don't nest.
        if (ts.scanning) return;
        ts.scanning = true;
        struct Done {
            bool& flag;
            ~Done() { flag = false; }
        } done{ts.scanning};
        detail::hazard::domain().scan(ts.retired);
    }

    // Objects the calling thread has retired that haven't been freed yet.
    static std::size_t retired_count() noexcept {
        return detail::hazard::thread_state().retired.size();
    }

    // Slots created so far, in use or not.
    static std::size_t record_count() noexcept { return detail::hazard::domain().record_count(); }

private:
    static void retire_erased(void* p, void (*reclaim)(void*)) {
        auto& ts = detail::hazard::thread_state();
        ts.retired.push_back({p, reclaim});
        if (ts.retired.size() >= detail::hazard::kScanBase + 2 * record_count()) scan();
    }

    detail::hazard::Record* record_;
};

} // namespace lockfree
