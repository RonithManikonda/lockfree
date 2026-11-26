// Hazard pointers vs epochs: what each costs a reader, and what each costs a
// thread that frees memory.
//
// Read side (BM_Read*): every thread repeatedly gets hold of the object behind
// one shared pointer and reads it. Nothing is ever retired, so this is the pure
// tax of being allowed to dereference:
//   - None:   an acquire load. Only safe because nothing here frees; it's the floor.
//   - Hazard: protect() = publish, seq_cst fence, re-load, compare; then reset().
//   - Epoch:  EpochGuard = store the pinned epoch, seq_cst fence; then unpin.
// Both schemes pay one full fence per read; the difference is what surrounds it.
// For a structure, the gap grows with the number of pointers an operation
// follows: hazard pointers protect (and fence) once per pointer, an epoch
// guard once per operation. BM_ReadChain* walks a 4-node chain to show that.
//
// Write side (BM_Update*): every thread allocates a node, swaps it into a
// shared slot and disposes of the node it swapped out. The exchange hands
// that node to exactly one thread, so plain `delete` is safe here and serves as
// the baseline; the others measure retire plus the scans/collections it
// triggers.
//
// items_per_second is per operation, summed over threads (state.threads()
// scaling, as in bench_mpmc.cpp).
//
// First numbers (M4 Pro, unpinned, two runs, ns per op at 1 / 8 threads):
//   ReadNone          3.9-4.0 / 11.6-12.6
//   ReadHazard        3.8-3.9 /  9.2-10.1   the fence hides behind the load chain
//   ReadEpoch        15.4-17.7 / 37-43
//   ReadChainHazard  15.5-16.3 / 34-45      4 protects
//   ReadChainEpoch   22.2-25.5 / 63-71      1 pin
//   UpdateDelete     59-63    / 708-749
//   UpdateHazard     82-84    / 669-784    backlog stays under ~85 per thread
//   UpdateEpoch     106-107   / 467-480    backlog reaches 6.5-7.3k at 8 threads
// So on this machine a single protect is no dearer than a bare acquire load,
// and an uncontended pin costs ~12 ns more. The guard's constructor isn't
// inlined and makes two thread_local accesses (macOS resolves those through a
// call), whereas a HazardPointer is an object the caller already holds; that
// is the likely gap, not the fence. Per pointer followed, the chain walk
// shows HP's cost growing (~4 ns each) while the pin is paid once, but four
// hops still weren't enough to overtake it here.
// On the write side EBR's backlog grows with threads: collect() advances the
// epoch at most once per 64 retires, and with 8 threads pinning constantly an
// advance often fails. Worth revisiting (advance more eagerly, or collect on
// unpin) before the queue benches in Phase 8.

#include <benchmark/benchmark.h>

#include <atomic>
#include <cstdint>

#include "affinity.hpp"
#include "lockfree/epoch.hpp"
#include "lockfree/hazard_pointer.hpp"

namespace {

struct Node {
    std::uint64_t value = 0;
    std::atomic<Node*> next{nullptr};
};

void finish(benchmark::State& state, bool pinned) {
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) * state.threads());
    state.counters["pinned"] = benchmark::Counter(pinned ? 1 : 0, benchmark::Counter::kAvgThreads);
}

// One object that every reader reads; never freed while the benchmark runs.
// Nothing in this file ever stores to the slots below, so without the
// ClobberMemory in publish() clang folds their loads into constants and the
// "None" baseline measures nothing at all.
Node shared_node{42};
std::atomic<Node*> shared_slot{nullptr};

void publish() {
    shared_slot.store(&shared_node, std::memory_order_release);
    benchmark::ClobberMemory();
}

void BM_ReadNone(benchmark::State& state) {
    const bool pinned = lockfree::bench::pin_current_thread(state.thread_index());
    std::uint64_t sum = 0;
    for (auto _ : state) {
        Node* p = shared_slot.load(std::memory_order_acquire);
        sum += p->value;
    }
    benchmark::DoNotOptimize(sum);
    finish(state, pinned);
}

void BM_ReadHazard(benchmark::State& state) {
    const bool pinned = lockfree::bench::pin_current_thread(state.thread_index());
    lockfree::HazardPointer hp; // one per thread, reused: the intended usage
    std::uint64_t sum = 0;
    for (auto _ : state) {
        Node* p = hp.protect(shared_slot);
        sum += p->value;
        hp.reset();
    }
    benchmark::DoNotOptimize(sum);
    finish(state, pinned);
}

void BM_ReadEpoch(benchmark::State& state) {
    const bool pinned = lockfree::bench::pin_current_thread(state.thread_index());
    std::uint64_t sum = 0;
    for (auto _ : state) {
        lockfree::EpochGuard guard;
        Node* p = shared_slot.load(std::memory_order_acquire);
        sum += p->value;
    }
    benchmark::DoNotOptimize(sum);
    finish(state, pinned);
}

// A fixed chain of 4 nodes, walked from the head.
constexpr int kChain = 4;
Node chain[kChain];
std::atomic<Node*> chain_head{nullptr};

void link_chain() {
    for (int i = 0; i + 1 < kChain; ++i) chain[i].next.store(&chain[i + 1], std::memory_order_relaxed);
    for (int i = 0; i < kChain; ++i) chain[i].value = static_cast<std::uint64_t>(i);
    chain_head.store(&chain[0], std::memory_order_release);
    benchmark::ClobberMemory();
}

// Set up before main() runs any benchmark; the stores are what keep the
// loads honest, not the timing.
const bool setup_done = [] {
    publish();
    link_chain();
    return true;
}();

void BM_ReadChainHazard(benchmark::State& state) {
    const bool pinned = lockfree::bench::pin_current_thread(state.thread_index());
    // Hand-over-hand with two slots, the way a list traversal does it.
    lockfree::HazardPointer a, b;
    std::uint64_t sum = 0;
    for (auto _ : state) {
        lockfree::HazardPointer* cur = &a;
        lockfree::HazardPointer* nxt = &b;
        Node* p = cur->protect(chain_head);
        while (p != nullptr) {
            sum += p->value;
            Node* q = nxt->protect(p->next);
            std::swap(cur, nxt);
            p = q;
        }
        a.reset();
        b.reset();
    }
    benchmark::DoNotOptimize(sum);
    finish(state, pinned);
}

void BM_ReadChainEpoch(benchmark::State& state) {
    const bool pinned = lockfree::bench::pin_current_thread(state.thread_index());
    std::uint64_t sum = 0;
    for (auto _ : state) {
        lockfree::EpochGuard guard;
        for (Node* p = chain_head.load(std::memory_order_acquire); p != nullptr;
             p = p->next.load(std::memory_order_acquire)) {
            sum += p->value;
        }
    }
    benchmark::DoNotOptimize(sum);
    finish(state, pinned);
}

std::atomic<Node*> update_slot{nullptr};

template <typename Dispose>
void run_update(benchmark::State& state, Dispose dispose) {
    const bool pinned = lockfree::bench::pin_current_thread(state.thread_index());
    std::uint64_t i = 0;
    for (auto _ : state) {
        Node* old = update_slot.exchange(new Node{++i}, std::memory_order_acq_rel);
        if (old != nullptr) dispose(old);
    }
    finish(state, pinned);
    if (state.thread_index() == 0) {
        // The framework has joined the other threads' timed loops by now
        // (teardown runs after the end barrier), so the slot is ours.
        delete update_slot.exchange(nullptr, std::memory_order_acq_rel);
    }
}

void BM_UpdateDelete(benchmark::State& state) {
    run_update(state, [](Node* n) { delete n; });
}

void BM_UpdateHazard(benchmark::State& state) {
    run_update(state, [](Node* n) { lockfree::HazardPointer::retire(n); });
    state.counters["backlog"] = benchmark::Counter(
        static_cast<double>(lockfree::HazardPointer::retired_count()), benchmark::Counter::kAvgThreads);
    lockfree::HazardPointer::scan();
}

void BM_UpdateEpoch(benchmark::State& state) {
    run_update(state, [](Node* n) {
        // A real structure would retire inside the guard of the operation
        // that unlinked n; retire pins on its own, so this is the same cost.
        lockfree::Epoch::retire(n);
    });
    state.counters["backlog"] = benchmark::Counter(
        static_cast<double>(lockfree::Epoch::retired_count()), benchmark::Counter::kAvgThreads);
    for (int i = 0; i < 3; ++i) lockfree::Epoch::collect();
}

} // namespace

BENCHMARK(BM_ReadNone)->ThreadRange(1, 8)->UseRealTime();
BENCHMARK(BM_ReadHazard)->ThreadRange(1, 8)->UseRealTime();
BENCHMARK(BM_ReadEpoch)->ThreadRange(1, 8)->UseRealTime();
BENCHMARK(BM_ReadChainHazard)->ThreadRange(1, 8)->UseRealTime();
BENCHMARK(BM_ReadChainEpoch)->ThreadRange(1, 8)->UseRealTime();
BENCHMARK(BM_UpdateDelete)->ThreadRange(1, 8)->UseRealTime();
BENCHMARK(BM_UpdateHazard)->ThreadRange(1, 8)->UseRealTime();
BENCHMARK(BM_UpdateEpoch)->ThreadRange(1, 8)->UseRealTime();
