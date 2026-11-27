# Safe memory reclamation

`include/lockfree/hazard_pointer.hpp` and `include/lockfree/epoch.hpp` answer one question: when a
lock-free structure unlinks a node, when is it safe to `delete` it?

## The problem

Every structure so far avoided the question. The SPSC and MPMC queues are arrays and never free a
slot. `MpscQueue` frees, but only on the single consumer, which is the only thread that ever reads a
node after it is linked. `AtomicStack` keeps popped nodes on a free list forever.

A Michael–Scott queue (Phase 8) can't do any of that. Several consumers race on `head`:

```cpp
Node* h = head.load();         // consumer A
Node* next = h->next.load();   // <-- A is preempted just before this
```

Meanwhile consumer B pops `h` and deletes it. When A wakes up, `h->next` reads freed memory. If the
allocator has already handed that address out again, A may even read a plausible-looking pointer and
then win its CAS, which is ABA again (see `docs/atomic_stack.md`), only this time through `malloc`
instead of a node pool.

Tags don't help here. A tag can tell A its CAS is stale, but A already dereferenced `h` before the
CAS. The read itself is the bug. So the unlinking thread must not free `h` while some other thread
might still be holding it. Both schemes defer the free; they differ in how they decide "nobody holds
it any more".

## Hazard pointers: track every pointer

A reader announces each pointer it is about to dereference by storing it in a *hazard slot*, one
per `HazardPointer` object. A thread that unlinks a node calls `retire()`, which only puts it on a
thread-local list. Once that list is longer than `64 + 2 × slots`, `scan()` reads every slot and
frees the retired nodes nobody announced. The rest wait for the next scan.

Because of the `2 × slots` threshold, each scan frees at least half of what it looks at, so a retire
costs O(1) amortized, and the backlog per thread is bounded by the number of slots. A stuck reader
pins at most the one or two nodes it announced.

### Why protect() has to re-read the source

The naive version is wrong:

```cpp
Node* h = head.load();
hazard = h;        // announce
use(h);            // too late?
```

Between the load and the announcement, another thread could unlink `h`, retire it and scan. That scan
doesn't see the hazard yet, so it frees `h`, and the announcement protects freed memory. So
`protect()` publishes, then loads `head` again:

```cpp
p = src.load(relaxed);
for (;;) {
    hazard.store(p, release);
    atomic_thread_fence(seq_cst);       // (F1)
    again = src.load(acquire);
    if (again == p) return p;           // safe
    p = again;
}
```

and `scan()` puts a fence between the unlinks it is cleaning up after and its reads of the slots:

```cpp
// every node in the retired list was unlinked before this point
atomic_thread_fence(seq_cst);           // (F2)
for (slot : all slots) hazards.push_back(slot.load(acquire));
```

All seq_cst fences fall into one total order, so either F1 comes before F2 or F2 comes before F1:

- **F1 first.** The hazard store comes before F1, the slot reads come after F2, so the scanner sees
  the hazard and keeps `p`.
- **F2 first.** The unlink of `p` comes before F2, and the re-read of `src` comes after F1, so the
  reader sees that `p` is gone (`again != p`) and retries. It never returns `p`.

There is no third case, so if `protect()` returns `p`, no scan that could free it has missed the
hazard. The fence is not decoration: without it, the hazard store may sit in the store buffer while
the re-read goes ahead, which is a store→load reordering that ARM and even x86 allow. Then both
sides can miss each other.

The mutation check for this (2026-11-25): deleting the re-read loop turns the many-thread hazard
stress into an ASan heap-use-after-free in 3 out of 3 runs. The test hook
`hazard_protect_before_validate` lets `ProtectRetriesWhenTheSourceChangesBeforeValidation` drive the
"F2 first" case deterministically.

## Epochs: track when each reader started

Epoch-based reclamation (EBR) gives up per-pointer tracking. There is one global counter, the epoch.
An `EpochGuard` *pins* the thread for a whole operation by recording the epoch it saw. A retired node
is tagged with the epoch at the time it was unlinked. The global epoch only moves from `e` to `e + 1`
when every pinned thread has been seen at `e`.

So when the epoch is two past a node's tag, every thread that was pinned when the node was unlinked
has since unpinned, and any later reader started after the unlink and can't have found it. Each
thread keeps three bags indexed by `epoch % 3`; the bag for `e` expires exactly when its slot is
needed again for `e + 3`.

The pin has the same store→load shape as `protect()` (publish "I'm pinned", then load shared
pointers), so it is followed by a seq_cst fence that pairs with one in `try_advance()`. The argument
is the same two-case one: either the advancer sees the pin and won't move past it, or the reader's
loads see every unlink that came before the advancer's fence.

## The trade-offs

| | hazard pointers | epochs |
|---|---|---|
| read cost | a store + fence + re-load **per pointer** | a store + fence **per operation** |
| reader code | protect/validate loop, must know which pointers to protect | wrap the operation in a guard |
| memory held by one stuck thread | the 1–2 nodes it protects | **everything** retired by anyone since it pinned |
| backlog per thread | bounded (≈ 2 × slots) | unbounded in the worst case |
| when a node can be freed | as soon as nobody announces it | two epoch advances later |

Numbers from `bench/bench_reclamation.cpp` (M4 Pro, unpinned, ns per op at 1 / 8 threads):

| | 1 thread | 8 threads |
|---|---|---|
| read, bare acquire load (unsafe floor) | 4.0 | 12 |
| read, `protect()` + `reset()` | 3.9 | 9–10 |
| read, `EpochGuard` | 16 | 37–43 |
| 4-node chain, 4 protects | 16 | 34–45 |
| 4-node chain, 1 pin | 22–25 | 63–71 |
| update + `delete` (baseline) | 60 | 710–750 |
| update + HP retire | 83 | 670–780 |
| update + epoch retire | 106 | 470–480 |

Two things stand out. First, on this machine a single `protect()` is as cheap as a bare load: the
fence overlaps with the dependent load chain the reader has to wait for anyway. The pin is dearer
than it should be, most likely because `EpochGuard`'s constructor isn't inlined and makes two
`thread_local` lookups, which macOS resolves through a function call. The textbook expectation (EBR
wins once an operation follows more than one pointer) didn't show up even at four hops.

Second, the write side shows the classic EBR risk at small scale. With 8 threads pinning constantly,
`collect()` (one advance attempt per 64 retires) often fails, and each thread's backlog grows to
6.5–7.3k nodes. Hazard pointers stay under ~85. Nothing is stalled here; the epoch is simply slow to
move. One descheduled reader would make it unbounded, which is what
`EpochStress.StalledPinnedThreadBlocksReclamation` pins down.

Rule of thumb: EBR when operations are short, readers are many, and memory spikes are tolerable;
hazard pointers when a stalled thread must not be able to pin the heap, or memory is tight.

## What TSan taught us

TSan doesn't model `atomic_thread_fence(seq_cst)`. It only believes happens-before edges built from
release stores and acquire loads. The fence argument above is correct by the C++ memory model, but
the first epoch stress run still reported a race: `retire()` reused a bag slot and freed its old
contents based on a *relaxed* load of the global epoch.

That report was real in spirit. The decision to free has to be ordered after every reader's last
access, and the chain that does it is: reader's unpin (release) → advancer reads it (acquire) →
advancing CAS (release) → the freeing thread's epoch load (acquire). Every free path now loads the
epoch with acquire. Pins and hazard stores are release too: re-pinning overwrites the 0 the last
unpin wrote, and a collector that sees the new value must still be ordered after the reads that came
before it.

Takeaway for new code in this repo: every path that frees memory needs an acquire that pairs with
the release that said "I'm done", even where a fence would be enough on paper.

## Where the tests are

- `test/test_hazard_pointer.cpp`: protection survives scans, reprotect moves the hazard, records are
  reused, thread exit hands leftovers on, a deterministic validate race, and many-thread
  pointer-swap stress (ASan + TSan).
- `test/test_epoch.cpp`: two advances before a free, a pinned thread blocks the advance, nested
  guards, orphaned bags, the stalled-reader test, and the same stress shapes as the HP suite.
- Both suites also build as `_sched` variants with every atomic routed through the scheduler's shim.
