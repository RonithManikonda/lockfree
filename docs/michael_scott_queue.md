# MichaelScottQueue

`include/lockfree/michael_scott_queue.hpp` — an unbounded multi-producer / multi-consumer FIFO queue
(Michael and Scott, 1996). It's a linked list with a dummy node at the front. Every operation is a
short CAS loop. Freed nodes go through a pluggable reclaimer: `HazardPointerReclaim` (default) or
`EpochReclaim`, both from [reclamation.md](reclamation.md).

```cpp
lockfree::MichaelScottQueue<std::string> q;                    // hazard pointers
lockfree::MichaelScottQueue<int, lockfree::EpochReclaim> e;    // epochs
q.push("hello");
if (auto v = q.try_pop()) use(*v);
```

## Layout

```
head_ -> dummy -> n1 -> n2 -> n3 <- tail_
         (no value)  (front)      (back)
```

`head_` always points at a node whose value is gone: the initial dummy, or the node whose value
was popped last. The front element is in `head_->next`. With the dummy, `head_` and `tail_` are
never null. An empty queue is simply `head_ == tail_` with `next == nullptr`. `head_` and `tail_`
sit on separate cache lines, because producers hit one and consumers the other.

## Enqueue: link, then swing

```cpp
tail = protect(tail_);
next = tail->next.load(acquire);
if (next) { tail_.CAS(tail, next); continue; }   // tail_ lags: help, retry
if (tail->next.CAS(nullptr, node, release))       // 1. link   (linearization point)
    tail_.CAS(tail, node, release);                // 2. swing  (best effort)
```

Linking and swinging are two separate CASes, and nothing makes them atomic together. So the queue
lives with a weaker invariant.

### The lagging-tail invariant

> `tail_` is either the last node or the one right before it.

Between steps 1 and 2, `tail_` lags one node behind the real end. It can't fall further behind,
because a producer only links after a node whose `next` is null, which is the real end. If
`tail_->next` is set, that producer goes back and swings `tail_` first.

### Helping

The producer that linked might stop between the two CASes forever (preempted, or killed). If
everyone waited for it to swing `tail_`, the queue would just be a lock with extra steps. Instead,
any thread that sees `tail->next != nullptr` does the swing itself, with the same CAS. Exactly one
of those CASes wins, and the losers already know someone helped. Step 2 is best effort for the same
reason: if it fails, a helper got there first.

That's the difference between lock-free and obstruction-free. A stalled thread never stops the
others, because they finish its work for it.

## Dequeue: protect, check, CAS head

```cpp
head = protect(0, head_);
tail = tail_.load(acquire);
next = protect(1, head->next);
if (head != head_.load(acquire)) continue;   // stale snapshot, retry
if (!next) return empty;                     // head == tail, nothing after it
if (head == tail) { tail_.CAS(tail, next); continue; }   // help a lagging tail
if (head_.CAS(head, next, acquire)) { retire(head); return next; }
```

The winner owns `next`. It moves the value out, destroys it, and `next` becomes the new dummy.

### Why dequeue has to help the tail too

If the queue holds one element and its producer hasn't swung `tail_` yet, then `head == tail` but
`next != nullptr`. Moving `head_` forward right then would leave `tail_` pointing at a node that's
about to be retired. A later producer would `protect(tail_)` and link onto freed memory. So
dequeue swings `tail_` first. **`head_` never passes `tail_`**, and every retired node is
unreachable from both ends.

### Reading the value after the CAS, not before

The 1996 paper reads `next->value` *before* the head CAS, because once the CAS succeeds another
dequeue could free `next`. That's fine for an `int`. For a general `T` it's a problem: every loser
copies the value for nothing, and moving a non-trivial `T` out of a node another thread is about to
own isn't allowed at all.

Here the move happens *after* the CAS, which is safe for two reasons:

- Only the CAS winner owns `next`'s value. Nobody else touches it.
- `next` itself stays alive. With HP it sits in hazard slot 1. With EBR the operation is still
  pinned. Another dequeue can make `next` the old head and retire it, but it won't be freed while
  we hold it.

That's also why `T` only needs nothrow move construction: the node is already ours, so there's no
way to give it back.

### What the head re-check is (and isn't) for

`protect(1, head->next)` only proves `head->next == next`. That never changes once set, so it says
nothing about whether `next` was retired. The textbook argument for safety is the re-check: if
`head` was still the head *after* `next`'s hazard went up, `next` was reachable, so no scan can
have freed it.

In this version, though, the head CAS gives the same guarantee. `next` is only dereferenced
after `head_.CAS(head, next)` succeeds, and that needs `head` to still be the head. The tail-help
CAS only succeeds while `tail_ == head`, which means `head_` hasn't moved past `head` yet. So the
re-check mostly saves a doomed CAS. NOTES on 2026-11-26 found the same thing from the other side:
replacing `protect(1)` with a plain load wasn't caught by a stalled-dequeuer test, because the
stalled thread's CAS always failed.

The re-validation *inside* `protect()` is a different matter. Without it, `protect(0, head_)` can
publish a hazard on a head that was retired and freed in between. The very next read,
`head->next`, then touches freed memory.

## Memory orderings

| Access | Order | Why |
|---|---|---|
| `tail->next` CAS (link) | release | publishes the node's value to whoever acquires the link |
| `head->next` / `tail->next` loads | acquire | pairs with the link: reading `next` makes its value visible |
| `head_` CAS | acquire | the winner reads `next`'s value; also orders after the link that made `next` visible |
| `tail_` CAS (swing / help) | release | a later `protect(tail_)` + `tail->next` load sees a fully built node |
| hazard publish / re-read | release / seq_cst fence / acquire | see [reclamation.md](reclamation.md) |

## Verification

- **Unit and lifetime tests** (`test_michael_scott_queue.cpp`), typed over both reclaimers:
  FIFO order, move-only and instance-counting values, nothing leaked or double-destroyed.
- **Stress:** every {1,2,4} producer × {1,2,4} consumer shape plus 8×8, exactly-once delivery and per-producer
  FIFO, under ASan and TSan.
- **Linearizability:** 2×2 runs recorded and checked, including empty pops. Unlike `MpscQueue`, an
  empty answer here is always justified (see [mpsc_queue.md](mpsc_queue.md) for the contrast).
- **Scheduler** (`interleaving_msq_test.cpp`): both real reclaimers survive 3000 random and 3000
  PCT schedules of a pop race and a 2-producer/1-consumer scenario. A test-only hazard scheme that
  records "frees" instead of doing them, plus a strategy wrapper that flags any atomic access inside
  a freed node, catches a `protect()` without re-validation at PCT run 11. A random walk doesn't
  find it in 3000: the stalled thread has to sit through the other thread's whole pop.
- **Mutation check:** replacing `retire` with a plain `delete` gives an ASan heap-use-after-free
  in the stress tests right away.

## Performance

`bench_msq` (M4 Pro, unpinned, million items per second; every queue behind the same 1024-item
credit window, since the MS queue is unbounded):

| | 1p1c | 2p2c | 4p4c | 8p8c |
|---|---|---|---|---|
| MS queue, hazard pointers | 12 | 6.9 | 3.2 | 2.4 |
| MS queue, epochs | 10.8 | 6.0 | 3.4 | 1.8 |
| MPMC (same window) | 39 | 9.6 | 4.2 | 3.0 |
| mutex + deque | 40 | 16 | 19 | 29 |

The MS queue pays for a `new` per push and a retire per pop. MPMC reuses preallocated cells, so
it's ahead at low thread counts. The two reclaimers are close. HP's fences and EBR's pin roughly
cancel out, and EBR falls behind at 16 threads, where its backlog grows. The mutex queue wins once
threads outnumber what the cores can absorb. Same contention cliff as MPMC and the stack; see
`docs/results.md` (step 70). The window itself isn't free: MPMC without it runs at 154 M/s for
1p1c.

Pick it when you need an unbounded MPMC queue and can't size a ring buffer ahead of time. If you
can bound the queue, `MpmcQueue` is faster and never allocates.
