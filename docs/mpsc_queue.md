# MpscQueue

`include/lockfree/mpsc_queue.hpp` — an unbounded queue that any number of threads can push to and
exactly one thread pops from (Dmitry Vyukov's intrusive design). It comes in two layers:

- `MpscIntrusiveQueue` links caller-owned `MpscNode`s and never allocates.
- `MpscQueue<T>` wraps it for plain values: each push allocates a node, and the consumer frees it.

## Layout

```
producers' line : tail_          (atomic, every push exchanges it)
consumer's line : head_          (plain pointer, only the consumer touches it)
own line        : stub_          (a dummy node; producers write its next)

head_ -> stub/n1 -> n2 -> n3 <- tail_
```

Nodes go in at `tail_` and come out at `head_`. The list always has at least one node for `head_`
to sit on, and that's what `stub_` is for: when the queue is empty, `head_` and `tail_` both point
at it.

## Push: one exchange, then one store

```cpp
node->next = nullptr;                    // relaxed
prev = tail_.exchange(node, acq_rel);    // 1. swing the tail
prev->next.store(node, release);         // 2. link the old tail to us
```

The exchange can't fail, so there is no CAS loop and no retry. Every producer does exactly these
three steps no matter how many others are pushing. Two producers that push at the same moment are
ordered by the exchange: whoever goes second gets the first one's node as `prev` and links behind
it.

## Pop: follow `next`, step over the stub

The consumer reads `head_->next`. If it's set, the old head is done and goes back to the caller.
The stub is skipped whenever it's at the head, because it isn't an element.

A node can only leave once it has a successor, since `head_` must always point at a real node. So
when the list is down to its last element, pop pushes the stub back in behind it, exactly like a
producer would, and then the element can leave. The stub cycles through the list over and over;
`MpscIntrusive.StubCyclesThroughRepeatedly` checks that.

## The gap between exchange and link

Between steps 1 and 2 of a push, the list is cut in two:

```
head_ -> n1 -> n2    n3 -> n4 <- tail_       (n2->next not set yet)
```

The producer that pushed `n3` has swung `tail_` but hasn't written `n2->next`. Anything pushed
after `n3` links fine onto `n3`, but the consumer walks from `head_` and can't get past `n2`. So
pop reports **empty**, even though `n3` and `n4` are sitting in the queue.

Nothing is lost. When the stalled producer runs its link store, the whole tail becomes reachable
again in order. The tests reproduce this deterministically through `MpscIntrusiveQueueTestAccess`,
which does the exchange without the link:

- `UnlinkedPushHidesItsItem` and `UnlinkedPushHidesEverythingBehindIt`: the hidden items reappear,
  in order, after the link.
- `UnlinkedPushLeavesItsPredecessorStuck`: even `n2` can't come out, because it has no successor
  yet, and pop can't push the stub behind it since `tail_` has moved on.
- `StubBehindAStalledProducerIsSkipped`: pop's own stub push can land behind a stalled producer,
  and the stub is skipped later like normal.

Two consequences for callers:

- `try_pop` returning false means "nothing reachable right now", not "empty". It only means empty
  if no push is in progress, which is why the destructor requires that nothing is racing it.
- The per-producer order and exactly-once guarantees hold; only *when* an item becomes visible is
  affected.

## Why it isn't strictly lock-free

Lock-free means that if threads keep taking steps, some operation finishes, no matter how any
thread is delayed. Pushes always finish: three steps each, nothing to wait for. Pops are the
problem. If one producer is preempted between its exchange and its link, the consumer can spin
forever on an empty-looking queue while every other producer keeps adding items behind the gap.
Every item pushed after the stall is stuck until that one thread gets CPU time again.

So the consumer's progress depends on a specific other thread being scheduled, which is exactly
what a lock does. The window is two instructions long and the "lock" is never held across a
system call, so in practice it hardly ever matters. But a real-time consumer, or a producer that
might be killed mid-push, would see it. Vyukov's own write-up of the design points out the same
blocking window. The blocking `pop()` here doesn't hide it either: it spins with `Backoff` until an
item is reachable.

`MpmcQueue` has a similar but narrower gap (a claimed but unpublished cell; see
[mpmc_queue.md](mpmc_queue.md)). The MPSC version hides everything behind the stall, while the
MPMC one blocks one cell and things around it keep working until the queue wraps.

## Why each operation has its memory ordering

| Operation | Ordering | Reason |
|---|---|---|
| producer stores `node->next = nullptr` | relaxed | nobody can see the node yet; the exchange's release publishes it |
| `tail_.exchange` | **acq_rel** | release: the next producer, which links through our node, sees `next == nullptr` first. acquire: our link store to `prev->next` comes after `prev`'s producer stored nullptr there, so its nullptr can't overwrite our link |
| link store `prev->next = node` | **release** | publishes the node's contents (the `T` in `MpscQueue`) to the consumer |
| consumer loads `next` | **acquire** | pairs with the link store: the element is fully built before we read it |
| consumer loads `tail_` | relaxed | only compared, never dereferenced; it can't read older than `head` because `head`'s own exchange happened before the link we followed |

Mutation checks on Apple Silicon, under TSan on the `MpscProducers` stress tests: relaxing the
link store, relaxing the consumer's `next` loads, and weakening the exchange from `acq_rel` to
`release` each produce a race report.

## Why the consumer may free nodes right away

A node comes back from `pop` only after its successor was linked, and that link store is the last
thing any producer does to it. The consumer is the only reader. So once `pop` returns a node,
no other thread has or will get a pointer to it, and `MpscQueue` can `delete` it immediately.

That argument needs a single consumer. With several, consumer B could load `head_` and be about to
read `head->next` while consumer A pops and frees that same node. Avoiding that is what hazard
pointers and epochs are for (the Michael–Scott queue in Phase 8).

## Cost of allocating per push

`bench_mpsc` (M4 Pro, unpinned, N producers into one consumer, two runs, items/s):

| Variant | Throughput |
|---|---|
| `MpscIntrusiveQueue`, nodes recycled through the bench | 23–64M |
| `MpscQueue<uint64_t>`, `new` per push | 6–11M |
| `MpmcQueue`, capacity 1024 | 45–78M |
| mutex + deque | 8–16M |

The allocator, not the queue, is the bottleneck in `MpscQueue`: the intrusive core with no
allocation is several times faster. That makes the intrusive form the one to use on a hot path,
with nodes embedded in objects that already exist (a task, a message buffer).

Why the bounded `MpmcQueue` still beats the intrusive core isn't settled. Two candidates: every
push writes to the *previous* node's cache line (the link), so producers pull each other's lines
around; and the bench's credit traffic, which only the MPSC variants pay for, limits how far
ahead producers can run.
