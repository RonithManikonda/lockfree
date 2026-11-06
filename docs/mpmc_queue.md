# MpmcQueue

`include/lockfree/mpmc_queue.hpp` — a bounded queue any number of threads can push to and pop from
(Dmitry Vyukov's design). No locks and no allocation after construction. Each operation is one CAS
on a position counter plus one store to a per-cell sequence number.

## Layout

```
producers' line : enqueue_pos_
consumers' line : dequeue_pos_
cells_          : Cell[N]        { atomic seq; raw storage for one T }   (N a power of two)
```

As in `SpscRingBuffer`, the positions are free-running counters and position `p` lives in cell
`p & (N - 1)`. The difference is that several threads now race to advance each counter, so the
counter alone can't say whether a cell is ready. Winning the CAS only means "cell `p` is yours when
it's ready". The sequence number says when that is.

## The sequence invariant

For the cell that position `p` maps to:

| `seq` | Meaning | Who acts next |
|---|---|---|
| `p` | free for lap `p` | the producer that claims `p` |
| `p + 1` | holds the element pushed at `p` | the consumer that claims `p` |
| `p + N` | moved out and destroyed; this is `seq == p'` for `p' = p + N` | the producer of the next lap |

So each cell walks `p → p + 1 → p + N → p + N + 1 → …`, and every step is made by the one thread
that owns the cell for that step. `seq` is only ever stored, never CASed: ownership comes from the
position CAS, so there's never a second writer to race with.

Threads compare `seq` with the value they want and act on the signed difference. It's signed so
"behind" and "ahead" survive wraparound at 2^64.

| | difference < 0 | difference == 0 | difference > 0 |
|---|---|---|---|
| **Producer**, `seq - pos` | the cell is a lap behind: **full** | try to claim `pos` | someone took `pos` already: reload |
| **Consumer**, `seq - (pos + 1)` | nothing pushed here yet: **empty** | try to claim `pos` | someone took `pos` already: reload |

## Why a slot can't be reused before it's consumed

Say the element pushed at `p` is still in its cell, so `seq == p + 1`. The next producer to map to
that cell is the one at `p + N`, and it needs `seq == p + N`. The difference is `1 - N`, which is
negative for every `N >= 2`, so it reports full and never touches the cell. The only store that
can move `seq` to `p + N` is `release_after_pop`, and that runs after the element has been moved
out and destroyed.

This also covers a consumer that has claimed `p` but hasn't finished. The position has moved on,
but `seq` hasn't, so the producer of the next lap still sees full. A claim isn't a completion, and
only `seq` tracks completions.

## Why sequence numbers avoid ABA

ABA is when a thread reads a value, something else changes it and changes it back, and the thread's
later check can't tell. A plain "full / empty" flag per cell would have exactly this problem: a slow
producer could look at a cell, see "empty", get descheduled for a whole lap, and then write into a
cell that's now meant for someone else.

Here the value encodes the lap, so it never comes back. Between two visits, `seq` only grows, and a
given value appears once per 2^64 counts. The position CAS has the same property: a stale `pos`
can't succeed because `enqueue_pos_` has moved past it and won't return. A producer that stalled
after loading `pos` either fails its CAS and reloads, or wins it, in which case nobody else has
claimed `pos` and the `seq == pos` it read is still current.

Getting bitten would take a thread stalling across exactly 2^64 increments of the same counter.

## Why each operation has its memory ordering

| Operation | Ordering | Reason |
|---|---|---|
| load `enqueue_pos_` / `dequeue_pos_` | relaxed | just a hint of where to start; a stale value fails the sequence check or the CAS |
| position CAS | relaxed | hands out a ticket. It publishes nothing: the data is guarded by `seq` |
| producer loads `seq` | **acquire** | pairs with the consumer's release: the old element is gone before we construct over it |
| producer stores `seq = pos + 1` | **release** | publishes the constructed element |
| consumer loads `seq` | **acquire** | pairs with the producer's release: the element is fully constructed before we read it |
| consumer stores `seq = pos + N` | **release** | our read and destruction happen before the next lap's constructor |
| `size_approx()` loads | relaxed | only a snapshot; loads `dequeue_pos_` first and clamps to N |

The interesting one is the relaxed CAS. It's tempting to think that claiming a position needs to
synchronize with something, but the claim doesn't touch the cell's contents. Everything the claimant
is about to read or write was already ordered by its acquire load of `seq`, which happened *before*
the CAS. The CAS only has to be atomic, so that exactly one thread wins each position.

Mutation checks, run on Apple Silicon:

- Relaxing the producer's `seq` release store makes TSan report a race in
  `MpmcStress.MixedPushPopKeepsSumAndCount`.
- Relaxing the consumer's `seq` acquire load makes TSan report races in the producers x consumers
  test.

## FIFO: what the queue promises

Positions are handed out in order, so pushes and pops pair up by position: the pop that claims `p`
gets the element pushed at `p`. But claim order isn't finish order, and that shapes what callers
can observe:

- One producer's elements come out in the order it pushed them, as seen by any one consumer. The
  stress test checks this per producer and per consumer.
- Two consumers can finish out of order. If consumer A claims `p` and consumer B claims `p + 1`, B
  may return first, so there's no global order across consumers unless you add your own.

## Lock-free, not wait-free

Some thread can always make progress on the positions, but no individual thread is guaranteed to.
Every losing CAS means someone else won, which is the lock-free guarantee. Under heavy contention a
thread can lose many times in a row, so it isn't wait-free.

There's also a subtler catch. A producer that claims `p` and is then descheduled before storing `seq`
blocks that one cell:

- Consumers reaching `p` see "nothing pushed yet" and report **empty**, even if `p + 1`, `p + 2`, …
  are already filled by faster producers. Those elements are stuck behind the stalled one, because
  FIFO hands out positions strictly in order.
- Once the queue wraps around to that cell, producers report **full** even if other cells are free.

Nothing is lost or corrupted. It all drains once the stalled thread resumes, but until then
`try_pop` can return false on a non-empty queue. A thread that dies between claiming and publishing
would block that cell forever, which is why filling a cell has to be `noexcept`: the claim can't be
given back. `try_emplace` builds a temporary *before* claiming when `T`'s constructor can throw, and
the class requires a nothrow move constructor and destructor.

So the lock-free label really applies to the claim step. By the formal definition the queue as a
whole falls short of it, since one stalled thread can eventually leave everyone else unable to finish
a useful operation. In practice it behaves like a very short critical section per cell rather than one
big lock: the window is a few instructions long, and other cells keep working until the queue wraps
around to the stuck one.

## False sharing and contention

`enqueue_pos_`, `dequeue_pos_` and the start of the cell array sit on separate cache lines, so
producers and consumers don't invalidate each other's position. Within each group, though, every
operation still writes the same line. That's the cost of a single shared counter, and it can't be
padded away.

Lost CASes back off (`Backoff`), so threads that lost on the same `pos` don't all retry in lock step.
The loops that wait on a full or empty queue live in the blocking `push` / `pop`, not the claim loops:
a `try_` call that sees full or empty returns immediately.

First numbers from `bench_mpmc` (M4 Pro, unpinned, capacity 1024, items/s):

| Shape | `MpmcQueue` | mutex + deque |
|---|---|---|
| 1p1c | 154M | 60M |
| 2p2c | 93M | 24M |
| 4p4c | 6M | 29M |
| 8p8c | 18M | 40M |

Lock-free doesn't automatically mean fast. The 4p4c collapse isn't explained yet. It isn't the waiting
policy, and 128-byte alignment helps about 1.6x without fixing it. One hypothesis is that 8+ threads
span both of the M4 Pro's performance clusters, so the position line crosses clusters on nearly every
operation, while a mutex holder gets to run a few operations before the line moves.
