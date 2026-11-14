# AtomicStack

`include/lockfree/atomic_stack.hpp` is an unbounded lock-free LIFO stack: a Treiber stack whose head
is a tagged pointer. `include/lockfree/naive/treiber_stack.hpp` holds the textbook version without the
tag. It's deliberately broken and kept for tests, and later as a target for the scheduler.

## The Treiber stack in two CASes

```cpp
// push
node->next = head;                      // relaxed
CAS(head, node->next, node)             // release: publishes node and its value

// pop
h = head;                               // acquire
next = h->next;                         // relaxed
CAS(head, h, next)                      // acquire
```

Push and pop each try to swing `head` with one CAS and loop until it works. The CAS on push is a
release and the load/CAS on pop are acquires, so a popper that gets a node also sees the value the
pusher wrote into it. Nothing else synchronizes, and nothing else is needed for correctness *as long
as a node's address means the same node for the whole pop*. ABA is what happens when that stops
being true.

## ABA, the exact interleaving

Start with `s = X(1) -> Y(2)`. Nodes come from a pool that recycles them, the way malloc hands a
freed address back out.

| step | thread A (`s.try_pop()`)          | thread B                                  | `s` afterwards              |
|------|-----------------------------------|-------------------------------------------|-----------------------------|
| 1    | `h = X`, `next = Y`; preempted    |                                           | `X(1) -> Y(2)`              |
| 2    |                                   | `s.try_pop()` → 1, X back to the pool     | `Y(2)`                      |
| 3    |                                   | `s.try_pop()` → 2, Y back to the pool     | empty                       |
| 4    |                                   | `other.push(30)` reuses Y                 | empty (`other = Y(30)`)     |
| 5    |                                   | `s.push(40)` reuses X                     | `X(40)`                     |
| 6    | `CAS(head, X, Y)` **succeeds**    |                                           | `head = Y`, Y belongs to `other` |

At step 6 A compares `head` against `X`, finds `X`, and installs the `next` it read at step 1. That
`next` is stale: Y has been popped, reused and now sits in a different stack. A returns 40, which is
fine, but `s` should be empty and instead points into `other`. From here the two stacks share nodes
and both are corrupt.

The CAS can't tell. It compares bit patterns, and "the head is X" was true at step 1 and true again at
step 6. The head went A → B → A, hence the name.

`NaiveTreiberAba.StalePopCorruptsTheStack` replays exactly this. Thread A is the outer `try_pop`, and
thread B is a callback on the `treiber_pop_before_cas` test hook, which runs in the gap between
reading `next` and the CAS. So the test is deterministic and single-threaded, with no timing luck
needed.

### Why the test needs a second stack

`NaiveTreiberAba.SameInterleavingOnOneStackHappensToBeHarmless` runs the same steps, but step 4 pushes
onto `s` itself. Then the bug stays hidden. The pool is LIFO, so pushes take nodes back in the
reverse of the order the pops released them. When X is on top again, Y is right under it, and A's
stale `next` happens to be correct. With one stack, ABA happened but did no damage. Add a second
user of the pool and the nodes come back in a different arrangement. That's closer to a real program,
where every structure shares one allocator.

The smallest version of the problem is `CasStillSucceedsAfterHeadChangedAndChangedBack`: B pops X
and pushes 99, which lands in X again. Nothing is corrupted, but A's CAS succeeds without a retry
although the stack changed underneath it. That alone breaks the assumption every lock-free
algorithm makes about CAS: "if it succeeded, nothing moved".

## The fix: version the head

`detail::TaggedAtomic` stores `{pointer, tag}` and every successful CAS writes `tag + 1`. A popper
compares both halves. A read `{X, 2}` at step 1 (two pushes so far), and after B's two pops and one
push the head is `{X, 5}`. Same pointer, different version, so A's
CAS fails. The failed CAS hands A the current value, A reloads `next` from it, and retries correctly.

Two representations, picked at compile time:

- **Wide**: `{Node*, uint64_t}` in one 16-byte atomic. Lock-free where the hardware has a 16-byte CAS
  (ARM64 with LSE: `casp`, on every Apple Silicon core; x86-64 with `cmpxchg16b`, which needs
  `-mcx16` and in practice clang). A 64-bit tag never wraps.
- **Packed**: the pointer in the low 48 bits of a `uint64_t`, a 16-bit tag in the top 16. This relies
  on user-space addresses fitting in 48 bits (true on x86-64 with 4-level paging and on ARM64 with the
  usual 48-bit VA). 5-level paging or ARM top-byte tagging would break it, and an assert catches that
  in debug builds. GCC on x86 reports 16-byte atomics as not always lock-free, so Linux CI exercises
  this form. The tests also force it on the Mac with `TaggedAtomic<Node, false>`.

The packed tag wraps after 65536 changes. A thread stalled across *exactly* a multiple of that many
head changes, ending on the same pointer, would be fooled. That's very unlikely, but it isn't
impossible the way 2^64 is. `PackedTagWrapsAt16BitsWithoutTouchingThePointer` shows the wrap.

`AtomicStackAba.CasFailsWhenHeadChangedAndChangedBack` and `StalePopRetriesAfterADrainAndRefill` run
the naive interleavings against `AtomicStack` and assert the opposite outcome: the hook runs twice,
because A's first CAS failed and it retried.

## Why tags don't solve memory reclamation

Look at the line between the load and the CAS again:

```cpp
next = h->next;   // h may already have been popped by someone else
```

The tag makes sure a stale `next` is never *installed*. It does nothing about *reading* it. If B
popped `h` and called `delete h`, A's load reads freed memory before its CAS ever gets the chance to
fail. That's a use-after-free, and the tag can't prevent it. The tag works after the fact, and the read
happens first.

So `AtomicStack` never frees a node while the stack is alive. Popped nodes go to a free list (a
second tagged stack, `free_`), pushes take from it, and memory only goes back to the OS in the
destructor. A late reader always lands on some valid node, maybe holding garbage that the failing
CAS then throws away. It's also why `Node::next` is atomic: a losing popper can read it while the
node's new owner writes it.

The cost is that the stack never shrinks. Its memory is the peak number of live elements, forever.
That's fine for a stack, but it doesn't work for structures that must actually free memory, like the
Michael–Scott queue, or anything whose nodes are owned by the caller. Those need to know *when no
thread can still be reading a node*:

- **Hazard pointers**: each reader publishes "I'm about to read `h`", and a node is only freed once
  no published pointer matches it.
- **Epoch-based reclamation**: readers announce which epoch they're in, and retired nodes are freed
  once every active thread has moved past the epoch they were retired in.

Both are Phase 7. Once safe reclamation exists, tags become optional for many structures: if a node
can't be freed and reused while someone holds a pointer to it, A → B → A on that address can't happen.

## Orderings

| operation | order | why |
|-----------|-------|-----|
| `n->next = expected.ptr` | relaxed | published by the CAS that follows |
| push CAS (success) | release | the value and `next` must be visible to whoever pops `n` |
| pop's head load | acquire | pairs with that release before reading `next` and the value |
| `expected.ptr->next` load | relaxed | if it's stale, the CAS fails and the value is discarded |
| pop CAS (success and failure) | acquire | a failure reloads `expected`, whose `next` we read next |

The free list uses the same functions. Its release/acquire pair orders "the element was destroyed"
before "the node is handed to a new push".

## Contention

Every operation writes the same head line, and a push-pop pair touches two contended heads (items and
free list), so four CASes on two hot lines. Lost CASes back off with `Backoff`, which roughly doubled
throughput at 2–4 threads. First numbers from `bench_atomic_stack` (M4 Pro, unpinned, push+pop
pairs/s):

| threads | `AtomicStack` | mutex + vector |
|---------|---------------|----------------|
| 1 | 72M | 120M |
| 2 | 9–16M | 6–89M |
| 4 | 7–14M | 28M |
| 8 | 2–4M | 46M |

As with `MpmcQueue`, lock-free doesn't mean fast under contention. A mutex holder gets several
operations done while it owns the line, whereas every CAS here moves the line to a new core. A
per-thread node cache would cut the free-list traffic and leave two CASes per pair. Treiber stacks are
useful for their progress guarantee (a preempted thread never blocks the others), not for raw
throughput on a shared head.
