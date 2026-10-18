# SpscRingBuffer

`include/lockfree/spsc_ring_buffer.hpp` — a bounded queue for exactly one producer thread and
exactly one consumer thread. No locks, no CAS: each side does one atomic store per operation.

## Layout and counters

```
consumer's line : read_          write_cache_
producer's line : write_         read_cache_
buffer_         : Slot[N]        (N a power of two, raw storage)
```

`read_` and `write_` are free-running counters. They only increase, and a counter value `i`
lives in slot `i & (N - 1)`. Because the counters never wrap into the slot range:

- empty is `write_ == read_`
- full is `write_ - read_ == N`

so all N slots are usable (the classic "head == tail" ring wastes one slot to tell these apart).
Unsigned wraparound at 2^64 doesn't matter: N divides 2^64, so both the masked index and the
difference `write_ - read_` stay correct across the overflow.

## Invariants

1. `0 <= write_ - read_ <= N` at all times.
2. Only the producer stores `write_`; only the consumer stores `read_`. One writer per index is
   what lets plain `store` replace a CAS.
3. A slot holds a live `T` exactly when its counter value is in `[read_, write_)`. Push constructs
   before publishing `write_ + 1`; pop destroys before publishing `read_ + 1`.
4. `read_cache_ <= read_` and `write_cache_ <= write_`. The cached copies are only ever stale in the
   safe direction: the producer may think the queue is fuller than it is, the consumer may think it
   is emptier. Neither mistake can cause an overwrite or a read of an unconstructed slot.

## Why each operation has its memory ordering

| Operation | Ordering | Reason |
|---|---|---|
| producer loads its own `write_` | relaxed | only the producer writes it, so it always sees its own latest value |
| producer stores `write_ + 1` | **release** | publishes the element constructed in the slot before the new count |
| consumer reloads `write_` (into `write_cache_`) | **acquire** | pairs with that release: if the consumer sees the count, it sees the constructed element |
| consumer loads its own `read_` | relaxed | same reasoning as the producer's own index |
| consumer stores `read_ + 1` | **release** | the consumer's read and destruction of the slot happen before the count says it's free |
| producer reloads `read_` (into `read_cache_`) | **acquire** | pairs with that release: the producer never constructs over an element still being moved out |
| `size_approx()` loads | acquire | only a snapshot; loads `read_` first so the difference can't go negative, then clamps to N |

The cached fast path needs no extra synchronization. Every value in `read_cache_` came from an
acquire load of `read_`. So when the producer trusts a cached value, the happens-before edge from
that earlier load already covers the slots it says are free. The same holds for `write_cache_`.

## What breaks if you weaken them (and why ARM matters)

x86 is TSO (total store order): ordinary stores already become visible in program order and
ordinary loads aren't reordered with other loads. A missing release/acquire there is usually
harmless *in the hardware* (it's still a data race and UB, and the compiler can still reorder).
ARM64 is weakly ordered, so the bugs actually show up:

- **Relaxed `write_` store.** The core may make the new count visible before the bytes of the
  element. The consumer sees "one more item", reads the slot, and gets old or half-written data.
- **Relaxed `write_` reload on the consumer.** Even if the producer released correctly, the
  consumer's later reads of the slot can be satisfied before the load of the count. Same symptom.
- **Relaxed `read_` store / reload.** The producer can start constructing into a slot while the
  consumer is still moving the old element out, so the consumer moves out a mix of the old and new
  objects (or destroys the new one).

The two-thread tests catch this under TSan: relaxing the `write_` publish store, or the
consumer's `write_cache_` reload, makes it report races. The development machine is an Apple
M4 Pro (arm64), so a weakened ordering can misbehave there for real, not just on paper.

## False sharing

A cache line is the unit cores move between each other (64 bytes here). If `read_` and `write_`
shared a line, every push would invalidate the consumer's copy and every pop the producer's,
even though neither thread touches the other's counter. That's false sharing: the data is
separate, but the line is shared. So:

- `read_` + `write_cache_` sit on the consumer's line,
- `write_` + `read_cache_` sit on the producer's line,
- `buffer_` starts on a line of its own.

`detail::cache_line_size` is 64. Apple Silicon's L2 prefetcher pulls lines in 128-byte pairs, so if
profiling on M-series ever shows the two counters still contending, doubling it is the first fix.

## Cached indices

Without caches, every push reads `read_` and every pop reads `write_`. Each such load pulls the
other core's line over, and the owner's next store must invalidate the copy again: two
cross-core transfers per element. With caches, a side only reloads the other's counter when its
copy says the queue is full (producer) or empty (consumer). In a steady stream that is about once
per lap of the buffer, not once per element. The benefit shrinks when the queue hovers near empty
or near full, because then the copies go stale on almost every operation.

## Element lifetime

Slots are `alignas(T) std::byte[sizeof(T)]`, not `T`, so `T` needs no default constructor and
empty slots hold nothing. Push placement-news into the slot; pop moves out and calls `~T()`;
`std::launder` is used when reading a slot because the same bytes hold a different object each
lap. The destructor destroys anything left between `read_` and `write_`. `T` must be nothrow
destructible, since pop and the destructor have no way to recover from a throwing destructor.
If `T`'s constructor throws in `try_emplace`, nothing was published and the queue is unchanged.
