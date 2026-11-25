# The adversarial scheduler

`tools/scheduler/` is a small deterministic interleaving controller, in the spirit of Loom (Rust) and
CHESS. It runs a handful of threads one at a time, decides at every atomic access who goes next, and
looks for the schedule that breaks a test. When it finds one it hands back a seed that replays it, then
shrinks it to the few context switches that matter.

Stress tests and TSan find a lot, but they're at the mercy of the OS scheduler: the window between a
load and the CAS that depends on it is a few nanoseconds, and the run that hits it may come once a day.
Here the preemption is a choice the tool makes on purpose, so a bug that needs one particular
interleaving shows up in a few hundred runs and comes back every time you ask.

## How a run works

### Yield points come from the atomics

Every structure in `include/lockfree/` keeps its shared state in `lockfree::detail::atomic<T>`
(`detail/atomic_ref.hpp`). In a normal build that is just `std::atomic<T>`; an assembly diff of all
five structures against the pre-shim headers was byte-identical. With `LOCKFREE_SCHEDULER` defined
(the `lockfree_scheduler` library sets it), it becomes a wrapper that calls a hook before every
load, store, exchange, CAS and fetch-add:

```cpp
T load(std::memory_order order = std::memory_order_seq_cst,
       std::source_location where = std::source_location::current()) const noexcept {
    yield_point(AtomicOp::load, this, where);
    return a_.load(order);
}
```

Atomic accesses are exactly the points where another thread's step can change the outcome; plain
memory only matters through the atomics that publish it. The defaulted `source_location` is evaluated
at the call site, so each yield point knows the line in the *structure* it came from. That is what
the traces and the coverage report print.

### One thread at a time, a baton between them

`Scheduler::run()` starts one real `std::thread` per logical thread, but only one holds the baton.
When the running thread reaches a yield point it asks the `Strategy` who should go next:

```cpp
struct Decision {
    std::optional<ThreadId> current;   // who is at a yield point (nullopt: start, or someone just finished)
    const detail::YieldPoint* point;   // the access it's about to make
    std::span<const ThreadId> runnable;
    std::uint64_t step;
};
```

If the strategy returns `current`, the thread carries on. Otherwise it hands the baton over and
waits on its own condition variable, so a handoff wakes exactly the thread that was picked rather
than the whole herd.

Because only one thread moves at a time and every switch happens at a known point, **a run is a pure
function of the strategy's choices** (as long as the test code itself is deterministic: no clocks, no
`rand()`, no real-thread races outside the scheduler). Everything below builds on that.

A run ends `completed`, `failed` (some thread threw; `require(cond, msg)` is the usual way), or
`step_limit`. The last one is how livelock and "waiting for something that will never happen" show up:
with one thread at a time, a wait nobody can satisfy is simply a run that never ends.

### Writing a scenario

A scenario is a factory that builds fresh state for each run, plus an optional check that runs after
every thread has returned:

```cpp
CaseFactory factory = [] {
    auto q = std::make_shared<lockfree::MpmcQueue<int, 4>>();
    auto got = std::make_shared<std::vector<int>>();
    return Case{
        .threads = {[q] { q->try_push(1); q->try_push(2); },
                    [q, got] { int v; while (got->size() < 2) if (q->try_pop(v)) got->push_back(v); }},
        .check = [got] { require(*got == std::vector{1, 2}, "fifo order"); },
    };
};
auto result = explore(factory, pct, /*iterations=*/5000, /*seed=*/42);
```

State is owned through `shared_ptr` captures, not references to the test's stack, because a run that
hits the step limit can't be unwound (its threads are parked inside `noexcept` atomic operations) and
lives on, detached, for the rest of the process.

## Strategies

### Random walk

`RandomStrategy` picks uniformly among the runnable threads at every decision. Each run gets its own
seed derived from `(exploration seed, run index)`, and a failure prints that one number:

```
lockfree::sched: run 417 of 5000 failed after 32 steps: check: values not conserved: ...
  replay seed (pct): 0xe1f6e8f99e3ecf1c
```

`replay(factory, strategy, seed)` gets the exact run back. A random walk is good at shallow bugs, one
switch at one of a few places. It is bad at bugs that need a thread to stay *parked* for a long time:
every step is a fresh coin flip, so "T0 stays stopped through 25 of T1's steps" costs about 2^-25 per run.

### PCT

`PctStrategy` is Probabilistic Concurrency Testing (Burckhardt et al., ASPLOS 2010). Each thread gets a
random priority, and the highest-priority runnable thread always runs. On top of that, d − 1 random
"change points" each knock the running thread to the bottom. So a run has only a few preemptions, at
random places, and between them threads run uninterrupted. That's the shape of most real bugs: one or
two preemptions at the right lines, with everything else in some ordinary order. For a bug of depth d,
one PCT run hits it with probability at least 1 / (k · n^(d−1)) for k threads and n steps, polynomial
instead of exponential.

Two additions make it fit lock-free code:

- **Starvation guard.** Strict priorities let a high-priority thread spin forever waiting on a
  low-priority one. A thread that wins `max_run` decisions in a row is demoted, which is the "you've had
  your time slice" every real OS gives anyway.
- **CAS bias.** The interesting window is usually right *after* a successful CAS, before the follow-up
  writes (a claimed but unpublished MPMC cell, a popped node that's about to be reused). At the yield
  after a compare-exchange, the thread is demoted with probability `cas_preempt` (default 0.25).

### Bounded exhaustive search

For tiny scenarios, sampling isn't needed. `explore_exhaustive` enumerates every schedule with at most
`preemption_bound` preemptions (CHESS's iterative context bounding, Musuvathi & Qadeer, PLDI 2007).
Only switching away from a thread that could have kept going counts as a preemption; picking someone
after a thread finishes is free. With a fixed bound the schedule count grows polynomially with run
length, so two threads with two or three operations each finish in a fraction of a second:

| scenario                          | bound | schedules | result |
|-----------------------------------|-------|-----------|--------|
| SPSC, 3 `try_push` vs 3 `try_pop` | 2     | 90        | all pass |
| MPMC, push then pop on each side  | 2     | 142       | all pass |
| MPMC, 2 pushes then a pop each    | 2     | 254       | all pass |
| lost update (`load` + `store`)    | 0 → 1 | 2 → 2     | passes at 0, caught at 1 |

The search is stateless: every schedule is a fresh run steered down a recorded prefix and then left to
its default ("keep running the current thread"). After each run, the deepest decision with an untried
alternative inside the bound is advanced. If a replayed prefix ever meets a different decision than
before, the scenario isn't deterministic and the result says so instead of pretending to be exhaustive.

One catch: a spinning thread keeps the baton under the default choice, so once the bound is spent it
spins until the step limit. Exhaustive scenarios use the `try_` operations and treat "full" or "empty"
as a result, not something to wait out.

## Shrinking

A failing PCT run typically contains a few switches, a random-walk failure dozens, and almost all of
them are noise. `shrink()` turns the run's choice list into a list of switches and minimizes it with
delta debugging (ddmin, Zeller & Hildebrandt 2002).

The key decision is how a switch is named. "At global step 17, run T1" would break as soon as an earlier
switch is removed, since every later step number shifts. Instead a switch is keyed to the preempted
thread's own progress: "when T0 reaches its 3rd yield point, run T1". That's the same line of T0's code
however the rest of the schedule changes. Replaying a list follows each switch whose key comes up and
otherwise keeps the current thread running, so *any subset of switches is still a valid schedule*, and
deleting one just means "don't preempt there". ddmin then removes chunks while the run still fails
with the same outcome and message (so it can't drift onto a different bug).

The ABA run from `interleaving_test.cpp` shrinks from 3 switches to 1 in 6 replays:

```
T0  start
T0  load       treiber_stack.hpp:106     h = head
T0  load       treiber_stack.hpp:109     next = h->next
T1  <- preempts T0
T1  ...                                  pop, pop, push to other, push to s (26 steps)
T0  cas        treiber_stack.hpp:116     CAS(head, h, next) succeeds with a stale next
```

That's the interleaving `docs/atomic_stack.md` walks through by hand, found and minimized without
anyone saying where to look.

## Coverage

"5000 runs passed" says little if they were the same 50 schedules. `CoverageStrategy` wraps any
strategy and keeps two numbers:

- **Distinct schedules**, by hashing each run's choice list.
- **Context-switch pairs**: at every preemption, (line the running thread stopped at, line the next
  thread resumes at). Most interleaving bugs come down to one such pair, and the set is bounded by
  (sites)², so it saturates. When it stops growing, more runs of the same strategy won't help much.

The ABA scenario shows why both are useful. PCT's 417 runs had only 51 distinct schedules and 33 pairs,
and it found the bug. The random walk ran all 5000 runs, with 3567 distinct schedules and 85 pairs,
and never did: lots of variety, none of it in the one shape that matters (a single long preemption).

## Results so far

| target                                | strategy    | outcome |
|---------------------------------------|-------------|---------|
| `naive::TreiberStack` (ABA)           | random walk | 0 hits in 200k runs |
| `naive::TreiberStack` (ABA)           | PCT, seed 0 | found at run 417, shrunk to 1 preemption |
| `AtomicStack` (tagged), same scenario | PCT         | 5000 runs, all pass |
| MPMC with the `enqueue_pos_` CAS turned into a store | exhaustive, bound 2 | caught at run 45 / 47 |
| SPSC, MPMC small scenarios            | exhaustive, bound 2 | every schedule passes |

Run them with `ctest --test-dir build -L scheduler --output-on-failure`.

## Limitation: sequentially consistent interleavings only

The scheduler runs one thread at a time, and a handoff goes through a mutex and condition variable,
which synchronizes everything. So every run is some interleaving of whole atomic operations, each
seeing the latest value: a **sequentially consistent** execution. The `memory_order` arguments are
passed through to the real atomics, but nothing here can make them matter.

That means a whole class of bugs is invisible to it:

- A publish store that should be `release` but is `relaxed`. On ARM the consumer can see the index move
  before it sees the element; under the scheduler it never can.
- Missing `acquire` on the reading side, store-load reordering that only a `seq_cst` fence prevents,
  and anything else that needs the hardware or compiler to reorder.

Those are TSan's job (it models the C++ happens-before relation and reports a missing edge even when
the run happened to be fine) and the job of running the stress tests on a weakly ordered machine. This
project's dev machine is an M4 Pro, and the mutation checks recorded in `NOTES` (relaxing SPSC's
publish, MPMC's `seq` store, MPSC's link store) were all caught by TSan, not by the scheduler.

Two smaller limits:

- **Fences aren't yield points.** None of the structures synchronize through a fence alone, so no
  switch is lost, but a future structure that does would need the wrapper extended.
- **Only the code that uses `detail::atomic` is visible.** A test's own `std::atomic`s, mutexes or
  sleeps don't yield. A thread that blocks on one hangs the run, so scenarios stick to the structure's
  operations and plain data owned by one thread at a time.

The split is deliberate: the scheduler answers "is there an *order of operations* that breaks this?",
TSan answers "is every cross-thread access *ordered* at all?". Together with the linearizability
checker on recorded stress runs, that covers the three ways a lock-free structure usually goes wrong.
