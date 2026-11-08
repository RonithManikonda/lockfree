#include <lockfree/atomic_stack.hpp>
#include <lockfree/naive/treiber_stack.hpp>

#include "stress_harness.hpp"
#include "tracked.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

using lockfree::AtomicStack;
using lockfree::detail::ScopedTestHook;
using lockfree::test::Tracked;
using lockfree::naive::NodePool;
using lockfree::naive::TreiberStack;

// ---- Test hooks --------------------------------------------------------------

TEST(TestHook, FiresWithItsNameWhileInstalled) {
    TreiberStack<int> s;
    s.push(1);
    std::vector<std::string> seen;
    {
        ScopedTestHook hook([&](std::string_view where) { seen.emplace_back(where); });
        EXPECT_EQ(s.try_pop(), 1);
    }
    EXPECT_EQ(seen, std::vector<std::string>{"treiber_pop_before_cas"});

    s.push(2);
    EXPECT_EQ(s.try_pop(), 2);
    EXPECT_EQ(seen.size(), 1u) << "hook still ran after its scope ended";
}

TEST(TestHook, PopOnEmptyNeverReachesTheHook) {
    TreiberStack<int> s;
    int calls = 0;
    ScopedTestHook hook([&](std::string_view) { ++calls; });
    EXPECT_EQ(s.try_pop(), std::nullopt);
    EXPECT_EQ(calls, 0);
}

TEST(TestHook, IsOffWhileItsOwnCallbackRuns) {
    // The callback pops from the same stack. If it re-entered itself this
    // would recurse until the stack was empty.
    TreiberStack<int> s;
    for (int i = 0; i < 4; ++i) s.push(i);
    int calls = 0;
    {
        ScopedTestHook hook([&](std::string_view) {
            ++calls;
            if (calls == 1) EXPECT_EQ(s.try_pop(), 3);
        });
        // The outer pop's CAS fails (head moved) and it retries, so the hook
        // runs a second time, but never from inside itself.
        EXPECT_EQ(s.try_pop(), 2);
    }
    EXPECT_EQ(calls, 2);
    EXPECT_EQ(s.try_pop(), 1);
}

TEST(TestHook, ScopesNestAndRestore) {
    TreiberStack<int> s;
    std::string log;
    ScopedTestHook outer([&](std::string_view) { log += 'o'; });
    {
        ScopedTestHook inner([&](std::string_view) { log += 'i'; });
        s.push(1);
        s.try_pop();
    }
    s.push(2);
    s.try_pop();
    EXPECT_EQ(log, "io");
}

TEST(TestHook, OnlyFiresOnTheInstallingThread) {
    TreiberStack<int> s;
    int calls = 0;
    ScopedTestHook hook([&](std::string_view) { ++calls; });
    std::thread other([&] {
        s.push(7);
        EXPECT_EQ(s.try_pop(), 7);
    });
    other.join();
    EXPECT_EQ(calls, 0);
}

// ---- naive::TreiberStack, single-threaded contract ---------------------------

TEST(NaiveTreiber, PopOnEmptyFails) {
    TreiberStack<int> s;
    EXPECT_TRUE(s.empty_approx());
    EXPECT_EQ(s.try_pop(), std::nullopt);
}

TEST(NaiveTreiber, LastInFirstOut) {
    TreiberStack<int> s;
    for (int i = 0; i < 10; ++i) s.push(i);
    EXPECT_FALSE(s.empty_approx());
    for (int i = 9; i >= 0; --i) EXPECT_EQ(s.try_pop(), i);
    EXPECT_TRUE(s.empty_approx());
}

TEST(NaiveTreiber, PoolRecyclesNodes) {
    NodePool<int> pool;
    TreiberStack<int> s(pool);
    for (int round = 0; round < 100; ++round) {
        for (int i = 0; i < 8; ++i) s.push(i);
        for (int i = 0; i < 8; ++i) ASSERT_TRUE(s.try_pop());
    }
    EXPECT_EQ(pool.allocated(), 8u) << "pool should only grow to the peak live count";
}

TEST(NaiveTreiber, PoolHandsBackTheMostRecentlyReleasedNode) {
    // Address reuse is the raw material for ABA.
    NodePool<int> pool;
    auto* a = pool.acquire();
    auto* b = pool.acquire();
    EXPECT_NE(a, b);
    pool.release(a);
    pool.release(b);
    EXPECT_EQ(pool.acquire(), b);
    EXPECT_EQ(pool.acquire(), a);
    EXPECT_EQ(pool.allocated(), 2u);
}

TEST(NaiveTreiber, StacksSharingAPoolStayIndependent) {
    NodePool<int> pool;
    TreiberStack<int> a(pool);
    TreiberStack<int> b(pool);
    a.push(1);
    a.push(2);
    EXPECT_EQ(a.try_pop(), 2);
    b.push(10);  // reuses the node a just gave back
    a.push(3);
    EXPECT_EQ(a.try_pop(), 3);
    EXPECT_EQ(a.try_pop(), 1);
    EXPECT_EQ(b.try_pop(), 10);
    EXPECT_EQ(a.try_pop(), std::nullopt);
    EXPECT_EQ(b.try_pop(), std::nullopt);
    EXPECT_EQ(pool.allocated(), 3u);
}

// ---- ABA, reproduced deterministically ---------------------------------------
//
// "Thread A" is the outer try_pop. It reads head = X and next = Y, then stops
// at the hook. "Thread B" is the hook body and runs to completion in the gap.
// When A resumes, its CAS compares head against X, finds X, and succeeds.

TEST(NaiveTreiberAba, StalePopCorruptsTheStack) {
    NodePool<int> pool;
    TreiberStack<int> s(pool);
    TreiberStack<int> other(pool);  // another user of the same pool
    s.push(2);                      // Y
    s.push(1);                      // X: s = X(1) -> Y(2)

    std::optional<int> a_result;
    {
        ScopedTestHook hook([&](std::string_view) {
            EXPECT_EQ(s.try_pop(), 1);  // X goes back to the pool
            EXPECT_EQ(s.try_pop(), 2);  // so does Y, on top of X
            other.push(30);             // Y is reused by the other stack
            s.push(40);                 // X is reused here: s = X(40), next = null
        });
        a_result = s.try_pop();
    }

    // A's CAS saw X at the head and installed the next it read earlier, Y.
    EXPECT_EQ(a_result, 40);
    // s should now be empty. Instead its head is Y, a node owned by `other`.
    ASSERT_FALSE(s.empty_approx()) << "expected the naive stack to be corrupted";
    EXPECT_EQ(s.try_pop(), 30) << "s handed out a value that was only ever pushed to other";
    EXPECT_EQ(other.try_pop(), 30) << "and other hands out the same element again";
}

TEST(NaiveTreiberAba, SameInterleavingOnOneStackHappensToBeHarmless) {
    // With a single stack and a LIFO pool, pushes take nodes back in the
    // reverse of the order pops released them. So when X is on top again, Y is
    // right under it, the stale next is accidentally correct, and the bug
    // stays hidden. That's why the test above needs a second user of the pool.
    TreiberStack<int> s;
    s.push(2);
    s.push(1);

    std::optional<int> a_result;
    {
        ScopedTestHook hook([&](std::string_view) {
            EXPECT_EQ(s.try_pop(), 1);
            EXPECT_EQ(s.try_pop(), 2);
            s.push(30);  // reuses Y
            s.push(40);  // reuses X: s = X(40) -> Y(30)
        });
        a_result = s.try_pop();
    }
    EXPECT_EQ(a_result, 40);
    EXPECT_EQ(s.try_pop(), 30);
    EXPECT_EQ(s.try_pop(), std::nullopt);
}

TEST(NaiveTreiberAba, CasStillSucceedsAfterHeadChangedAndChangedBack) {
    // The smallest form of the problem: B pops X and pushes a new value, which
    // lands in the same node. head went X -> Y -> X, A can't tell, and its CAS
    // succeeds without a retry even though the stack changed underneath it.
    TreiberStack<int> s;
    s.push(2);
    s.push(1);
    int hook_calls = 0;
    {
        ScopedTestHook hook([&](std::string_view) {
            if (++hook_calls > 1) return;
            EXPECT_EQ(s.try_pop(), 1);
            s.push(99);
        });
        EXPECT_EQ(s.try_pop(), 99);
    }
    EXPECT_EQ(hook_calls, 1) << "a retry would mean the CAS noticed the change";
    EXPECT_EQ(s.try_pop(), 2);
}

// ---- TaggedAtomic ------------------------------------------------------------

namespace {

struct Dummy {
    int x = 0;
};

template <typename Tagged>
void expect_cas_bumps_tag_and_rejects_stale_tag() {
    Dummy a, b;
    Tagged t;
    auto v = t.load(std::memory_order_relaxed);
    EXPECT_EQ(v.ptr, nullptr);
    EXPECT_EQ(v.tag, 0u);

    ASSERT_TRUE(t.compare_exchange_weak(v, &a, std::memory_order_relaxed, std::memory_order_relaxed));
    auto stale = t.load(std::memory_order_relaxed);  // {a, 1}
    EXPECT_EQ(stale.ptr, &a);
    EXPECT_EQ(stale.tag, 1u);

    // a -> b -> a: same pointer, two versions later.
    auto cur = stale;
    ASSERT_TRUE(t.compare_exchange_weak(cur, &b, std::memory_order_relaxed, std::memory_order_relaxed));
    cur = t.load(std::memory_order_relaxed);
    ASSERT_TRUE(t.compare_exchange_weak(cur, &a, std::memory_order_relaxed, std::memory_order_relaxed));

    auto expected = stale;
    EXPECT_FALSE(t.compare_exchange_weak(expected, nullptr, std::memory_order_relaxed,
                                         std::memory_order_relaxed))
        << "a CAS with the old tag must fail even though the pointer matches";
    EXPECT_EQ(expected.ptr, &a);
    EXPECT_EQ(expected.tag, 3u) << "a failed CAS reports the current version";
}

} // namespace

TEST(TaggedAtomic, PackedRejectsStaleTag) {
    expect_cas_bumps_tag_and_rejects_stale_tag<lockfree::detail::TaggedAtomic<Dummy, false>>();
}

TEST(TaggedAtomic, DefaultRepresentationRejectsStaleTag) {
    expect_cas_bumps_tag_and_rejects_stale_tag<lockfree::detail::TaggedAtomic<Dummy>>();
}

TEST(TaggedAtomic, PackedTagWrapsAt16BitsWithoutTouchingThePointer) {
    using Packed = lockfree::detail::TaggedAtomic<Dummy, false>;
    Dummy d;
    Packed t;
    auto v = t.load(std::memory_order_relaxed);
    ASSERT_TRUE(t.compare_exchange_weak(v, &d, std::memory_order_relaxed, std::memory_order_relaxed));
    for (int i = 1; i < 0x10000; ++i) {
        v = t.load(std::memory_order_relaxed);
        while (!t.compare_exchange_weak(v, &d, std::memory_order_relaxed, std::memory_order_relaxed)) {
        }
    }
    v = t.load(std::memory_order_relaxed);
    EXPECT_EQ(v.ptr, &d);
    EXPECT_EQ(v.tag, 0u) << "65536 changes bring a 16-bit tag back around";
}

// ---- AtomicStack, single-threaded contract -----------------------------------

TEST(AtomicStack, PopOnEmptyFails) {
    AtomicStack<int> s;
    EXPECT_TRUE(s.empty_approx());
    EXPECT_EQ(s.try_pop(), std::nullopt);
    int out = -1;
    EXPECT_FALSE(s.try_pop(out));
    EXPECT_EQ(out, -1);
}

TEST(AtomicStack, LastInFirstOutThroughBothPopForms) {
    AtomicStack<int> s;
    for (int i = 0; i < 10; ++i) s.push(i);
    for (int i = 9; i >= 0; i -= 2) {
        EXPECT_EQ(s.try_pop(), i);
        int out = -1;
        ASSERT_TRUE(s.try_pop(out));
        EXPECT_EQ(out, i - 1);
    }
    EXPECT_TRUE(s.empty_approx());
}

TEST(AtomicStack, ManyRoundsReuseNodes) {
    AtomicStack<int> s;
    for (int round = 0; round < 1000; ++round) {
        for (int i = 0; i < 5; ++i) s.push(round * 10 + i);
        for (int i = 4; i >= 0; --i) ASSERT_EQ(s.try_pop(), round * 10 + i);
    }
}

TEST(AtomicStack, MoveOnlyType) {
    AtomicStack<std::unique_ptr<int>> s;
    s.push(std::make_unique<int>(1));
    s.emplace(new int(2));
    auto top = s.try_pop();
    ASSERT_TRUE(top && *top);
    EXPECT_EQ(**top, 2);
    std::unique_ptr<int> out;
    ASSERT_TRUE(s.try_pop(out));
    EXPECT_EQ(*out, 1);
}

TEST(AtomicStack, ThrowingConstructorLeavesStackUnchanged) {
    struct Picky {
        explicit Picky(int v) : value(v) {
            if (v < 0) throw std::invalid_argument("negative");
        }
        Picky(Picky&&) noexcept = default;
        int value;
    };
    AtomicStack<Picky> s;
    s.emplace(1);
    EXPECT_THROW(s.emplace(-1), std::invalid_argument);
    s.emplace(2);  // takes the node the failed emplace gave back
    auto a = s.try_pop();
    auto b = s.try_pop();
    ASSERT_TRUE(a && b);
    EXPECT_EQ(a->value, 2);
    EXPECT_EQ(b->value, 1);
    EXPECT_FALSE(s.try_pop());
}

class AtomicStackLifetime : public lockfree::test::TrackedTest {};

TEST_F(AtomicStackLifetime, PopDestroysTheStoredObject) {
    AtomicStack<Tracked> s;
    s.emplace(1);
    s.emplace(2);
    EXPECT_EQ(Tracked::live.load(), 2);
    {
        auto v = s.try_pop();
        ASSERT_TRUE(v);
        EXPECT_EQ(v->value, 2);
        EXPECT_EQ(Tracked::live.load(), 2) << "the node's copy should be gone, only *v and 1 left";
    }
    EXPECT_EQ(Tracked::live.load(), 1);
}

TEST_F(AtomicStackLifetime, DestructorDestroysWhatsLeft) {
    {
        AtomicStack<Tracked> s;
        for (int i = 0; i < 5; ++i) s.emplace(i);
        Tracked out(0);
        ASSERT_TRUE(s.try_pop(out));
        EXPECT_EQ(out.value, 4);
    }
    // TearDown checks nothing leaked and nothing was destroyed twice.
}

// ---- AtomicStack under the same ABA interleavings ----------------------------

TEST(AtomicStackAba, CasFailsWhenHeadChangedAndChangedBack) {
    // Same as NaiveTreiberAba.CasStillSucceedsAfterHeadChangedAndChangedBack:
    // B pops X and pushes 99, which lands in the same node. head's pointer is
    // X again, but its tag moved on, so A's CAS fails and A retries.
    AtomicStack<int> s;
    s.push(2);
    s.push(1);
    int hook_calls = 0;
    {
        ScopedTestHook hook([&](std::string_view where) {
            if (where != "atomic_stack_pop_before_cas") return;
            if (++hook_calls > 1) return;
            EXPECT_EQ(s.try_pop(), 1);
            s.push(99);
        });
        EXPECT_EQ(s.try_pop(), 99);
    }
    EXPECT_EQ(hook_calls, 2) << "A should have retried after its CAS failed";
    EXPECT_EQ(s.try_pop(), 2);
    EXPECT_EQ(s.try_pop(), std::nullopt);
}

TEST(AtomicStackAba, StalePopRetriesAfterADrainAndRefill) {
    // B empties the stack and refills it. The free list hands the nodes back
    // in reverse, so X ends up on top again, three versions later.
    AtomicStack<int> s;
    s.push(3);
    s.push(2);
    s.push(1);  // X(1) -> Y(2) -> Z(3); A reads head = X, next = Y
    int hook_calls = 0;
    {
        ScopedTestHook hook([&](std::string_view where) {
            if (where != "atomic_stack_pop_before_cas") return;
            if (++hook_calls > 1) return;
            EXPECT_EQ(s.try_pop(), 1);
            EXPECT_EQ(s.try_pop(), 2);
            EXPECT_EQ(s.try_pop(), 3);
            s.push(30);
            s.push(40);
            s.push(50);  // X(50) -> Y(40) -> Z(30)
        });
        EXPECT_EQ(s.try_pop(), 50);
    }
    EXPECT_EQ(hook_calls, 2) << "A should have retried after its CAS failed";
    EXPECT_EQ(s.try_pop(), 40);
    EXPECT_EQ(s.try_pop(), 30);
    EXPECT_EQ(s.try_pop(), std::nullopt);
}

// ---- AtomicStack under many threads ------------------------------------------

class AtomicStackThreads : public lockfree::test::StressTest,
                           public ::testing::WithParamInterface<std::size_t> {};

TEST_P(AtomicStackThreads, MixedPushPopDeliversEveryValueExactlyOnce) {
    // Every thread pushes its own range and pops whatever it finds, so pushes
    // and pops of the same nodes overlap constantly.
    const std::size_t threads = GetParam();
    const std::uint64_t per_thread = lockfree::test::scaled(20000);
    const std::uint64_t total = threads * per_thread;

    AtomicStack<std::uint64_t> s;
    std::vector<std::vector<std::uint64_t>> popped(threads);

    lockfree::test::run_stress(threads, [&](std::size_t index, lockfree::test::StressRng& rng) {
        auto& mine = popped[index];
        mine.reserve(per_thread);
        for (std::uint64_t i = 0; i < per_thread; ++i) {
            s.push(index * per_thread + i);
            lockfree::test::maybe_delay(rng);
            // Pop about as often as we push, sometimes twice in a row, so the
            // stack keeps swinging between nearly empty and a little full.
            for (int pops = static_cast<int>(rng.below(3)); pops > 0; --pops) {
                if (auto v = s.try_pop()) mine.push_back(*v);
            }
        }
    });

    std::vector<std::uint8_t> seen(total, 0);
    std::uint64_t bad_values = 0, count = 0;
    auto record = [&](std::uint64_t v) {
        ++count;
        if (v >= total) {
            ++bad_values;
            return;
        }
        ++seen[v];
    };
    for (const auto& mine : popped) {
        for (auto v : mine) record(v);
    }
    while (auto v = s.try_pop()) record(*v);

    EXPECT_EQ(bad_values, 0u);
    EXPECT_EQ(count, total);
    std::uint64_t wrong = 0;
    for (auto n : seen) wrong += n != 1;
    EXPECT_EQ(wrong, 0u) << "values missing or popped more than once";
}

TEST_P(AtomicStackThreads, TokensSurviveHeavyNodeReuse) {
    // A handful of tokens bounce between the stack and the threads. With so
    // few nodes, the same few addresses come back to the top over and over,
    // which is exactly when an untagged CAS would lose or duplicate a token.
    const std::size_t threads = GetParam();
    const std::size_t tokens = threads;
    const std::uint64_t rounds = lockfree::test::scaled(50000);

    AtomicStack<std::uint64_t> s;
    for (std::size_t t = 0; t < tokens; ++t) s.push(t);
    std::vector<std::uint64_t> handled(tokens, 0);  // per token, written only by its holder

    lockfree::test::run_stress(threads, [&](std::size_t, lockfree::test::StressRng& rng) {
        for (std::uint64_t r = 0; r < rounds; ++r) {
            auto t = s.try_pop();
            if (!t) continue;
            // Holding the token means no other thread has it, so this plain
            // increment is race-free, and TSan would say so if the stack ever
            // handed one token to two threads.
            if (*t < tokens) ++handled[*t];
            lockfree::test::maybe_delay(rng);
            s.push(*t);
        }
    });

    std::vector<std::uint8_t> seen(tokens, 0);
    std::size_t count = 0;
    while (auto t = s.try_pop()) {
        ++count;
        if (*t < tokens) ++seen[*t];
    }
    EXPECT_EQ(count, tokens);
    for (std::size_t t = 0; t < tokens; ++t) EXPECT_EQ(seen[t], 1) << "token " << t;
    std::uint64_t total_handled = 0;
    for (auto h : handled) total_handled += h;
    EXPECT_GT(total_handled, 0u);
}

INSTANTIATE_TEST_SUITE_P(Threads, AtomicStackThreads, ::testing::Values(2, 4, 8),
                         [](const auto& info) { return std::to_string(info.param) + "threads"; });

class AtomicStackThreadedLifetime : public lockfree::test::TrackedTest {};

TEST_F(AtomicStackThreadedLifetime, RacingPushPopThenDestructorDrains) {
    const std::uint64_t per_thread = lockfree::test::scaled(5000);
    {
        AtomicStack<Tracked> s;
        lockfree::test::run_stress(4, [&](std::size_t index, lockfree::test::StressRng& rng) {
            for (std::uint64_t i = 0; i < per_thread; ++i) {
                s.emplace(static_cast<int>(index));
                if (rng.below(4) != 0) s.try_pop();
            }
        });
        EXPECT_GT(Tracked::live.load(), 0) << "some elements should be left for the destructor";
    }
    // TearDown checks the destructor destroyed exactly what was left.
}
