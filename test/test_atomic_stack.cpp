#include <lockfree/atomic_stack.hpp>
#include <lockfree/naive/treiber_stack.hpp>

#include "tracked.hpp"

#include <gtest/gtest.h>

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
