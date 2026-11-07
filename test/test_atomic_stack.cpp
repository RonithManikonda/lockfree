#include <lockfree/naive/treiber_stack.hpp>

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

using lockfree::detail::ScopedTestHook;
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
