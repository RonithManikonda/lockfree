#include <lockfree/naive/treiber_stack.hpp>

#include <gtest/gtest.h>

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
