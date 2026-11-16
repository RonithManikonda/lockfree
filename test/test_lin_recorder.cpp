#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <optional>
#include <set>
#include <thread>
#include <vector>

#include "linearizability/history.hpp"
#include "linearizability/recorder.hpp"
#include "lockfree/mpmc_queue.hpp"
#include "stress_harness.hpp"

using lockfree::lin::happens_before;
using lockfree::lin::History;
using lockfree::lin::OpKind;
using lockfree::lin::Recorder;

TEST(LinRecorder, RecordsPushesAndPopsWithTheirResults) {
    Recorder rec(1);
    auto& log = rec.log(0);

    EXPECT_TRUE(log.push(7, [] { return true; }));
    EXPECT_EQ(log.pop([] { return std::optional<std::int64_t>(7); }), 7);
    EXPECT_EQ(log.pop([] { return std::optional<std::int64_t>(); }), std::nullopt);

    History h = rec.merge();
    ASSERT_EQ(h.size(), 3u);
    EXPECT_EQ(h[0].kind, OpKind::push);
    EXPECT_EQ(h[0].value, 7);
    EXPECT_EQ(h[1].kind, OpKind::pop);
    EXPECT_EQ(h[1].result, 7);
    EXPECT_EQ(h[2].kind, OpKind::pop);
    EXPECT_EQ(h[2].result, std::nullopt);
    EXPECT_EQ(h.check_well_formed(), std::nullopt);
}

TEST(LinRecorder, FailedPushIsNotRecorded) {
    Recorder rec(1);
    EXPECT_FALSE(rec.log(0).push(1, [] { return false; }));
    EXPECT_TRUE(rec.merge().empty());
}

TEST(LinRecorder, StampsBracketTheCallAndIncreaseOnOneThread) {
    Recorder rec(1);
    auto& log = rec.log(0);
    std::uint64_t inside = 0;
    log.push(1, [&] {
        inside = log.now();
        return true;
    });
    log.push(2, [] { return true; });

    History h = rec.merge();
    ASSERT_EQ(h.size(), 2u);
    EXPECT_LE(h[0].invoke_ts, inside);
    EXPECT_LE(inside, h[0].response_ts);
    EXPECT_LE(h[0].response_ts, h[1].invoke_ts);
}

TEST(LinRecorder, TagsOpsWithTheirThreadAndMergesInInvokeOrder) {
    Recorder rec(3);
    // Interleave by hand so the merge has to reorder across buffers.
    rec.log(2).push(20, [] { return true; });
    rec.log(0).push(0, [] { return true; });
    rec.log(1).push(10, [] { return true; });
    rec.log(2).push(21, [] { return true; });

    History h = rec.merge();
    ASSERT_EQ(h.size(), 4u);
    const std::uint32_t threads[] = {2, 0, 1, 2};
    const std::int64_t values[] = {20, 0, 10, 21};
    for (std::size_t i = 0; i < h.size(); ++i) {
        EXPECT_EQ(h[i].thread, threads[i]) << i;
        EXPECT_EQ(h[i].value, values[i]) << i;
        if (i > 0) EXPECT_LE(h[i - 1].invoke_ts, h[i].invoke_ts) << i;
    }
}

// The one property the checker depends on: if one op really finished before
// another started (here, ordered through an acquire/release flag), the stamps
// must never claim the opposite order. Equal stamps are fine; they read as
// overlapping.
TEST(LinRecorder, StampsNeverContradictRealTimeOrderAcrossThreads) {
    constexpr int rounds = 200;
    Recorder rec(2, rounds);
    std::atomic<int> turn{0};

    std::thread a([&] {
        for (int r = 0; r < rounds; ++r) {
            while (turn.load(std::memory_order_acquire) != 2 * r) std::this_thread::yield();
            rec.log(0).push(r, [] { return true; });
            turn.store(2 * r + 1, std::memory_order_release);
        }
    });
    std::thread b([&] {
        for (int r = 0; r < rounds; ++r) {
            while (turn.load(std::memory_order_acquire) != 2 * r + 1) std::this_thread::yield();
            rec.log(1).pop([r] { return std::optional<std::int64_t>(r); });
            turn.store(2 * r + 2, std::memory_order_release);
        }
    });
    a.join();
    b.join();

    History h = rec.merge();
    ASSERT_EQ(h.size(), 2u * rounds);
    ASSERT_EQ(h.check_well_formed(), std::nullopt);
    std::vector<const lockfree::lin::Operation*> pushes(rounds), pops(rounds);
    for (const auto& op : h.ops()) {
        if (op.kind == OpKind::push) pushes[static_cast<std::size_t>(op.value)] = &op;
        else pops[static_cast<std::size_t>(*op.result)] = &op;
    }
    for (int r = 0; r < rounds; ++r) {
        const auto i = static_cast<std::size_t>(r);
        EXPECT_FALSE(happens_before(*pops[i], *pushes[i])) << r;
        if (r > 0) EXPECT_FALSE(happens_before(*pushes[i], *pops[i - 1])) << r;
    }
}

class LinRecorderStress : public lockfree::test::StressTest {};

// Record a real MPMC run. This doesn't check linearizability yet (that's the
// checker's job); it checks the recorder hands it a sane history: well formed,
// nothing lost, and every popped value was actually pushed exactly once.
TEST_F(LinRecorderStress, RecordsAConcurrentMpmcRun) {
    constexpr std::size_t threads = 4;
    const std::uint64_t per_thread = lockfree::test::scaled(2000);
    lockfree::MpmcQueue<std::int64_t, 64> q;
    Recorder rec(threads, 2 * per_thread);
    std::vector<std::uint64_t> pushed(threads), popped(threads);

    lockfree::test::run_stress(threads, [&](std::size_t i, lockfree::test::StressRng& rng) {
        auto& log = rec.log(i);
        for (std::uint64_t k = 0; k < per_thread; ++k) {
            // Unique across threads: thread in the high bits.
            const auto v = static_cast<std::int64_t>((i << 32) | k);
            if (log.push(v, [&] { return q.try_push(v); })) ++pushed[i];
            lockfree::test::maybe_delay(rng);
            if (log.pop([&] { return q.try_pop(); })) ++popped[i];
        }
    });

    // Drain on one thread so every pushed value shows up in the history.
    Recorder drain(1);
    while (drain.log(0).pop([&] { return q.try_pop(); })) {
    }

    History h = rec.merge();
    ASSERT_EQ(h.check_well_formed(), std::nullopt);

    std::uint64_t pushes = 0, pops = 0;
    for (std::size_t i = 0; i < threads; ++i) {
        pushes += pushed[i];
        pops += popped[i];
    }
    std::set<std::int64_t> pushed_values, popped_values;
    std::uint64_t empty_pops = 0, value_pops = 0;
    for (const auto& op : h.ops()) {
        if (op.kind == OpKind::push) {
            EXPECT_TRUE(pushed_values.insert(op.value).second) << to_string(op);
        } else if (op.result) {
            ++value_pops;
            EXPECT_TRUE(popped_values.insert(*op.result).second) << to_string(op);
        } else {
            ++empty_pops;
        }
    }
    // Keep the History alive: looping over merge().ops() directly would
    // iterate a vector owned by a temporary that's already gone.
    const History drained = drain.merge();
    for (const auto& op : drained.ops()) {
        if (op.result) EXPECT_TRUE(popped_values.insert(*op.result).second);
    }
    EXPECT_EQ(pushed_values.size(), pushes);
    EXPECT_EQ(value_pops, pops);
    EXPECT_EQ(value_pops + empty_pops, threads * per_thread);
    EXPECT_EQ(popped_values, pushed_values);
}
