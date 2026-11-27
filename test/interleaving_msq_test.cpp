#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <lockfree/michael_scott_queue.hpp>

#include "scheduler/explore.hpp"
#include "scheduler/pct.hpp"

// The Michael-Scott queue under the scheduler.
//
// The real reclaimers run here too (every schedule has to conserve values),
// but a missing protect step doesn't break conservation: the freed node is
// read, not corrupted, and the run usually carries on. ASan sees that, the
// scheduler alone doesn't. So the mutation tests use a small instrumented
// hazard-pointer scheme instead: "freeing" a node only records its address
// range, and a strategy wrapper flags every atomic access that lands inside a
// freed node. Every node read the queue does goes through `next`, an atomic,
// so a use-after-free is always a yield point we can look at.
//
// With that, dropping the re-read in protect() (publish the hazard, but don't
// check the source still holds the pointer) is found within a few hundred
// schedules by PCT, and the real protect survives the same budget.

using lockfree::MichaelScottQueue;
using lockfree::sched::Case;
using lockfree::sched::CaseFactory;
using lockfree::sched::Decision;
using lockfree::sched::ExploreResult;
using lockfree::sched::Outcome;
using lockfree::sched::PctStrategy;
using lockfree::sched::RandomStrategy;
using lockfree::sched::require;
using lockfree::sched::SeededStrategy;
using lockfree::sched::ThreadId;

namespace {

constexpr std::size_t kBudget = 3000;
constexpr std::uint64_t kSeed = 0x5eed'45;
constexpr std::size_t kThreads = 2;

// --- Instrumented hazard pointers ----------------------------------------

// Per-case state: kThreads pairs of hazard slots, the retired list, and the
// address ranges of the nodes "freed" so far. Nothing is really deleted until
// the domain goes away, so a buggy schedule reads stale memory, not unmapped
// memory, and the run stays well defined.
struct CheckedDomain {
    std::array<std::array<lockfree::detail::atomic<void*>, 2>, kThreads> hazards{};

    struct Retired {
        void* p;
        std::size_t size;
        void (*destroy)(void*);
        bool freed;
    };
    std::vector<Retired> retired;

    ~CheckedDomain() {
        for (auto& r : retired) r.destroy(r.p);
    }

    bool is_freed(const void* addr) const noexcept {
        auto a = reinterpret_cast<std::uintptr_t>(addr);
        for (const auto& r : retired) {
            auto base = reinterpret_cast<std::uintptr_t>(r.p);
            if (r.freed && a >= base && a < base + r.size) return true;
        }
        return false;
    }

    // Scans on every retire, so a node is freed at the first moment the
    // scheme allows. That's the most hostile reclaimer possible, which is
    // what we want when looking for a missing protect.
    //
    // By index, not by reference: every hazard load is a yield point, and the
    // thread that runs meanwhile may retire too and grow the vector.
    void scan() {
        for (std::size_t i = 0; i < retired.size(); ++i) {
            if (retired[i].freed) continue;
            const void* p = retired[i].p;
            bool hazardous = false;
            for (auto& pair : hazards)
                for (auto& h : pair)
                    if (h.load(std::memory_order_acquire) == p) hazardous = true;
            retired[i].freed = !hazardous;
        }
    }
};

// One case at a time (the scheduler allows one run per process), so the
// policy reaches its domain and the logical thread id through globals that
// each case sets up.
CheckedDomain* g_domain = nullptr;
thread_local std::size_t tl_thread = 0;

// What the wrapper below found during the current run.
struct UafReport {
    bool hit = false;
    ThreadId thread = 0;
    const char* file = "";
    std::uint_least32_t line = 0;
};
UafReport g_uaf;

template <bool Validate>
struct CheckedHazardReclaim {
    class Guard {
    public:
        Guard() : slots_(g_domain->hazards[tl_thread]) {}
        ~Guard() {
            for (auto& h : slots_) h.store(nullptr, std::memory_order_release);
        }
        Guard(const Guard&) = delete;
        Guard& operator=(const Guard&) = delete;

        template <typename T>
        T* protect(std::size_t slot, const lockfree::detail::atomic<T*>& src) noexcept {
            T* p = src.load(std::memory_order_acquire);
            for (;;) {
                slots_[slot].store(p, std::memory_order_release);
                // The step under test. Without it, p may have been unlinked
                // and freed between the load above and the publish.
                if constexpr (!Validate) return p;
                T* again = src.load(std::memory_order_acquire);
                if (again == p) return p;
                p = again;
            }
        }

    private:
        std::array<lockfree::detail::atomic<void*>, 2>& slots_;
    };

    template <typename T>
    static void retire(T* p) {
        g_domain->retired.push_back({p, sizeof(T), [](void* q) { delete static_cast<T*>(q); }, false});
        g_domain->scan();
    }
};

// Passes every choice through to `inner` and checks the access the current
// thread is about to make. Strategies can't throw, so a hit is only noted;
// the case's check turns it into a failure.
class UafDetector : public SeededStrategy {
public:
    explicit UafDetector(SeededStrategy& inner) noexcept : inner_(inner) {}

    void reseed(std::uint64_t seed) noexcept override { inner_.reseed(seed); }
    const char* name() const noexcept override { return inner_.name(); }

    void begin_run(std::size_t thread_count) override {
        g_uaf = {};
        inner_.begin_run(thread_count);
    }

    ThreadId pick(const Decision& d) override {
        if (d.point != nullptr && !g_uaf.hit && g_domain != nullptr &&
            g_domain->is_freed(d.point->address))
            g_uaf = {true, *d.current, d.point->where.file_name(), d.point->where.line()};
        return inner_.pick(d);
    }

private:
    SeededStrategy& inner_;
};

// --- Scenarios ------------------------------------------------------------

template <typename Reclaimer>
struct Shared {
    std::shared_ptr<CheckedDomain> domain; // only for the checked policies
    MichaelScottQueue<int, Reclaimer> q;
    std::optional<int> t0_pop;
    std::optional<int> t1_pops[2];
};

// The queue starts as [1, 2]. T0 pops once; T1 pops twice and pushes 3.
// For a missing re-validation, T0 has to load head_, stop before its hazard
// is published, and stay stopped while T1 dequeues (retiring and, here,
// immediately freeing that head). T0 then reads the freed node's next.
//
// Check: no access to a freed node, and 1, 2, 3 each come out exactly once
// between the pops and a final drain.
template <typename Reclaimer, bool Checked>
Case pop_race() {
    auto sh = std::make_shared<Shared<Reclaimer>>();
    if constexpr (Checked) {
        // The domain must be live before the queue's first push (it retires
        // nothing, but the guard claims slots).
        sh->domain = std::make_shared<CheckedDomain>();
        g_domain = sh->domain.get();
    }
    tl_thread = 0; // the calling thread borrows T0's slots; nothing runs yet
    sh->q.push(1);
    sh->q.push(2);

    auto t0 = [sh] {
        tl_thread = 0;
        sh->t0_pop = sh->q.try_pop();
    };
    auto t1 = [sh] {
        tl_thread = 1;
        sh->t1_pops[0] = sh->q.try_pop();
        sh->t1_pops[1] = sh->q.try_pop();
        sh->q.push(3);
    };
    auto check = [sh] {
        if constexpr (Checked) {
            if (g_uaf.hit) {
                static std::string msg;
                std::string file = g_uaf.file;
                file = file.substr(file.find_last_of('/') + 1);
                msg = "use after free: T" + std::to_string(g_uaf.thread) + " touched a freed node at " + file +
                      ":" + std::to_string(g_uaf.line);
                require(false, msg.c_str());
            }
        }
        tl_thread = 0;
        std::vector<int> seen;
        auto note = [&](const std::optional<int>& v) {
            if (v) seen.push_back(*v);
        };
        note(sh->t0_pop);
        for (const auto& v : sh->t1_pops) note(v);
        for (int i = 0; i < 8; ++i) {
            auto v = sh->q.try_pop();
            if (!v) break;
            seen.push_back(*v);
        }
        std::sort(seen.begin(), seen.end());
        require(seen == std::vector<int>{1, 2, 3}, "values not conserved");
    };
    return Case{{t0, t1}, check};
}

// Two producers and a consumer on an empty queue, for the real policies:
// enqueue's tail helping and dequeue's lagging-tail path both get exercised
// (the consumer can find head == tail with a successor). The consumer's pops
// may come up empty; whatever they miss has to be in the final drain, and
// each producer's values must come out in its push order.
template <typename Reclaimer>
Case producers_and_consumer() {
    auto sh = std::make_shared<Shared<Reclaimer>>();
    auto consumed = std::make_shared<std::vector<int>>();
    auto producer = [sh](int base) {
        return [sh, base] {
            sh->q.push(base + 1);
            sh->q.push(base + 2);
        };
    };
    auto consumer = [sh, consumed] {
        for (int i = 0; i < 3; ++i)
            if (auto v = sh->q.try_pop()) consumed->push_back(*v);
    };
    auto check = [sh, consumed] {
        std::vector<int> order = *consumed;
        while (auto v = sh->q.try_pop()) order.push_back(*v);
        auto pos = [&](int v) { return std::find(order.begin(), order.end(), v) - order.begin(); };
        require(pos(11) < pos(12) && pos(21) < pos(22), "a producer's values came out of order");
        std::sort(order.begin(), order.end());
        require(order == std::vector<int>{11, 12, 21, 22}, "values not conserved");
    };
    return Case{{producer(10), producer(20), consumer}, check};
}

ExploreResult explore_with(const CaseFactory& factory, SeededStrategy& strategy) {
    auto result = lockfree::sched::explore(factory, strategy, kBudget, kSeed);
    std::printf("[%s] %zu run%s, %s\n", strategy.name(), result.runs, result.runs == 1 ? "" : "s",
                result.ok() ? "no failure" : result.failure.failure.c_str());
    return result;
}

class MsqInterleaving : public ::testing::Test {
protected:
    void TearDown() override { g_domain = nullptr; }
};

TEST_F(MsqInterleaving, PctCatchesProtectWithoutRevalidation) {
    PctStrategy pct(0);
    UafDetector detect(pct);
    auto factory = pop_race<CheckedHazardReclaim<false>, true>;
    auto result = explore_with(factory, detect);
    ASSERT_FALSE(result.ok()) << "missing re-validation not caught in " << kBudget << " PCT schedules";
    EXPECT_NE(result.failure.failure.find("use after free"), std::string::npos) << result.failure.failure;

    // The seed alone brings it back, detector included.
    auto again = lockfree::sched::replay(factory, detect, *result.failing_seed);
    EXPECT_EQ(again.outcome, Outcome::failed);
    EXPECT_EQ(again.choices, result.failure.choices);
}

TEST_F(MsqInterleaving, RandomWalkForComparison) {
    // Printed next to PCT's count, not asserted. The window is one preemption
    // between T0's load of head_ and its hazard publish, but T0 then has to
    // stay parked through all of T1's pop, which with this reclaimer's
    // retire-time scan is a few dozen yield points; coin flips rarely manage
    // that (none in 3000 runs when this was written). PCT needs one change
    // point in the window and a lower priority for T0 afterwards.
    RandomStrategy random(0);
    UafDetector detect(random);
    auto result = explore_with(pop_race<CheckedHazardReclaim<false>, true>, detect);
    if (!result.ok()) EXPECT_NE(result.failure.failure.find("use after free"), std::string::npos);
}

TEST_F(MsqInterleaving, ValidatedProtectSurvivesTheBudget) {
    for (bool use_pct : {false, true}) {
        RandomStrategy random(0);
        PctStrategy pct(0);
        SeededStrategy& inner = use_pct ? static_cast<SeededStrategy&>(pct) : random;
        UafDetector detect(inner);
        auto result = explore_with(pop_race<CheckedHazardReclaim<true>, true>, detect);
        EXPECT_TRUE(result.ok()) << result.failure.failure << " (replay seed "
                                 << result.failing_seed.value_or(0) << ")";
        EXPECT_EQ(result.runs, kBudget);
    }
}

template <typename Reclaimer>
void survives(const CaseFactory& factory) {
    RandomStrategy random(0);
    PctStrategy pct(0);
    for (SeededStrategy* s : {static_cast<SeededStrategy*>(&random), static_cast<SeededStrategy*>(&pct)}) {
        auto result = explore_with(factory, *s);
        EXPECT_TRUE(result.ok()) << result.failure.failure << " (replay seed "
                                 << result.failing_seed.value_or(0) << ")";
        EXPECT_EQ(result.runs, kBudget);
    }
}

TEST_F(MsqInterleaving, RealHazardPointersSurvive) {
    survives<lockfree::HazardPointerReclaim>(pop_race<lockfree::HazardPointerReclaim, false>);
    survives<lockfree::HazardPointerReclaim>(producers_and_consumer<lockfree::HazardPointerReclaim>);
}

TEST_F(MsqInterleaving, RealEpochsSurvive) {
    survives<lockfree::EpochReclaim>(pop_race<lockfree::EpochReclaim, false>);
    survives<lockfree::EpochReclaim>(producers_and_consumer<lockfree::EpochReclaim>);
}

} // namespace
