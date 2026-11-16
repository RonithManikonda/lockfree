#pragma once
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "lockfree/detail/cache_line.hpp"
#include "linearizability/history.hpp"

// Records a History while a stress test runs.
//
//   Recorder rec(threads);
//   run_stress(threads, [&](std::size_t i, StressRng&) {
//       auto& log = rec.log(i);
//       log.push(v, [&] { return q.try_push(v); });
//       log.pop([&] { return q.try_pop(); });
//   });
//   History h = rec.merge();
//
// Each thread appends only to its own buffer, so recording adds no shared
// writes (and no extra synchronization) to the run being observed. A shared
// buffer would serialize the threads on its lock or its index and hide
// exactly the interleavings we want to see. Buffers sit on separate cache
// lines so neighbouring threads' appends don't false-share.
//
// merge() reads every buffer, so call it only after the recording threads
// have been joined (run_stress returning is enough).
namespace lockfree::lin {

class Recorder {
public:
    class ThreadLog {
    public:
        // Wraps one push. fn() does the push and returns whether it took
        // effect; a push that found the structure full isn't recorded (see
        // Operation::result). Returns fn()'s result.
        template <typename Fn>
        bool push(std::int64_t value, Fn&& fn) {
            const std::uint64_t invoke = now();
            const bool ok = std::forward<Fn>(fn)();
            const std::uint64_t response = now();
            if (ok) ops_.push_back({thread_, OpKind::push, value, std::nullopt, invoke, response});
            return ok;
        }

        // Wraps one pop. fn() returns the popped value or nullopt for empty;
        // both outcomes are recorded, since an empty pop is a claim the
        // checker has to verify too.
        template <typename Fn>
        std::optional<std::int64_t> pop(Fn&& fn) {
            const std::uint64_t invoke = now();
            const std::optional<std::int64_t> result = std::forward<Fn>(fn)();
            const std::uint64_t response = now();
            ops_.push_back({thread_, OpKind::pop, 0, result, invoke, response});
            return result;
        }

        std::uint32_t thread() const noexcept { return thread_; }
        std::size_t size() const noexcept { return ops_.size(); }
        void reserve(std::size_t n) { ops_.reserve(n); }

        // Nanoseconds since the recorder was created, on the clock every
        // thread shares.
        //
        // The fences keep the operation's own loads and stores from moving
        // across the clock read, so [invoke, response] really brackets them.
        // That's best effort: reading the timer isn't a memory access, and
        // on ARM the counter read can still drift by a few cycles without an
        // ISB. The slack is far below one 42 ns tick of steady_clock on
        // Apple Silicon, and happens_before() treats equal stamps as
        // overlapping, so the checker never relies on an order the clock
        // can't actually tell apart.
        std::uint64_t now() const noexcept {
            std::atomic_thread_fence(std::memory_order_seq_cst);
            const auto t = std::chrono::steady_clock::now();
            std::atomic_thread_fence(std::memory_order_seq_cst);
            return static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t - *start_).count());
        }

    private:
        friend class Recorder;

        std::uint32_t thread_ = 0;
        const std::chrono::steady_clock::time_point* start_ = nullptr;
        std::vector<Operation> ops_;
    };

    // `reserve_per_thread` preallocates each buffer so the first few
    // thousand appends don't call malloc in the middle of the run.
    explicit Recorder(std::size_t threads, std::size_t reserve_per_thread = 0)
        : start_(std::chrono::steady_clock::now()), logs_(threads) {
        for (std::size_t i = 0; i < threads; ++i) {
            logs_[i].log.thread_ = static_cast<std::uint32_t>(i);
            logs_[i].log.start_ = &start_;
            logs_[i].log.reserve(reserve_per_thread);
        }
    }

    // Logs point back at start_, so the recorder stays put.
    Recorder(const Recorder&) = delete;
    Recorder& operator=(const Recorder&) = delete;

    std::size_t threads() const noexcept { return logs_.size(); }

    // Only thread `index` may touch log(index) while recording.
    ThreadLog& log(std::size_t index) noexcept { return logs_[index].log; }

    // Every recorded op from every thread, in invocation order.
    History merge() const {
        std::size_t total = 0;
        for (const Slot& s : logs_) total += s.log.ops_.size();
        std::vector<Operation> all;
        all.reserve(total);
        for (const Slot& s : logs_) all.insert(all.end(), s.log.ops_.begin(), s.log.ops_.end());
        History h(std::move(all));
        h.sort_by_invoke();
        return h;
    }

private:
    struct alignas(detail::cache_line_size) Slot {
        ThreadLog log;
    };

    std::chrono::steady_clock::time_point start_;
    std::vector<Slot> logs_;
};

} // namespace lockfree::lin
