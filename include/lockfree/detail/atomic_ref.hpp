#pragma once
#include <atomic>

// lockfree::detail::atomic<T>: the atomic type every structure uses for its
// shared state.
//
// In a normal build it's an alias for std::atomic<T>, so it costs nothing and
// the structures compile exactly as before.
//
// With LOCKFREE_SCHEDULER defined it becomes a wrapper whose every load,
// store and read-modify-write first calls a yield point. The adversarial
// scheduler (tools/scheduler) installs a hook there and decides which thread
// runs next, so it can force the interleavings that matter: a preemption
// between a load and the CAS that depends on it, a producer stopped halfway
// through a push, and so on. Every atomic access is a place where another
// thread's step can change the outcome, which is exactly the set of points
// worth switching at; plain memory accesses only matter through the atomics
// that publish them.
//
// The scheduler runs one thread at a time, so it explores sequentially
// consistent interleavings only. The memory_order arguments are passed
// through unchanged (the run is still real code on real atomics), but bugs
// that need a weak-memory reordering are TSan's and the ARM runs' job.
//
// Fences (std::atomic_thread_fence) aren't wrapped; none of the structures
// synchronize through a fence alone.
#if defined(LOCKFREE_SCHEDULER)
#include <cstdint>
#include <source_location>
#include <type_traits>
#endif

namespace lockfree::detail {

#if !defined(LOCKFREE_SCHEDULER)

template <typename T>
using atomic = std::atomic<T>;

#else

enum class AtomicOp : std::uint8_t {
    load,
    store,
    exchange,
    compare_exchange,
    fetch_add,
    fetch_sub,
};

// What a yield point knows: the kind of access, the atomic it touches, and
// the source line it came from (the scheduler's coverage counts switches
// between pairs of these sites).
struct YieldPoint {
    AtomicOp op;
    const void* address;
    std::source_location where;
};

using YieldHook = void (*)(const YieldPoint&) noexcept;

// The installed hook, or nullptr. A function-local static in an inline
// function, so every translation unit shares one. Install it before starting
// the threads it should see; it isn't meant to change while they run.
inline YieldHook& yield_hook() noexcept {
    static YieldHook hook = nullptr;
    return hook;
}

inline void yield_point(AtomicOp op, const void* address, std::source_location where) noexcept {
    if (YieldHook hook = yield_hook()) hook(YieldPoint{op, address, where});
}

// Same interface as the parts of std::atomic<T> the structures use, plus a
// trailing defaulted source_location on each operation. Callers never pass
// it; the default argument is evaluated at the call site, which is what
// makes it name the line in the structure rather than this header.
template <typename T>
class atomic {
    using loc = std::source_location;

public:
    using value_type = T;
    static constexpr bool is_always_lock_free = std::atomic<T>::is_always_lock_free;

    constexpr atomic() noexcept(std::is_nothrow_default_constructible_v<T>) = default;
    constexpr atomic(T desired) noexcept : a_(desired) {}
    atomic(const atomic&) = delete;
    atomic& operator=(const atomic&) = delete;

    bool is_lock_free() const noexcept { return a_.is_lock_free(); }

    T load(std::memory_order order = std::memory_order_seq_cst,
           loc where = loc::current()) const noexcept {
        yield_point(AtomicOp::load, this, where);
        return a_.load(order);
    }

    void store(T desired, std::memory_order order = std::memory_order_seq_cst,
               loc where = loc::current()) noexcept {
        yield_point(AtomicOp::store, this, where);
        a_.store(desired, order);
    }

    T exchange(T desired, std::memory_order order = std::memory_order_seq_cst,
               loc where = loc::current()) noexcept {
        yield_point(AtomicOp::exchange, this, where);
        return a_.exchange(desired, order);
    }

    bool compare_exchange_weak(T& expected, T desired, std::memory_order success,
                               std::memory_order failure, loc where = loc::current()) noexcept {
        yield_point(AtomicOp::compare_exchange, this, where);
        return a_.compare_exchange_weak(expected, desired, success, failure);
    }

    bool compare_exchange_weak(T& expected, T desired,
                               std::memory_order order = std::memory_order_seq_cst,
                               loc where = loc::current()) noexcept {
        yield_point(AtomicOp::compare_exchange, this, where);
        return a_.compare_exchange_weak(expected, desired, order);
    }

    bool compare_exchange_strong(T& expected, T desired, std::memory_order success,
                                 std::memory_order failure, loc where = loc::current()) noexcept {
        yield_point(AtomicOp::compare_exchange, this, where);
        return a_.compare_exchange_strong(expected, desired, success, failure);
    }

    bool compare_exchange_strong(T& expected, T desired,
                                 std::memory_order order = std::memory_order_seq_cst,
                                 loc where = loc::current()) noexcept {
        yield_point(AtomicOp::compare_exchange, this, where);
        return a_.compare_exchange_strong(expected, desired, order);
    }

    // Integral types only, like std::atomic's (pointer arithmetic isn't used
    // anywhere, so it isn't offered).
    T fetch_add(T arg, std::memory_order order = std::memory_order_seq_cst,
                loc where = loc::current()) noexcept
        requires std::is_integral_v<T>
    {
        yield_point(AtomicOp::fetch_add, this, where);
        return a_.fetch_add(arg, order);
    }

    T fetch_sub(T arg, std::memory_order order = std::memory_order_seq_cst,
                loc where = loc::current()) noexcept
        requires std::is_integral_v<T>
    {
        yield_point(AtomicOp::fetch_sub, this, where);
        return a_.fetch_sub(arg, order);
    }

    // Implicit conversion and assignment are seq_cst load/store, as with
    // std::atomic. Their yield points name this header, not the caller; the
    // structures always spell out the call so this only matters for tests.
    operator T() const noexcept { return load(); }
    T operator=(T desired) noexcept {
        store(desired);
        return desired;
    }

private:
    std::atomic<T> a_;
};

#endif

} // namespace lockfree::detail
