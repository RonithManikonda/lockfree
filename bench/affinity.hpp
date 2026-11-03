#pragma once

// Best-effort thread pinning for benchmarks.
//
// Linux: pthread_setaffinity_np binds the calling thread to one CPU; the
// scheduler will not move it.
//
// macOS has no way to bind a thread to a core. The closest thing is
// THREAD_AFFINITY_POLICY, which is only a hint: threads with the same tag
// are placed to share an L2, threads with different tags are spread apart.
// On Apple Silicon the kernel doesn't implement it at all and returns
// KERN_NOT_SUPPORTED, so there pin_current_thread() reports false and the
// benches simply run unpinned. Numbers from a Mac therefore include whatever
// the scheduler decides, including putting a thread on an efficiency core.
//
// LOCKFREE_BENCH_NO_PIN=1 turns pinning off everywhere, to compare.

#include <cstdlib>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <mach/thread_policy.h>
#endif

namespace lockfree::bench {

inline bool pinning_disabled() noexcept {
    const char* v = std::getenv("LOCKFREE_BENCH_NO_PIN");
    return v != nullptr && *v != '\0' && *v != '0';
}

// Pins (Linux) or hints (macOS) the calling thread onto `cpu`. Returns true
// only if the OS accepted the request.
inline bool pin_current_thread(int cpu) noexcept {
    if (pinning_disabled() || cpu < 0) return false;
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#elif defined(__APPLE__)
    // Tag 0 means "no affinity", so shift by one. Distinct tags ask for
    // distinct L2s, which is the opposite of what a producer/consumer pair
    // wants for latency, but it's the only knob there is.
    thread_affinity_policy_data_t policy{cpu + 1};
    const mach_port_t thread = mach_thread_self();
    const kern_return_t kr = thread_policy_set(
        thread, THREAD_AFFINITY_POLICY, reinterpret_cast<thread_policy_t>(&policy),
        THREAD_AFFINITY_POLICY_COUNT);
    mach_port_deallocate(mach_task_self(), thread);
    return kr == KERN_SUCCESS;
#else
    return false;
#endif
}

} // namespace lockfree::bench
