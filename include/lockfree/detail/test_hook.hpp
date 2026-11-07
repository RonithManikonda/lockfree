#pragma once

// Named points inside an algorithm where a test can run code of its own, so a
// race that needs an exact interleaving can be replayed on one thread:
//
//   ScopedTestHook hook([&](std::string_view where) {
//       if (where == "treiber_pop_before_cas") { /* act as another thread */ }
//   });
//
// LOCKFREE_TEST_HOOK(name) compiles to nothing unless LOCKFREE_TEST_HOOKS is
// defined (the test targets define it), so release code pays nothing.
//
// The callback is per thread: other threads in a stress test never see it.
// It's also switched off while it runs, so the structure's own operations
// called from inside the callback don't re-enter it.

#if defined(LOCKFREE_TEST_HOOKS)

#include <functional>
#include <string_view>
#include <utility>

namespace lockfree::detail {

using TestHookFn = std::function<void(std::string_view)>;

inline TestHookFn& current_test_hook() noexcept {
    thread_local TestHookFn hook;
    return hook;
}

inline void run_test_hook(std::string_view where) {
    TestHookFn& slot = current_test_hook();
    if (!slot) return;
    TestHookFn fn = std::exchange(slot, nullptr);
    fn(where);
    slot = std::move(fn);
}

// Installs a callback on this thread for the lifetime of the object.
class ScopedTestHook {
public:
    explicit ScopedTestHook(TestHookFn fn) : previous_(std::exchange(current_test_hook(), std::move(fn))) {}
    ~ScopedTestHook() { current_test_hook() = std::move(previous_); }
    ScopedTestHook(const ScopedTestHook&) = delete;
    ScopedTestHook& operator=(const ScopedTestHook&) = delete;

private:
    TestHookFn previous_;
};

} // namespace lockfree::detail

#define LOCKFREE_TEST_HOOK(name) ::lockfree::detail::run_test_hook(name)

#else

#define LOCKFREE_TEST_HOOK(name) ((void)0)

#endif
