# Unit + stress tests (GTest). Fetched if not found on the system.
include(FetchContent)
find_package(GTest QUIET)
if(NOT GTest_FOUND)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2)
  FetchContent_MakeAvailable(googletest)
endif()

set(LOCKFREE_TESTS
  test_spsc_ring_buffer
  test_mpsc_queue
  test_mpmc_queue
  test_atomic_stack
  test_hazard_pointer
  test_epoch
  test_michael_scott_queue
  test_seqlock
  test_locks
  interleaving_test
  test_latency_histogram
  test_atomic_backoff
  test_atomic_shim
  test_lin_history
  test_lin_recorder
  test_lin_checker
  test_lin_catalog
  test_lin_stress
  test_lin_log
)

foreach(test ${LOCKFREE_TESTS})
  add_executable(${test} ${test}.cpp)
  target_link_libraries(${test} PRIVATE lockfree GTest::gtest_main)
  target_include_directories(${test} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  # Turns LOCKFREE_TEST_HOOK(name) points into callbacks tests can install.
  target_compile_definitions(${test} PRIVATE LOCKFREE_TEST_HOOKS)
  add_test(NAME ${test} COMMAND ${test})
endforeach()

# The latency histogram lives with the benches but is plain logic worth testing.
target_include_directories(test_latency_histogram PRIVATE ${PROJECT_SOURCE_DIR}/bench)

# Linearizability history model, specs, recorder and checker (header-only under tools/).
# `ctest -L linearizability` runs just these.
foreach(test test_lin_history test_lin_recorder test_lin_checker test_lin_catalog test_lin_stress
            test_lin_log)
  target_include_directories(${test} PRIVATE ${PROJECT_SOURCE_DIR}/tools)
  set_tests_properties(${test} PROPERTIES LABELS linearizability)
endforeach()

# The same shim test with LOCKFREE_SCHEDULER, where detail::atomic becomes the
# yielding wrapper.
add_executable(test_atomic_shim_sched test_atomic_shim.cpp)
target_link_libraries(test_atomic_shim_sched PRIVATE lockfree GTest::gtest_main)
target_compile_definitions(test_atomic_shim_sched PRIVATE LOCKFREE_SCHEDULER)
add_test(NAME test_atomic_shim_sched COMMAND test_atomic_shim_sched)

# The structures' own suites again, with every atomic access going through the
# yielding wrapper (no hook installed, so it only adds a null check). Shows the
# port didn't change behaviour; test_shim_ported checks the yields themselves.
foreach(test test_spsc_ring_buffer test_mpmc_queue test_mpsc_queue test_atomic_stack
            test_shim_ported)
  if(test STREQUAL "test_shim_ported")
    set(target ${test})
  else()
    set(target ${test}_sched)
  endif()
  add_executable(${target} ${test}.cpp)
  target_link_libraries(${target} PRIVATE lockfree GTest::gtest_main)
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(${target} PRIVATE LOCKFREE_SCHEDULER LOCKFREE_TEST_HOOKS)
  add_test(NAME ${target} COMMAND ${target})
endforeach()

# Scheduler core (baton handoff, step limit), random/PCT exploration and shrinking. Links the scheduler
# library, which brings LOCKFREE_SCHEDULER with it.
if(LOCKFREE_BUILD_TOOLS)
  foreach(test test_scheduler test_explore test_pct test_shrink)
    add_executable(${test} ${test}.cpp)
    target_link_libraries(${test} PRIVATE lockfree_scheduler GTest::gtest_main)
    add_test(NAME ${test} COMMAND ${test})
    set_tests_properties(${test} PROPERTIES LABELS scheduler)
  endforeach()
endif()
