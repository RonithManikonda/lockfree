# Verification tooling.
#   scheduler        - deterministic interleaving controller ("mini-Loom"):
#                      seed-driven replay, shrinking, coverage output.
#   linearizability  - offline checker over timestamped operation logs.

add_library(lockfree_scheduler STATIC
  scheduler/scheduler.cpp
  scheduler/explore.cpp
  scheduler/pct.cpp
  scheduler/shrink.cpp
  scheduler/coverage.cpp
  scheduler/exhaustive.cpp)
target_include_directories(lockfree_scheduler PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lockfree_scheduler PUBLIC lockfree)
# The scheduler only exists for the yielding detail::atomic, so anything that
# links it gets that build of the structures too.
target_compile_definitions(lockfree_scheduler PUBLIC LOCKFREE_SCHEDULER)

add_library(lockfree_linearizability STATIC
  linearizability/checker.cpp)
target_include_directories(lockfree_linearizability PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lockfree_linearizability PUBLIC lockfree)

# Command-line front end: lin_check [--spec queue|stack] FILE
add_executable(lin_check linearizability/lin_check.cpp)
target_link_libraries(lin_check PRIVATE lockfree_linearizability)

# lin_check end to end on the sample logs in test/data/lin.
if(LOCKFREE_BUILD_TESTS)
  add_test(NAME lin_check_accepts_a_linearizable_log
           COMMAND lin_check ${PROJECT_SOURCE_DIR}/test/data/lin/queue_ok.log)
  # Exit status 1 is the expected answer here, so match the output instead.
  add_test(NAME lin_check_shrinks_a_reordered_queue
           COMMAND lin_check ${PROJECT_SOURCE_DIR}/test/data/lin/queue_reordered.log)
  set_tests_properties(lin_check_shrinks_a_reordered_queue PROPERTIES
    PASS_REGULAR_EXPRESSION "minimal failing window \\(4 ops")
  add_test(NAME lin_check_rejects_a_bad_log COMMAND lin_check --spec stack /nonexistent.log)
  set_tests_properties(lin_check_rejects_a_bad_log PROPERTIES WILL_FAIL TRUE)
  set_tests_properties(lin_check_accepts_a_linearizable_log lin_check_shrinks_a_reordered_queue
    lin_check_rejects_a_bad_log PROPERTIES LABELS linearizability)
endif()
