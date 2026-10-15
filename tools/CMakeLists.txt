# Verification tooling.
#   scheduler        - deterministic interleaving controller ("mini-Loom"):
#                      seed-driven replay, shrinking, coverage output.
#   linearizability  - offline checker over timestamped operation logs.

add_library(lockfree_scheduler STATIC
  scheduler/scheduler.cpp
  scheduler/shrink.cpp
  scheduler/coverage.cpp)
target_include_directories(lockfree_scheduler PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lockfree_scheduler PUBLIC lockfree)

add_library(lockfree_linearizability STATIC
  linearizability/checker.cpp)
target_include_directories(lockfree_linearizability PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lockfree_linearizability PUBLIC lockfree)
