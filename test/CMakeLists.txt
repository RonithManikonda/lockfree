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
)

foreach(test ${LOCKFREE_TESTS})
  add_executable(${test} ${test}.cpp)
  target_link_libraries(${test} PRIVATE lockfree GTest::gtest_main)
  target_include_directories(${test} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME ${test} COMMAND ${test})
endforeach()
