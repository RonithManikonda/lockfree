# Benchmarks (Google Benchmark). Fetched if not found.
# bench_reference also compares against moodycamel::ConcurrentQueue and
# boost::lockfree when those are available (guarded by compile definitions).
include(FetchContent)
find_package(benchmark QUIET)
if(NOT benchmark_FOUND)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.0)
  FetchContent_MakeAvailable(googlebenchmark)
endif()

set(LOCKFREE_BENCHMARKS
  bench_spsc
  bench_mpsc
  bench_mpmc
  bench_atomic_stack
  bench_reference
)

foreach(bench ${LOCKFREE_BENCHMARKS})
  add_executable(${bench} ${bench}.cpp)
  target_link_libraries(${bench} PRIVATE lockfree benchmark::benchmark)
endforeach()
