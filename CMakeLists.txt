cmake_minimum_required(VERSION 3.20)
project(lockfree LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(LOCKFREE_BUILD_TESTS "Build unit + stress tests" ON)
option(LOCKFREE_BUILD_BENCH  "Build benchmarks" ON)
option(LOCKFREE_BUILD_TOOLS  "Build tools (scheduler, linearizability checker)" ON)
set(LOCKFREE_SANITIZER "none" CACHE STRING "Sanitizer: none|thread|address|undefined")
set_property(CACHE LOCKFREE_SANITIZER PROPERTY STRINGS none thread address undefined)

# ---- Header-only core library ------------------------------------------------
add_library(lockfree INTERFACE)
target_include_directories(lockfree INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(lockfree INTERFACE cxx_std_20)

find_package(Threads REQUIRED)
target_link_libraries(lockfree INTERFACE Threads::Threads)

# ---- Sanitizers (run the suite on ARM to catch missing acquire/release) -------
if(NOT LOCKFREE_SANITIZER STREQUAL "none")
  add_compile_options(-fsanitize=${LOCKFREE_SANITIZER} -g -O1 -fno-omit-frame-pointer)
  add_link_options(-fsanitize=${LOCKFREE_SANITIZER})
endif()

if(LOCKFREE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

if(LOCKFREE_BUILD_BENCH)
  add_subdirectory(bench)
endif()

if(LOCKFREE_BUILD_TOOLS)
  add_subdirectory(tools)
endif()
