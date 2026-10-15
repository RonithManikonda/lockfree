#pragma once

#include <cstddef>

namespace lockfree::detail {

// L1 line size on effectively every x86_64 and ARM64 core, including Apple
// Silicon and Graviton. Apple Silicon's L2 prefetcher operates on 128-byte
// pairs, so if profiling on M-series ever shows two "separated" atomics still
// contending, doubling this is the first thing to try.
inline constexpr std::size_t cache_line_size = 64;

} // namespace lockfree::detail
