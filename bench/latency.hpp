#pragma once

// Latency histogram for per-operation samples, reported as percentiles.
//
// Buckets are log-linear, the same idea as HdrHistogram: values below 64 get
// one bucket each, and every power of two above that is split into 32 equal
// sub-buckets. So the bucket width is at most 1/32 of the value (~3%), the
// table is a fixed ~2000 counters no matter how many samples go in, and
// record() is a couple of shifts and an increment, cheap enough to sit inside
// the timed loop.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace lockfree::bench {

class LatencyHistogram {
    static constexpr unsigned sub_bits = 5;
    static constexpr std::size_t sub_buckets = std::size_t{1} << sub_bits;  // 32
    static constexpr std::uint64_t linear_limit = 2 * sub_buckets;         // 64

public:
    // shift ranges over 1..58 for 64-bit values.
    static constexpr std::size_t bucket_count = linear_limit + 58 * sub_buckets;

    void record(std::uint64_t value) noexcept {
        ++counts_[bucket_of(value)];
        ++total_;
        max_ = std::max(max_, value);
    }

    std::uint64_t count() const noexcept { return total_; }
    std::uint64_t max() const noexcept { return max_; }

    // Smallest bucket upper bound such that at least `p` (0..1] of the
    // samples are <= it. Reporting the upper bound errs on the slow side,
    // which is the honest direction for a latency number. Capped at the
    // exact max so p = 1 is the true maximum. 0 if empty.
    std::uint64_t percentile(double p) const noexcept {
        if (total_ == 0) return 0;
        const auto target = static_cast<std::uint64_t>(std::ceil(p * static_cast<double>(total_)));
        const std::uint64_t rank = std::clamp<std::uint64_t>(target, 1, total_);
        std::uint64_t seen = 0;
        for (std::size_t i = 0; i < bucket_count; ++i) {
            seen += counts_[i];
            if (seen >= rank) return std::min(upper_bound_of(i), max_);
        }
        return max_;
    }

    void reset() noexcept {
        counts_.fill(0);
        total_ = 0;
        max_ = 0;
    }

    static constexpr std::size_t bucket_of(std::uint64_t v) noexcept {
        if (v < linear_limit) return static_cast<std::size_t>(v);
        // shift >= 1. v >> shift lands in [32, 64): the top six bits of v.
        const unsigned shift = static_cast<unsigned>(std::bit_width(v)) - (sub_bits + 1);
        const std::uint64_t top = v >> shift;
        return linear_limit + (shift - 1) * sub_buckets + static_cast<std::size_t>(top - sub_buckets);
    }

    static constexpr std::uint64_t upper_bound_of(std::size_t bucket) noexcept {
        if (bucket < linear_limit) return bucket;
        const std::size_t k = bucket - linear_limit;
        const unsigned shift = static_cast<unsigned>(k / sub_buckets) + 1;
        const std::uint64_t top = sub_buckets + k % sub_buckets;
        // For the very last bucket (top + 1) << shift is 2^64, which wraps
        // to 0, and 0 - 1 is the right answer anyway.
        return ((top + 1) << shift) - 1;
    }

private:
    std::array<std::uint64_t, bucket_count> counts_{};
    std::uint64_t total_ = 0;
    std::uint64_t max_ = 0;
};

} // namespace lockfree::bench
