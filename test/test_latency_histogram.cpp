#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

#include "latency.hpp"

using lockfree::bench::LatencyHistogram;

TEST(LatencyHistogram, EmptyReportsZero) {
    LatencyHistogram h;
    EXPECT_EQ(h.count(), 0u);
    EXPECT_EQ(h.percentile(0.5), 0u);
    EXPECT_EQ(h.percentile(1.0), 0u);
}

TEST(LatencyHistogram, SmallValuesAreExact) {
    LatencyHistogram h;
    for (std::uint64_t v = 1; v <= 50; ++v) h.record(v);
    EXPECT_EQ(h.count(), 50u);
    EXPECT_EQ(h.percentile(0.5), 25u);
    EXPECT_EQ(h.percentile(0.98), 49u);
    EXPECT_EQ(h.percentile(1.0), 50u);
}

TEST(LatencyHistogram, BucketsAreContiguousAndMonotonic) {
    // Every value maps to a bucket whose upper bound is >= the value, and the
    // previous bucket's bound is below it: no gaps, no overlaps.
    for (std::uint64_t v : {0ull, 1ull, 63ull, 64ull, 65ull, 127ull, 128ull, 1000ull,
                            123456789ull, 1ull << 40, (1ull << 63) + 5,
                            std::numeric_limits<std::uint64_t>::max()}) {
        const auto b = LatencyHistogram::bucket_of(v);
        ASSERT_LT(b, LatencyHistogram::bucket_count) << v;
        EXPECT_GE(LatencyHistogram::upper_bound_of(b), v) << v;
        if (b > 0) EXPECT_LT(LatencyHistogram::upper_bound_of(b - 1), v) << v;
    }
    EXPECT_EQ(LatencyHistogram::bucket_of(std::numeric_limits<std::uint64_t>::max()),
              LatencyHistogram::bucket_count - 1);
}

TEST(LatencyHistogram, RelativeErrorStaysSmall) {
    // Bucket width is at most 1/32 of the value above the linear range.
    for (std::uint64_t v = 64; v < (1ull << 30); v = v * 3 / 2 + 7) {
        const auto upper = LatencyHistogram::upper_bound_of(LatencyHistogram::bucket_of(v));
        EXPECT_LE(upper - v, v / 32) << v;
    }
}

TEST(LatencyHistogram, TailPercentilesSeeOutliers) {
    LatencyHistogram h;
    for (int i = 0; i < 9990; ++i) h.record(100);
    for (int i = 0; i < 10; ++i) h.record(1'000'000);
    EXPECT_LE(h.percentile(0.5), 101u);
    EXPECT_LE(h.percentile(0.99), 101u);
    // 10 of 10000 samples are slow, so p99.9 is exactly on the boundary
    // and p99.95 must be in the slow bucket.
    EXPECT_LE(h.percentile(0.999), 101u);
    EXPECT_GE(h.percentile(0.9995), 1'000'000u);
    EXPECT_EQ(h.percentile(1.0), 1'000'000u);
    EXPECT_EQ(h.max(), 1'000'000u);
}

TEST(LatencyHistogram, ResetClears) {
    LatencyHistogram h;
    h.record(5);
    h.reset();
    EXPECT_EQ(h.count(), 0u);
    EXPECT_EQ(h.max(), 0u);
}
