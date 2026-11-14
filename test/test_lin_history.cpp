#include <gtest/gtest.h>

#include <optional>
#include <vector>

#include "linearizability/history.hpp"
#include "linearizability/spec.hpp"

using lockfree::lin::FifoQueueSpec;
using lockfree::lin::first_illegal;
using lockfree::lin::History;
using lockfree::lin::LifoStackSpec;
using lockfree::lin::Operation;
using lockfree::lin::OpKind;

namespace {

Operation push(std::int64_t v, std::uint64_t inv = 0, std::uint64_t resp = 0,
               std::uint32_t thread = 0) {
    return {thread, OpKind::push, v, std::nullopt, inv, resp};
}

Operation pop(std::optional<std::int64_t> r, std::uint64_t inv = 0, std::uint64_t resp = 0,
              std::uint32_t thread = 0) {
    return {thread, OpKind::pop, 0, r, inv, resp};
}

} // namespace

// ---- Sequential specs --------------------------------------------------------

TEST(FifoQueueSpec, AcceptsFirstInFirstOut) {
    std::vector ops{push(1), push(2), pop(1), push(3), pop(2), pop(3), pop(std::nullopt)};
    EXPECT_EQ(first_illegal<FifoQueueSpec>(ops), std::nullopt);
}

TEST(FifoQueueSpec, RejectsLastInFirstOut) {
    std::vector ops{push(1), push(2), pop(2)};
    EXPECT_EQ(first_illegal<FifoQueueSpec>(ops), 2u);
}

TEST(FifoQueueSpec, RejectsEmptyPopOnNonEmptyQueue) {
    std::vector ops{push(1), pop(std::nullopt)};
    EXPECT_EQ(first_illegal<FifoQueueSpec>(ops), 1u);
}

TEST(FifoQueueSpec, RejectsValueFromEmptyQueue) {
    std::vector ops{pop(5)};
    EXPECT_EQ(first_illegal<FifoQueueSpec>(ops), 0u);
}

TEST(FifoQueueSpec, RejectsSecondPopOfTheSameValue) {
    std::vector ops{push(1), pop(1), pop(1)};
    EXPECT_EQ(first_illegal<FifoQueueSpec>(ops), 2u);
}

TEST(LifoStackSpec, AcceptsLastInFirstOut) {
    std::vector ops{push(1), push(2), pop(2), push(3), pop(3), pop(1), pop(std::nullopt)};
    EXPECT_EQ(first_illegal<LifoStackSpec>(ops), std::nullopt);
}

TEST(LifoStackSpec, RejectsFirstInFirstOut) {
    std::vector ops{push(1), push(2), pop(1)};
    EXPECT_EQ(first_illegal<LifoStackSpec>(ops), 2u);
}

TEST(LifoStackSpec, RejectsEmptyPopOnNonEmptyStack) {
    std::vector ops{push(1), pop(std::nullopt)};
    EXPECT_EQ(first_illegal<LifoStackSpec>(ops), 1u);
}

// The checker relies on a rejected apply leaving the state as it was, so it
// can try the next candidate without copying the state first.
template <typename Spec>
void expect_rejected_apply_changes_nothing() {
    Spec s;
    ASSERT_TRUE(s.apply(push(1)));
    ASSERT_TRUE(s.apply(push(2)));
    Spec before = s;
    EXPECT_FALSE(s.apply(pop(42)));
    EXPECT_FALSE(s.apply(pop(std::nullopt)));
    EXPECT_EQ(s, before);
    EXPECT_EQ(s.hash(), before.hash());
    EXPECT_EQ(s.size(), 2u);
}

TEST(FifoQueueSpec, RejectedApplyChangesNothing) {
    expect_rejected_apply_changes_nothing<FifoQueueSpec>();
}

TEST(LifoStackSpec, RejectedApplyChangesNothing) {
    expect_rejected_apply_changes_nothing<LifoStackSpec>();
}

template <typename Spec>
void expect_state_equality_follows_contents() {
    Spec a, b;
    // Different paths to the same contents compare equal (that's what makes
    // memoization pay off).
    ASSERT_TRUE(a.apply(push(7)));
    ASSERT_TRUE(b.apply(push(9)));
    ASSERT_TRUE(b.apply(pop(9)));
    ASSERT_TRUE(b.apply(push(7)));
    EXPECT_EQ(a, b);
    EXPECT_EQ(a.hash(), b.hash());

    Spec c;
    ASSERT_TRUE(c.apply(push(8)));
    EXPECT_NE(a, c);
    EXPECT_NE(a.hash(), c.hash());

    // Order matters too: [1, 2] and [2, 1] are different states.
    Spec d, e;
    ASSERT_TRUE(d.apply(push(1)));
    ASSERT_TRUE(d.apply(push(2)));
    ASSERT_TRUE(e.apply(push(2)));
    ASSERT_TRUE(e.apply(push(1)));
    EXPECT_NE(d, e);
    EXPECT_NE(d.hash(), e.hash());
}

TEST(FifoQueueSpec, StateEqualityFollowsContents) {
    expect_state_equality_follows_contents<FifoQueueSpec>();
}

TEST(LifoStackSpec, StateEqualityFollowsContents) {
    expect_state_equality_follows_contents<LifoStackSpec>();
}

// ---- Real-time order -----------------------------------------------------------

TEST(LinOperation, HappensBeforeNeedsAStrictGap) {
    Operation a = push(1, 10, 20);
    Operation b = pop(1, 21, 30);
    EXPECT_TRUE(happens_before(a, b));
    EXPECT_FALSE(happens_before(b, a));
    EXPECT_FALSE(overlaps(a, b));

    // Same tick: the clock can't tell which came first, so neither is forced.
    Operation c = pop(1, 20, 30);
    EXPECT_FALSE(happens_before(a, c));
    EXPECT_TRUE(overlaps(a, c));
}

TEST(LinOperation, NestedIntervalsOverlap) {
    Operation outer = push(1, 0, 100);
    Operation inner = pop(std::nullopt, 40, 50, 1);
    EXPECT_TRUE(overlaps(outer, inner));
    EXPECT_TRUE(overlaps(inner, outer));
}

TEST(LinOperation, ToStringIsReadable) {
    EXPECT_EQ(to_string(push(3, 90, 95, 0)), "t0 push 3 [90, 95]");
    EXPECT_EQ(to_string(pop(7, 100, 140, 2)), "t2 pop -> 7 [100, 140]");
    EXPECT_EQ(to_string(pop(std::nullopt, 1, 2, 1)), "t1 pop -> empty [1, 2]");
}

// ---- History -------------------------------------------------------------------

TEST(LinHistory, SortByInvokeBreaksTiesByThread) {
    History h;
    h.add(pop(1, 50, 60, 0));
    h.add(push(1, 10, 20, 2));
    h.add(push(2, 10, 15, 1));
    h.sort_by_invoke();
    ASSERT_EQ(h.size(), 3u);
    EXPECT_EQ(h[0].thread, 1u);
    EXPECT_EQ(h[1].thread, 2u);
    EXPECT_EQ(h[2].thread, 0u);
}

TEST(LinHistory, WellFormedConcurrentHistory) {
    History h({push(1, 0, 10, 0), push(2, 5, 12, 1), pop(1, 11, 20, 0), pop(2, 13, 13, 1),
               pop(std::nullopt, 13, 14, 1)});
    EXPECT_EQ(h.check_well_formed(), std::nullopt);
}

TEST(LinHistory, RejectsResponseBeforeInvoke) {
    History h({push(1, 10, 5)});
    auto why = h.check_well_formed();
    ASSERT_TRUE(why.has_value());
    EXPECT_NE(why->find("responds before invoked"), std::string::npos) << *why;
}

TEST(LinHistory, RejectsOverlappingOpsOnOneThread) {
    // Recorded out of order on purpose; the check sorts before comparing.
    History h({pop(1, 15, 30, 3), push(1, 10, 20, 3)});
    auto why = h.check_well_formed();
    ASSERT_TRUE(why.has_value());
    EXPECT_NE(why->find("overlapping ops on one thread"), std::string::npos) << *why;
}

TEST(LinHistory, RejectsPushWithAResult) {
    Operation bad = push(1, 0, 1);
    bad.result = 1;
    History h({bad});
    EXPECT_TRUE(h.check_well_formed().has_value());
}
