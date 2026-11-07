#pragma once
#include <atomic>
#include <cassert>
#include <cstdint>

namespace lockfree::detail {

struct WideTaggedPair {
    void* ptr;
    std::uint64_t tag;
};

// True where a 16-byte CAS is a single instruction: ARM64 with LSE (`casp`,
// every Apple Silicon core), or x86-64 built with -mcx16 under clang. GCC on
// x86 reports false even with -mcx16 because it routes 16-byte atomics through
// libatomic, so Linux builds usually take the packed path below.
inline constexpr bool wide_cas_is_lock_free = std::atomic<WideTaggedPair>::is_always_lock_free;

// An atomic {pointer, tag} where every successful CAS bumps the tag. Two loads
// that see the same pointer but different tags saw different versions of the
// head, so a CAS that compares both can't be fooled by a node that was popped
// and pushed back in between (ABA).
//
// Wide:   {Node*, uint64_t} in one 16-byte atomic. The tag wraps after 2^64
//         changes, which never happens.
// Packed: one 64-bit word, the tag in the top 16 bits and the pointer in the
//         low 48. That relies on user-space addresses fitting in 48 bits,
//         which holds on x86-64 with 4-level paging and on ARM64 with the
//         usual 48-bit VA (macOS, default Linux). 5-level paging (LA57) or
//         ARM64 top-byte tagging would break it, and the assert in pack()
//         catches that in debug builds. The tag wraps after 65536 changes, so
//         a thread stalled across exactly a multiple of that many could still
//         be fooled. Unlikely, but not impossible the way 2^64 is.
template <typename Node, bool Wide = wide_cas_is_lock_free>
class TaggedAtomic;

template <typename Node>
struct TaggedValue {
    Node* ptr = nullptr;
    std::uint64_t tag = 0;

    friend bool operator==(const TaggedValue&, const TaggedValue&) = default;
};

template <typename Node>
class TaggedAtomic<Node, true> {
public:
    using Value = TaggedValue<Node>;
    static constexpr std::uint64_t tag_mask = ~std::uint64_t{0};

    TaggedAtomic() noexcept = default;

    Value load(std::memory_order order) const noexcept {
        WideTaggedPair raw = word_.load(order);
        return {static_cast<Node*>(raw.ptr), raw.tag};
    }

    // On success stores {desired, expected.tag + 1}. On failure updates
    // `expected` with what's there now.
    bool compare_exchange_weak(Value& expected, Node* desired, std::memory_order success,
                               std::memory_order failure) noexcept {
        WideTaggedPair want{expected.ptr, expected.tag};
        bool ok = word_.compare_exchange_weak(want, WideTaggedPair{desired, expected.tag + 1},
                                              success, failure);
        if (!ok) expected = {static_cast<Node*>(want.ptr), want.tag};
        return ok;
    }

private:
    std::atomic<WideTaggedPair> word_{WideTaggedPair{nullptr, 0}};
};

template <typename Node>
class TaggedAtomic<Node, false> {
    static_assert(sizeof(void*) == 8, "the packed form needs 64-bit pointers");

public:
    using Value = TaggedValue<Node>;
    static constexpr int ptr_bits = 48;
    static constexpr std::uint64_t ptr_mask = (std::uint64_t{1} << ptr_bits) - 1;
    static constexpr std::uint64_t tag_mask = 0xFFFF;

    TaggedAtomic() noexcept = default;

    Value load(std::memory_order order) const noexcept { return unpack(word_.load(order)); }

    bool compare_exchange_weak(Value& expected, Node* desired, std::memory_order success,
                               std::memory_order failure) noexcept {
        std::uint64_t want = pack(expected);
        bool ok = word_.compare_exchange_weak(want, pack({desired, expected.tag + 1}), success,
                                              failure);
        if (!ok) expected = unpack(want);
        return ok;
    }

private:
    static std::uint64_t pack(Value v) noexcept {
        auto bits = reinterpret_cast<std::uintptr_t>(v.ptr);
        assert((bits & ~ptr_mask) == 0 && "pointer doesn't fit in 48 bits");
        return ((v.tag & tag_mask) << ptr_bits) | bits;
    }

    static Value unpack(std::uint64_t word) noexcept {
        return {reinterpret_cast<Node*>(static_cast<std::uintptr_t>(word & ptr_mask)),
                word >> ptr_bits};
    }

    std::atomic<std::uint64_t> word_{0};
};

} // namespace lockfree::detail
