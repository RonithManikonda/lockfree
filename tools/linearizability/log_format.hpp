#pragma once
#include <cstdint>
#include <istream>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "linearizability/history.hpp"

// Plain-text history logs, one operation per line:
//
//   # anything after '#' is a comment; blank lines are ignored
//   <thread> push <value> <invoke_ns> <response_ns>
//   <thread> pop <value|empty> <invoke_ns> <response_ns>
//
// e.g.
//   0 push 7 100 140
//   1 pop 7 120 180
//   1 pop empty 190 200
//
// Text so a failing run can be pasted into an issue, trimmed by hand, and
// fed back to lin_check; one op per line so `sort -k4 -n`, grep and diff
// all work on it. Lines can be in any order (the checker sorts).
namespace lockfree::lin {

inline void write_op(std::ostream& out, const Operation& op) {
    out << op.thread << (op.kind == OpKind::push ? " push " : " pop ");
    if (op.kind == OpKind::push) {
        out << op.value;
    } else if (op.result) {
        out << *op.result;
    } else {
        out << "empty";
    }
    out << ' ' << op.invoke_ts << ' ' << op.response_ts << '\n';
}

inline void write_log(std::ostream& out, const History& h) {
    for (const Operation& op : h.ops()) write_op(out, op);
}

struct ParseError {
    std::size_t line = 0; // 1-based
    std::string message;
};

struct ParseResult {
    History history;
    std::optional<ParseError> error;
};

namespace detail {

// Whole-token integer parse: "12x" and "" are errors, not 12 and 0.
// Unsigned fields refuse a sign: istream would read "-1" as 2^64 - 1.
template <typename Int>
bool parse_int(const std::string& token, Int& out) {
    if (token.empty()) return false;
    if (std::is_unsigned_v<Int> && (token[0] == '-' || token[0] == '+')) return false;
    std::istringstream in(token);
    in >> out;
    return !in.fail() && in.peek() == std::char_traits<char>::eof();
}

} // namespace detail

// Stops at the first bad line and reports it; the history holds the ops
// parsed before it.
inline ParseResult read_log(std::istream& in) {
    ParseResult result;
    std::string line;
    for (std::size_t lineno = 1; std::getline(in, line); ++lineno) {
        if (auto hash = line.find('#'); hash != std::string::npos) line.erase(hash);
        std::istringstream fields(line);
        std::vector<std::string> tok;
        for (std::string t; fields >> t;) tok.push_back(t);
        if (tok.empty()) continue;

        auto fail = [&](std::string msg) {
            result.error = ParseError{lineno, std::move(msg)};
            return result;
        };
        if (tok.size() != 5) {
            return fail("expected 5 fields (thread kind value invoke response), got " +
                        std::to_string(tok.size()));
        }
        Operation op;
        if (!detail::parse_int(tok[0], op.thread)) return fail("bad thread id '" + tok[0] + "'");
        if (tok[1] == "push") {
            op.kind = OpKind::push;
            if (!detail::parse_int(tok[2], op.value)) return fail("bad push value '" + tok[2] + "'");
        } else if (tok[1] == "pop") {
            op.kind = OpKind::pop;
            if (tok[2] != "empty") {
                std::int64_t v = 0;
                if (!detail::parse_int(tok[2], v)) {
                    return fail("bad pop result '" + tok[2] + "' (a number or 'empty')");
                }
                op.result = v;
            }
        } else {
            return fail("unknown op '" + tok[1] + "' (push or pop)");
        }
        if (!detail::parse_int(tok[3], op.invoke_ts)) return fail("bad invoke time '" + tok[3] + "'");
        if (!detail::parse_int(tok[4], op.response_ts)) {
            return fail("bad response time '" + tok[4] + "'");
        }
        result.history.add(op);
    }
    return result;
}

} // namespace lockfree::lin
