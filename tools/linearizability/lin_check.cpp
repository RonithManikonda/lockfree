// lin_check: check a recorded history log for linearizability.
//
//   lin_check [--spec queue|stack] [--max-configs N] [--no-minimize] [FILE]
//
// FILE (or stdin when absent or "-") is in the format described in
// log_format.hpp. When the history fails, the smallest piece of it that still
// fails is printed in the same format, so it can be saved and rerun.
//
// Exit status: 0 linearizable, 1 not linearizable, 2 bad usage / unreadable
// or ill-formed log, 3 gave up (raise --max-configs).
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

#include "linearizability/checker.hpp"
#include "linearizability/log_format.hpp"
#include "linearizability/minimize.hpp"
#include "linearizability/spec.hpp"

namespace {

using namespace lockfree::lin;

constexpr int kLinearizable = 0;
constexpr int kNotLinearizable = 1;
constexpr int kBadInput = 2;
constexpr int kGaveUp = 3;

int usage(const char* why) {
    std::cerr << "lin_check: " << why << "\n"
              << "usage: lin_check [--spec queue|stack] [--max-configs N] [--no-minimize] [FILE]\n";
    return kBadInput;
}

template <typename Spec>
int run(const History& h, const CheckOptions& options, bool minimize_window) {
    const CheckResult r = check<Spec>(h, options);
    std::cout << to_string(r.verdict) << " (" << h.size() << " ops, " << r.segments
              << " segments, " << r.configurations << " configurations)\n";
    switch (r.verdict) {
    case Verdict::linearizable: return kLinearizable;
    case Verdict::ill_formed: std::cout << r.message << "\n"; return kBadInput;
    case Verdict::gave_up: std::cout << r.message << "\n"; return kGaveUp;
    case Verdict::not_linearizable: break;
    }

    std::cout << r.message << "\n";
    if (minimize_window) {
        if (auto m = minimize<Spec>(h, options)) {
            std::cout << "\nminimal failing window (" << m->window.size() << " ops, "
                      << m->checks << " checks); dropping any one value from it passes:\n";
            write_log(std::cout, m->window);
            return kNotLinearizable;
        }
        std::cout << "(minimizing hit the configuration limit; showing the failing segment)\n";
    }
    std::cout << "\nfailing segment:\n";
    write_log(std::cout, History(r.failing_segment));
    std::cout << "\nlongest legal order through it:\n";
    write_log(std::cout, History(r.best_order));
    return kNotLinearizable;
}

} // namespace

int main(int argc, char** argv) {
    std::string spec = "queue";
    std::string path = "-";
    CheckOptions options;
    bool minimize_window = true;
    bool have_path = false;

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--spec" && i + 1 < argc) {
            spec = argv[++i];
        } else if (arg == "--max-configs" && i + 1 < argc) {
            char* end = nullptr;
            options.max_configurations = std::strtoull(argv[++i], &end, 10);
            if (*end != '\0' || options.max_configurations == 0) return usage("bad --max-configs");
        } else if (arg == "--no-minimize") {
            minimize_window = false;
        } else if (arg == "-h" || arg == "--help") {
            usage("check a history log for linearizability");
            return kLinearizable;
        } else if (!arg.empty() && arg[0] == '-' && arg != "-") {
            return usage("unknown option");
        } else if (have_path) {
            return usage("more than one input file");
        } else {
            path = argv[i];
            have_path = true;
        }
    }
    if (spec != "queue" && spec != "stack") return usage("--spec must be queue or stack");

    ParseResult parsed;
    if (path == "-") {
        parsed = read_log(std::cin);
    } else {
        std::ifstream in(path);
        if (!in) {
            std::cerr << "lin_check: can't open " << path << "\n";
            return kBadInput;
        }
        parsed = read_log(in);
    }
    if (parsed.error) {
        std::cerr << "lin_check: " << path << ":" << parsed.error->line << ": "
                  << parsed.error->message << "\n";
        return kBadInput;
    }

    return spec == "queue" ? run<FifoQueueSpec>(parsed.history, options, minimize_window)
                           : run<LifoStackSpec>(parsed.history, options, minimize_window);
}
