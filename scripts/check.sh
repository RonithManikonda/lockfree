#!/usr/bin/env bash
# Build and test every verification config: plain Debug, TSan, ASan, UBSan.
#
#   scripts/check.sh                # all configs
#   scripts/check.sh tsan asan      # just these
#
# Each config gets its own build-<name>/ directory (plain Debug uses build/) so
# reruns are incremental. Exits non-zero if any config fails to build or test;
# the remaining configs still run so one failure doesn't hide another.
#
# Env: JOBS (parallelism, default: CPU count), LOCKFREE_STRESS_SCALE and
# LOCKFREE_SEED are passed through to the tests.
set -u

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

if [[ -z "${JOBS:-}" ]]; then
    JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
fi

# Sanitizers should stop at the first report with a stack trace; a report that
# scrolls past while the test still exits 0 is as good as no report.
export TSAN_OPTIONS="${TSAN_OPTIONS:-halt_on_error=1:second_deadlock_stack=1}"
export ASAN_OPTIONS="${ASAN_OPTIONS:-abort_on_error=1:strict_string_checks=1:detect_stack_use_after_return=1}"
export UBSAN_OPTIONS="${UBSAN_OPTIONS:-halt_on_error=1:print_stacktrace=1}"

all_configs=(default tsan asan ubsan)
configs=("$@")
[[ ${#configs[@]} -eq 0 ]] && configs=("${all_configs[@]}")

configure_args() {
    case "$1" in
        default) echo "-DCMAKE_BUILD_TYPE=Debug" ;;
        # Benches are skipped under sanitizers: they'd only measure the
        # instrumentation, and Google Benchmark adds minutes to every build.
        tsan)  echo "-DCMAKE_BUILD_TYPE=Debug -DLOCKFREE_SANITIZER=thread -DLOCKFREE_BUILD_BENCH=OFF" ;;
        asan)  echo "-DCMAKE_BUILD_TYPE=Debug -DLOCKFREE_SANITIZER=address -DLOCKFREE_BUILD_BENCH=OFF" ;;
        ubsan) echo "-DCMAKE_BUILD_TYPE=Debug -DLOCKFREE_SANITIZER=undefined -DLOCKFREE_BUILD_BENCH=OFF" ;;
        *) return 1 ;;
    esac
}

build_dir() {
    if [[ "$1" == default ]]; then echo build; else echo "build-$1"; fi
}

passed=()
failed=()

for cfg in "${configs[@]}"; do
    if ! args="$(configure_args "$cfg")"; then
        echo "check.sh: unknown config '$cfg' (expected: ${all_configs[*]})" >&2
        exit 2
    fi
    dir="$(build_dir "$cfg")"
    echo "==> [$cfg] $dir"

    # shellcheck disable=SC2086  # args is a deliberate word list
    if cmake -S . -B "$dir" $args >/dev/null \
        && cmake --build "$dir" -j "$JOBS" \
        && ctest --test-dir "$dir" --output-on-failure -j "$JOBS"; then
        passed+=("$cfg")
    else
        echo "==> [$cfg] FAILED" >&2
        failed+=("$cfg")
    fi
done

echo
echo "passed: ${passed[*]:-none}"
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "failed: ${failed[*]}" >&2
    exit 1
fi
