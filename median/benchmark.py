"""Benchmark all three median algorithms on generated test cases.

Logs each algorithm's answer and wall-clock time to benchmark.log,
then runs the checker to verify correctness.
"""

import os
import sys
import time

from algorithms import randomized_median, deterministic_median, sort_median, sampling_median
from checker import check_from_log

TESTCASE_DIR = os.path.join(os.path.dirname(__file__), "testcases")
LOGFILE = os.path.join(os.path.dirname(__file__), "benchmark.log")

ALGORITHMS = [
    ("Randomized-QuickSelect",  randomized_median),
    ("Deterministic-MoM",       deterministic_median),
    ("Sampling-n^{3/4}",        sampling_median),
    ("Sort-Baseline",           sort_median),
]

# Per-algorithm size limits for slow pure-Python implementations
ALGO_MAX_N = {
    "Deterministic-MoM": 500_000,
}


def load_testcase(path):
    with open(path) as f:
        n = int(f.readline().strip())
        data = list(map(int, f.readline().split()))
    assert len(data) == n
    return data


def run_benchmarks():
    # Ensure test cases exist
    if not os.path.isdir(TESTCASE_DIR) or not os.listdir(TESTCASE_DIR):
        print("No test cases found. Generating...")
        from generate_testcases import generate
        generate()

    tc_files = sorted(f for f in os.listdir(TESTCASE_DIR) if f.endswith(".txt"))
    print(f"\n{'='*72}")
    print(f"  Benchmarking {len(ALGORITHMS)} algorithms on {len(tc_files)} test cases")
    print(f"{'='*72}\n")

    with open(LOGFILE, "w") as log:
        log.write("# Median-Finding Benchmark Log\n")
        log.write(f"# {'='*68}\n\n")

        for tc_name in tc_files:
            tc_path = os.path.join(TESTCASE_DIR, tc_name)
            arr = load_testcase(tc_path)
            n = len(arr)

            header = f"--- {tc_name}  (n={n:,}) ---"
            print(header)
            log.write(header + "\n")

            for algo_name, algo_fn in ALGORITHMS:
                max_n = ALGO_MAX_N.get(algo_name)
                if max_n and n > max_n:
                    skip_msg = f"  [{algo_name}] {tc_name} | SKIPPED (n>{max_n:,})"
                    print(skip_msg)
                    log.write(skip_msg + "\n")
                    continue

                t0 = time.perf_counter()
                median_val = algo_fn(arr)
                elapsed = time.perf_counter() - t0

                line = f"  [{algo_name}] {tc_name} | median={median_val} | time={elapsed:.6f}s"
                print(line)
                log.write(line + "\n")

            log.write("\n")
            print()

    print(f"Log written to {LOGFILE}\n")

    # Verify correctness
    print("="*72)
    print("  Checking correctness of all logged results...")
    print("="*72)
    check_from_log(LOGFILE, TESTCASE_DIR)


if __name__ == "__main__":
    sys.setrecursionlimit(50_000)  # MoM can recurse on large inputs
    run_benchmarks()
