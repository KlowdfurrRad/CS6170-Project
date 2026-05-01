"""Checker: verify that a reported median is correct for a given array."""


def check_median(arr, reported_median):
    """Return True if reported_median equals the lower median of arr."""
    expected = sorted(arr)[(len(arr) - 1) // 2]
    return reported_median == expected


def check_from_log(logfile, testcase_dir):
    """Parse a benchmark log and verify every reported answer."""
    import os, re

    # Load all test case arrays keyed by filename
    arrays = {}
    for fname in sorted(os.listdir(testcase_dir)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(testcase_dir, fname)
        with open(path) as f:
            n = int(f.readline().strip())
            data = list(map(int, f.readline().split()))
        arrays[fname] = data

    # Parse log lines like:  [algo] tc_01_n100.txt | median=42 | time=0.001s
    pattern = re.compile(
        r"\[(?P<algo>[^\]]+)\]\s+(?P<tc>\S+\.txt)\s+\|\s+median=(?P<med>-?\d+)"
    )

    all_ok = True
    checked = 0
    with open(logfile) as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            algo = m.group("algo")
            tc = m.group("tc")
            med = int(m.group("med"))
            if tc not in arrays:
                print(f"  WARN: {tc} not found in testcase dir")
                continue
            ok = check_median(arrays[tc], med)
            checked += 1
            if not ok:
                expected = sorted(arrays[tc])[(len(arrays[tc]) - 1) // 2]
                print(f"  FAIL: [{algo}] {tc}  reported={med}  expected={expected}")
                all_ok = False

    if checked == 0:
        print("  No results found in log to check.")
        return False

    if all_ok:
        print(f"  ALL {checked} results CORRECT.")
    return all_ok


if __name__ == "__main__":
    import sys, os

    logfile = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "benchmark.log"
    )
    tc_dir = os.path.join(os.path.dirname(__file__), "testcases")
    check_from_log(logfile, tc_dir)
