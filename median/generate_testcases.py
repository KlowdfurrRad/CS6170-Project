"""Generate 15 test cases with sizes from 100 to 50,000,000."""

import os
import random
import struct

TESTCASE_DIR = os.path.join(os.path.dirname(__file__), "testcases")
SIZES = [
    100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000,
    2_000_000, 5_000_000, 10_000_000, 25_000_000, 50_000_000,
]


def generate():
    os.makedirs(TESTCASE_DIR, exist_ok=True)

    for i, n in enumerate(SIZES, 1):
        path = os.path.join(TESTCASE_DIR, f"tc_{i:02d}_n{n}.txt")
        if os.path.exists(path):
            print(f"  Skipped  tc_{i:02d} | n={n:>10,}  (already exists)")
            continue
        # Use a mix: random ints in a wide range
        data = [random.randint(-(10**9), 10**9) for _ in range(n)]
        with open(path, "w") as f:
            f.write(f"{n}\n")
            f.write(" ".join(map(str, data)) + "\n")
        print(f"  Generated tc_{i:02d} | n={n:>10,}")

    print(f"\n  All {len(SIZES)} test cases written to {TESTCASE_DIR}/")


if __name__ == "__main__":
    generate()
