"""Four median-finding algorithms:
  1. Randomized QuickSelect
  2. Deterministic Median-of-Medians (BFPRT)
  3. Sort-based (baseline)
  4. Randomized Sampling (n^{3/4} sample)
"""

import math
import random


# ---------------------------------------------------------------------------
# 1. Randomized QuickSelect  –  expected O(n), worst-case O(n^2)
# ---------------------------------------------------------------------------
def _randomized_partition(arr, lo, hi):
    pivot_idx = random.randint(lo, hi)
    arr[pivot_idx], arr[hi] = arr[hi], arr[pivot_idx]
    pivot = arr[hi]
    store = lo
    for i in range(lo, hi):
        if arr[i] < pivot:
            arr[store], arr[i] = arr[i], arr[store]
            store += 1
    arr[store], arr[hi] = arr[hi], arr[store]
    return store


def randomized_median(arr):
    """Return the lower median using randomized QuickSelect."""
    a = arr[:]  # work on a copy
    n = len(a)
    k = (n - 1) // 2  # 0-indexed lower median

    lo, hi = 0, n - 1
    while lo < hi:
        p = _randomized_partition(a, lo, hi)
        if p == k:
            return a[p]
        elif p < k:
            lo = p + 1
        else:
            hi = p - 1
    return a[lo]


# ---------------------------------------------------------------------------
# 2. Deterministic Median-of-Medians (BFPRT)  –  worst-case O(n)
# ---------------------------------------------------------------------------
def _insertion_sort(arr, lo, hi):
    for i in range(lo + 1, hi + 1):
        key = arr[i]
        j = i - 1
        while j >= lo and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


def _median_of_medians(arr, lo, hi):
    """Return the index of the median-of-medians pivot."""
    n = hi - lo + 1
    if n <= 5:
        _insertion_sort(arr, lo, hi)
        return lo + n // 2

    # Split into groups of 5, find each group's median
    num_groups = n // 5
    for i in range(num_groups):
        g_lo = lo + i * 5
        g_hi = g_lo + 4
        _insertion_sort(arr, g_lo, g_hi)
        # Move group median to the front block
        mid = g_lo + 2
        arr[lo + i], arr[mid] = arr[mid], arr[lo + i]

    # Recursively find the median of the group medians
    return _median_of_medians(arr, lo, lo + num_groups - 1)


def _mom_partition(arr, lo, hi, pivot_idx):
    arr[pivot_idx], arr[hi] = arr[hi], arr[pivot_idx]
    pivot = arr[hi]
    store = lo
    for i in range(lo, hi):
        if arr[i] < pivot:
            arr[store], arr[i] = arr[i], arr[store]
            store += 1
    arr[store], arr[hi] = arr[hi], arr[store]
    return store


def deterministic_median(arr):
    """Return the lower median using Median-of-Medians (BFPRT)."""
    a = arr[:]  # work on a copy
    n = len(a)
    k = (n - 1) // 2

    lo, hi = 0, n - 1
    while lo < hi:
        pivot_idx = _median_of_medians(a, lo, hi)
        p = _mom_partition(a, lo, hi, pivot_idx)
        if p == k:
            return a[p]
        elif p < k:
            lo = p + 1
        else:
            hi = p - 1
    return a[lo]


# ---------------------------------------------------------------------------
# 3. Sort-based baseline  –  O(n log n)
# ---------------------------------------------------------------------------
def sort_median(arr):
    """Return the lower median by sorting (baseline)."""
    s = sorted(arr)
    return s[(len(s) - 1) // 2]


# ---------------------------------------------------------------------------
# 4. Randomized Sampling with n^{3/4} sample  –  O(n) w.h.p.
#
#    Algorithm (from Motwani & Raghavan / Floyd-Rivest style):
#    1. Draw a random sample S of size ceil(n^{3/4}).
#    2. Sort S.
#    3. Pick two "bracket" elements from S that, with high probability,
#       surround the true median: positions (|S|/2 ± sqrt(n)) in S.
#    4. In one linear scan count elements < L, filter elements in [L, R].
#    5. If the median falls inside the filtered set C and |C| is small
#       (O(n^{3/4})), sort C and read off the answer.
#    6. Otherwise retry (happens with probability O(n^{-1/4})).
# ---------------------------------------------------------------------------
def sampling_median(arr):
    """Return the lower median using the n^{3/4} random-sampling algorithm."""
    n = len(arr)
    k = (n - 1) // 2  # 0-indexed lower median rank

    while True:
        # Step 1: sample
        sample_size = math.ceil(n ** 0.75)
        S = sorted(random.sample(arr, sample_size))

        # Step 2: bracket positions in the sample
        # The median has rank k in the full array.  Its expected rank in S
        # is k * |S| / n ≈ |S|/2.  We widen by √n to get high-probability
        # containment.
        half = len(S) // 2
        offset = math.ceil(n ** 0.5)
        lo_idx = max(0, half - offset)
        hi_idx = min(len(S) - 1, half + offset)
        L = S[lo_idx]
        R = S[hi_idx]

        # Step 3: single linear scan – count < L, collect elements in [L, R]
        count_less = 0
        C = []
        for x in arr:
            if x < L:
                count_less += 1
            elif x <= R:
                C.append(x)

        # Step 4: check feasibility
        # The median (rank k) should land inside C.
        # rank of L in arr >= count_less, rank of max(C) <= count_less+|C|-1
        if count_less <= k < count_less + len(C) and len(C) <= 4 * sample_size:
            C.sort()
            return C[k - count_less]
        # else: bad sample, retry (very unlikely)
