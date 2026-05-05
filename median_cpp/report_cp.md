# Randomized Median Finding: Algorithm Analysis and Benchmarks (C++)

## 1. Problem Statement

Given an unsorted array of $n$ floating-point numbers, find the **lower median** — the element at rank $\lfloor (n-1)/2 \rfloor$ in the sorted order. We implement and benchmark three algorithms with fundamentally different design philosophies (one randomized partition, one deterministic, one randomized sampling) across five dataset types that stress different algorithmic properties.

---

## 2. Algorithms

### 2.1 Randomized QuickSelect — Expected $O(n)$, Worst-case $O(n^2)$

A selection algorithm based on partitioning. Our C++ implementation uses a **three-way (Dutch National Flag) partition** — critical for correctness and efficiency on inputs with many duplicates:

1. Pick a **random pivot**.
2. Partition into three regions: elements $< \text{pivot}$, $= \text{pivot}$, $> \text{pivot}$.
3. If the target rank $k$ falls in the equal region, return the pivot immediately.
4. Otherwise recurse into the relevant side.

The 3-way partition reduces duplicate-heavy inputs (e.g., `random01`) from a potential $O(n^2)$ disaster to $O(n)$ in the best case — when the pivot value equals the median, a single pass suffices.

**Expected runtime:** $\approx 3.39n$ comparisons on average.

### 2.2 Deterministic Median-of-Medians (BFPRT) — Worst-case $O(n)$

The Blum-Floyd-Pratt-Rivest-Tarjan algorithm guarantees $O(n)$ worst-case by selecting a provably balanced pivot:

1. Divide the array into $\lceil n/5 \rceil$ groups of 5.
2. Sort each group with insertion sort (at most 10 comparisons, fully unrollable by the compiler).
3. Extract each group's median; recursively find the median of these $n/5$ medians.
4. Three-way partition around this pivot; recurse on the relevant side.

**Recurrence:** $T(n) = T(n/5) + T(7n/10) + O(n)$. Since $1/5 + 7/10 = 9/10 < 1$, this solves to $T(n) = O(n)$ with constant $\approx 24n$.

The implementation now uses a **3-way partition** (not the textbook 2-way), so duplicate-heavy inputs no longer cause runtime to degrade — the asymptotic guarantee is preserved while the practical constant on `random01` drops by orders of magnitude.

### 2.3 LazySelect — Lazy Sampling, $O(n)$ with High Probability

This is the textbook lazy sampling algorithm of Motwani–Raghavan (Sec. 3.3 of *Randomized Algorithms*):

1. Draw a multiset $R$ of $\lceil n^{3/4}\rceil$ elements **uniformly at random with replacement** from $S$. Sort $R$.
2. Set $x = k\, n^{-1/4}$, $\ell = \max(\lfloor x - \sqrt{n}\rfloor, 1)$, $h = \min(\lceil x + \sqrt{n}\rceil, n^{3/4})$, and use the order statistics $a = R_{(\ell)}$, $b = R_{(h)}$ as fences.
3. In a single linear scan over $S$, compute $r_a = \mathrm{rank}_S(a)$ and the candidate set $P = \{y \in S : a \le y \le b\}$.
4. Verify: if $r_a \le k \le r_b$ and $|P| \le 4n^{3/4} + 2$, sort $P$ and return its element of rank $k - r_a + 1$. Otherwise, restart.

**Total work (typical case):** sample + sort sample $O(n^{3/4}\log n)$, one linear scan $O(n)$, sort $P$ $O(n^{3/4}\log n)$. Each pass uses $2n + o(n)$ comparisons; the expected number of passes is $1 + o(1)$, matching the information-theoretic lower bound up to lower-order terms.

---

## 3. Datasets

All inputs are arrays of `double` of sizes $n \in \{10^3, 10^4, 10^5, 5{\times}10^5, 10^6, 5{\times}10^6, 10^7\}$.

| Dataset | Description | Challenge |
|---|---|---|
| **random** | Uniform floats in $[-10^9, 10^9]$ | Baseline; no adversarial structure |
| **random01** | $n/2$ zeros and $n/2$ ones, shuffled | Extreme duplicates; tests partition collapse |
| **organpipe** | $0,1,\ldots,\lfloor n/2\rfloor-1,\lfloor n/2\rfloor-1,\ldots,1,0$ | Ascending then descending; moderate duplicates |
| **sorted** | $0,1,\ldots,n-1$ | Already sorted; tests pivot-selection sensitivity |
| **rotated** | $1,2,\ldots,n-1,0$ | Nearly sorted; classic adversary for naive QuickSort |

---

## 4. Benchmark Results

All medians are verified against ground truth (`std::sort` + index). Plots in `plots_mom.pdf` show **speed of each algorithm relative to MoM-BFPRT** across all datasets and input sizes.

The complete numerical results are stored in `benchmark_results.csv`. The presentation in this report focuses on the relative-speed comparison (vs. MoM) consistent with the plots in `plots_mom.pdf`.

---

## 5. Implementation Notes

### Correctness Under Duplicates

A textbook 2-way partition collapses to $O(n^2)$ on `random01` (every partition removes only one element equal to the pivot). All three algorithms in this benchmark — QuickSelect, MoM, and the implicit linear scan in LazySelect — use partitioning schemes that handle duplicates cleanly. QuickSelect and MoM both employ Dutch National Flag (3-way) partitioning; LazySelect's sample-and-fence step naturally handles multiplicity through the candidate set $P$.

### LazySelect Implementation Details

The implementation samples **with replacement** (matching the textbook description), uses 0-indexed ranks throughout, and restarts on verification failure rather than widening the bracket. The `nth_element`-style fall-back was removed in favor of a clean restart loop, so the runtime distribution is exactly the geometric distribution analyzed in the report.

### Reproducing the Results

```
g++ -O2 -std=c++17 -o median_benchmark median_benchmark.cpp
./median_benchmark > benchmark_results.csv
python3 plot_speed_vs_mom.py
```
