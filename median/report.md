# Randomized Median Finding: Algorithm Analysis and Benchmarks

## 1. Problem Statement

Given an unsorted array of $n$ integers, find the **lower median** — the element at rank $\lfloor (n-1)/2 \rfloor$ in the sorted order. We implement and benchmark four algorithms with fundamentally different approaches to this problem.

## 2. Algorithms

### 2.1 Sort Baseline — $O(n \log n)$

Sort the array, index the middle element. This is the simplest approach and delegates all work to Python's Timsort (implemented in C), making it an important practical baseline despite being asymptotically suboptimal.

### 2.2 Randomized QuickSelect — Expected $O(n)$, Worst-case $O(n^2)$

A selection algorithm based on the partitioning step of QuickSort:
1. Pick a **random pivot**.
2. Partition the array into elements $< \text{pivot}$ and $\geq \text{pivot}$.
3. Recurse into the side that contains the target rank.

**Expected runtime analysis:** Let $T(n)$ be the expected comparisons. A random pivot has a $1/n$ chance of landing at each position. If it lands at position $i$, we recurse on $\max(i, n-1-i)$ elements. The key insight is that a pivot in the "central half" (ranks $n/4$ to $3n/4$) reduces the problem size to at most $3n/4$. A random pivot falls in this range with probability $1/2$, so in expectation we need 2 rounds to get a good pivot. This gives:

$$T(n) \leq n + n \cdot \frac{3}{4} + n \cdot \left(\frac{3}{4}\right)^2 + \cdots = 4n$$

More precisely, the expected number of comparisons is $\approx 3.39n$. The variance is low: by a Chernoff-style argument, $\Pr[T(n) > cn] \leq 2^{-\Omega(c)}$.

**Worst-case $O(n^2)$:** If every pivot is the min or max, each partition removes only 1 element, leading to $n + (n-1) + \cdots + 1 = O(n^2)$. This event has probability $O(2^{-n})$ — negligible in practice.

### 2.3 Deterministic Median-of-Medians (BFPRT) — Worst-case $O(n)$

The Blum-Floyd-Pratt-Rivest-Tarjan (1973) algorithm guarantees $O(n)$ in the worst case by choosing a **provably good pivot**:

1. **Divide** the array into $\lceil n/5 \rceil$ groups of 5.
2. **Sort** each group (insertion sort, $O(1)$ per group since group size is constant).
3. **Extract** the median of each group.
4. **Recursively** find the median of these $n/5$ group medians — this is the pivot.
5. **Partition** using this pivot, then recurse on the relevant side.

**Why groups of 5 and insertion sort?** Each group must be sorted to extract its median. Insertion sort on 5 elements takes at most 10 comparisons — it is optimal for tiny fixed-size inputs (no recursion overhead, branch-predictor friendly). The choice of 5 is the smallest odd group size that makes the recurrence converge: with groups of 3, the recurrence $T(n) = T(n/3) + T(2n/3) + O(n)$ solves to $O(n \log n)$, not $O(n)$.

**Recurrence and constant factor:** The pivot is the median of $n/5$ medians. By construction, at least $3/10$ of all elements are $\leq$ pivot and at least $3/10$ are $\geq$ pivot. So the worst-case recursive call is on at most $7n/10$ elements. The full recurrence is:

$$T(n) = \underbrace{T(n/5)}_{\text{find pivot}} + \underbrace{T(7n/10)}_{\text{recurse on larger side}} + \underbrace{O(n)}_{\text{partition}}$$

Since $1/5 + 7/10 = 9/10 < 1$, this solves to $T(n) = O(n)$. But unwinding the recurrence precisely:

$$T(n) = cn \sum_{k=0}^{\infty} \left(\frac{9}{10}\right)^k = 10cn$$

The constant $c$ itself includes the overhead of $n/5$ insertion sorts per level, making the true constant roughly **$\sim 24n$** total comparisons — about 6-8x more than QuickSelect's expected $\sim 3.4n$.

**Why it is slow in practice despite being $O(n)$:**
- The $T(n/5)$ recursive call to find the pivot is pure overhead — QuickSelect picks a pivot in $O(1)$.
- At every recursion level, the algorithm performs $n/5$ insertion sorts, element rearrangements to collect group medians, and a full partition scan.
- The recursion tree has overlapping branches ($T(n/5)$ and $T(7n/10)$) that both touch the array, leading to poor cache locality.
- In pure Python, every comparison, swap, and function call has interpreter overhead. The $\sim 24n$ operations are all executed in slow bytecode, whereas Sort Baseline delegates to C.

### 2.4 Randomized Sampling with $n^{3/4}$ — $O(n)$ with High Probability

This algorithm (Motwani & Raghavan; Floyd-Rivest style) uses a sublinear-size random sample to bracket the median:

1. **Sample** $s = \lceil n^{3/4} \rceil$ elements uniformly at random.
2. **Sort** the sample.
3. **Bracket:** In the sorted sample, the median's expected rank is $\approx s/2$. Pick elements at positions $s/2 \pm \lceil\sqrt{n}\rceil$ as lower and upper bounds $L, R$.
4. **Filter:** In one linear scan, count elements $< L$ and collect elements in $[L, R]$ into a set $C$.
5. **Extract:** If the median's rank falls within $C$ and $|C| \leq 4n^{3/4}$, sort $C$ and return the answer. Otherwise retry.

**Why $n^{3/4}$?** This is the critical sample size that balances two competing requirements:
- **Large enough** that the sample median concentrates tightly around the true median. By the DKW inequality, the empirical CDF of a sample of size $s$ deviates from the true CDF by at most $O(1/\sqrt{s})$ with high probability. With $s = n^{3/4}$, this deviation is $O(n^{-3/8})$, which maps to a rank error of $O(n^{5/8})$ in the full array.
- **Small enough** that sorting the sample costs $o(n)$: sorting $n^{3/4}$ elements takes $O(n^{3/4} \log n)$, which is sublinear.

**Why $\sqrt{n}$ offset for the brackets?** The bracket width of $\pm\sqrt{n}$ positions in the sample translates (via the concentration bound) to a window in the full array that contains $O(n^{3/4})$ elements. The probability that the true median falls outside this window is $O(n^{-1/4})$ — it shrinks polynomially, so expected retries are $1 + O(n^{-1/4}) \approx 1$.

**Total work per attempt:**
- Sampling: $O(n^{3/4})$
- Sorting the sample: $O(n^{3/4} \log n)$
- Linear scan (the dominant term): $O(n)$
- Sorting $C$: $O(n^{3/4} \log n)$

**Total: $O(n)$ with high probability.** The key advantage is the constant factor — only about $\sim 2n$ operations touch the full array (one linear scan), and the sorts are on a set of size $n^{3/4}$ which is tiny relative to $n$. Furthermore, the heavy operations (`random.sample`, `sorted`) are implemented in C in Python's standard library, so the constant is small in practice.

**Failure probability:** The event that the brackets fail to contain the median has probability $O(n^{-1/4})$. By a union bound with the event that $|C|$ is too large, the total failure probability per attempt is $O(n^{-1/4})$. After $t$ retries, the failure probability is $O(n^{-t/4})$, so the algorithm succeeds in one attempt with overwhelming probability for large $n$.

## 3. Benchmark Results

All algorithms were benchmarked on 15 test cases with uniformly random integers in $[-10^9, 10^9]$, sizes ranging from $n = 100$ to $n = 50{,}000{,}000$. Wall-clock times measured via `time.perf_counter()`. All 54 results verified correct by an independent checker.

| $n$ | QuickSelect | Median-of-Medians | Sampling $n^{3/4}$ | Sort Baseline |
|---:|---:|---:|---:|---:|
| 100 | 0.000038s | 0.000052s | 0.000055s | **0.000007s** |
| 500 | 0.000088s | 0.000213s | **0.000084s** | 0.000034s |
| 1,000 | 0.000170s | 0.000525s | **0.000113s** | 0.000073s |
| 5,000 | 0.000681s | 0.002754s | 0.000468s | **0.000442s** |
| 10,000 | 0.001499s | 0.005311s | **0.000866s** | 0.000963s |
| 50,000 | 0.004451s | 0.027886s | **0.003539s** | 0.005779s |
| 100,000 | 0.016271s | 0.056831s | **0.007046s** | 0.013383s |
| 250,000 | 0.046222s | 0.137821s | **0.016495s** | 0.042812s |
| 500,000 | 0.108009s | 0.314501s | **0.035327s** | 0.097844s |
| 1,000,000 | 0.166369s | — | **0.063986s** | 0.212221s |
| 2,000,000 | 0.825778s | — | **0.132894s** | 0.557089s |
| 5,000,000 | 1.497698s | — | **0.292195s** | 1.324037s |
| 10,000,000 | 1.406438s | — | **0.573525s** | 3.289212s |
| 25,000,000 | 5.718849s | — | **1.516740s** | 8.763385s |
| 50,000,000 | 15.227346s | — | **2.688555s** | 19.872041s |

Deterministic MoM was skipped for $n > 500{,}000$ due to excessive runtime in pure Python (would take minutes for the largest inputs).

## 4. Analysis of Results

### 4.1 Constant Factors Matter More Than Asymptotics

All three selection algorithms (QuickSelect, MoM, Sampling) are $O(n)$, yet their practical performance differs by an order of magnitude. At $n = 500{,}000$, where all four ran:

| Algorithm | Time | Effective constant (relative to Sampling) |
|---|---|---|
| Sampling $n^{3/4}$ | 0.035s | 1.0x |
| Sort Baseline | 0.098s | 2.8x |
| QuickSelect | 0.108s | 3.1x |
| Median-of-Medians | 0.315s | 8.9x |

The $\sim$9x gap between MoM and Sampling — both theoretically $O(n)$ — illustrates that asymptotic class alone does not determine practical performance. The constant factor, which Big-O notation deliberately hides, dominates at every realistic input size.

### 4.2 Why Median-of-Medians Is Slowest

MoM's overhead comes from its pivot selection mechanism:
- **$n/5$ insertion sorts per recursion level:** Even though each is $O(1)$, there are $n/5$ of them, each involving memory reads, comparisons, and writes in pure Python.
- **Recursive pivot finding:** The $T(n/5)$ call to find the median of medians is pure overhead with no analog in the other algorithms. It itself triggers another round of $n/25$ insertion sorts, and so on.
- **Poor cache behavior:** The recursion tree branches into $T(n/5)$ and $T(7n/10)$, both of which sweep over overlapping regions of the array, causing cache thrashing.
- **Python interpreter overhead:** Every comparison and swap goes through Python's bytecode interpreter. With $\sim 24n$ operations, this cost is 24x the cost of a single optimized C pass.

### 4.3 Why Sampling $n^{3/4}$ Wins at Scale

The sampling algorithm's advantage is structural — it converts the selection problem into:
1. One call to `random.sample()` — implemented in C, $O(n^{3/4})$.
2. One call to `sorted()` on $n^{3/4}$ elements — C Timsort, $O(n^{3/4} \log n)$.
3. One pure-Python linear scan — $O(n)$, but with only a comparison and occasional list append per element (no swaps, no partitioning).
4. One call to `sorted()` on $O(n^{3/4})$ elements — again C Timsort.

The only $O(n)$ step (the linear scan) has the lightest possible per-element work: a comparison against two bounds and a conditional append. Contrast this with QuickSelect, whose $O(n)$ partition step involves comparisons *and* swaps (array element exchanges) in pure Python.

At $n = 50{,}000{,}000$:
- Sampling: $n^{3/4} \approx 5{,}946$ elements to sort. The two sorts take microseconds. The linear scan dominates at $\sim$2.7s.
- QuickSelect: Multiple full partition passes ($\sim$3-4 on average), each doing $O(n)$ swaps in Python. Total $\sim$15.2s.
- Sort Baseline: Timsort in C on all 50M elements. Despite C optimization, $O(n \log n)$ with $\log_2(50M) \approx 26$ makes it $\sim$19.9s.

### 4.4 Scaling Behavior

Approximate empirical scaling from $n = 1M$ to $n = 50M$ (a 50x increase):

| Algorithm | Time at 1M | Time at 50M | Growth factor | Expected (theory) |
|---|---|---|---|---|
| Sampling $n^{3/4}$ | 0.064s | 2.689s | 42x | 50x (linear) |
| QuickSelect | 0.166s | 15.227s | 92x | 50x (linear) |
| Sort Baseline | 0.212s | 19.872s | 94x | 85x ($n \log n$) |

Sampling scales closest to the theoretical linear prediction. QuickSelect and Sort show super-linear empirical growth due to cache effects and Python's memory management overhead on large arrays.

### 4.5 The Tradeoff Space

| Algorithm | Worst-case | Expected | Constant | Randomness needed | Practical |
|---|---|---|---|---|---|
| Sort Baseline | $O(n \log n)$ | $O(n \log n)$ | Very low (C) | None | Best for $n < 5K$ |
| QuickSelect | $O(n^2)$ | $O(n)$ | Low ($\sim 3.4n$) | $O(\log n)$ pivots | Good general purpose |
| MoM (BFPRT) | $O(n)$ | $O(n)$ | High ($\sim 24n$) | None | Theoretical interest |
| Sampling $n^{3/4}$ | $O(n)$ w.h.p. | $O(n)$ | Very low ($\sim 2n$) | One sample | Best for large $n$ |

The sampling algorithm achieves the best practical performance by exploiting concentration of measure: a sublinear random sample carries enough information to locate the median with high probability, reducing the problem to a single linear scan and a tiny sort. This is a powerful demonstration of how randomization can simultaneously improve both theoretical guarantees and practical constants.

## 5. Correctness Verification

All reported medians were verified by an independent checker that computes the ground truth via sorting and compares it against each algorithm's output. All **54 results across 15 test cases and 4 algorithms** were confirmed correct.
