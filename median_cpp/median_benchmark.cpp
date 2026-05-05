/**
 * median_benchmark.cpp
 *
 * Three median-finding algorithms benchmarked across five dataset types:
 *   random, random01, organpipe, sorted, rotated
 *
 * Algorithms (each implemented to match the description in the
 * accompanying LaTeX report, Section 3 -- Background & Related Work):
 *   1. Randomized QuickSelect (Hoare's FIND, with 3-way Dutch National
 *      Flag partition for correctness on duplicate-heavy inputs).
 *   2. Deterministic Median-of-Medians (Blum, Floyd, Pratt, Rivest,
 *      Tarjan -- BFPRT) with 3-way partition.
 *   3. LazySelect -- the n^{3/4} lazy sampling algorithm of
 *      Motwani-Raghavan (Randomized Algorithms, Sec. 3.3).
 *      Sampling is performed *with replacement*, fences are placed
 *      at x +/- sqrt(n) where x = k * n^{-1/4}, and the algorithm
 *      restarts if the verification step fails.
 *
 * Compile:
 *   g++ -O2 -std=c++17 -o median_benchmark median_benchmark.cpp
 */

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// ============================================================
//  Utility: lower median rank (0-indexed)
// ============================================================
static int lower_median_rank(int n) { return (n - 1) / 2; }

// Single global RNG (seeded from std::random_device).
static std::mt19937_64 rng(std::random_device{}());

// ============================================================
//  1. Randomized QuickSelect  --  expected O(n), worst O(n^2)
//
//  Matches Algorithm "QuickSelect(S, k)" in the report:
//     pick a uniform random pivot p in S,
//     partition into L = { y < p }, E = { y = p }, H = { y > p },
//     recurse into the side that contains the k-th order statistic.
//
//  We use Dutch National Flag (3-way) partitioning in place, which
//  is exactly the L/E/H decomposition above, done in O(n) time.
// ============================================================
static void three_way_partition(std::vector<double>& a, int lo, int hi,
                                int& lt, int& gt) {
    int pivot_idx = std::uniform_int_distribution<int>(lo, hi)(rng);
    double pivot = a[pivot_idx];
    int i = lo;
    lt = lo;
    gt = hi;
    while (i <= gt) {
        if      (a[i] < pivot) std::swap(a[lt++], a[i++]);
        else if (a[i] > pivot) std::swap(a[i],    a[gt--]);
        else                   ++i;
    }
}

double randomized_median(std::vector<double> a) {
    int n  = (int)a.size();
    int k  = lower_median_rank(n);
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int lt, gt;
        three_way_partition(a, lo, hi, lt, gt);
        if      (k < lt) hi = lt - 1;     // recurse into L
        else if (k > gt) lo = gt + 1;     // recurse into H
        else             return a[k];     // k lies in E -- return pivot
    }
    return a[lo];
}

// ============================================================
//  2. Deterministic Median-of-Medians (BFPRT)  --  worst O(n)
//
//  Matches Algorithm "Select(S, k) -- Median-of-Medians" in the
//  report:
//     - groups of 5, find each group's median by insertion sort,
//     - recursively compute the median of medians as pivot,
//     - 3-way partition around that pivot, recurse on the side
//       containing the k-th order statistic.
//
//  Using a 3-way partition (rather than the 2-way partition in the
//  textbook) keeps duplicate-heavy inputs (e.g. random01) linear
//  rather than quadratic, while leaving the asymptotic guarantee
//  T(n) = T(n/5) + T(7n/10) + O(n) = O(n) intact.
// ============================================================
static void insertion_sort(std::vector<double>& a, int lo, int hi) {
    for (int i = lo + 1; i <= hi; ++i) {
        double key = a[i];
        int j = i - 1;
        while (j >= lo && a[j] > key) { a[j + 1] = a[j]; --j; }
        a[j + 1] = key;
    }
}

// Returns the index in [lo, hi] of the median-of-medians pivot.
static int median_of_medians(std::vector<double>& a, int lo, int hi) {
    int n = hi - lo + 1;
    if (n <= 5) {
        insertion_sort(a, lo, hi);
        return lo + n / 2;
    }
    int num_groups = n / 5;
    for (int i = 0; i < num_groups; ++i) {
        int g_lo = lo + i * 5, g_hi = g_lo + 4;
        insertion_sort(a, g_lo, g_hi);
        // Move each group's median to the front block [lo, lo+num_groups).
        std::swap(a[lo + i], a[g_lo + 2]);
    }
    return median_of_medians(a, lo, lo + num_groups - 1);
}

// 3-way partition around the value at `pivot_idx`, restricted to [lo, hi].
// Returns (lt, gt) such that a[lo..lt-1] < pivot, a[lt..gt] == pivot,
// a[gt+1..hi] > pivot.
static void mom_three_way(std::vector<double>& a, int lo, int hi,
                          int pivot_idx, int& lt, int& gt) {
    double pivot = a[pivot_idx];
    int i = lo;
    lt = lo;
    gt = hi;
    while (i <= gt) {
        if      (a[i] < pivot) std::swap(a[lt++], a[i++]);
        else if (a[i] > pivot) std::swap(a[i],    a[gt--]);
        else                   ++i;
    }
}

double deterministic_median(std::vector<double> a) {
    int n  = (int)a.size();
    int k  = lower_median_rank(n);
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int pivot_idx = median_of_medians(a, lo, hi);
        int lt, gt;
        mom_three_way(a, lo, hi, pivot_idx, lt, gt);
        if      (k < lt) hi = lt - 1;
        else if (k > gt) lo = gt + 1;
        else             return a[k];
    }
    return a[lo];
}

// ============================================================
//  3. LazySelect  --  O(n) with high probability
//
//  Faithful implementation of the algorithm in the report
//  (Motwani-Raghavan, lazy sampling):
//     repeat:
//       1. R = multiset of ceil(n^{3/4}) elements drawn uniformly
//          at random *with replacement* from S; sort R.
//       2. Set x = k * n^{-1/4}; ell = max(floor(x - sqrt n), 1);
//          h   = min(ceil(x + sqrt n), n^{3/4}).
//          a = R_{(ell)}, b = R_{(h)} are the order-statistic fences.
//       3. Single linear scan over S: compute r_a = rank_S(a)
//          and P = { y in S : a <= y <= b }.
//       4. If r_a <= k <= r_b and |P| <= 4 n^{3/4} + 2, sort P
//          and return P with rank k - r_a (0-indexed).
//          Else restart.
//
//  Using 0-indexed ranks throughout (k = (n-1)/2).
// ============================================================
//
//  Practical note: the textbook algorithm assumes the elements of S
//  are distinct (or very nearly so). On adversarial inputs with
//  extreme multiplicity -- e.g. a binary {0,1} array -- the candidate
//  set P deterministically equals all of S in every pass, so |P| > 4
//  n^{3/4}+2 and the algorithm restarts forever. To keep the
//  benchmark driver from looping indefinitely on such inputs, we add
//  a safety bound: after MAX_PASSES failed verification attempts we
//  fall back to a deterministic median via std::sort. This fallback
//  is *never* triggered on the inputs covered by the textbook
//  analysis (Section 3.3) -- it is a runtime safety net only.
//
static constexpr int LAZY_SELECT_MAX_PASSES = 32;

double lazy_select(const std::vector<double>& arr) {
    int n = (int)arr.size();
    int k = lower_median_rank(n);
    int sample_size =
        std::max(1, (int)std::ceil(std::pow((double)n, 0.75)));
    int max_p = 4 * sample_size + 2;

    std::uniform_int_distribution<int> uni(0, n - 1);

    for (int pass = 0; pass < LAZY_SELECT_MAX_PASSES; ++pass) {
        // ---- Step 1: Sample R with replacement and sort. ---------
        std::vector<double> R(sample_size);
        for (int i = 0; i < sample_size; ++i)
            R[i] = arr[uni(rng)];
        std::sort(R.begin(), R.end());

        // ---- Step 2: Choose fences. ------------------------------
        double x      = (double)k * std::pow((double)n, -0.25);
        double sqrt_n = std::sqrt((double)n);
        int ell = std::max((int)std::floor(x - sqrt_n), 0);
        int h   = std::min((int)std::ceil (x + sqrt_n), sample_size - 1);
        double a = R[ell];
        double b = R[h];

        // ---- Step 3: One linear scan over S. ---------------------
        // r_a = #{ y in S : y < a }     (0-indexed rank of a)
        // P   = { y in S : a <= y <= b }
        int r_a = 0;
        std::vector<double> P;
        P.reserve(max_p + 4);
        for (double y : arr) {
            if      (y < a) ++r_a;
            else if (y <= b) P.push_back(y);
        }
        int p_size = (int)P.size();

        // ---- Step 4: Verify and finish. --------------------------
        // We need k in [r_a, r_a + |P| - 1] (i.e. r_a <= k <= r_b)
        // and |P| <= 4 n^{3/4} + 2.
        if (r_a <= k && k < r_a + p_size && p_size <= max_p) {
            std::sort(P.begin(), P.end());
            return P[k - r_a];
        }
        // Otherwise restart this pass.
    }

    // Safety fallback: only reached on inputs with degenerate
    // multiplicity (e.g. random01) where the textbook algorithm
    // cannot terminate. Returns the correct median via std::sort.
    std::vector<double> tmp(arr);
    std::sort(tmp.begin(), tmp.end());
    return tmp[k];
}

// ============================================================
//  Dataset generators
// ============================================================
std::vector<double> gen_random(int n) {
    std::uniform_real_distribution<double> dist(-1e9, 1e9);
    std::vector<double> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

std::vector<double> gen_random01(int n) {
    std::vector<double> v(n);
    for (int i = 0; i < n; ++i) v[i] = (i < n / 2) ? 0.0 : 1.0;
    std::shuffle(v.begin(), v.end(), rng);
    return v;
}

std::vector<double> gen_organpipe(int n) {
    // 0,1,2,...,n/2-1, n/2-1,...,1,0
    std::vector<double> v(n);
    for (int i = 0; i < n; ++i)
        v[i] = (i < n / 2) ? (double)i : (double)(n - 1 - i);
    return v;
}

std::vector<double> gen_sorted(int n) {
    std::vector<double> v(n);
    std::iota(v.begin(), v.end(), 0.0);
    return v;
}

std::vector<double> gen_rotated(int n) {
    // 1,2,...,n-1,0
    std::vector<double> v(n);
    for (int i = 0; i < n - 1; ++i) v[i] = (double)(i + 1);
    v[n - 1] = 0.0;
    return v;
}

// ============================================================
//  Ground-truth checker
// ============================================================
double ground_truth(std::vector<double> a) {
    std::sort(a.begin(), a.end());
    return a[lower_median_rank((int)a.size())];
}

// ============================================================
//  Main benchmark driver
// ============================================================
using AlgoFn = std::function<double(std::vector<double>)>;

struct AlgoEntry {
    std::string name;
    AlgoFn fn;
    int max_n;   // 0 = no cap
};

using DataFn = std::function<std::vector<double>(int)>;

int main(int argc, char* argv[]) {
    std::string ds_filter = (argc > 1) ? std::string(argv[1]) : "";

    // No size caps: the user explicitly requested running every
    // algorithm at every size up to 10^7, even when slow.
    std::vector<AlgoEntry> algos = {
        {"QuickSelect", AlgoFn(randomized_median),    0},
        {"MoM-BFPRT",   AlgoFn(deterministic_median), 0},
        {"LazySelect",  AlgoFn(lazy_select),          0},
    };

    std::vector<std::pair<std::string, DataFn>> datasets = {
        {"random",    DataFn(gen_random)},
        {"random01",  DataFn(gen_random01)},
        {"organpipe", DataFn(gen_organpipe)},
        {"sorted",    DataFn(gen_sorted)},
        {"rotated",   DataFn(gen_rotated)},
    };

    // Sizes up to 10^7, log-spaced (with 5x10^k checkpoints).
    std::vector<int> sizes = {
        1000, 10000, 100000, 500000,
        1000000, 5000000, 10000000
    };

    // CSV header
    std::cout << "dataset,n,algorithm,time_s,correct\n";
    std::cerr << std::fixed << std::setprecision(6);

    for (auto& [ds_name, ds_gen] : datasets) {
        if (!ds_filter.empty() && ds_name != ds_filter) continue;
        for (int n : sizes) {
            std::vector<double> arr = ds_gen(n);
            double gt = ground_truth(arr);

            std::cerr << "[" << ds_name << " n=" << n << "]\n";

            for (auto& algo : algos) {
                if (algo.max_n > 0 && n > algo.max_n) {
                    std::cout << ds_name << "," << n << ","
                              << algo.name << ",SKIP,SKIP\n";
                    continue;
                }

                auto t0 = std::chrono::high_resolution_clock::now();
                double result = algo.fn(arr);
                auto t1 = std::chrono::high_resolution_clock::now();
                double elapsed =
                    std::chrono::duration<double>(t1 - t0).count();

                bool ok = (std::abs(result - gt) < 1e-9);
                std::cout << ds_name << "," << n << ","
                          << algo.name << ","
                          << std::fixed << std::setprecision(6) << elapsed
                          << "," << (ok ? "OK" : "FAIL") << "\n";
                std::cout.flush();
            }
        }
    }
    return 0;
}
