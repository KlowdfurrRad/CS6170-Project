/*
 * cuda_median.cu -- GPU-accelerated median finding via n^{3/4} sampling.
 *
 * Algorithm (Motwani & Raghavan / Floyd-Rivest style, GPU-parallel):
 *   1. Draw a random sample S of size ceil(n^{3/4}) from the input.
 *   2. Sort S on the GPU using thrust::sort.
 *   3. Pick bracket elements L = S[|S|/2 - sqrt(n)], R = S[|S|/2 + sqrt(n)].
 *   4. In parallel (thrust): count elements < L, copy elements in [L, R] into C.
 *   5. If the median rank falls within C and |C| <= 4*n^{3/4}, sort C and extract.
 *   6. Otherwise retry (probability O(n^{-1/4})).
 *
 * The GPU accelerates: sorting the sample, the linear scan (count + filter),
 * and sorting the filtered set C.
 *
 * Usage: cuda_median.exe <testcase_file>
 * Output: median=<value> time=<seconds>s
 *
 * Build: nvcc -O3 -o cuda_median.exe cuda_median.cu
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <random>
#include <algorithm>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/functional.h>

// ---------------------------------------------------------------------------
// Host: read test case file
// ---------------------------------------------------------------------------
int* read_testcase(const char* path, int* out_n) {
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", path);
        exit(1);
    }
    int n;
    if (fscanf(f, "%d", &n) != 1) {
        fprintf(stderr, "Cannot read n from %s\n", path);
        exit(1);
    }
    int* data = (int*)malloc((size_t)n * sizeof(int));
    if (!data) {
        fprintf(stderr, "Malloc failed for n=%d\n", n);
        exit(1);
    }
    for (int i = 0; i < n; i++) {
        if (fscanf(f, "%d", &data[i]) != 1) {
            fprintf(stderr, "Cannot read element %d from %s\n", i, path);
            exit(1);
        }
    }
    fclose(f);
    *out_n = n;
    return data;
}

// ---------------------------------------------------------------------------
// Functors for thrust operations
// ---------------------------------------------------------------------------
struct less_than_val {
    int L;
    __host__ __device__ bool operator()(int x) const { return x < L; }
};

struct in_bracket {
    int L, R;
    __host__ __device__ bool operator()(int x) const { return x >= L && x <= R; }
};

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <testcase_file>\n", argv[0]);
        return 1;
    }

    // Read input
    int n;
    int* h_data = read_testcase(argv[1], &n);
    int k = (n - 1) / 2;  // 0-indexed lower median rank

    // Warm up: force CUDA context initialization before timing
    cudaFree(0);

    // Start timing (after file I/O and CUDA init)
    auto t_start = std::chrono::high_resolution_clock::now();

    // Copy full array to device
    thrust::device_vector<int> d_data(h_data, h_data + n);

    // Sampling parameters
    int sample_size = (int)ceil(pow((double)n, 0.75));
    int offset = (int)ceil(sqrt((double)n));
    int max_C_size = 4 * sample_size;

    std::mt19937 rng(42);

    while (true) {
        // Step 1: draw a random sample of size n^{3/4} (on host)
        // Fisher-Yates partial shuffle to pick sample_size indices
        thrust::host_vector<int> h_sample(sample_size);
        {
            // Copy indices into a temporary array for sampling
            std::vector<int> indices(n);
            for (int i = 0; i < n; i++) indices[i] = i;
            for (int i = 0; i < sample_size; i++) {
                std::uniform_int_distribution<int> dist(i, n - 1);
                int j = dist(rng);
                std::swap(indices[i], indices[j]);
                h_sample[i] = h_data[indices[i]];
            }
        }

        // Step 2: sort the sample on GPU
        thrust::device_vector<int> d_sample = h_sample;
        thrust::sort(d_sample.begin(), d_sample.end());

        // Step 3: pick bracket elements
        int half = sample_size / 2;
        int lo_idx = half - offset;
        if (lo_idx < 0) lo_idx = 0;
        int hi_idx = half + offset;
        if (hi_idx >= sample_size) hi_idx = sample_size - 1;

        int L = d_sample[lo_idx];
        int R = d_sample[hi_idx];

        // Step 4: count elements < L (on GPU, parallel)
        less_than_val lt_pred;
        lt_pred.L = L;
        int count_less = (int)thrust::count_if(d_data.begin(), d_data.end(), lt_pred);

        // Step 5: check if median rank falls in [count_less, count_less + |C| - 1]
        //         and estimate |C| first via count
        in_bracket bracket_pred;
        bracket_pred.L = L;
        bracket_pred.R = R;
        int C_size = (int)thrust::count_if(d_data.begin(), d_data.end(), bracket_pred);

        if (count_less <= k && k < count_less + C_size && C_size <= max_C_size) {
            // Step 6: copy elements in [L, R] into C on GPU
            thrust::device_vector<int> d_C(C_size);
            thrust::copy_if(d_data.begin(), d_data.end(), d_C.begin(), bracket_pred);

            // Step 7: sort C on GPU
            thrust::sort(d_C.begin(), d_C.end());

            // Extract the median
            int median = d_C[k - count_less];

            auto t_end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t_end - t_start).count();
            printf("median=%d time=%.6fs\n", median, elapsed);

            free(h_data);
            return 0;
        }

        // Bad sample (extremely rare), retry with a new seed
        rng.seed(rng());
    }

    free(h_data);
    return 0;
}
