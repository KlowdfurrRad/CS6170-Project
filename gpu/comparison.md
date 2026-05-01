# CPU vs GPU Median-Finding Benchmark Comparison

GPU: NVIDIA GeForce RTX 3050 Laptop GPU, CUDA 12.6

Algorithm key:
- **QS**: Randomized QuickSelect (CPU, pure Python)
- **MoM**: Deterministic Median-of-Medians (CPU, pure Python)
- **Sample-CPU**: Sampling n^{3/4} (CPU, pure Python)
- **Sort**: Sort Baseline (CPU, Python Timsort in C)
- **Sample-GPU**: Sampling n^{3/4} (GPU, CUDA + Thrust)

All times in seconds. Best time per row in **bold**.

| n | QS | MoM | Sample-CPU | Sort | Sample-GPU | GPU Speedup vs Best CPU |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.000038 | 0.000052 | 0.000055 | **0.000007** | 0.005205 | 0.00x (CPU faster) |
| 500 | 0.000088 | 0.000213 | 0.000084 | **0.000034** | 0.005160 | 0.01x (CPU faster) |
| 1,000 | 0.000170 | 0.000525 | 0.000113 | **0.000073** | 0.007540 | 0.01x (CPU faster) |
| 5,000 | 0.000681 | 0.002754 | 0.000468 | **0.000442** | 0.004785 | 0.09x (CPU faster) |
| 10,000 | 0.001499 | 0.005311 | **0.000866** | 0.000963 | 0.005205 | 0.17x (CPU faster) |
| 50,000 | 0.004451 | 0.027886 | **0.003539** | 0.005779 | 0.005398 | 0.66x (CPU faster) |
| 100,000 | 0.016271 | 0.056831 | **0.007046** | 0.013383 | 0.008648 | 0.81x (CPU faster) |
| 250,000 | 0.046222 | 0.137821 | 0.016495 | 0.042812 | **0.005594** | **2.95x** |
| 500,000 | 0.108009 | 0.314501 | 0.035327 | 0.097844 | **0.008010** | **4.41x** |
| 1,000,000 | 0.166369 | skipped | 0.063986 | 0.212221 | **0.009510** | **6.73x** |
| 2,000,000 | 0.825778 | skipped | 0.132894 | 0.557089 | **0.011449** | **11.61x** |
| 5,000,000 | 1.497698 | skipped | 0.292195 | 1.324037 | **0.019142** | **15.26x** |
| 10,000,000 | 1.406438 | skipped | 0.573525 | 3.289212 | **0.032345** | **17.73x** |
| 25,000,000 | 5.718849 | skipped | 1.516740 | 8.763385 | **0.068869** | **22.02x** |
| 50,000,000 | 15.227346 | skipped | 2.688555 | 19.872041 | **0.152598** | **17.62x** |

## Observations

- At small n, the GPU version is slower due to kernel launch overhead and host-to-device transfer latency.
- As n grows, the GPU's massively parallel count/filter/sort operations amortize the fixed overhead and overtake all CPU methods.
- The sampling n^{3/4} algorithm is especially well-suited for GPU acceleration because its dominant operations (count_if, copy_if, sort) map directly to Thrust primitives that saturate GPU memory bandwidth.

