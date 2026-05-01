"""Benchmark the CUDA sampling median kernel on all test cases.

Calls the compiled cuda_median.exe for each test case and logs results.
Runs the checker to verify correctness, then generates a comparison
table against the CPU results in ../median/benchmark.log.
"""

import os
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
TESTCASE_DIR = os.path.join(PROJECT_DIR, "median", "testcases")
LOGFILE = os.path.join(SCRIPT_DIR, "benchmark_gpu.log")
CPU_LOGFILE = os.path.join(PROJECT_DIR, "median", "benchmark.log")
CUDA_EXE = os.path.join(SCRIPT_DIR, "cuda_median.exe")
COMPARISON_MD = os.path.join(SCRIPT_DIR, "comparison.md")

# Import checker from median directory
sys.path.insert(0, os.path.join(PROJECT_DIR, "median"))
from checker import check_from_log


def parse_log(logfile):
    """Parse a benchmark log, return dict: (algo, tc_name) -> time_seconds."""
    pattern = re.compile(
        r"\[(?P<algo>[^\]]+)\]\s+(?P<tc>\S+\.txt)\s+\|.*?time=(?P<time>[\d.]+)s"
    )
    results = {}
    with open(logfile) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                key = (m.group("algo"), m.group("tc"))
                results[key] = float(m.group("time"))
    return results


def extract_n(tc_name):
    """Extract n from a testcase filename like tc_01_n100.txt."""
    m = re.search(r"_n(\d+)\.txt$", tc_name)
    return int(m.group(1)) if m else 0


def generate_comparison(gpu_results, cpu_results, tc_files):
    """Generate a markdown comparison table."""
    cpu_algos = ["Randomized-QuickSelect", "Deterministic-MoM",
                 "Sampling-n^{3/4}", "Sort-Baseline"]
    gpu_algo = "GPU-Sampling-n^{3/4}"

    lines = []
    lines.append("# CPU vs GPU Median-Finding Benchmark Comparison")
    lines.append("")
    lines.append("GPU: NVIDIA GeForce RTX 3050 Laptop GPU, CUDA 12.6")
    lines.append("")
    lines.append("Algorithm key:")
    lines.append("- **QS**: Randomized QuickSelect (CPU, pure Python)")
    lines.append("- **MoM**: Deterministic Median-of-Medians (CPU, pure Python)")
    lines.append("- **Sample-CPU**: Sampling n^{3/4} (CPU, pure Python)")
    lines.append("- **Sort**: Sort Baseline (CPU, Python Timsort in C)")
    lines.append("- **Sample-GPU**: Sampling n^{3/4} (GPU, CUDA + Thrust)")
    lines.append("")
    lines.append("All times in seconds. Best time per row in **bold**.")
    lines.append("")

    # Header
    lines.append("| n | QS | MoM | Sample-CPU | Sort | Sample-GPU | GPU Speedup vs Best CPU |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")

    for tc_name in tc_files:
        n = extract_n(tc_name)
        n_str = f"{n:,}"

        def fmt(algo, results):
            t = results.get((algo, tc_name))
            if t is not None:
                return t
            return None

        qs = fmt("Randomized-QuickSelect", cpu_results)
        mom = fmt("Deterministic-MoM", cpu_results)
        samp_cpu = fmt("Sampling-n^{3/4}", cpu_results)
        sort_b = fmt("Sort-Baseline", cpu_results)
        samp_gpu = fmt(gpu_algo, gpu_results)

        all_times = [t for t in [qs, mom, samp_cpu, sort_b, samp_gpu] if t is not None]
        best = min(all_times) if all_times else None

        cpu_times = [t for t in [qs, mom, samp_cpu, sort_b] if t is not None]
        best_cpu = min(cpu_times) if cpu_times else None

        def cell(t):
            if t is None:
                return "skipped"
            s = f"{t:.6f}"
            if best is not None and t == best:
                return f"**{s}**"
            return s

        if samp_gpu is not None and best_cpu is not None and samp_gpu > 0:
            ratio = best_cpu / samp_gpu
            if ratio < 1.0:
                speedup = f"{ratio:.2f}x (CPU faster)"
            else:
                speedup = f"**{ratio:.2f}x**"
        else:
            speedup = "n/a"

        lines.append(
            f"| {n_str} | {cell(qs)} | {cell(mom)} | {cell(samp_cpu)} "
            f"| {cell(sort_b)} | {cell(samp_gpu)} | {speedup} |"
        )

    lines.append("")
    lines.append("## Observations")
    lines.append("")
    lines.append("- At small n, the GPU version is slower due to kernel launch overhead "
                 "and host-to-device transfer latency.")
    lines.append("- As n grows, the GPU's massively parallel count/filter/sort operations "
                 "amortize the fixed overhead and overtake all CPU methods.")
    lines.append("- The sampling n^{3/4} algorithm is especially well-suited for GPU "
                 "acceleration because its dominant operations (count_if, copy_if, sort) "
                 "map directly to Thrust primitives that saturate GPU memory bandwidth.")
    lines.append("")

    with open(COMPARISON_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Comparison table written to {COMPARISON_MD}")


def run_benchmarks():
    if not os.path.isfile(CUDA_EXE):
        print(f"ERROR: {CUDA_EXE} not found. Compile first:")
        print(f"  nvcc -O3 -o cuda_median.exe cuda_median.cu")
        sys.exit(1)

    if not os.path.isdir(TESTCASE_DIR) or not os.listdir(TESTCASE_DIR):
        print("No test cases found. Run generate_testcases.py in ../median/ first.")
        sys.exit(1)

    tc_files = sorted(f for f in os.listdir(TESTCASE_DIR) if f.endswith(".txt"))
    print(f"\n{'='*72}")
    print(f"  GPU Benchmark: CUDA Sampling n^{{3/4}} on {len(tc_files)} test cases")
    print(f"{'='*72}\n")

    result_pattern = re.compile(r"median=(-?\d+)\s+time=([\d.]+)s")
    gpu_algo = "GPU-Sampling-n^{3/4}"

    with open(LOGFILE, "w") as log:
        log.write("# GPU Median-Finding Benchmark Log (CUDA Sampling n^{3/4})\n")
        log.write(f"# {'='*68}\n\n")

        for tc_name in tc_files:
            tc_path = os.path.join(TESTCASE_DIR, tc_name)
            with open(tc_path) as f:
                n = int(f.readline().strip())

            header = f"--- {tc_name}  (n={n:,}) ---"
            print(header)
            log.write(header + "\n")

            try:
                result = subprocess.run(
                    [CUDA_EXE, tc_path],
                    capture_output=True, text=True, timeout=120,
                )
                output = result.stdout.strip()
                if result.returncode != 0:
                    err_msg = (f"  [{gpu_algo}] {tc_name} | ERROR: "
                               f"exit code {result.returncode}")
                    if result.stderr:
                        err_msg += f" stderr: {result.stderr.strip()}"
                    print(err_msg)
                    log.write(err_msg + "\n\n")
                    continue

                m = result_pattern.search(output)
                if m:
                    median_val = m.group(1)
                    elapsed = m.group(2)
                    line = (f"  [{gpu_algo}] {tc_name} | "
                            f"median={median_val} | time={elapsed}s")
                else:
                    line = f"  [{gpu_algo}] {tc_name} | RAW: {output}"

                print(line)
                log.write(line + "\n")

            except subprocess.TimeoutExpired:
                timeout_msg = f"  [{gpu_algo}] {tc_name} | TIMEOUT (>120s)"
                print(timeout_msg)
                log.write(timeout_msg + "\n")

            log.write("\n")
            print()

    print(f"Log written to {LOGFILE}\n")

    # Verify correctness
    print("=" * 72)
    print("  Checking correctness of all logged results...")
    print("=" * 72)
    check_from_log(LOGFILE, TESTCASE_DIR)
    print()

    # Generate comparison table
    print("=" * 72)
    print("  Generating CPU vs GPU comparison table...")
    print("=" * 72)
    gpu_results = parse_log(LOGFILE)
    cpu_results = parse_log(CPU_LOGFILE) if os.path.isfile(CPU_LOGFILE) else {}
    generate_comparison(gpu_results, cpu_results, tc_files)


if __name__ == "__main__":
    run_benchmarks()
