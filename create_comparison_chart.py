#!/usr/bin/env python3
"""
Create performance comparison chart similar to the reference image
Comparing Reference vs Optimized Submissions on H100
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import math

# Benchmark configurations - calculate total operations
BENCHMARK_CONFIGS = [
    {"seqlen": 256, "bs": 2, "dim": 128, "hiddendim": 128},
    {"seqlen": 768, "bs": 1, "dim": 128, "hiddendim": 128},
    {"seqlen": 256, "bs": 2, "dim": 384, "hiddendim": 128},
    {"seqlen": 512, "bs": 1, "dim": 128, "hiddendim": 128},
    {"seqlen": 1024, "bs": 1, "dim": 128, "hiddendim": 128},
    {"seqlen": 768, "bs": 1, "dim": 384, "hiddendim": 128},
    {"seqlen": 1024, "bs": 1, "dim": 384, "hiddendim": 128}
]

def calculate_total_operations(seqlen, dim):
    """Calculate total operations: seqlen^2 * dim (simplified)"""
    return seqlen**2 * dim

def load_results():
    """Load benchmark results from README data"""
    results = {}

    # CUDA Naive implementation (from README.md)
    results["cuda_naive"] = {
        "geo_mean": 35.107,
        "times": [6.554, 44.610, 14.358, 15.014, 87.692, 79.493, 149.596]
    }

    # PyTorch Reference (baseline PyTorch implementation)
    results["pytorch_reference"] = {
        "geo_mean": 12.308,
        "times": [3.355, 16.043, 5.314, 6.049, 29.843, 21.378, 38.758]
    }

    # PyTorch Optimized (submission.py / submission_v2.py)
    results["pytorch_optimized"] = {
        "geo_mean": 4.201,
        "times": [1.311, 5.180, 1.606, 2.308, 10.754, 6.482, 13.164]
    }

    # PyTorch Hybrid with Triton (submission_improved_triton.py)
    results["pytorch_hybrid"] = {
        "geo_mean": 2.399,
        "times": [0.411, 3.945, 0.733, 0.882, 6.878, 5.954, 10.657]
    }

    return results

def create_chart():
    """Create comparison chart"""
    results = load_results()

    # Calculate total operations for each benchmark
    total_ops = [calculate_total_operations(c["seqlen"], c["dim"]) for c in BENCHMARK_CONFIGS]

    # Calculate geometric means
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON - H100 GPU")
    print("="*60)
    for name, data in sorted(results.items(), key=lambda x: x[1]["geo_mean"]):
        print(f"{name:30} {data['geo_mean']:6.3f} ms")
    print()

    # Calculate speedups vs CUDA Naive baseline
    baseline_geo_mean = results["cuda_naive"]["geo_mean"]
    print("Speedup vs CUDA Naive (Baseline):")
    print("-"*40)
    for name, data in sorted(results.items(), key=lambda x: x[1]["geo_mean"]):
        if name != "cuda_naive":
            speedup = baseline_geo_mean / data["geo_mean"]
            print(f"{name:30} {speedup:5.2f}x")
    print()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Sort total_ops and corresponding times for smooth lines
    sorted_indices = np.argsort(total_ops)
    sorted_ops = np.array(total_ops)[sorted_indices]

    # Plot each implementation
    colors = {
        "cuda_naive": "#FF6B6B",
        "pytorch_reference": "#FFA07A",
        "pytorch_optimized": "#4ECDC4",
        "pytorch_hybrid": "#45B7D1"
    }

    linestyles = {
        "cuda_naive": "-",
        "pytorch_reference": "--",
        "pytorch_optimized": "-",
        "pytorch_hybrid": "-"
    }

    markers = {
        "cuda_naive": "o",
        "pytorch_reference": "x",
        "pytorch_optimized": "s",
        "pytorch_hybrid": "^"
    }

    labels = {
        "cuda_naive": f"CUDA Naive (Geo Mean: {results['cuda_naive']['geo_mean']:.2f}ms)",
        "pytorch_reference": f"PyTorch Reference (Geo Mean: {results['pytorch_reference']['geo_mean']:.2f}ms)",
        "pytorch_optimized": f"PyTorch Optimized (Geo Mean: {results['pytorch_optimized']['geo_mean']:.2f}ms)",
        "pytorch_hybrid": f"PyTorch Hybrid + Triton (Geo Mean: {results['pytorch_hybrid']['geo_mean']:.2f}ms)"
    }

    for name, data in results.items():
        # Sort times according to sorted ops for smooth lines
        sorted_times = np.array(data["times"])[sorted_indices]
        ax.plot(sorted_ops, sorted_times,
                marker=markers[name],
                linestyle=linestyles[name],
                color=colors[name],
                label=labels[name],
                linewidth=2,
                markersize=8)

    # Calculate speedup annotation
    baseline_mean = results["cuda_naive"]["geo_mean"]
    best_submission = min([(n, d["geo_mean"]) for n, d in results.items() if n != "cuda_naive"],
                          key=lambda x: x[1])
    speedup_vs_cuda = baseline_mean / best_submission[1]
    speedup_vs_pytorch = results["pytorch_optimized"]["geo_mean"] / best_submission[1]

    # Add speedup annotation
    ax.text(0.02, 0.98,
            f'PyTorch Hybrid: {speedup_vs_cuda:.2f}x faster than CUDA\n'
            f'PyTorch Hybrid: {speedup_vs_pytorch:.2f}x faster than PyTorch Optimized',
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Formatting
    ax.set_xlabel('Total Operations (seqlen² × dim)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Execution Time (ms)', fontsize=14, fontweight='bold')
    ax.set_title('TriMul GPU Performance: CUDA vs PyTorch Implementations\nH100 GPU Benchmark Comparison',
                 fontsize=16, fontweight='bold', pad=20)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=11, loc='upper left', framealpha=0.9)

    plt.tight_layout()
    plt.savefig('trimul_performance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Chart saved to: trimul_performance_comparison.png")

    return fig

if __name__ == "__main__":
    create_chart()
    # plt.show()  # Disabled for non-interactive environments
