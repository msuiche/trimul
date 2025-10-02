#!/usr/bin/env python3
"""
Create performance comparison chart for PyTorch Reference, CUDA Naive, and Triton Naive
on H100 GPU
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import re

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
    """Calculate total operations: seqlen^2 * dim"""
    return seqlen**2 * dim

def parse_benchmark_output(text):
    """Parse Modal benchmark output and extract times"""
    benchmarks = []
    geo_mean = None

    # Extract individual benchmark results
    benchmark_pattern = r'Benchmark #\d+:.*?Mean: ([\d.]+) ms'
    matches = re.findall(benchmark_pattern, text, re.DOTALL)

    benchmarks = [float(m) for m in matches]

    # Extract geometric mean
    geo_mean_match = re.search(r'Geometric Mean: ([\d.]+) ms', text)
    if geo_mean_match:
        geo_mean = float(geo_mean_match.group(1))

    return benchmarks, geo_mean

def load_benchmark_data():
    """Load or generate benchmark data for all implementations"""
    results = {}

    # Try to load existing benchmark data
    implementations = {
        "PyTorch Reference": None,  # We'll use manual values
        "CUDA Naive": "cuda_naive",
        "Triton Naive": "triton_naive"
    }

    # CUDA Naive - actual H100 benchmark
    results["CUDA Naive"] = {
        "geo_mean": 35.107,
        "times": [6.554, 44.610, 14.358, 15.014, 87.692, 79.493, 149.596],
        "color": "#FF6B6B",
        "marker": "s",
        "linestyle": "--"
    }

    # Triton Naive - actual H100 benchmark
    results["Triton Naive"] = {
        "geo_mean": 9.637,
        "times": [1.886, 16.638, 2.631, 4.086, 31.047, 19.968, 36.939],
        "color": "#4ECDC4",
        "marker": "^",
        "linestyle": "-"
    }

    # PyTorch Optimized - actual H100 benchmark (submission_v2.py with FP16+TF32)
    results["PyTorch Optimized"] = {
        "geo_mean": 4.201,
        "times": [1.311, 5.180, 1.606, 2.308, 10.754, 6.482, 13.164],
        "color": "#95E1D3",
        "marker": "o",
        "linestyle": "-."
    }

    return results

def create_comparison_chart():
    """Create performance comparison chart"""
    results = load_benchmark_data()

    # Calculate total operations for each benchmark
    total_ops = [calculate_total_operations(c["seqlen"], c["dim"]) for c in BENCHMARK_CONFIGS]

    # Print results
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON - H100 GPU")
    print("PyTorch Optimized vs CUDA Naive vs Triton Naive")
    print("="*70)
    for name in ["PyTorch Optimized", "Triton Naive", "CUDA Naive"]:
        if name in results:
            data = results[name]
            print(f"{name:25} {data['geo_mean']:7.3f} ms")
    print()

    # Calculate speedups
    pytorch_geo_mean = results["PyTorch Optimized"]["geo_mean"]
    cuda_geo_mean = results["CUDA Naive"]["geo_mean"]
    triton_geo_mean = results["Triton Naive"]["geo_mean"]

    print("Speedup Analysis:")
    print("-"*50)
    print(f"{'PyTorch vs Triton Naive:':<35} {triton_geo_mean / pytorch_geo_mean:5.2f}x")
    print(f"{'PyTorch vs CUDA Naive:':<35} {cuda_geo_mean / pytorch_geo_mean:5.2f}x")
    print(f"{'Triton vs CUDA Naive:':<35} {cuda_geo_mean / triton_geo_mean:5.2f}x")
    print()

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 9))

    # Sort by total operations for smooth lines
    sorted_indices = np.argsort(total_ops)
    sorted_ops = np.array(total_ops)[sorted_indices]

    # Plot each implementation
    for name in ["PyTorch Optimized", "Triton Naive", "CUDA Naive"]:
        if name in results:
            data = results[name]
            sorted_times = np.array(data["times"])[sorted_indices]

            label = f"{name} (Geo Mean: {data['geo_mean']:.2f}ms)"
            ax.plot(sorted_ops, sorted_times,
                    marker=data["marker"],
                    linestyle=data["linestyle"],
                    color=data["color"],
                    label=label,
                    linewidth=2.5,
                    markersize=10)

    # Add speedup annotations
    speedup_pytorch_vs_cuda = cuda_geo_mean / pytorch_geo_mean
    speedup_pytorch_vs_triton = triton_geo_mean / pytorch_geo_mean

    annotation_text = (
        f'PyTorch Optimized vs CUDA Naive: {speedup_pytorch_vs_cuda:.2f}x faster\\n'
        f'PyTorch Optimized vs Triton: {speedup_pytorch_vs_triton:.2f}x faster'
    )

    ax.text(0.02, 0.98, annotation_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Formatting
    ax.set_xlabel('Total Operations (seqlen² × dim)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Execution Time (ms)', fontsize=14, fontweight='bold')
    ax.set_title('TriMul Implementation Comparison on H100 GPU\\nPyTorch Optimized (FP16+TF32) vs Triton vs CUDA',
                 fontsize=16, fontweight='bold', pad=20)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both', linestyle=':')
    ax.legend(fontsize=12, loc='upper left', framealpha=0.95)

    plt.tight_layout()
    output_file = 'implementation_comparison_h100.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Chart saved to: {output_file}")

    return fig

if __name__ == "__main__":
    create_comparison_chart()
    plt.show()
