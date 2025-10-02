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
    """Load benchmark results from JSON files"""
    results = {}

    # Load submission results
    files = [
        ("submission_pt_4174", "results_pt_4174.json"),
        ("submission_pt_4189_final", "results_pt_4189_final.json"),
        ("submission_v2", "results_v2.json"),
    ]

    for name, filename in files:
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                results[name] = {
                    "geo_mean": data["geometric_mean_ms"],
                    "times": [b["mean_ms"] for b in data["benchmarks"]]
                }
        except Exception as e:
            print(f"Warning: Could not load {filename}: {e}")

    # Add reference implementation (from manual benchmark)
    results["reference"] = {
        "geo_mean": 12.308,
        "times": [3.355, 16.043, 5.314, 6.049, 29.843, 21.378, 38.758]
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

    # Calculate speedups vs reference
    ref_geo_mean = results["reference"]["geo_mean"]
    print("Speedup vs Reference:")
    print("-"*40)
    for name, data in sorted(results.items(), key=lambda x: x[1]["geo_mean"]):
        if name != "reference":
            speedup = ref_geo_mean / data["geo_mean"]
            print(f"{name:30} {speedup:5.2f}x")
    print()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Sort total_ops and corresponding times for smooth lines
    sorted_indices = np.argsort(total_ops)
    sorted_ops = np.array(total_ops)[sorted_indices]

    # Plot each implementation
    colors = {
        "reference": "#FF6B6B",
        "submission_pt_4174": "#4ECDC4",
        "submission_pt_4189_final": "#45B7D1",
        "submission_v2": "#95E1D3"
    }

    linestyles = {
        "reference": "-",
        "submission_pt_4174": "--",
        "submission_pt_4189_final": "-",
        "submission_v2": "-."
    }

    markers = {
        "reference": "o",
        "submission_pt_4174": "s",
        "submission_pt_4189_final": "^",
        "submission_v2": "D"
    }

    labels = {
        "reference": f"Reference (Geo Mean: {results['reference']['geo_mean']:.2f}ms)",
        "submission_pt_4174": f"Submission PT 4174 (Geo Mean: {results['submission_pt_4174']['geo_mean']:.2f}ms)",
        "submission_pt_4189_final": f"Submission PT 4189 Final (Geo Mean: {results['submission_pt_4189_final']['geo_mean']:.2f}ms)",
        "submission_v2": f"Submission V2 (Geo Mean: {results['submission_v2']['geo_mean']:.2f}ms)"
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
    ref_mean = results["reference"]["geo_mean"]
    best_submission = min([(n, d["geo_mean"]) for n, d in results.items() if n != "reference"],
                          key=lambda x: x[1])
    speedup = ref_mean / best_submission[1]

    # Add speedup annotation
    ax.text(0.02, 0.98, f'Submission v2 vs Reference: {speedup:.2f}x',
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Formatting
    ax.set_xlabel('Total Operations (seqlen² × dim)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Execution Time (ms)', fontsize=14, fontweight='bold')
    ax.set_title('TriMul GPU Performance: Reference vs Submission v2\nH100-Optimized FP16 Implementation',
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
    plt.show()
