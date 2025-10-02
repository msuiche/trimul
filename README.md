# TriMul - Triangle Multiplicative Update Kernel

Optimized implementation of the Triangle Multiplicative Update (TriMul) operator, a core operation in AlphaFold3, Chai, Protenix, and other protein structure prediction models.

## Overview

The TriMul operator processes 4D tensors of shape `[B, N, N, C]` representing pairwise features in protein structure prediction. This repository provides two implementations: **PyTorch Optimized** (recommended for production) and CUDA Naive (educational).

**Problem:** https://tinyurl.com/gpumode-trimul

## Performance

### Benchmark Results (H100 GPU)

![Implementation Comparison](implementation_comparison_h100.png)

| Implementation | Geometric Mean | Speedup vs Best | Tests Passed | Code Complexity |
|----------------|----------------|-----------------|--------------|-----------------|
| **PyTorch Optimized** ⭐ | **4.201 ms** | **1.00x** | ✓ 18/18 | Very Low (~100 LOC) |
| CUDA Naive | 35.107 ms | 0.12x | ✓ 18/18 | High (~700 LOC) |

**Key Findings:**
- ✅ **PyTorch Optimized is 8.36x faster than CUDA Naive**
- ✅ PyTorch uses FP16 precision, kernel fusion, and TF32 tensor cores
- ✅ Simplest code wins: PyTorch ~100 LOC vs CUDA ~700 LOC

### Detailed Benchmark Breakdown

| Config | PyTorch Opt | CUDA Naive | PyTorch Speedup |
|--------|-------------|------------|-----------------|
| seqlen:256, bs:2, dim:128 | **1.311ms** | 6.554ms | **5.0x** |
| seqlen:768, bs:1, dim:128 | **5.180ms** | 44.610ms | **8.6x** |
| seqlen:256, bs:2, dim:384 | **1.606ms** | 14.358ms | **8.9x** |
| seqlen:512, bs:1, dim:128 | **2.308ms** | 15.014ms | **6.5x** |
| seqlen:1024, bs:1, dim:128 | **10.754ms** | 87.692ms | **8.2x** |
| seqlen:768, bs:1, dim:384 | **6.482ms** | 79.493ms | **12.3x** |
| seqlen:1024, bs:1, dim:384 | **13.164ms** | 149.596ms | **11.4x** |

**Average Speedup:** PyTorch Optimized is **8.7x faster** than CUDA Naive across all benchmarks

### Directory Structure

```
.
├── pytorch/                         # PyTorch Optimized ⭐ RECOMMENDED
│   ├── submission.py                # FP16+TF32 optimized implementation
│   └── README.md                    # Documentation
│
├── cuda_naive/                      # CUDA Baseline (Educational)
│   ├── submission.py                # Simple CUDA kernels
│   ├── cuda_simple_kernels.cu       # Per-thread matmul
│   ├── cuda_simple_wrapper.cpp      # C++ wrapper
│   └── README.md                    # Documentation
│
├── eval.py                          # Evaluation harness
├── reference.py                     # Reference PyTorch implementation
├── task.py                          # Type definitions
├── task.yml                         # Task configuration
├── utils.py                         # Utility functions
├── run_modal.py                     # Modal deployment script
├── compare_implementations.py       # Generate comparison chart
└── implementation_comparison_h100.png  # Performance visualization
```

## Get Started

### Prerequisites
- Modal account (for H100 GPU access)
- Python 3.11+
- Modal CLI: `pip install modal`

### Running Tests

Test all implementations to verify correctness:

```bash
# Test PyTorch Optimized (recommended)
cd pytorch
MODAL_MODE=test MODAL_GPU=H100 MODAL_TASK=task.yml modal run ../run_modal.py

# Test CUDA Naive
cd cuda_naive
MODAL_MODE=test MODAL_GPU=H100 MODAL_TASK=task.yml modal run ../run_modal.py
```

### Running Benchmarks

Compare performance between implementations:

```bash
# Benchmark PyTorch Optimized (fastest)
cd pytorch
MODAL_JSON=1 MODAL_MODE=benchmark MODAL_GPU=H100 MODAL_TASK=task.yml modal run ../run_modal.py

# Benchmark CUDA Naive
cd cuda_naive
MODAL_JSON=1 MODAL_MODE=benchmark MODAL_GPU=H100 MODAL_TASK=task.yml modal run ../run_modal.py

# Generate comparison chart
cd ..
python compare_implementations.py
```

### Verbose Output

Enable detailed logging for debugging:

```bash
MODAL_VERBOSE=true MODAL_MODE=test MODAL_GPU=H100 MODAL_TASK=task.yml modal run ../run_modal.py
```

## Algorithm

The TriMul operator implements:

1. **Input LayerNorm** - Normalize input tensor `[B, N, N, D]`
2. **Projection** - Project to hidden dimension `H` via 5 linear layers
3. **Gating** - Apply sigmoid gates with optional masking
4. **Einsum** - Core triangle multiplication: `bhik,bhjk->bhij`
5. **Output Norm** - LayerNorm on result
6. **Final Projection** - Project back to dimension `D`

### Reference Implementation (PyTorch)
```python
x = LayerNorm(x)                          # [B, N, N, D]
left = sigmoid(left_gate(x)) * left_proj(x) * mask
right = sigmoid(right_gate(x)) * right_proj(x) * mask
out = einsum('bhik,bhjk->bhij', left, right)  # [B, H, N, N]
out = sigmoid(out_gate(x)) * LayerNorm(out)   # [B, N, N, H]
return to_out(out)                        # [B, N, N, D]
```

## Implementations

### pytorch (Optimized) ⭐ RECOMMENDED
- **Language:** Pure PyTorch with FP16+TF32
- **Strategy:** Kernel fusion, mixed precision, TF32 tensor cores
- **Optimizations:** FP16 matmuls, fused operations, minimal reshapes
- **Performance:** 4.201 ms geometric mean (FASTEST!)
- **Code:** ~100 lines (simplest!)
- **Use case:** Production deployment - best performance with simplest code

### cuda_naive
- **Language:** Pure CUDA C++
- **Kernels:** Simple per-thread computation
- **Optimization:** None (baseline)
- **Performance:** 35.107 ms geometric mean (8.4x slower than PyTorch)
- **Code:** ~700 lines (most complex)
- **Use case:** Educational, understanding GPU fundamentals

## Test Suite

### Tests (18 configurations)
- Sequence lengths: 32, 64, 128, 256, 512, 768, 1024
- Batch sizes: 1-2
- Dimensions: 128, 256, 384, 768
- Hidden dim: 128
- Distributions: Normal, Cauchy
- Masking: With/without

### Benchmarks (7 configurations)
```python
{"seqlen": 256,  "bs": 2, "dim": 128, "hiddendim": 128}
{"seqlen": 768,  "bs": 1, "dim": 128, "hiddendim": 128}
{"seqlen": 256,  "bs": 2, "dim": 384, "hiddendim": 128}
{"seqlen": 512,  "bs": 1, "dim": 128, "hiddendim": 128}
{"seqlen": 1024, "bs": 1, "dim": 128, "hiddendim": 128}
{"seqlen": 768,  "bs": 1, "dim": 384, "hiddendim": 128}
{"seqlen": 1024, "bs": 1, "dim": 384, "hiddendim": 128}
```

Ranking: Geometric mean of execution times across all benchmarks

## Environment Variables

```bash
MODAL_MODE=test|benchmark              # Execution mode (required)
MODAL_GPU=H100|A100                    # GPU type (default: H100)
MODAL_TASK=task.yml                    # Task configuration file (required when in subdirectory)
MODAL_JSON=1                           # Output results as JSON (REQUIRED for saving benchmarks)
MODAL_VERBOSE=true                     # Enable verbose output (optional)
```

**Key Variable:**
- `MODAL_JSON=1` - **Always use this when running benchmarks** to save results in JSON format. Without this flag, results are only printed to console and cannot be used for comparison charts.

## Why PyTorch Optimized Wins

The PyTorch Optimized implementation achieves **8.36x speedup over hand-written CUDA** while being the **simplest implementation**:

### Performance Advantages
1. **FP16 Mixed Precision**
   - Uses H100's TF32 tensor cores for matrix multiplications
   - FP16 for intermediate computations
   - FP32 only where precision matters (LayerNorm)

2. **Kernel Fusion**
   - Single bmm operation for einsum
   - Fused gating operations
   - Minimal memory traffic

3. **Optimal Memory Layout**
   - Contiguous tensors
   - Minimal reshapes
   - Efficient use of H100 memory hierarchy

### Code Simplicity - PyTorch WINS!
| Metric | PyTorch Opt | CUDA Naive |
|--------|-------------|------------|
| Lines of Code | ~100 | ~700 |
| Files | 1 (.py) | 3 (.cu, .cpp, .py) |
| Complexity | **Very Low** | High |
| Maintenance | **Easiest** | Difficult |
| Performance | **4.2ms (BEST)** | 35.1ms |

### Developer Experience
- ✅ **Pure PyTorch** - No custom kernels needed
- ✅ **Framework optimizations** - PyTorch handles everything
- ✅ **Instant testing** - No compilation needed
- ✅ **Most maintainable** - Standard PyTorch operations
- ✅ **Best performance** - 8.4x faster than CUDA!

### The Surprising Result
**PyTorch with proper optimization (FP16+TF32) beats hand-written CUDA!**

This shows that:
- H100's tensor cores are incredibly powerful
- PyTorch's cuBLAS integration is highly optimized
- Simple mixed-precision can outperform naive custom kernels
- Framework defaults + FP16 often win over complex custom code!

### Conclusion
For production deployment of TriMul, **use PyTorch Optimized**. It delivers:
- ✅ Best performance (4.2ms)
- ✅ Simplest code (~100 LOC)
- ✅ Easiest to maintain
- ✅ No custom kernels needed

## References

- AlphaFold3 Paper: [Nature 2024](https://www.nature.com/articles/s41586-024-07487-w)
- Reference Implementation: https://github.com/lucidrains/triangle-multiplicative-module
- GPU Mode: https://tinyurl.com/gpumode-trimul
