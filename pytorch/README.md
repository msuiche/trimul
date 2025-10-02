# Submissions Directory

This directory contains optimized PyTorch implementations for the TriMul kernel benchmark.

## Performance Comparison (H100)

| Implementation | Geometric Mean | Description |
|----------------|---------------|-------------|
| submission_v2.py | 4.195 ms | Clean FP16 implementation |

## Implementation Details

---

### submission_v2.py
**Strategy:** Clean H100 optimization with identical core logic

**Key Features:**
- Same core implementation as submission_pt_4189_final.py
- Respects `DisableCuDNNTF32()` context
- FP16 computation path with selective FP32 conversions
- Optimized for H100 Tensor Cores

**Performance:** 4.195 ms geometric mean on H100

---

## Common Optimization Techniques

All implementations share these optimization strategies:

1. **Weight Caching:** Concatenate projection weights once (`__W_h16__`, `__Wt_h16__`)
2. **Mixed Precision:** FP16 for compute, FP32 for LayerNorm
3. **Memory Layout:** Minimal reshapes, contiguous tensors for kernels
4. **Fused Operations:** Single projection for all gates and values
5. **H100 Features:** Tensor Cores, reduced precision reductions, Flash Attention

## Benchmark Details

All benchmarks run on NVIDIA H100 80GB HBM3 with:
- 7 different test configurations (varying seqlen, batch size, dimensions)
- Multiple runs per configuration for statistical significance
- Geometric mean calculated across all benchmarks

## Usage

To benchmark any submission:

```bash
cp submissions/submission_v2.py submission.py
MODAL_MODE=benchmark MODAL_GPU=H100 modal run run_modal.py
```
