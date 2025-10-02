# Submissions Directory

This directory contains optimized PyTorch implementations for the TriMul kernel benchmark.

## Performance Comparison (H100)

| Implementation | Geometric Mean | Description |
|----------------|---------------|-------------|
| submission_improved_triton.py | **2.399 ms** | Three-way hybrid: Arseni + optimized routing |
| submission_v2.py | 4.195 ms | Clean FP16 implementation |

**Best Performance:** submission_improved_triton.py achieves 2.399ms (43% faster than v2)

## Implementation Details

---

### submission_improved_triton.py
**Strategy:** Three-tier routing based on Arseni Ivanov's hybrid approach

**Key Features:**
- Small inputs (seqlen ≤ 256): Arseni's lightweight PyTorch path
- Medium inputs (256 < seqlen ≤ 512): Optimized W @ x.t() memory layout (47% speedup)
- Large inputs (seqlen > 512): Fused Triton kernels (LayerNorm + MatMul + Gating)
- Empirical benchmark-driven routing strategy

**Performance:** 2.399 ms geometric mean on H100
- 11.5% faster than Arseni's original implementation
- 43% faster than baseline v2

**Credits:** Based on Arseni Ivanov's work (https://arseniivanov.github.io/blog.html)

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
