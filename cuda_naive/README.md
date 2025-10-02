# CUDA Naive Implementation

Simple, straightforward CUDA implementation of TriMul operator. Serves as baseline for optimization comparisons.

## Performance

**H100 GPU Benchmark:** 35.149 ms geometric mean

### Individual Benchmark Results

| Config | Mean (ms) | Std Dev (ms) | Best (ms) | Worst (ms) |
|--------|-----------|--------------|-----------|-----------|
| seqlen:256, bs:2, dim:128 | 6.563 | 0.017 | 6.545 | 6.602 |
| seqlen:768, bs:1, dim:128 | 44.819 | 0.025 | 44.804 | 44.849 |
| seqlen:256, bs:2, dim:384 | 14.343 | 0.017 | 14.330 | 14.362 |
| seqlen:512, bs:1, dim:128 | 15.028 | 0.020 | 15.012 | 15.050 |
| seqlen:1024, bs:1, dim:128 | 87.657 | 0.077 | 87.608 | 87.746 |
| seqlen:768, bs:1, dim:384 | 79.703 | 0.028 | 79.671 | 79.726 |
| seqlen:1024, bs:1, dim:384 | 149.636 | 0.078 | 149.546 | 149.681 |

## Implementation

### Kernel Design

**Philosophy:** Simple, easy-to-understand kernels. One output element per thread for matmul.

#### 1. LayerNorm Kernel
- One block per row
- Block-level reduction for mean and variance
- Standard Welford algorithm

#### 2. Matrix Multiplication
- **Grid:** `(M, (N + BLOCK_SIZE - 1) / BLOCK_SIZE)`
- **Block:** `BLOCK_SIZE` threads
- Each thread computes one output element
- Simple dot product loop over K dimension

```cuda
int m = blockIdx.x;
int n = threadIdx.x + blockIdx.y * blockDim.x;
float sum = 0.0f;
for (int k = 0; k < K; k++) {
    sum += A[m * K + k] * B[k * N + n];
}
C[m * N + n] = sum;
```

#### 3. Gating Kernel
- Fused sigmoid + multiply + optional masking
- One thread per output element

#### 4. Einsum Kernel
- Batched matrix multiply: `bhik,bhjk->bhij`
- Per-batch-head computation

#### 5. Output Norm + Gate
- Fused LayerNorm and gating
- Row-wise normalization

#### 6. Final Projection
- Standard matrix multiply

### Memory Management

- All tensors made `.contiguous()` before CUDA kernels
- No stride handling - assumes contiguous layout
- Explicit memory layout for simplicity

### Files

- `submission.py` - PyTorch wrapper and kernel dispatch
- `cuda_simple_kernels.cu` - CUDA kernel implementations
- `cuda_simple_wrapper.cpp` - C++ extension wrapper

## Limitations

1. **No shared memory tiling** - Inefficient for large matrices
2. **Global memory bound** - High bandwidth usage
3. **No vectorization** - Scalar loads/stores
4. **No warp-level primitives** - Simple thread-level operations

## Use Cases

- **Learning:** Clear, readable CUDA code
- **Baseline:** Reference for optimization efforts
- **Debugging:** Simple execution model for validation

## Running

```bash
cd cuda_naive

# Run tests
MODAL_MODE=test MODAL_GPU=H100 MODAL_TASK=task.yml modal run ../run_modal.py

# Run benchmarks
MODAL_JSON=1 MODAL_MODE=benchmark MODAL_GPU=H100 MODAL_TASK=task.yml modal run ../run_modal.py
```

## Next Steps

See `cuda_optimized/` for performance improvements via:
- Shared memory tiling
- Better memory access patterns
- Reduced global memory bandwidth
