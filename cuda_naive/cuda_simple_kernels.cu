// Ultra-simple CUDA kernels - correctness over performance
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define BLOCK_SIZE 256

// ============================================================================
// Kernel 1: LayerNorm (simple, one row per block)
// ============================================================================

__global__ void simple_layernorm_kernel(
    const float* __restrict__ input,      // [M, D]
    const float* __restrict__ weight,     // [D]
    const float* __restrict__ bias,       // [D]
    float* __restrict__ output,           // [M, D]
    int M, int D
) {
    int m = blockIdx.x;
    if (m >= M) return;

    // Compute mean
    float sum = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        sum += input[m * D + d];
    }

    // Block reduce
    __shared__ float shared_sum[BLOCK_SIZE];
    shared_sum[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    float mean = shared_sum[0] / D;
    __syncthreads();

    // Compute variance
    float var_sum = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float diff = input[m * D + d] - mean;
        var_sum += diff * diff;
    }

    shared_sum[threadIdx.x] = var_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    float variance = shared_sum[0] / D;
    float rstd = rsqrtf(variance + 1e-5f);

    // Normalize
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float val = input[m * D + d];
        output[m * D + d] = (val - mean) * rstd * weight[d] + bias[d];
    }
}

// ============================================================================
// Kernel 2: Matrix multiply (simple, one output per thread)
// ============================================================================

__global__ void simple_matmul_kernel(
    const float* __restrict__ A,          // [M, K]
    const half* __restrict__ B,           // [K, N]
    half* __restrict__ C,                 // [M, N]
    int M, int K, int N
) {
    int m = blockIdx.x;
    int n = threadIdx.x + blockIdx.y * blockDim.x;

    if (m >= M || n >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[m * K + k] * __half2float(B[k * N + n]);
    }
    C[m * N + n] = __float2half(sum);
}

// ============================================================================
// Kernel 3: Sigmoid + multiply + mask (element-wise)
// ============================================================================

__global__ void simple_gating_kernel(
    const half* __restrict__ gate_proj,   // [H, M]
    const half* __restrict__ value_proj,  // [H, M]
    const float* __restrict__ mask,       // [M]
    half* __restrict__ output,            // [H, M]
    int H, int M, bool apply_mask
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * M;

    if (idx >= total) return;

    int m = idx % M;

    float gate_val = __half2float(gate_proj[idx]);
    float value_val = __half2float(value_proj[idx]);

    float sig = 1.0f / (1.0f + expf(-gate_val));
    float result = sig * value_val;

    if (apply_mask) {
        result *= mask[m];
    }

    output[idx] = __float2half(result);
}

// ============================================================================
// Kernel 4: Einsum (reuse from before - we know this works)
// ============================================================================

#define EINSUM_BLOCK_M 64
#define EINSUM_BLOCK_N 64
#define EINSUM_BLOCK_K 64

__global__ void einsum_kernel(
    const half* __restrict__ left,
    const half* __restrict__ right,
    half* __restrict__ output,
    int BH, int N
) {
    const int bh = blockIdx.z;
    const int block_i = blockIdx.y;
    const int block_j = blockIdx.x;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int ti = threadIdx.y * 8;
    const int tj = threadIdx.x * 8;

    const int i_base = block_i * EINSUM_BLOCK_M;
    const int j_base = block_j * EINSUM_BLOCK_N;

    float acc[8][8];
    #pragma unroll
    for (int ii = 0; ii < 8; ii++) {
        #pragma unroll
        for (int jj = 0; jj < 8; jj++) {
            acc[ii][jj] = 0.0f;
        }
    }

    __shared__ half smem_left[EINSUM_BLOCK_M][EINSUM_BLOCK_K + 8];
    __shared__ half smem_right[EINSUM_BLOCK_N][EINSUM_BLOCK_K + 8];

    for (int k_start = 0; k_start < N; k_start += EINSUM_BLOCK_K) {
        for (int load_iter = tid; load_iter < EINSUM_BLOCK_M * EINSUM_BLOCK_K; load_iter += 64) {
            int load_i = load_iter / EINSUM_BLOCK_K;
            int load_k = load_iter % EINSUM_BLOCK_K;
            int i = i_base + load_i;
            int k = k_start + load_k;

            if (i < N && k < N) {
                smem_left[load_i][load_k] = left[bh * N * N + i * N + k];
            } else {
                smem_left[load_i][load_k] = __float2half(0.0f);
            }
        }

        for (int load_iter = tid; load_iter < EINSUM_BLOCK_N * EINSUM_BLOCK_K; load_iter += 64) {
            int load_j = load_iter / EINSUM_BLOCK_K;
            int load_k = load_iter % EINSUM_BLOCK_K;
            int j = j_base + load_j;
            int k = k_start + load_k;

            if (j < N && k < N) {
                smem_right[load_j][load_k] = right[bh * N * N + j * N + k];
            } else {
                smem_right[load_j][load_k] = __float2half(0.0f);
            }
        }

        __syncthreads();

        #pragma unroll 8
        for (int k = 0; k < EINSUM_BLOCK_K; k++) {
            half left_vals[8];
            half right_vals[8];

            #pragma unroll
            for (int ii = 0; ii < 8; ii++) {
                left_vals[ii] = smem_left[ti + ii][k];
            }

            #pragma unroll
            for (int jj = 0; jj < 8; jj++) {
                right_vals[jj] = smem_right[tj + jj][k];
            }

            #pragma unroll
            for (int ii = 0; ii < 8; ii++) {
                #pragma unroll
                for (int jj = 0; jj < 8; jj++) {
                    acc[ii][jj] += __half2float(left_vals[ii]) * __half2float(right_vals[jj]);
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int ii = 0; ii < 8; ii++) {
        #pragma unroll
        for (int jj = 0; jj < 8; jj++) {
            int i = i_base + ti + ii;
            int j = j_base + tj + jj;
            if (i < N && j < N) {
                output[bh * N * N + i * N + j] = __float2half(acc[ii][jj]);
            }
        }
    }
}

// ============================================================================
// Kernel 5: Output LayerNorm + Gating (combined)
// ============================================================================

__global__ void output_norm_gate_kernel(
    const half* __restrict__ input,       // [M, H] float16
    const half* __restrict__ gate,        // [M, H] float16
    const float* __restrict__ weight,     // [H] float32
    const float* __restrict__ bias,       // [H] float32
    float* __restrict__ output,           // [M, H] float32
    int M, int H
) {
    int m = blockIdx.x;
    if (m >= M) return;

    // Compute mean
    float sum = 0.0f;
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        sum += __half2float(input[m * H + h]);
    }

    // Block reduce
    __shared__ float shared_sum[BLOCK_SIZE];
    shared_sum[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    float mean = shared_sum[0] / H;
    __syncthreads();

    // Compute variance
    float var_sum = 0.0f;
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        float val = __half2float(input[m * H + h]);
        float diff = val - mean;
        var_sum += diff * diff;
    }

    shared_sum[threadIdx.x] = var_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    float variance = shared_sum[0] / H;
    float rstd = rsqrtf(variance + 1e-5f);

    // Normalize, apply weight/bias, and multiply by gate
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        float val = __half2float(input[m * H + h]);
        float gate_val = __half2float(gate[m * H + h]);
        float normalized = (val - mean) * rstd * weight[h] + bias[h];
        output[m * H + h] = normalized * gate_val;
    }
}

// ============================================================================
// Kernel 6: Final Projection (matrix multiply with FP16 weights)
// ============================================================================

__global__ void final_projection_kernel(
    const float* __restrict__ input,      // [M, H] float32
    const half* __restrict__ weight,      // [H, D] float16
    float* __restrict__ output,           // [M, D] float32
    int M, int H, int D
) {
    int m = blockIdx.x;
    int d = threadIdx.x + blockIdx.y * blockDim.x;

    if (m >= M || d >= D) return;

    float sum = 0.0f;
    for (int h = 0; h < H; h++) {
        sum += input[m * H + h] * __half2float(weight[h * D + d]);
    }
    output[m * D + d] = sum;
}

// ============================================================================
// Host wrappers
// ============================================================================

extern "C" {

void launch_simple_layernorm(
    const void* input, const void* weight, const void* bias, void* output,
    int M, int D, cudaStream_t stream
) {
    simple_layernorm_kernel<<<M, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<const float*>(weight),
        reinterpret_cast<const float*>(bias),
        reinterpret_cast<float*>(output),
        M, D
    );
}

void launch_simple_matmul(
    const void* A, const void* B, void* C,
    int M, int K, int N, cudaStream_t stream
) {
    dim3 grid(M, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    simple_matmul_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<const float*>(A),
        reinterpret_cast<const half*>(B),
        reinterpret_cast<half*>(C),
        M, K, N
    );
}

void launch_simple_gating(
    const void* gate_proj, const void* value_proj, const void* mask, void* output,
    int H, int M, bool apply_mask, cudaStream_t stream
) {
    int total = H * M;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    simple_gating_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<const half*>(gate_proj),
        reinterpret_cast<const half*>(value_proj),
        reinterpret_cast<const float*>(mask),
        reinterpret_cast<half*>(output),
        H, M, apply_mask
    );
}

void launch_einsum(
    const void* left, const void* right, void* output,
    int BH, int N, cudaStream_t stream
) {
    dim3 block(8, 8);
    dim3 grid(
        (N + EINSUM_BLOCK_N - 1) / EINSUM_BLOCK_N,
        (N + EINSUM_BLOCK_M - 1) / EINSUM_BLOCK_M,
        BH
    );

    einsum_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(left),
        reinterpret_cast<const half*>(right),
        reinterpret_cast<half*>(output),
        BH, N
    );
}

void launch_output_norm_gate(
    const void* input, const void* gate,
    const void* weight, const void* bias,
    void* output, int M, int H, cudaStream_t stream
) {
    output_norm_gate_kernel<<<M, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<const half*>(input),
        reinterpret_cast<const half*>(gate),
        reinterpret_cast<const float*>(weight),
        reinterpret_cast<const float*>(bias),
        reinterpret_cast<float*>(output),
        M, H
    );
}

void launch_final_projection(
    const void* input, const void* weight, void* output,
    int M, int H, int D, cudaStream_t stream
) {
    dim3 grid(M, (D + BLOCK_SIZE - 1) / BLOCK_SIZE);
    final_projection_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<const half*>(weight),
        reinterpret_cast<float*>(output),
        M, H, D
    );
}

// ============================================================================
// Utility: Sigmoid-only kernel (no multiply)
// ============================================================================

__global__ void sigmoid_only_kernel(
    const half* __restrict__ input,    // [H, M]
    half* __restrict__ output,         // [H, M]
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    float val = __half2float(input[idx]);
    float sig = 1.0f / (1.0f + expf(-val));
    output[idx] = __float2half(sig);
}

// ============================================================================
// Utility kernels for reshaping
// ============================================================================

// Reshape [H, B, N, N] -> [B, H, N, N] (permute dims 0,1)
__global__ void reshape_HBNN_to_BHNN_kernel(
    const half* __restrict__ input,   // [H, B, N, N]
    half* __restrict__ output,        // [B, H, N, N]
    int B, int H, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * N * N;

    if (idx >= total) return;

    // Decompose idx in [B, H, N, N] layout
    int n2 = idx % N;
    int n1 = (idx / N) % N;
    int h = (idx / (N * N)) % H;
    int b = idx / (H * N * N);

    // Compute source index in [H, B, N, N] layout
    int src_idx = h * (B * N * N) + b * (N * N) + n1 * N + n2;

    output[idx] = input[src_idx];
}

// Reshape [B, H, N, N] -> [B, N, N, H] (permute to move H to last dim)
__global__ void reshape_BHNN_to_BNNH_kernel(
    const half* __restrict__ input,   // [B, H, N, N]
    half* __restrict__ output,        // [B, N, N, H]
    int B, int H, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * N * N;

    if (idx >= total) return;

    // Decompose idx in [B, N, N, H] output layout
    int h = idx % H;
    int n2 = (idx / H) % N;
    int n1 = (idx / (H * N)) % N;
    int b = idx / (H * N * N);

    // Compute source index in [B, H, N, N] layout
    int src_idx = b * (H * N * N) + h * (N * N) + n1 * N + n2;

    output[idx] = input[src_idx];
}

// Reshape [H, B, N, N] -> [B, N, N, H] (combined permutation)
__global__ void reshape_HBNN_to_BNNH_kernel(
    const half* __restrict__ input,   // [H, B, N, N]
    half* __restrict__ output,        // [B, N, N, H]
    int B, int H, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * N * N;

    if (idx >= total) return;

    // Decompose idx in [B, N, N, H] output layout
    int h = idx % H;
    int n2 = (idx / H) % N;
    int n1 = (idx / (H * N)) % N;
    int b = idx / (H * N * N);

    // Compute source index in [H, B, N, N] layout
    int src_idx = h * (B * N * N) + b * (N * N) + n1 * N + n2;

    output[idx] = input[src_idx];
}

// Transpose [M, 5H] -> [5H, M]
__global__ void transpose_M_5H_kernel(
    const half* __restrict__ input,   // [M, 5*H]
    half* __restrict__ output,        // [5*H, M]
    int M, int H
) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int fh = blockIdx.y * blockDim.y + threadIdx.y;
    int FH = 5 * H;

    if (m >= M || fh >= FH) return;

    output[fh * M + m] = input[m * FH + fh];
}

// ============================================================================
// UNIFIED PIPELINE - All stages in one host function
// ============================================================================

void launch_trimul_full_pipeline(
    const void* input,           // [B, N, N, D] -> [M, D] float32
    const void* mask,            // [B, N, N] -> [M] float32
    const void* weights_5HD,     // [5*H, D] float16 (concatenated projections)
    const void* weights_out,     // [H, D] float16 (output projection)
    const void* norm1_w,         // [D] float32
    const void* norm1_b,         // [D] float32
    const void* norm2_w,         // [H] float32
    const void* norm2_b,         // [H] float32
    void* output,                // [B, N, N, D] -> [M, D] float32
    void* temp_proj_M5H,         // [M, 5*H] float16 - matmul output
    void* temp_proj_5HM,         // [5*H, M] float16 - transposed
    void* temp_left_HM,          // [H, M] float16 - gating output
    void* temp_right_HM,         // [H, M] float16 - gating output
    void* temp_gate_HM,          // [H, M] float16 - gating output
    void* temp_left_BHNN,        // [B*H, N, N] float16 - reshaped for einsum
    void* temp_right_BHNN,       // [B*H, N, N] float16 - reshaped for einsum
    void* temp_ein_BHNN,         // [B*H, N, N] float16 - einsum output
    void* temp_ein_MH,           // [M, H] float16 - reshaped
    void* temp_gate_MH,          // [M, H] float16 - reshaped
    void* temp_gated,            // [M, H] float32 - norm+gate output
    int B, int N, int D, int H,
    bool has_mask,
    cudaStream_t stream
) {
    const int M = B * N * N;
    const int BH = B * H;

    // ===== STAGE 1: Input LayerNorm =====
    // [M, D] -> [M, D]
    simple_layernorm_kernel<<<M, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<const float*>(norm1_w),
        reinterpret_cast<const float*>(norm1_b),
        reinterpret_cast<float*>(output),  // reuse output buffer temporarily
        M, D
    );

    // ===== STAGE 2: 5x Projection =====
    // [M, D] x [D, 5*H] -> [M, 5*H]
    dim3 proj_grid(M, (5*H + BLOCK_SIZE - 1) / BLOCK_SIZE);
    simple_matmul_kernel<<<proj_grid, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<const float*>(output),  // normalized input
        reinterpret_cast<const half*>(weights_5HD),
        reinterpret_cast<half*>(temp_proj_M5H),
        M, D, 5*H
    );

    // Transpose [M, 5*H] -> [5*H, M]
    dim3 transpose_block(16, 16);
    dim3 transpose_grid((M + 15) / 16, (5*H + 15) / 16);
    transpose_M_5H_kernel<<<transpose_grid, transpose_block, 0, stream>>>(
        reinterpret_cast<const half*>(temp_proj_M5H),
        reinterpret_cast<half*>(temp_proj_5HM),
        M, H
    );

    // ===== STAGE 3: Gating =====
    // Process [5*H, M] as [5, H, M] to create LEFT, RIGHT, OUT_GATE
    int total = H * M;
    int gating_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // LEFT = sigmoid(proj[2]) * proj[0] * mask
    simple_gating_kernel<<<gating_blocks, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<const half*>(temp_proj_5HM) + 2 * H * M,  // proj[2, :, :]
        reinterpret_cast<const half*>(temp_proj_5HM) + 0 * H * M,  // proj[0, :, :]
        reinterpret_cast<const float*>(mask),
        reinterpret_cast<half*>(temp_left_HM),
        H, M, has_mask
    );

    // RIGHT = sigmoid(proj[3]) * proj[1]
    simple_gating_kernel<<<gating_blocks, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<const half*>(temp_proj_5HM) + 3 * H * M,
        reinterpret_cast<const half*>(temp_proj_5HM) + 1 * H * M,
        reinterpret_cast<const float*>(mask),
        reinterpret_cast<half*>(temp_right_HM),
        H, M, false
    );

    // OUT_GATE = sigmoid(proj[4]) - just sigmoid, no multiply
    sigmoid_only_kernel<<<gating_blocks, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<const half*>(temp_proj_5HM) + 4 * H * M,
        reinterpret_cast<half*>(temp_gate_HM),
        total
    );

    // ===== STAGE 4: Reshape for Einsum =====
    // [H, M] = [H, B, N, N] -> [B, H, N, N] -> [B*H, N, N]
    int reshape_blocks = (B * H * N * N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    reshape_HBNN_to_BHNN_kernel<<<reshape_blocks, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<const half*>(temp_left_HM),
        reinterpret_cast<half*>(temp_left_BHNN),
        B, H, N
    );

    reshape_HBNN_to_BHNN_kernel<<<reshape_blocks, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<const half*>(temp_right_HM),
        reinterpret_cast<half*>(temp_right_BHNN),
        B, H, N
    );

    // ===== STAGE 5: Einsum (Batch Matrix Multiply) =====
    // [B*H, N, N] x [B*H, N, N] -> [B*H, N, N]
    dim3 ein_block(8, 8);
    dim3 ein_grid(
        (N + EINSUM_BLOCK_N - 1) / EINSUM_BLOCK_N,
        (N + EINSUM_BLOCK_M - 1) / EINSUM_BLOCK_M,
        BH
    );

    einsum_kernel<<<ein_grid, ein_block, 0, stream>>>(
        reinterpret_cast<const half*>(temp_left_BHNN),
        reinterpret_cast<const half*>(temp_right_BHNN),
        reinterpret_cast<half*>(temp_ein_BHNN),
        BH, N
    );

    // ===== STAGE 6: Reshape for Output =====
    // [B*H, N, N] = [B, H, N, N] -> [B, N, N, H] = [M, H]
    reshape_BHNN_to_BNNH_kernel<<<reshape_blocks, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<const half*>(temp_ein_BHNN),
        reinterpret_cast<half*>(temp_ein_MH),
        B, H, N
    );

    // Reshape out_gate: [H, M] = [H, B, N, N] -> [B, N, N, H] = [M, H]
    reshape_HBNN_to_BNNH_kernel<<<reshape_blocks, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<const half*>(temp_gate_HM),
        reinterpret_cast<half*>(temp_gate_MH),
        B, H, N
    );

    // ===== STAGE 7: Output LayerNorm + Gating =====
    // [M, H] -> [M, H]
    output_norm_gate_kernel<<<M, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<const half*>(temp_ein_MH),
        reinterpret_cast<const half*>(temp_gate_MH),
        reinterpret_cast<const float*>(norm2_w),
        reinterpret_cast<const float*>(norm2_b),
        reinterpret_cast<float*>(temp_gated),
        M, H
    );

    // ===== STAGE 8: Final Projection =====
    // [M, H] x [H, D] -> [M, D]
    dim3 final_grid(M, (D + BLOCK_SIZE - 1) / BLOCK_SIZE);
    final_projection_kernel<<<final_grid, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<const float*>(temp_gated),
        reinterpret_cast<const half*>(weights_out),
        reinterpret_cast<float*>(output),
        M, H, D
    );
}

} // extern "C"
