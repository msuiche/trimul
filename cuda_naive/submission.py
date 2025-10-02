"""
PURE CUDA IMPLEMENTATION - All operations in CUDA
Fixed: Ensure all tensors are contiguous before passing to CUDA
"""
import torch
from task import input_t, output_t
from utils import DisableCuDNNTF32
import os
import sys

HAS_CUDA = False

if not torch.cuda.is_available():
    raise RuntimeError("CUDA required!")

try:
    from torch.utils.cpp_extension import load
    current_dir = os.path.dirname(os.path.abspath(__file__))

    if 'TORCH_CUDA_ARCH_LIST' not in os.environ:
        os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'

    print("Compiling PURE CUDA kernels...", file=sys.stderr, flush=True)

    cuda = load(
        name='trimul_pure_cuda',
        sources=[
            os.path.join(current_dir, 'cuda_simple_wrapper.cpp'),
            os.path.join(current_dir, 'cuda_simple_kernels.cu')
        ],
        extra_cuda_cflags=['-O3', '--use_fast_math', '-gencode=arch=compute_90,code=sm_90'],
        verbose=False,
        with_cuda=True
    )
    HAS_CUDA = True
    print("✓ PURE CUDA kernels compiled!", file=sys.stderr, flush=True)

except Exception as e:
    print(f"✗ Failed: {e}", file=sys.stderr)
    raise


def custom_kernel(data: input_t) -> output_t:
    """PURE CUDA implementation - all compute in CUDA"""
    input_tensor, mask, weights, config = data
    B, N, _, D = input_tensor.shape
    H = config["hidden_dim"]
    M = B * N * N

    # Prepare weights - ENSURE CONTIGUOUS
    W_5H_D = "__W_5H_D__"
    if W_5H_D not in weights:
        weights[W_5H_D] = torch.cat([
            weights['left_proj.weight'],
            weights['right_proj.weight'],
            weights['left_gate.weight'],
            weights['right_gate.weight'],
            weights['out_gate.weight'],
        ], dim=0).half().contiguous()  # CONTIGUOUS

    W_out = "__W_out__"
    if W_out not in weights:
        weights[W_out] = weights['to_out.weight'].t().contiguous().half()  # CONTIGUOUS

    # === STAGE 1: LayerNorm (CUDA) ===
    x_flat = input_tensor.view(M, D).contiguous()
    x_norm = cuda.layernorm(x_flat, weights["norm.weight"], weights["norm.bias"])  # [M, D] float32

    # === STAGE 2: Projection (CUDA) ===
    # A=[M,D] float32, B=[5H,D] float16 transposed -> B^T=[D,5H], so we want A @ B^T = [M,5H]
    # But matmul expects A=[M,K], B=[K,N] -> C=[M,N]
    # So we need B transposed: [D, 5H]
    W_transposed = weights[W_5H_D].t().contiguous()  # [D, 5*H] CONTIGUOUS
    P = cuda.matmul(x_norm, W_transposed)  # [M, 5*H] float16
    P = P.t().contiguous().view(5, H, M)  # [5, H, M]

    # === STAGE 3: Gating (CUDA) ===
    has_mask = mask.min().item() < 1.0
    mask_flat = mask.view(M).contiguous()

    LEFT_T = cuda.gating(P[2].contiguous(), P[0].contiguous(), mask_flat, has_mask)  # [H, M]
    RIGHT_T = cuda.gating(P[3].contiguous(), P[1].contiguous(), mask_flat, False)  # [H, M]

    # OG needs sigmoid only
    ones = torch.ones_like(P[4])
    OG_T = cuda.gating(P[4].contiguous(), ones, mask_flat, False)  # [H, M]

    # === STAGE 4: Einsum (CUDA) ===
    LEFT_bhnn = LEFT_T.view(H, B, N, N).permute(1, 0, 2, 3).contiguous()
    RIGHT_bhnn = RIGHT_T.view(H, B, N, N).permute(1, 0, 2, 3).contiguous()
    LEFT_flat = LEFT_bhnn.view(B * H, N, N).contiguous()
    RIGHT_flat = RIGHT_bhnn.view(B * H, N, N).contiguous()

    EIN_flat = cuda.einsum(LEFT_flat, RIGHT_flat)  # [BH, N, N] float16
    EIN = EIN_flat.view(B, H, N, N).permute(0, 2, 3, 1).contiguous()  # [B, N, N, H]

    # === STAGE 5: Output Norm + Gate (CUDA) ===
    EIN_flat_half = EIN.view(M, H).contiguous()
    OG_flat = OG_T.view(H, B, N, N).permute(1, 2, 3, 0).contiguous().view(M, H)

    G = cuda.output_norm_gate(EIN_flat_half, OG_flat, weights['to_out_norm.weight'], weights['to_out_norm.bias'])  # [M, H] float32

    # === STAGE 6: Final Projection (CUDA) ===
    OUT = cuda.final_projection(G, weights[W_out])  # [M, D] float32

    return OUT.view(B, N, N, D)
