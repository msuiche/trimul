#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

extern "C" {
    void launch_simple_layernorm(const void*, const void*, const void*, void*, int, int, cudaStream_t);
    void launch_simple_matmul(const void*, const void*, void*, int, int, int, cudaStream_t);
    void launch_simple_gating(const void*, const void*, const void*, void*, int, int, bool, cudaStream_t);
    void launch_einsum(const void*, const void*, void*, int, int, cudaStream_t);
    void launch_output_norm_gate(const void*, const void*, const void*, const void*, void*, int, int, cudaStream_t);
    void launch_final_projection(const void*, const void*, void*, int, int, int, cudaStream_t);

    void launch_trimul_full_pipeline(
        const void* input, const void* mask,
        const void* weights_5HD, const void* weights_out,
        const void* norm1_w, const void* norm1_b,
        const void* norm2_w, const void* norm2_b,
        void* output,
        void* temp_proj_M5H, void* temp_proj_5HM,
        void* temp_left_HM, void* temp_right_HM, void* temp_gate_HM,
        void* temp_left_BHNN, void* temp_right_BHNN,
        void* temp_ein_BHNN, void* temp_ein_MH, void* temp_gate_MH,
        void* temp_gated,
        int B, int N, int D, int H, bool has_mask,
        cudaStream_t stream
    );
}

torch::Tensor simple_layernorm(
    const torch::Tensor& input,    // [M, D] float32
    const torch::Tensor& weight,   // [D] float32
    const torch::Tensor& bias      // [D] float32
) {
    int M = input.size(0);
    int D = input.size(1);

    auto output = torch::empty_like(input);

    launch_simple_layernorm(
        input.data_ptr(), weight.data_ptr(), bias.data_ptr(), output.data_ptr(),
        M, D, c10::cuda::getCurrentCUDAStream()
    );

    return output;
}

torch::Tensor simple_matmul(
    const torch::Tensor& A,   // [M, K] float32
    const torch::Tensor& B    // [K, N] float16
) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::empty({M, N}, torch::dtype(torch::kFloat16).device(A.device()));

    launch_simple_matmul(
        A.data_ptr(), B.data_ptr(), C.data_ptr(),
        M, K, N, c10::cuda::getCurrentCUDAStream()
    );

    return C;
}

torch::Tensor simple_gating(
    const torch::Tensor& gate_proj,   // [H, M] float16
    const torch::Tensor& value_proj,  // [H, M] float16
    const torch::Tensor& mask,        // [M] float32
    bool apply_mask
) {
    int H = gate_proj.size(0);
    int M = gate_proj.size(1);

    auto output = torch::empty_like(gate_proj);

    launch_simple_gating(
        gate_proj.data_ptr(), value_proj.data_ptr(), mask.data_ptr(), output.data_ptr(),
        H, M, apply_mask, c10::cuda::getCurrentCUDAStream()
    );

    return output;
}

torch::Tensor simple_einsum(
    const torch::Tensor& left,    // [BH, N, N] float16
    const torch::Tensor& right    // [BH, N, N] float16
) {
    int BH = left.size(0);
    int N = left.size(1);

    auto output = torch::empty_like(left);

    launch_einsum(
        left.data_ptr(), right.data_ptr(), output.data_ptr(),
        BH, N, c10::cuda::getCurrentCUDAStream()
    );

    return output;
}

torch::Tensor output_norm_gate(
    const torch::Tensor& input,    // [M, H] float16
    const torch::Tensor& gate,     // [M, H] float16
    const torch::Tensor& weight,   // [H] float32
    const torch::Tensor& bias      // [H] float32
) {
    int M = input.size(0);
    int H = input.size(1);

    auto output = torch::empty({M, H}, torch::dtype(torch::kFloat32).device(input.device()));

    launch_output_norm_gate(
        input.data_ptr(), gate.data_ptr(), weight.data_ptr(), bias.data_ptr(), output.data_ptr(),
        M, H, c10::cuda::getCurrentCUDAStream()
    );

    return output;
}

torch::Tensor final_projection(
    const torch::Tensor& input,    // [M, H] float32
    const torch::Tensor& weight    // [H, D] float16
) {
    int M = input.size(0);
    int H = input.size(1);
    int D = weight.size(1);

    auto output = torch::empty({M, D}, torch::dtype(torch::kFloat32).device(input.device()));

    launch_final_projection(
        input.data_ptr(), weight.data_ptr(), output.data_ptr(),
        M, H, D, c10::cuda::getCurrentCUDAStream()
    );

    return output;
}

// UNIFIED TRIMUL - Single function call
torch::Tensor trimul_full(
    const torch::Tensor& input,       // [B, N, N, D] float32
    const torch::Tensor& mask,        // [B, N, N] float32
    const torch::Tensor& weights_5HD, // [5*H, D] float16
    const torch::Tensor& weights_out, // [H, D] float16
    const torch::Tensor& norm1_w,     // [D] float32
    const torch::Tensor& norm1_b,     // [D] float32
    const torch::Tensor& norm2_w,     // [H] float32
    const torch::Tensor& norm2_b,     // [H] float32
    int H
) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    int B = input.size(0);
    int N = input.size(1);
    int D = input.size(3);
    int M = B * N * N;

    // Ensure inputs are contiguous
    auto input_c = input.contiguous();
    auto mask_c = mask.contiguous();

    // Flatten views (memory layout is same, just reinterpret shape)
    auto input_flat = input_c.view({M, D});
    auto mask_flat = mask_c.view({M});

    // Allocate output
    auto output = torch::empty({M, D}, torch::dtype(torch::kFloat32).device(input.device()));

    // Allocate all temporary buffers
    auto temp_proj_M5H = torch::empty({M, 5*H}, torch::dtype(torch::kFloat16).device(input.device()));
    auto temp_proj_5HM = torch::empty({5*H, M}, torch::dtype(torch::kFloat16).device(input.device()));
    auto temp_left_HM = torch::empty({H, M}, torch::dtype(torch::kFloat16).device(input.device()));
    auto temp_right_HM = torch::empty({H, M}, torch::dtype(torch::kFloat16).device(input.device()));
    auto temp_gate_HM = torch::empty({H, M}, torch::dtype(torch::kFloat16).device(input.device()));
    auto temp_left_BHNN = torch::empty({B*H, N, N}, torch::dtype(torch::kFloat16).device(input.device()));
    auto temp_right_BHNN = torch::empty({B*H, N, N}, torch::dtype(torch::kFloat16).device(input.device()));
    auto temp_ein_BHNN = torch::empty({B*H, N, N}, torch::dtype(torch::kFloat16).device(input.device()));
    auto temp_ein_MH = torch::empty({M, H}, torch::dtype(torch::kFloat16).device(input.device()));
    auto temp_gate_MH = torch::empty({M, H}, torch::dtype(torch::kFloat16).device(input.device()));
    auto temp_gated = torch::empty({M, H}, torch::dtype(torch::kFloat32).device(input.device()));

    // Compute has_mask
    bool has_mask = mask_c.min().item<float>() < 1.0f;

    // Call unified pipeline
    launch_trimul_full_pipeline(
        input_flat.data_ptr(), mask_flat.data_ptr(),
        weights_5HD.data_ptr(), weights_out.data_ptr(),
        norm1_w.data_ptr(), norm1_b.data_ptr(),
        norm2_w.data_ptr(), norm2_b.data_ptr(),
        output.data_ptr(),
        temp_proj_M5H.data_ptr(), temp_proj_5HM.data_ptr(),
        temp_left_HM.data_ptr(), temp_right_HM.data_ptr(), temp_gate_HM.data_ptr(),
        temp_left_BHNN.data_ptr(), temp_right_BHNN.data_ptr(),
        temp_ein_BHNN.data_ptr(), temp_ein_MH.data_ptr(), temp_gate_MH.data_ptr(),
        temp_gated.data_ptr(),
        B, N, D, H, has_mask,
        c10::cuda::getCurrentCUDAStream()
    );

    // Reshape output back to [B, N, N, D]
    return output.view({B, N, N, D});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layernorm", &simple_layernorm);
    m.def("matmul", &simple_matmul);
    m.def("gating", &simple_gating);
    m.def("einsum", &simple_einsum);
    m.def("output_norm_gate", &output_norm_gate);
    m.def("final_projection", &final_projection);
    m.def("trimul_full", &trimul_full, "Complete TriMul in Single Call (CUDA)");
}
