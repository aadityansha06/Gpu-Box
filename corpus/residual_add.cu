// residual_add.cu - karpathy/llm.c Residual Stream Addition Kernel
__global__ void residual_forward_kernel(float* out, const float* inp1, const float* inp2, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[i] = inp1[i] + inp2[i];
    }
}
