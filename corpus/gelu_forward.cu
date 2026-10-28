// gelu_forward.cu - karpathy/llm.c GELU Activation Forward Kernel
#define GELU_SCALING_FACTOR 0.7978845608f // sqrt(2 / M_PI)

__global__ void gelu_forward_kernel(float* out, const float* inp, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
    }
}
