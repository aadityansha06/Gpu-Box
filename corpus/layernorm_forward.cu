// layernorm_forward.cu - karpathy/llm.c LayerNorm Forward Kernel
__global__ void layernorm_forward_kernel(float* out, float* mean, float* rstd,
                                         const float* inp, const float* weight, const float* bias,
                                         int N, int C) {
    int idx = blockIdx.x; // each block processes one token of size C
    int tid = threadIdx.x;
    const float* x = inp + idx * C;

    // 1. Calculate mean using shared memory
    __shared__ float s_mean[128];
    float sum = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        sum += x[i];
    }
    s_mean[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_mean[tid] += s_mean[tid + stride];
        }
        __syncthreads();
    }
    float m = s_mean[0] / C;
    if (tid == 0 && mean != NULL) {
        mean[idx] = m;
    }
    __syncthreads();

    // 2. Calculate variance
    __shared__ float s_var[128];
    float var_sum = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        float diff = x[i] - m;
        var_sum += diff * diff;
    }
    s_var[tid] = var_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_var[tid] += s_var[tid + stride];
        }
        __syncthreads();
    }
    float s = 1.0f / sqrtf(s_var[0] / C + 1e-5f);
    if (tid == 0 && rstd != NULL) {
        rstd[idx] = s;
    }
    __syncthreads();

    // 3. Normalize, scale, and shift
    float* o = out + idx * C;
    for (int i = tid; i < C; i += blockDim.x) {
        float n = s * (x[i] - m);
        o[i] = n * weight[i] + bias[i];
    }
}
