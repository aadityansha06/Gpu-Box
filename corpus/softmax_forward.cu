// softmax_forward.cu - karpathy/llm.c Softmax Forward Kernel
__global__ void softmax_forward_kernel(float* out, const float* inp, int N, int C) {
    int idx = blockIdx.x; // token/row index
    int tid = threadIdx.x;
    const float* x = inp + idx * C;
    float* o = out + idx * C;

    // 1. Max reduction for numerical stability
    __shared__ float s_max[128];
    float maxval = -10000.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        if (x[i] > maxval) maxval = x[i];
    }
    s_max[tid] = maxval;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            if (s_max[tid + stride] > s_max[tid]) s_max[tid] = s_max[tid + stride];
        }
        __syncthreads();
    }
    float m = s_max[0];
    __syncthreads();

    // 2. Compute exp and sum
    __shared__ float s_sum[128];
    float sumval = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        float ex = expf(x[i] - m);
        o[i] = ex;
        sumval += ex;
    }
    s_sum[tid] = sumval;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }
    float total_sum = s_sum[0];
    __syncthreads();

    // 3. Normalize
    for (int i = tid; i < C; i += blockDim.x) {
        o[i] = o[i] / total_sum;
    }
}
