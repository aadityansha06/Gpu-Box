// dot_product.cu - Dot product with shared memory reduction (NVIDIA/cuda-samples)
__global__ void dotProduct(const float *a, const float *b, float *c, int n) {
    __shared__ float cache[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float temp = 0.0f;
    while (i < n) {
        temp += a[i] * b[i];
        i += blockDim.x * gridDim.x;
    }
    cache[tid] = temp;
    __syncthreads();

    int ib = blockDim.x / 2;
    while (ib != 0) {
        if (tid < ib) {
            cache[tid] += cache[tid + ib];
        }
        __syncthreads();
        ib /= 2;
    }

    if (tid == 0) {
        atomicAdd(c, cache[0]);
    }
}
