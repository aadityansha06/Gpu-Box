// prefix_sum.cu - Inclusive Scan / Prefix Sum (Hillis-Steele)
__global__ void scanInclusive(float *g_data, int n) {
    __shared__ float temp[128];
    int tid = threadIdx.x;
    int pout = 0, pin = 1;

    temp[tid] = (tid < n) ? g_data[tid] : 0.0f;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        float val = 0.0f;
        if (tid >= offset) {
            val = temp[tid - offset];
        }
        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }

    if (tid < n) {
        g_data[tid] = temp[tid];
    }
}
