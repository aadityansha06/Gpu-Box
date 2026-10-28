// warp_shuffle_reduce.cu - Warp shuffle tree reduction (NVIDIA/cuda-samples)
__global__ void warpReduceSum(const float *in, float *out, int n) {
    int tid = threadIdx.x;
    float val = (tid < n) ? in[tid] : 0.0f;

    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset, 32);
    }

    if (tid == 0) {
        out[0] = val;
    }
}
