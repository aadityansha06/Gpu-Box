// conv1d.cu - 1D Convolution with radius 2
#define RADIUS 2

__global__ void conv1dKernel(const float *in, const float *mask, float *out, int n, int mask_width) {
    __shared__ float tile[64 + 4]; // blockDim.x + 2 * RADIUS

    int gindex = blockDim.x * blockIdx.x + threadIdx.x;
    int lindex = threadIdx.x + RADIUS;

    tile[lindex] = (gindex < n) ? in[gindex] : 0.0f;
    if (threadIdx.x < RADIUS) {
        tile[lindex - RADIUS] = (gindex >= RADIUS) ? in[gindex - RADIUS] : 0.0f;
        tile[lindex + blockDim.x] = (gindex + blockDim.x < n) ? in[gindex + blockDim.x] : 0.0f;
    }
    __syncthreads();

    if (gindex < n) {
        float val = 0.0f;
        for (int j = 0; j < mask_width; j++) {
            val += tile[threadIdx.x + j] * mask[j];
        }
        out[gindex] = val;
    }
}
