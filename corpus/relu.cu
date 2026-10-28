// relu.cu - Rectified Linear Unit activation
__global__ void reluKernel(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = in[i];
        out[i] = v > 0.0f ? v : 0.0f;
    }
}
