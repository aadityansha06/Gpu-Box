// histogram256.cu - 256-bin histogram (NVIDIA/cuda-samples)
__global__ void histogram256Kernel(const float *data, float *histo, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float val = data[i];
        int bin = val;
        atomicAdd(&histo[bin], 1.0f);
    }
}
