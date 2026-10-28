// dynamic_parallelism.cu - Child kernel launch inside parent kernel (Unsupported)
__global__ void childKernel(float *data) {
    data[threadIdx.x] += 1.0f;
}

__global__ void parentKernel(float *data) {
    if (threadIdx.x == 0) {
        // Nested kernel launch is dynamic parallelism
        childKernel<<<1, 32>>>(data);
        cudaDeviceSynchronize();
    }
}
