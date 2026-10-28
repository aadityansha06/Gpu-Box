// scalar_mul.cu - In-place scalar multiplication
__global__ void scalarMul(float *data, float factor, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = data[i] * factor;
    }
}
