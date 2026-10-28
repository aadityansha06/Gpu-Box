// matrix_mul.cu - Tiled matrix multiplication (NVIDIA/cuda-samples)
#define TILE_WIDTH 16

__global__ void matrixMulKernel(float *C, const float *A, const float *B, int wA, int wB) {
    __shared__ float ds_A[16][16];
    __shared__ float ds_B[16][16];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * 16 + ty;
    int col = bx * 16 + tx;

    float pValue = 0.0f;

    for (int p = 0; p < (wA / 16); ++p) {
        ds_A[ty][tx] = A[row * wA + p * 16 + tx];
        ds_B[ty][tx] = B[(p * 16 + ty) * wB + col];
        __syncthreads();

        for (int k = 0; k < 16; ++k) {
            pValue += ds_A[ty][k] * ds_B[k][tx];
        }
        __syncthreads();
    }

    C[row * wB + col] = pValue;
}
