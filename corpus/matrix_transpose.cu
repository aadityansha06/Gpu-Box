// matrix_transpose.cu - Shared memory matrix transpose (NVIDIA/cuda-samples)
#define TILE_DIM 16
#define BLOCK_ROWS 16

__global__ void transposeCoalesced(float *odata, const float *idata, int width, int height) {
    __shared__ float tile[16][17]; // Pad by 1 to avoid bank conflicts

    int x = blockIdx.x * 16 + threadIdx.x;
    int y = blockIdx.y * 16 + threadIdx.y;

    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = idata[y * width + x];
    }
    __syncthreads();

    x = blockIdx.y * 16 + threadIdx.x;
    y = blockIdx.x * 16 + threadIdx.y;

    if (x < height && y < width) {
        odata[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}
