// BUG: rank 3 forgets to join the collective (a classic cause of hung training jobs).
// On a real cluster ranks 0-2 would block forever. Gpubox reports the deadlock instead.
// Fix: let every rank call ncclAllReduce.
#include <cuda_runtime.h>
#include <nccl.h>
#include <cstdio>

int main() {
    const int gpus = 4;
    const int n = 256;
    ncclComm_t comms[4];
    ncclCommInitAll(comms, gpus);

    float *send[4], *recv[4];
    for (int r = 0; r < gpus; r++) {
        cudaSetDevice(r);
        cudaMalloc(&send[r], n * sizeof(float));
        cudaMalloc(&recv[r], n * sizeof(float));
        cudaMemset(send[r], 0, n * sizeof(float));
    }

    for (int r = 0; r < gpus - 1; r++) {                 // BUG: should be r < gpus
        cudaSetDevice(r);
        ncclAllReduce(send[r], recv[r], n, ncclFloat, ncclSum, comms[r], 0);
    }
    printf("AllReduce issued\n");
    return 0;
}
