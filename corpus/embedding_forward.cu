// embedding_forward.cu - karpathy/llm.c Embedding Lookup Forward Kernel
__global__ void encoder_forward_kernel(float* out,
                                       const int* inp, const float* wte, const float* wpe,
                                       int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T * C;
    if (idx < N) {
        int bt = idx / C;
        int b = bt / T;
        int t = bt % T;
        int c = idx % C;

        int ix = inp[b * T + t];
        float val = wte[ix * C + c];
        if (wpe != NULL) {
            val += wpe[t * C + c];
        }
        out[idx] = val;
    }
}
