// cross_entropy_forward.cu - karpathy/llm.c Cross Entropy Loss Forward Kernel
__global__ void crossentropy_forward_kernel(float* losses,
                                            const float* probs, const int* targets,
                                            int B, int T, int V) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B * T) {
        const float* p = probs + i * V;
        int target = targets[i];
        losses[i] = -logf(p[target]);
    }
}
