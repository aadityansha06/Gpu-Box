// inline_ptx.cu - Inline PTX assembly (Unsupported)
__global__ void inlinePtxKernel(float *d_out, float *d_in) {
    int tid = threadIdx.x;
    float in_val = d_in[tid];
    float out_val;
    asm("mov.f32 %0, %1;" : "=f"(out_val) : "f"(in_val));
    d_out[tid] = out_val;
}
