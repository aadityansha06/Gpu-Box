/**
 * Kernels used by the real-GPU parity check (scripts/gpu-parity.js + gpu-parity.py).
 * The same CUDA source runs on the simulator and on a physical NVIDIA GPU (via NVRTC);
 * outputs must agree. Each case also records what the simulator reports for
 * bytes moved / FLOPs so they can be checked against analytic expectations.
 */
let seed = 7;
const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296 * 2 - 1; };
const vec = (n, scale = 1) => Array.from({ length: n }, () => Math.fround(rnd() * scale));
const ivec = (n, mod) => Array.from({ length: n }, () => Math.floor(Math.abs(rnd()) * 1e6) % mod);

const cases = [];

{ const N = 3000;
  cases.push({
    name: 'saxpy', kernel: 'saxpy', grid: [24, 1, 1], block: [128, 1, 1],
    src: `extern "C" __global__ void saxpy(float a, const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}`,
    args: [{ t: 'f32', v: 2.5 }, { t: 'buf', dtype: 'f32', data: vec(N) }, { t: 'buf', dtype: 'f32', data: vec(N), out: true }, { t: 'i32', v: N }],
    expect: { flops: 2 * N, globalLoadBytes: 8 * N, globalStoreBytes: 4 * N, h2d: 8 * N, d2h: 4 * N } });
}
{ const N = 5000;
  cases.push({
    name: 'reduce_sum (shared tree + atomicAdd)', kernel: 'red', grid: [Math.ceil(N / 256), 1, 1], block: [256, 1, 1],
    src: `extern "C" __global__ void red(const float* in, float* out, int n) {
  __shared__ float s[256]; int t = threadIdx.x; int i = blockIdx.x * blockDim.x + t;
  s[t] = i < n ? in[i] : 0.0f; __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) { if (t < st) s[t] += s[t + st]; __syncthreads(); }
  if (t == 0) atomicAdd(out, s[0]);
}`,
    args: [{ t: 'buf', dtype: 'f32', data: vec(N) }, { t: 'buf', dtype: 'f32', data: [0], out: true }, { t: 'i32', v: N }], tol: 2e-4 });
}
{ const R = 8, C = 100;
  cases.push({
    name: 'softmax rows', kernel: 'sm', grid: [R, 1, 1], block: [64, 1, 1],
    src: `extern "C" __global__ void sm(const float* in, float* out, int C) {
  __shared__ float sh[64]; int r = blockIdx.x, t = threadIdx.x;
  float m = -1e30f; for (int c = t; c < C; c += blockDim.x) m = fmaxf(m, in[r * C + c]);
  sh[t] = m; __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (t < s) sh[t] = fmaxf(sh[t], sh[t + s]); __syncthreads(); }
  m = sh[0]; __syncthreads();
  float sum = 0.0f; for (int c = t; c < C; c += blockDim.x) sum += expf(in[r * C + c] - m);
  sh[t] = sum; __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (t < s) sh[t] += sh[t + s]; __syncthreads(); }
  sum = sh[0];
  for (int c = t; c < C; c += blockDim.x) out[r * C + c] = expf(in[r * C + c] - m) / sum;
}`,
    args: [{ t: 'buf', dtype: 'f32', data: vec(R * C, 12) }, { t: 'buf', dtype: 'f32', data: new Array(R * C).fill(0), out: true }, { t: 'i32', v: C }], tol: 1e-5 });
}
{ const R = 6, C = 96;
  cases.push({
    name: 'layernorm', kernel: 'ln', grid: [R, 1, 1], block: [32, 1, 1],
    src: `extern "C" __global__ void ln(const float* in, float* out, int C) {
  __shared__ float sh[32]; int r = blockIdx.x, t = threadIdx.x; float s = 0.0f;
  for (int c = t; c < C; c += blockDim.x) s += in[r * C + c];
  sh[t] = s; __syncthreads();
  for (int k = 16; k > 0; k >>= 1) { if (t < k) sh[t] += sh[t + k]; __syncthreads(); }
  float mean = sh[0] / C; __syncthreads(); float v = 0.0f;
  for (int c = t; c < C; c += blockDim.x) { float d = in[r * C + c] - mean; v += d * d; }
  sh[t] = v; __syncthreads();
  for (int k = 16; k > 0; k >>= 1) { if (t < k) sh[t] += sh[t + k]; __syncthreads(); }
  float inv = rsqrtf(sh[0] / C + 1e-5f);
  for (int c = t; c < C; c += blockDim.x) out[r * C + c] = (in[r * C + c] - mean) * inv;
}`,
    args: [{ t: 'buf', dtype: 'f32', data: vec(R * C, 3) }, { t: 'buf', dtype: 'f32', data: new Array(R * C).fill(0), out: true }, { t: 'i32', v: C }], tol: 2e-4 });
}
{ const M = 48, K = 40, N = 56;
  cases.push({
    name: 'tiled matmul 48x40x56', kernel: 'mm', grid: [Math.ceil(N / 16), Math.ceil(M / 16), 1], block: [16, 16, 1],
    src: `#define T 16
extern "C" __global__ void mm(const float* A, const float* B, float* C, int M, int K, int N) {
  __shared__ float As[T][T]; __shared__ float Bs[T][T];
  int row = blockIdx.y * T + threadIdx.y, col = blockIdx.x * T + threadIdx.x; float acc = 0.0f;
  for (int t = 0; t < (K + T - 1) / T; t++) {
    As[threadIdx.y][threadIdx.x] = (row < M && t * T + threadIdx.x < K) ? A[row * K + t * T + threadIdx.x] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] = (t * T + threadIdx.y < K && col < N) ? B[(t * T + threadIdx.y) * N + col] : 0.0f;
    __syncthreads();
    for (int k = 0; k < T; k++) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = acc;
}`,
    args: [{ t: 'buf', dtype: 'f32', data: vec(M * K) }, { t: 'buf', dtype: 'f32', data: vec(K * N) }, { t: 'buf', dtype: 'f32', data: new Array(M * N).fill(0), out: true },
      { t: 'i32', v: M }, { t: 'i32', v: K }, { t: 'i32', v: N }], tol: 1e-4, expect: { minFlops: 2 * M * K * N } });
}
{ const W = 64;
  cases.push({
    name: 'transpose (padded shared)', kernel: 'tr', grid: [W / 32, W / 32, 1], block: [32, 32, 1],
    src: `extern "C" __global__ void tr(const float* in, float* out, int n) {
  __shared__ float t[32][33]; int x = blockIdx.x * 32 + threadIdx.x, y = blockIdx.y * 32 + threadIdx.y;
  t[threadIdx.y][threadIdx.x] = in[y * n + x]; __syncthreads();
  x = blockIdx.y * 32 + threadIdx.x; y = blockIdx.x * 32 + threadIdx.y; out[y * n + x] = t[threadIdx.x][threadIdx.y];
}`,
    args: [{ t: 'buf', dtype: 'f32', data: vec(W * W) }, { t: 'buf', dtype: 'f32', data: new Array(W * W).fill(0), out: true }, { t: 'i32', v: W }], tol: 0,
    expect: { bankConflicts: 0 } });
}
{ const W = 24, H = 18;
  cases.push({
    name: 'conv 3x3 (halo bounds)', kernel: 'conv', grid: [3, 3, 1], block: [8, 8, 1],
    src: `extern "C" __global__ void conv(const float* in, float* out, int W, int H) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y; if (x >= W || y >= H) return;
  float s = 0.0f;
  for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
    int yy = y + dy, xx = x + dx;
    if (yy >= 0 && yy < H && xx >= 0 && xx < W) s += in[yy * W + xx] * (float)((2 - abs(dy)) * (2 - abs(dx)));
  }
  out[y * W + x] = s / 16.0f;
}`,
    args: [{ t: 'buf', dtype: 'f32', data: vec(W * H) }, { t: 'buf', dtype: 'f32', data: new Array(W * H).fill(0), out: true }, { t: 'i32', v: W }, { t: 'i32', v: H }], tol: 1e-5 });
}
{ const N = 4096;
  cases.push({
    name: 'histogram (shared privatised, int atomics)', kernel: 'hist', grid: [16, 1, 1], block: [256, 1, 1],
    src: `extern "C" __global__ void hist(const int* in, int* bins, int n) {
  __shared__ int loc[32]; if (threadIdx.x < 32) loc[threadIdx.x] = 0; __syncthreads();
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) atomicAdd(&loc[in[i]], 1); __syncthreads();
  if (threadIdx.x < 32) atomicAdd(&bins[threadIdx.x], loc[threadIdx.x]);
}`,
    args: [{ t: 'buf', dtype: 'i32', data: ivec(N, 32) }, { t: 'buf', dtype: 'i32', data: new Array(32).fill(0), out: true }, { t: 'i32', v: N }], tol: 0 });
}
{ const N = 64;
  cases.push({
    name: 'inclusive scan (Hillis-Steele, pointer swap)', kernel: 'scan', grid: [1, 1, 1], block: [64, 1, 1],
    src: `extern "C" __global__ void scan(const float* in, float* out, int n) {
  __shared__ float a[64]; __shared__ float b[64]; int t = threadIdx.x; a[t] = in[t]; __syncthreads();
  float* src = a; float* dst = b;
  for (int o = 1; o < n; o <<= 1) { if (t >= o) dst[t] = src[t] + src[t - o]; else dst[t] = src[t]; __syncthreads(); float* tmp = src; src = dst; dst = tmp; }
  out[t] = src[t];
}`,
    args: [{ t: 'buf', dtype: 'f32', data: vec(N) }, { t: 'buf', dtype: 'f32', data: new Array(N).fill(0), out: true }, { t: 'i32', v: N }], tol: 1e-5 });
}
{ const N = 2048;
  cases.push({
    name: 'gelu (tanhf) elementwise', kernel: 'gelu', grid: [8, 1, 1], block: [256, 1, 1],
    src: `extern "C" __global__ void gelu(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { float v = x[i]; y[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v))); }
}`,
    args: [{ t: 'buf', dtype: 'f32', data: vec(N, 4) }, { t: 'buf', dtype: 'f32', data: new Array(N).fill(0), out: true }, { t: 'i32', v: N }], tol: 2e-5 });
}
{ const N = 1500;
  cases.push({
    name: 'int division / modulo indexing', kernel: 'idx', grid: [6, 1, 1], block: [256, 1, 1],
    src: `extern "C" __global__ void idx(const float* in, float* out, int n, int w) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { int row = i / w; int col = i % w; int half = (n - 1) / 2; out[i] = in[row * w + col] + (float)(half / 7) + (float)(-i / 3); }
}`,
    args: [{ t: 'buf', dtype: 'f32', data: vec(N) }, { t: 'buf', dtype: 'f32', data: new Array(N).fill(0), out: true }, { t: 'i32', v: N }, { t: 'i32', v: 37 }], tol: 1e-5 });
}

module.exports = { cases };
