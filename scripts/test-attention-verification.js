const assert = require('assert');
global.window = global;
require('../demo/gpu-profiles.js');
require('../demo/cuda-runtime.js');
require('../demo/cuda-transpiler.js');

function cpuAttention(Q, K, V, N, D) {
  const scores = new Float32Array(N * N);
  const out = new Float32Array(N * D);
  for (let i = 0; i < N; i++) {
    let maxVal = -Infinity;
    for (let j = 0; j < N; j++) {
      let acc = 0;
      for (let d = 0; d < D; d++) acc += Q[i * D + d] * K[j * D + d];
      scores[i * N + j] = acc / Math.sqrt(D);
      if (scores[i * N + j] > maxVal) maxVal = scores[i * N + j];
    }
    let sumExp = 0;
    for (let j = 0; j < N; j++) {
      scores[i * N + j] = Math.exp(scores[i * N + j] - maxVal);
      sumExp += scores[i * N + j];
    }
    for (let j = 0; j < N; j++) scores[i * N + j] /= sumExp;
  }
  for (let i = 0; i < N; i++) {
    for (let d = 0; d < D; d++) {
      let acc = 0;
      for (let j = 0; j < N; j++) acc += scores[i * N + j] * V[j * D + d];
      out[i * D + d] = acc;
    }
  }
  return out;
}

const testDims = [
  { N: 2, D: 2 },
  { N: 7, D: 8 },
  { N: 16, D: 16 }
];

testDims.forEach(({ N, D }) => {
  const code = `
  #include <cuda_runtime.h>
  #include <math.h>

  __global__ void scores_kernel(const float* Q, const float* K, float* scores, int N, int D) {
      int i = blockIdx.y * blockDim.y + threadIdx.y;
      int j = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < N && j < N) {
          float acc = 0.0f;
          for (int d = 0; d < D; d++) acc += Q[i * D + d] * K[j * D + d];
          scores[i * N + j] = acc / sqrtf((float)D);
      }
  }

  __global__ void softmax_rows_kernel(float* scores, int N) {
      int row = blockIdx.x;
      if (row >= N) return;
      float* r = scores + row * N;
      float m = -1e30f;
      for (int j = 0; j < N; j++) if (r[j] > m) m = r[j];
      float sum = 0.0f;
      for (int j = 0; j < N; j++) {
          r[j] = expf(r[j] - m);
          sum += r[j];
      }
      float inv = 1.0f / sum;
      for (int j = 0; j < N; j++) r[j] *= inv;
  }

  __global__ void av_kernel(const float* attn, const float* V, float* output, int N, int D) {
      int i = blockIdx.y * blockDim.y + threadIdx.y;
      int d = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < N && d < D) {
          float acc = 0.0f;
          for (int j = 0; j < N; j++) acc += attn[i * N + j] * V[j * D + d];
          output[i * D + d] = acc;
      }
  }

  int main() {
      int N = ${N};
      int D = ${D};
      float *d_Q, *d_K, *d_V, *d_scores, *d_output;
      cudaMalloc(&d_Q, N * D * 4);
      cudaMalloc(&d_K, N * D * 4);
      cudaMalloc(&d_V, N * D * 4);
      cudaMalloc(&d_scores, N * N * 4);
      cudaMalloc(&d_output, N * D * 4);

      float h_Q[${N * D}];
      float h_K[${N * D}];
      float h_V[${N * D}];
      for (int i = 0; i < N * D; i++) {
          h_Q[i] = (float)((i % 5) + 1);
          h_K[i] = (float)((i % 3) + 1);
          h_V[i] = (float)((i % 7) + 1);
      }
      cudaMemcpy(d_Q, h_Q, N * D * 4, cudaMemcpyHostToDevice);
      cudaMemcpy(d_K, h_K, N * D * 4, cudaMemcpyHostToDevice);
      cudaMemcpy(d_V, h_V, N * D * 4, cudaMemcpyHostToDevice);

      dim3 block(16, 16);
      dim3 gridScores((N + 15) / 16, (N + 15) / 16);
      scores_kernel<<<gridScores, block>>>(d_Q, d_K, d_scores, N, D);

      softmax_rows_kernel<<<N, 1>>>(d_scores, N);

      dim3 gridOut((D + 15) / 16, (N + 15) / 16);
      av_kernel<<<gridOut, block>>>(d_scores, d_V, d_output, N, D);
      cudaDeviceSynchronize();

      float h_output[${N * D}];
      cudaMemcpy(h_output, d_output, N * D * 4, cudaMemcpyDeviceToHost);
      for (int i = 0; i < N * D; i++) {
          printf("%.6f ", h_output[i]);
      }
      return 0;
  }
  `;

  const trans = window.CudaTranspiler.transpile(code);
  const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile('A100'));
  new Function('__rt', trans.code)(rt);

  const h_Q = new Float32Array(N * D);
  const h_K = new Float32Array(N * D);
  const h_V = new Float32Array(N * D);
  for (let i = 0; i < N * D; i++) {
    h_Q[i] = (i % 5) + 1;
    h_K[i] = (i % 3) + 1;
    h_V[i] = (i % 7) + 1;
  }
  const expected = cpuAttention(h_Q, h_K, h_V, N, D);

  const diags = rt.getDiagnostics().filter(d => d.type === 'printf');
  const actualStr = diags.map(d => d.message.trim()).join(' ').split(/\s+/).filter(Boolean);
  const actual = actualStr.map(parseFloat);

  assert.strictEqual(actual.length, N * D);
  let maxDiff = 0;
  for (let i = 0; i < N * D; i++) {
    const diff = Math.abs(actual[i] - expected[i]);
    if (diff > maxDiff) maxDiff = diff;
  }
  console.log(`Dim N=${N}, D=${D}: Max Diff vs CPU Reference = ${maxDiff.toExponential(4)}`);
  assert.ok(maxDiff < 1e-4, `Max diff ${maxDiff} must be < 1e-4`);
});
