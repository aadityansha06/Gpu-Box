/**
 * Engine stress test: realistic kernels vs. independent CPU references.
 * Each case: CUDA source -> transpile -> run on simulator -> compare device
 * buffers with a CPU reference, and sanity-check counters (bytes, flops).
 *   node scripts/stress-engine.js [filter]
 */
global.window = global;
require('../demo/fp16-bf16.js');
require('../demo/race-detector.js');
require('../demo/sparse-memory.js');
require('../demo/topology-model.js');
require('../demo/desync-detector.js');
require('../demo/gpu-profiles.js');
require('../demo/cuda-runtime.js');
require('../demo/cuda-transpiler.js');

const filter = process.argv[2] || '';
const cases = [];
const F = Math.fround;
let seed = 42;
const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296 * 2 - 1; };

function run(code, { profile = 'A100', devices = 1 } = {}) {
  const t = window.CudaTranspiler.transpile(code);
  if (!t.success) return { ok: false, stage: 'transpile', errors: t.errors };
  const rt = new window.CudaRuntime(window.GPU_PROFILES.getProfile(profile));
  rt.deviceCount = devices; rt.clusterSelectCount = devices;
  if (devices > 1 && rt.initMultiGPU) rt.initMultiGPU(devices);
  const logs = [];
  const ol = console.log; console.log = (...a) => logs.push(a.join(' '));
  let thrown = null;
  try { new Function('__rt', t.code)(rt); } catch (e) { thrown = e.message; }
  console.log = ol;
  let diags = [];
  try { diags = rt.getDiagnostics(); } catch (e) {}
  const errD = diags.filter(d => d.type === 'error');
  if (!thrown && errD.length) thrown = 'device error: ' + (errD[0].message || JSON.stringify(errD[0]));
  return { ok: !thrown, thrown, rt, logs, diags, counters: rt.getCounters().toJSON() };
}
const bufs = (rt, d = 0) => {
  const mem = rt._multiGPU ? rt._multiGPU.getDevice(d).memory : rt._memory;
  return Array.from(mem._allocations.values()).map(a => new Float32Array(a.buffer.slice(0)));
};
const lit = (arr) => '{' + Array.from(arr).join(',') + '}';
function close(a, b, tol, label) {
  let worst = 0, wi = -1;
  for (let i = 0; i < b.length; i++) {
    const d = Math.abs(a[i] - b[i]) / Math.max(1, Math.abs(b[i]));
    if (!(d <= worst)) { worst = d; wi = i; }
    if (Number.isNaN(a[i])) { worst = Infinity; wi = i; break; }
  }
  if (worst > tol) throw new Error(`${label}: idx ${wi} sim=${a[wi]} ref=${b[wi]} rel=${worst.toExponential(2)}`);
}
const add = (name, fn) => cases.push({ name, fn });

// 1 --------------------------------------------------------------------------
add('saxpy grid-stride N=1000', () => {
  const N = 1000, a = [], b = [], ref = [];
  for (let i = 0; i < N; i++) { a.push(F(rnd())); b.push(F(rnd())); }
  for (let i = 0; i < N; i++) ref.push(F(F(2.5 * a[i]) + b[i]));
  const r = run(`
#include <cuda_runtime.h>
__global__ void saxpy(float a, const float* x, float* y, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    y[i] = a * x[i] + y[i];
}
int main(){ int n=${N}; float hx[${N}] = ${lit(a)}; float hy[${N}] = ${lit(b)};
 float *dx,*dy; cudaMalloc(&dx,n*4); cudaMalloc(&dy,n*4);
 cudaMemcpy(dx,hx,n*4,cudaMemcpyHostToDevice); cudaMemcpy(dy,hy,n*4,cudaMemcpyHostToDevice);
 saxpy<<<4,128>>>(2.5f,dx,dy,n);
 cudaMemcpy(hy,dy,n*4,cudaMemcpyDeviceToHost); return 0; }`);
  if (!r.ok) throw new Error(r.thrown || JSON.stringify(r.errors));
  close(bufs(r.rt)[1], ref, 1e-6, 'y');
  if (r.counters.memcpyH2DBytes !== 8000) throw new Error('H2D bytes ' + r.counters.memcpyH2DBytes);
  if (r.counters.memcpyD2HBytes !== 4000) throw new Error('D2H bytes ' + r.counters.memcpyD2HBytes);
  if (r.counters.globalLoadBytes < 8000) throw new Error('global load bytes too low ' + r.counters.globalLoadBytes);
  if (r.counters.flopCount < 1000) throw new Error('flops too low ' + r.counters.flopCount);
});

// 2 --------------------------------------------------------------------------
add('block reduction (shared + atomicAdd)', () => {
  const N = 5000, x = []; let ref = 0;
  for (let i = 0; i < N; i++) { x.push(F(rnd())); ref += x[i]; }
  const r = run(`
#include <cuda_runtime.h>
__global__ void red(const float* in, float* out, int n){
  __shared__ float s[256]; int t=threadIdx.x; int i=blockIdx.x*blockDim.x+t;
  s[t] = i<n ? in[i] : 0.0f; __syncthreads();
  for (int st=blockDim.x/2; st>0; st>>=1){ if(t<st) s[t]+=s[t+st]; __syncthreads(); }
  if(t==0) atomicAdd(out, s[0]);
}
int main(){ int n=${N}; float hx[${N}] = ${lit(x)}; float h=0; float *dx,*d;
 cudaMalloc(&dx,n*4); cudaMalloc(&d,4); cudaMemset(d,0,4);
 cudaMemcpy(dx,hx,n*4,cudaMemcpyHostToDevice);
 red<<<(n+255)/256,256>>>(dx,d,n); cudaMemcpy(&h,d,4,cudaMemcpyDeviceToHost); return 0; }`);
  if (!r.ok) throw new Error(r.thrown || JSON.stringify(r.errors));
  close([bufs(r.rt)[1][0]], [ref], 1e-4, 'sum');
  if (r.counters.syncthreadsCount === 0) throw new Error('no syncthreads counted');
  if (r.counters.atomicOps !== 20) throw new Error('atomicOps ' + r.counters.atomicOps);
});

// 3 --------------------------------------------------------------------------
add('warp shuffle reduction', () => {
  const N = 256, x = []; let ref = 0;
  for (let i = 0; i < N; i++) { x.push(F(rnd())); ref += x[i]; }
  const r = run(`
#include <cuda_runtime.h>
__global__ void wred(const float* in, float* out){
  float v = in[blockIdx.x*blockDim.x+threadIdx.x];
  for (int o=16;o>0;o>>=1) v += __shfl_down_sync(0xffffffff, v, o);
  if ((threadIdx.x & 31)==0) atomicAdd(out, v);
}
int main(){ float hx[${N}] = ${lit(x)}; float *dx,*d; cudaMalloc(&dx,1024); cudaMalloc(&d,4); cudaMemset(d,0,4);
 cudaMemcpy(dx,hx,1024,cudaMemcpyHostToDevice); wred<<<2,128>>>(dx,d); return 0; }`);
  if (!r.ok) throw new Error(r.thrown || JSON.stringify(r.errors));
  close([bufs(r.rt)[1][0]], [ref], 1e-4, 'sum');
});

// 4 --------------------------------------------------------------------------
add('tiled matmul 40x24x56 (non-multiples of 16)', () => {
  const M = 40, K = 24, N = 56, A = [], B = [], C = [];
  for (let i = 0; i < M * K; i++) A.push(F(rnd())); for (let i = 0; i < K * N; i++) B.push(F(rnd()));
  for (let i = 0; i < M; i++) for (let j = 0; j < N; j++) { let s = 0; for (let k = 0; k < K; k++) s += A[i * K + k] * B[k * N + j]; C.push(s); }
  const r = run(`
#include <cuda_runtime.h>
#define T 16
__global__ void mm(const float* A,const float* B,float* C,int M,int K,int N){
  __shared__ float As[T][T]; __shared__ float Bs[T][T];
  int row=blockIdx.y*T+threadIdx.y, col=blockIdx.x*T+threadIdx.x; float acc=0.0f;
  for (int t=0;t<(K+T-1)/T;t++){
    As[threadIdx.y][threadIdx.x] = (row<M && t*T+threadIdx.x<K) ? A[row*K+t*T+threadIdx.x] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] = (t*T+threadIdx.y<K && col<N) ? B[(t*T+threadIdx.y)*N+col] : 0.0f;
    __syncthreads();
    for (int k=0;k<T;k++) acc += As[threadIdx.y][k]*Bs[k][threadIdx.x];
    __syncthreads();
  }
  if(row<M && col<N) C[row*N+col]=acc;
}
int main(){ int M=${M},K=${K},N=${N}; float hA[${M * K}] = ${lit(A)}; float hB[${K * N}] = ${lit(B)}; float hC[${M * N}];
 float *dA,*dB,*dC; cudaMalloc(&dA,M*K*4); cudaMalloc(&dB,K*N*4); cudaMalloc(&dC,M*N*4);
 cudaMemcpy(dA,hA,M*K*4,cudaMemcpyHostToDevice); cudaMemcpy(dB,hB,K*N*4,cudaMemcpyHostToDevice);
 dim3 b(T,T); dim3 g((N+T-1)/T,(M+T-1)/T); mm<<<g,b>>>(dA,dB,dC,M,K,N);
 cudaMemcpy(hC,dC,M*N*4,cudaMemcpyDeviceToHost); return 0; }`);
  if (!r.ok) throw new Error(r.thrown || JSON.stringify(r.errors));
  close(bufs(r.rt)[2], C, 1e-4, 'C');
  const ideal = 2 * M * K * N;
  if (r.counters.flopCount < ideal * 0.5) throw new Error(`flops ${r.counters.flopCount} vs ideal ${ideal}`);
});

// 5 --------------------------------------------------------------------------
add('softmax rows with expf/fmaxf', () => {
  const R = 6, C = 50, x = [], ref = [];
  for (let i = 0; i < R * C; i++) x.push(F(rnd() * 20));
  for (let r = 0; r < R; r++) { let m = -Infinity, s = 0; for (let c = 0; c < C; c++) m = Math.max(m, x[r * C + c]);
    for (let c = 0; c < C; c++) s += Math.exp(x[r * C + c] - m); for (let c = 0; c < C; c++) ref.push(Math.exp(x[r * C + c] - m) / s); }
  const r = run(`
#include <cuda_runtime.h>
#include <math.h>
__global__ void sm(const float* in,float* out,int C){
  __shared__ float sh[64]; int r=blockIdx.x,t=threadIdx.x;
  float m=-INFINITY; for(int c=t;c<C;c+=blockDim.x) m=fmaxf(m,in[r*C+c]);
  sh[t]=m; __syncthreads();
  for(int s=blockDim.x/2;s>0;s>>=1){ if(t<s) sh[t]=fmaxf(sh[t],sh[t+s]); __syncthreads(); }
  m=sh[0]; __syncthreads();
  float sum=0.0f; for(int c=t;c<C;c+=blockDim.x) sum+=expf(in[r*C+c]-m);
  sh[t]=sum; __syncthreads();
  for(int s=blockDim.x/2;s>0;s>>=1){ if(t<s) sh[t]+=sh[t+s]; __syncthreads(); }
  sum=sh[0];
  for(int c=t;c<C;c+=blockDim.x) out[r*C+c]=expf(in[r*C+c]-m)/sum;
}
int main(){ int R=${R},C=${C}; float h[${R * C}] = ${lit(x)}; float *di,*dout; cudaMalloc(&di,R*C*4); cudaMalloc(&dout,R*C*4);
 cudaMemcpy(di,h,R*C*4,cudaMemcpyHostToDevice); sm<<<R,64>>>(di,dout,C); cudaMemcpy(h,dout,R*C*4,cudaMemcpyDeviceToHost); return 0; }`);
  if (!r.ok) throw new Error(r.thrown || JSON.stringify(r.errors));
  close(bufs(r.rt)[1], ref, 1e-4, 'softmax');
});

// 6 --------------------------------------------------------------------------
add('histogram atomics (uint, shared privatization)', () => {
  const N = 2048, x = [], ref = new Array(16).fill(0);
  for (let i = 0; i < N; i++) { const v = (Math.abs(rnd()) * 1000 | 0) % 16; x.push(v); ref[v]++; }
  const r = run(`
#include <cuda_runtime.h>
__global__ void hist(const int* in,int* bins,int n){
  __shared__ int loc[16]; if(threadIdx.x<16) loc[threadIdx.x]=0; __syncthreads();
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) atomicAdd(&loc[in[i]],1); __syncthreads();
  if(threadIdx.x<16) atomicAdd(&bins[threadIdx.x],loc[threadIdx.x]);
}
int main(){ int n=${N}; int h[${N}] = ${lit(x)}; int *d,*b; cudaMalloc(&d,n*4); cudaMalloc(&b,64); cudaMemset(b,0,64);
 cudaMemcpy(d,h,n*4,cudaMemcpyHostToDevice); hist<<<8,256>>>(d,b,n); return 0; }`);
  if (!r.ok) throw new Error(r.thrown || JSON.stringify(r.errors));
  const mem = Array.from(r.rt._memory._allocations.values())[1];
  const got = Array.from(new Float32Array(mem.buffer));
  if (JSON.stringify(got) !== JSON.stringify(ref)) throw new Error(`bins ${got} vs ${ref}`);
});

// 7 --------------------------------------------------------------------------
add('device functions, templates, structs, __constant__', () => {
  const r = run(`
#include <cuda_runtime.h>
struct P { float x; float y; };
__constant__ float coef[4];
__device__ float sq(float v){ return v*v; }
template<typename T> __device__ T mx(T a,T b){ return a>b?a:b; }
__global__ void k(float* out,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  P p; p.x=(float)i; p.y=sq((float)i);
  out[i]=mx(p.x,p.y)+coef[i&3];
}
int main(){ float c[4]={1,2,3,4}; cudaMemcpyToSymbol(coef,c,16); float *d; cudaMalloc(&d,64*4); k<<<2,32>>>(d,64); return 0; }`);
  if (!r.ok) throw new Error(r.thrown || JSON.stringify(r.errors));
  const ref = []; for (let i = 0; i < 64; i++) ref.push(Math.max(i, i * i) + [1, 2, 3, 4][i & 3]);
  close(bufs(r.rt)[0], ref, 1e-6, 'out');
});

// 8 --------------------------------------------------------------------------
add('transpose with padded shared (bank conflicts down)', () => {
  const mk = (pad) => `
#include <cuda_runtime.h>
__global__ void tr(const float* in,float* out,int n){
  __shared__ float t[32][32+${pad}]; int x=blockIdx.x*32+threadIdx.x,y=blockIdx.y*32+threadIdx.y;
  t[threadIdx.y][threadIdx.x]=in[y*n+x]; __syncthreads();
  x=blockIdx.y*32+threadIdx.x; y=blockIdx.x*32+threadIdx.y; out[y*n+x]=t[threadIdx.x][threadIdx.y];
}
int main(){ int n=64; float *a,*b; cudaMalloc(&a,n*n*4); cudaMalloc(&b,n*n*4); cudaMemset(a,0,n*n*4);
 dim3 bl(32,32); dim3 g(2,2); tr<<<g,bl>>>(a,b,n); return 0; }`;
  const a = run(mk(0)), b = run(mk(1));
  if (!a.ok || !b.ok) throw new Error(a.thrown || b.thrown || 'compile');
  if (!(a.counters.bankConflicts > b.counters.bankConflicts)) throw new Error(`conflicts unpadded=${a.counters.bankConflicts} padded=${b.counters.bankConflicts}`);
});

// 9 --------------------------------------------------------------------------
add('coalesced vs strided global transactions', () => {
  const mk = (idx) => `
#include <cuda_runtime.h>
__global__ void k(const float* a,float* b){ int i=blockIdx.x*blockDim.x+threadIdx.x; b[i]=a[${idx}]; }
int main(){ float *a,*b; cudaMalloc(&a,1<<20); cudaMalloc(&b,1<<20); cudaMemset(a,0,1<<20); k<<<8,128>>>(a,b); return 0; }`;
  const c = run(mk('i')), s = run(mk('(i*33)%1024'));
  if (!(s.counters.globalTransactions > c.counters.globalTransactions * 2)) throw new Error(`coalesced=${c.counters.globalTransactions} strided=${s.counters.globalTransactions}`);
});

// 10 -------------------------------------------------------------------------
add('divergence + prefix scan (Hillis-Steele)', () => {
  const N = 64, x = [], ref = []; let s = 0;
  for (let i = 0; i < N; i++) { x.push(F(i % 7 - 3)); s += x[i]; ref.push(s); }
  const r = run(`
#include <cuda_runtime.h>
__global__ void scan(const float* in,float* out,int n){
  __shared__ float a[64]; __shared__ float b[64]; int t=threadIdx.x; a[t]=in[t]; __syncthreads();
  float* src=a; float* dst=b;
  for(int o=1;o<n;o<<=1){ if(t>=o) dst[t]=src[t]+src[t-o]; else dst[t]=src[t]; __syncthreads(); float* tmp=src; src=dst; dst=tmp; }
  out[t]=src[t];
}
int main(){ float h[${N}] = ${lit(x)}; float *di,*d; cudaMalloc(&di,256); cudaMalloc(&d,256); cudaMemcpy(di,h,256,cudaMemcpyHostToDevice);
 scan<<<1,64>>>(di,d,64); return 0; }`);
  if (!r.ok) throw new Error(r.thrown || JSON.stringify(r.errors));
  close(bufs(r.rt)[1], ref, 1e-6, 'scan');
  if (r.counters.divergentBranches === 0) throw new Error('no divergence recorded');
});

// 11 -------------------------------------------------------------------------
add('layernorm (rsqrtf, 2-pass)', () => {
  const R = 4, C = 96, x = [], ref = [];
  for (let i = 0; i < R * C; i++) x.push(F(rnd() * 3));
  for (let r = 0; r < R; r++) { let m = 0; for (let c = 0; c < C; c++) m += x[r * C + c]; m /= C; let v = 0;
    for (let c = 0; c < C; c++) v += (x[r * C + c] - m) ** 2; v /= C; for (let c = 0; c < C; c++) ref.push((x[r * C + c] - m) / Math.sqrt(v + 1e-5)); }
  const r = run(`
#include <cuda_runtime.h>
#include <math.h>
__global__ void ln(const float* in,float* out,int C){
  __shared__ float sh[32]; int r=blockIdx.x,t=threadIdx.x; float s=0;
  for(int c=t;c<C;c+=blockDim.x) s+=in[r*C+c]; sh[t]=s; __syncthreads();
  for(int k=16;k>0;k>>=1){ if(t<k) sh[t]+=sh[t+k]; __syncthreads(); }
  float mean=sh[0]/C; __syncthreads(); float v=0;
  for(int c=t;c<C;c+=blockDim.x){ float d=in[r*C+c]-mean; v+=d*d; } sh[t]=v; __syncthreads();
  for(int k=16;k>0;k>>=1){ if(t<k) sh[t]+=sh[t+k]; __syncthreads(); }
  float inv=rsqrtf(sh[0]/C+1e-5f);
  for(int c=t;c<C;c+=blockDim.x) out[r*C+c]=(in[r*C+c]-mean)*inv;
}
int main(){ int R=${R},C=${C}; float h[${R * C}] = ${lit(x)}; float *di,*d; cudaMalloc(&di,R*C*4); cudaMalloc(&d,R*C*4);
 cudaMemcpy(di,h,R*C*4,cudaMemcpyHostToDevice); ln<<<R,32>>>(di,d,C); return 0; }`);
  if (!r.ok) throw new Error(r.thrown || JSON.stringify(r.errors));
  close(bufs(r.rt)[1], ref, 1e-3, 'ln');
});

// 12 -------------------------------------------------------------------------
add('streams, events, async copy, sync, error API', () => {
  const r = run(`
#include <cuda_runtime.h>
#include <stdio.h>
__global__ void inc(float* p,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) p[i]+=1.0f; }
int main(){ float h[64]; for(int i=0;i<64;i++) h[i]=0; float* d; cudaMalloc(&d,256);
 cudaStream_t s; cudaStreamCreate(&s); cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
 cudaMemcpyAsync(d,h,256,cudaMemcpyHostToDevice,s); cudaEventRecord(a,s);
 inc<<<2,32,0,s>>>(d,64); cudaEventRecord(b,s); cudaStreamSynchronize(s); cudaDeviceSynchronize();
 float ms=0; cudaEventElapsedTime(&ms,a,b); cudaMemcpy(h,d,256,cudaMemcpyDeviceToHost);
 cudaError_t e = cudaGetLastError(); if(e!=cudaSuccess) printf("ERR %s\\n", cudaGetErrorString(e));
 printf("h0=%f ok\\n", h[0]); cudaFree(d); cudaStreamDestroy(s); return 0; }`);
  if (!r.ok) throw new Error(r.thrown || JSON.stringify(r.errors));
  const out = r.logs.join('\n') + r.diags.filter(d => d.type === 'printf').map(d => d.message).join('');
  if (!/h0=1\.0+ ok/.test(out)) throw new Error('printf output: ' + JSON.stringify(out) + ' diags=' + JSON.stringify(r.diags.slice(0, 3)));
});

// 13 -------------------------------------------------------------------------
add('uint/int/bit ops, 64-bit, casts', () => {
  const r = run(`
#include <cuda_runtime.h>
__global__ void k(unsigned int* o,int* s){
  int i=threadIdx.x; unsigned int u=0x9e3779b9u*(i+1); o[i]=(u>>3)^(u<<5);
  s[i]=(-i)/3 + (i%5) + (int)(2.7f*i) + __popc(u) + (int)(u & 0xff);
}
int main(){ unsigned int* o; int* s; cudaMalloc(&o,128); cudaMalloc(&s,128); k<<<1,32>>>(o,s); return 0; }`);
  if (!r.ok) throw new Error(r.thrown || JSON.stringify(r.errors));
  const al = Array.from(r.rt._memory._allocations.values());
  const o = new Float32Array(al[0].buffer), s = new Float32Array(al[1].buffer);
  const pop = (v) => { let c = 0; v >>>= 0; while (v) { c += v & 1; v >>>= 1; } return c; };
  for (let i = 0; i < 32; i++) {
    const u = Math.imul(0x9e3779b9, i + 1) >>> 0;
    const eo = (((u >>> 3) ^ (u << 5)) >>> 0);
    const es = Math.trunc(-i / 3) + (i % 5) + Math.trunc(Math.fround(2.7 * i)) + pop(u) + (u & 0xff);
    if (Math.abs(o[i] - eo) > eo * 1.2e-7 + 1) throw new Error(`o[${i}]=${o[i]} want ${eo}`);
    if (s[i] !== es) throw new Error(`s[${i}]=${s[i]} want ${es}`);
  }
});

// 14 -------------------------------------------------------------------------
add('2D conv 3x3 with halo bounds', () => {
  const W = 20, H = 12, x = [], ref = [], k = [1, 2, 1, 2, 4, 2, 1, 2, 1];
  for (let i = 0; i < W * H; i++) x.push(F(rnd()));
  for (let y = 0; y < H; y++) for (let xx = 0; xx < W; xx++) { let s = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const yy = y + dy, x2 = xx + dx; if (yy >= 0 && yy < H && x2 >= 0 && x2 < W) s += x[yy * W + x2] * k[(dy + 1) * 3 + dx + 1]; } ref.push(s / 16); }
  const r = run(`
#include <cuda_runtime.h>
__constant__ float K[9];
__global__ void conv(const float* in,float* out,int W,int H){
  int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y; if(x>=W||y>=H) return;
  float s=0; for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++){ int yy=y+dy,xx=x+dx; if(yy>=0&&yy<H&&xx>=0&&xx<W) s+=in[yy*W+xx]*K[(dy+1)*3+dx+1]; }
  out[y*W+x]=s/16.0f;
}
int main(){ int W=${W},H=${H}; float kk[9]=${lit(k)}; cudaMemcpyToSymbol(K,kk,36); float h[${W * H}] = ${lit(x)};
 float *a,*b; cudaMalloc(&a,W*H*4); cudaMalloc(&b,W*H*4); cudaMemcpy(a,h,W*H*4,cudaMemcpyHostToDevice);
 dim3 bl(8,8); dim3 g((W+7)/8,(H+7)/8); conv<<<g,bl>>>(a,b,W,H); return 0; }`);
  if (!r.ok) throw new Error(r.thrown || JSON.stringify(r.errors));
  close(bufs(r.rt)[1], ref, 1e-5, 'conv');
});

// ---- bug detection -----------------------------------------------------------
add('detects OOB write', () => {
  const r = run(`#include <cuda_runtime.h>
__global__ void k(float* p){ p[threadIdx.x + 1000] = 1.0f; }
int main(){ float* d; cudaMalloc(&d,64); k<<<1,32>>>(d); cudaDeviceSynchronize(); return 0; }`);
  const errs = r.diags.filter(d => d.type === 'error');
  if (!errs.length && r.ok) throw new Error('OOB not detected');
});
add('detects shared race without __syncthreads', () => {
  const r = run(`#include <cuda_runtime.h>
__global__ void k(float* o){ __shared__ float s[64]; int t=threadIdx.x; s[t]=t; o[t]=s[(t+1)%64]; }
int main(){ float* d; cudaMalloc(&d,256); k<<<1,64>>>(d); return 0; }`);
  if (!r.diags.length && r.ok) throw new Error('race not detected');
});
add('detects divergent __syncthreads', () => {
  const r = run(`#include <cuda_runtime.h>
__global__ void k(float* o){ if(threadIdx.x<16) __syncthreads(); o[threadIdx.x]=1; }
int main(){ float* d; cudaMalloc(&d,256); k<<<1,64>>>(d); return 0; }`);
  if (!r.diags.length && r.ok) throw new Error('divergent barrier not detected');
});
add('detects double free / leak', () => {
  const r = run(`#include <cuda_runtime.h>
int main(){ float* d; cudaMalloc(&d,64); cudaFree(d); cudaFree(d); return 0; }`);
  if (!r.diags.length && r.ok) throw new Error('double free not detected');
});

// ---- compiler errors ---------------------------------------------------------
const compileCases = {
  'stray token': 'int main(){ int x = 1 @@ 2; return 0; }',
  'unbalanced brace': '__global__ void k(float* p){ p[0]=1; \nint main(){ return 0; }',
  'undeclared identifier in kernel': '#include <cuda_runtime.h>\n__global__ void k(float* p){ p[0]=undefinedVar; }\nint main(){ float*d; cudaMalloc(&d,4); k<<<1,1>>>(d); return 0; }',
  'unknown kernel launch': '#include <cuda_runtime.h>\nint main(){ nokernel<<<1,1>>>(); return 0; }',
  'unsupported inline PTX': '__global__ void k(){ asm("mov.u32 %0, %laneid;" : "=r"(x)); }\nint main(){return 0;}',
  'unsupported wmma': '#include <mma.h>\n__global__ void k(){ nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,16,16,16,half,nvcuda::wmma::row_major> a; }\nint main(){return 0;}',
};
for (const [n, src] of Object.entries(compileCases)) {
  add('compile error: ' + n, () => {
    const t = window.CudaTranspiler.transpile(src);
    let rejected = !t.success;
    if (!rejected) { const r = run(src); rejected = !r.ok || r.diags.some(d => d.type === 'error'); }
    if (!rejected) throw new Error('accepted bad program silently');
    if (!t.success && !(t.errors && t.errors.length && String(t.errors[0]).length > 5)) throw new Error('no useful message');
  });
}

// ---- multi-GPU ---------------------------------------------------------------
for (const nGpu of [2, 4, 8, 16]) {
  add(`multi-GPU ${nGpu}x ncclAllReduce SUM + traffic + time`, () => {
    const N = 64;
    const r = run(`
#include <cuda_runtime.h>
#include <nccl.h>
__global__ void fill(float* p,int n,float v){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) p[i]=v+i; }
int main(){ int nd=${nGpu}; int n=${N}; ncclComm_t comms[${nGpu}]; int devs[${nGpu}]; float* s[${nGpu}]; float* rcv[${nGpu}];
 for(int i=0;i<nd;i++) devs[i]=i;
 ncclCommInitAll(comms,nd,devs);
 for(int i=0;i<nd;i++){ cudaSetDevice(i); cudaMalloc(&s[i],n*4); cudaMalloc(&rcv[i],n*4); fill<<<1,64>>>(s[i],n,(float)i); }
 ncclGroupStart(); for(int i=0;i<nd;i++) ncclAllReduce(s[i],rcv[i],n,ncclFloat,ncclSum,comms[i],0); ncclGroupEnd();
 for(int i=0;i<nd;i++){ cudaSetDevice(i); cudaDeviceSynchronize(); } return 0; }`, { devices: nGpu });
    if (!r.ok) throw new Error(r.thrown || JSON.stringify(r.errors));
    const per = []; for (let d = 0; d < nGpu; d++) per.push(bufs(r.rt, d));
    // per-device allocs: [send, recv]
    for (let d = 0; d < nGpu; d++) {
      const rcv = per[d][1]; if (!rcv) throw new Error('device ' + d + ' has no recv buffer');
      for (let i = 0; i < N; i++) {
        const want = (nGpu * (nGpu - 1)) / 2 + nGpu * i;
        if (Math.abs(rcv[i] - want) > 1e-3) throw new Error(`dev${d}[${i}]=${rcv[i]} want ${want}`);
      }
    }
    const c = r.counters;
    const ringPer = 2 * (nGpu - 1) / nGpu * N * 4;
    if (!(c.crossGpuBytes > 0)) throw new Error('crossGpuBytes = 0');
    if (!(c.ncclOps >= 1)) throw new Error('ncclOps = 0');
    const exp = ringPer * nGpu;
    if (Math.abs(c.crossGpuBytes - exp) / exp > 0.01) throw new Error(`crossGpuBytes=${c.crossGpuBytes} expected ring ${exp}`);
    if (!(c.estimatedTimeMs > 0)) throw new Error('estimatedTimeMs = 0');
  });
}
add('multi-GPU desync: rank skips allreduce is reported', () => {
  const r = run(`
#include <cuda_runtime.h>
#include <nccl.h>
int main(){ int nd=4; ncclComm_t comms[4]; int devs[4]={0,1,2,3}; float* s[4]; float* rc[4];
 ncclCommInitAll(comms,nd,devs);
 for(int i=0;i<nd;i++){ cudaSetDevice(i); cudaMalloc(&s[i],64); cudaMalloc(&rc[i],64); }
 ncclGroupStart(); for(int i=0;i<nd-1;i++) ncclAllReduce(s[i],rc[i],16,ncclFloat,ncclSum,comms[i],0); ncclGroupEnd();
 for(int i=0;i<nd;i++){ cudaSetDevice(i); cudaDeviceSynchronize(); } return 0; }`, { devices: 4 });
  const flagged = !r.ok || r.diags.some(d => d.type === 'error' || /desync|deadlock|mismatch/i.test(JSON.stringify(d)));
  if (!flagged) throw new Error('missing-rank collective not flagged');
});
add('multi-GPU P2P cudaMemcpyPeer accounts bytes', () => {
  const r = run(`
#include <cuda_runtime.h>
int main(){ float *a,*b; cudaSetDevice(0); cudaMalloc(&a,4096); cudaMemset(a,0,4096); cudaSetDevice(1); cudaMalloc(&b,4096);
 cudaMemcpyPeer(b,1,a,0,4096); cudaDeviceSynchronize(); return 0; }`, { devices: 2 });
  if (!r.ok) throw new Error(r.thrown || JSON.stringify(r.errors));
  if (r.counters.crossGpuBytes < 4096) throw new Error('crossGpuBytes=' + r.counters.crossGpuBytes);
});

// ---- runner -----------------------------------------------------------------
let pass = 0, fail = 0; const failures = [];
for (const c of cases) {
  if (filter && !c.name.includes(filter)) continue;
  try { c.fn(); pass++; console.log('PASS ', c.name); }
  catch (e) { fail++; failures.push([c.name, e.message]); console.log('FAIL ', c.name, '\n      ->', e.message.split('\n')[0]); }
}
console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
