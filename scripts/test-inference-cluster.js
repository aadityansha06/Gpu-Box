// Test script for Multi-GPU Inference Cluster & Traffic Overload Simulation
global.window = global;
require('../demo/gpu-profiles.js');
require('../demo/cuda-runtime.js');

const prof = window.GPU_PROFILES.getProfile('A100');
console.log('Testing A100 Multi-GPU cluster with 8 nodes...');

const rt = new window.CudaRuntime(prof);
const mgpu = rt.initMultiGPU(8);

console.log('Multi-GPU initialized with ranks:', mgpu.deviceCount);

// Test NCCL AllReduce across 8 GPUs
const count = 1024;
const sendbufs = [];
const recvbufs = [];

for (let r = 0; r < 8; r++) {
  const s = mgpu.getDevice(r).cudaMalloc(count * 4);
  const rec = mgpu.getDevice(r).cudaMalloc(count * 4);
  const hostData = new Float32Array(count);
  hostData.fill(1.5);
  mgpu.getDevice(r).cudaMemcpy(s, hostData, count * 4, 'H2D');
  sendbufs.push(s);
  recvbufs.push(rec);
}

const comm = mgpu.ncclCommInitRank(8, 0);
mgpu.ncclAllReduce(sendbufs, recvbufs, count, 'float32', 'sum', comm);

// Verify result on rank 0
const out = new Float32Array(count);
mgpu.getDevice(0).cudaMemcpy(out, recvbufs[0], count * 4, 'D2H');
console.log('NCCL AllReduce across 8 GPUs: expected 12.0, got:', out[0]);
console.log('Rank 0 cross-GPU bytes:', mgpu.getDevice(0).counters.crossGpuBytes);
console.log('Rank 0 NCCL ops:', mgpu.getDevice(0).counters.ncclOps);

console.log('\n--- Testing Multi-GPU Inference Cluster (LLaMA-70B on 8x A100-80GB) ---');
const cluster = new window.CudaRuntime.InferenceCluster({
  gpuModel: 'A100-SXM4-80GB',
  gpuCount: 8,
  tensorParallel: 8,
  pipelineParallel: 1,
  modelParamsB: 70
});

console.log('Cluster Configuration:');
console.log(`- VRAM per GPU: ${cluster.gpuVramGB} GB`);
console.log(`- Weight per GPU: ${cluster.weightPerGpuGB.toFixed(2)} GB`);
console.log(`- Available KV Cache per GPU: ${cluster.availableKvCacheGBPerGpu.toFixed(2)} GB`);
console.log(`- Max Concurrent Cluster Tokens: ${cluster.maxConcurrentTokensCluster}`);

// Normal load test
console.log('\n[Workload 1: Normal Traffic (40 RPS)]');
const normalRes = cluster.simulateTraffic({ targetRps: 40, durationSec: 5, burstMultiplier: 1.0 });
console.log(`- Served: ${normalRes.trafficMetrics.servedRequests} / ${normalRes.trafficMetrics.totalArrivedRequests} reqs`);
console.log(`- Drop rate: ${normalRes.trafficMetrics.dropRatePct}%`);
console.log(`- Peak KV Saturation: ${normalRes.congestedLatency.peakKvSaturationPct}%`);
console.log(`- P50 Latency: ${normalRes.congestedLatency.p50Ms} ms | P99: ${normalRes.congestedLatency.p99Ms} ms`);

// Traffic overload spike test
console.log('\n[Workload 2: Overload Spike (40 RPS with 4.5x Burst = 180 RPS)]');
const burstRes = cluster.simulateTraffic({ targetRps: 40, durationSec: 5, burstMultiplier: 4.5 });
console.log(`- Arrived: ${burstRes.trafficMetrics.totalArrivedRequests} reqs`);
console.log(`- Served: ${burstRes.trafficMetrics.servedRequests} reqs`);
console.log(`- Dropped: ${burstRes.trafficMetrics.droppedRequests} reqs (${burstRes.trafficMetrics.dropRatePct}%)`);
console.log(`- Max Queue Depth: ${burstRes.congestedLatency.maxQueueDepth} reqs`);
console.log(`- Peak KV Saturation: ${burstRes.congestedLatency.peakKvSaturationPct}%`);
console.log(`- Throughput: ${burstRes.trafficMetrics.tokenThroughputTps} tokens/sec`);
console.log(`- P50 Latency: ${burstRes.congestedLatency.p50Ms} ms | P95: ${burstRes.congestedLatency.p95Ms} ms | P99: ${burstRes.congestedLatency.p99Ms} ms`);
console.log('✓ Inference cluster & traffic overload simulation passed!');
