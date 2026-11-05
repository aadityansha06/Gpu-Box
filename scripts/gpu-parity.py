"""
Real-GPU parity check (called by scripts/gpu-parity.js).
Compiles each kernel with NVRTC, runs it on the physical NVIDIA GPU, and compares
the output buffers with what the simulator produced for identical inputs.
"""
import ctypes
import json
import os
import sys

nvrtc_bin = os.path.join(sys.prefix, 'Lib', 'site-packages', 'nvidia', 'cuda_nvrtc', 'bin')
if os.path.exists(nvrtc_bin):
    os.add_dll_directory(nvrtc_bin)
    os.environ['PATH'] = nvrtc_bin + os.pathsep + os.environ['PATH']

import numpy as np
from cuda.bindings import driver, nvrtc

spec = json.load(open(sys.argv[1]))

def ok(res, what):
    err = res[0] if isinstance(res, tuple) else res
    if int(err) != 0:
        raise RuntimeError(f"{what} failed: {err}")
    return res[1:] if isinstance(res, tuple) and len(res) > 1 else None

ok(driver.cuInit(0), 'cuInit')
(dev,) = ok(driver.cuDeviceGet(0), 'cuDeviceGet')
(name,) = ok(driver.cuDeviceGetName(128, dev), 'name')
dev_name = name.decode().rstrip('\x00').strip()
(major,) = ok(driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev), 'cc')
(minor,) = ok(driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev), 'cc')
(ctx,) = ok(driver.cuCtxCreate(None, 0, dev), 'ctx')

print("=" * 78)
print(f"  GPU PARITY: simulator vs physical {dev_name} (sm_{major}{minor})")
print("=" * 78)

def compile_ptx(src):
    full = ("#define abs_i(x) abs(x)\n" + src).encode()
    (prog,) = ok(nvrtc.nvrtcCreateProgram(full, b"k.cu", 0, [], []), 'nvrtcCreate')
    opt = f"--gpu-architecture=compute_{major}{minor}".encode()
    res = nvrtc.nvrtcCompileProgram(prog, 1, [opt])
    err = res[0] if isinstance(res, tuple) else res
    if int(err) != 0:
        (sz,) = nvrtc.nvrtcGetProgramLogSize(prog)[1:]
        buf = b" " * sz
        nvrtc.nvrtcGetProgramLog(prog, buf)
        raise RuntimeError("NVRTC: " + buf.decode(errors='ignore'))
    (sz,) = nvrtc.nvrtcGetPTXSize(prog)[1:]
    ptx = b" " * sz
    nvrtc.nvrtcGetPTX(prog, ptx)
    return ptx

results = []
for c in spec:
    row = {"name": c["name"], "status": "FAIL", "detail": ""}
    results.append(row)
    if c.get("simError"):
        row["detail"] = "simulator error: " + c["simError"]
        continue
    try:
        ptx = compile_ptx(c["src"])
        (mod,) = ok(driver.cuModuleLoadData(ptx), 'moduleLoad')
        (fn,) = ok(driver.cuModuleGetFunction(mod, c["kernel"].encode()), 'getFunction')
        dev_bufs, host_bufs, params, keep = {}, {}, [], []
        for i, a in enumerate(c["args"]):
            if a["t"] == "buf":
                dt = np.int32 if a["dtype"] == "i32" else np.float32
                arr = np.array(a["data"], dtype=dt)
                (dptr,) = ok(driver.cuMemAlloc(arr.nbytes), 'alloc')
                ok(driver.cuMemcpyHtoD(dptr, arr.ctypes.data, arr.nbytes), 'h2d')
                dev_bufs[i] = dptr
                host_bufs[i] = arr
                keep.append(ctypes.c_uint64(int(dptr)))
            elif a["t"] == "i32":
                keep.append(ctypes.c_int(int(a["v"])))
            else:
                keep.append(ctypes.c_float(float(a["v"])))
        params = (ctypes.c_void_p * len(keep))(*[ctypes.cast(ctypes.byref(k), ctypes.c_void_p) for k in keep])
        g, b = c["grid"], c["block"]
        launch = lambda: driver.cuLaunchKernel(fn, g[0], g[1], g[2], b[0], b[1], b[2], 0, 0, int(ctypes.cast(params, ctypes.c_void_p).value), 0)
        ok(launch(), 'launch')
        ok(driver.cuCtxSynchronize(), 'sync')
        worst = 0.0
        detail = []
        for i, sim in c["simOut"].items():
            i = int(i)
            out = np.zeros_like(host_bufs[i])
            ok(driver.cuMemcpyDtoH(out.ctypes.data, dev_bufs[i], out.nbytes), 'd2h')
            ref = out.astype(np.float64)
            got = np.array(sim, dtype=np.float64)
            denom = np.maximum(1.0, np.abs(ref))
            rel = np.abs(got - ref) / denom
            worst = max(worst, float(rel.max()) if rel.size else 0.0)
            if rel.size and rel.max() > c["tol"]:
                j = int(rel.argmax())
                detail.append(f"idx {j}: sim={got[j]:.8g} gpu={ref[j]:.8g}")
        # timing (informational)
        import time
        (e0,) = ok(driver.cuEventCreate(0), 'ev')
        (e1,) = ok(driver.cuEventCreate(0), 'ev')
        for _ in range(3):
            launch()
        driver.cuCtxSynchronize()
        driver.cuEventRecord(e0, 0)
        for _ in range(20):
            launch()
        driver.cuEventRecord(e1, 0)
        driver.cuEventSynchronize(e1)
        (ms,) = ok(driver.cuEventElapsedTime(e0, e1), 'elapsed')
        gpu_us = ms * 1000.0 / 20
        k = c.get("counters", {})
        row.update({
            "maxRelErr": worst, "gpuTimeUs": round(gpu_us, 2), "simTimeUs": round(k.get("estimatedTimeMs", 0) * 1000, 3),
            "simFlops": k.get("flopCount"), "simGlobalBytes": (k.get("globalLoadBytes", 0) or 0) + (k.get("globalStoreBytes", 0) or 0),
            "simBankConflicts": k.get("bankConflicts"), "simDivergentBranches": k.get("divergentBranches"),
            "simGlobalTransactions": k.get("globalTransactions"),
        })
        problems = c.get("metricProblems", [])
        if detail:
            row["detail"] = "; ".join(detail[:3])
        elif problems:
            row["detail"] = "metric check: " + "; ".join(problems)
        else:
            row["status"] = "PASS"
        for d in dev_bufs.values():
            driver.cuMemFree(d)
        driver.cuModuleUnload(mod)
    except Exception as e:
        row["detail"] = str(e)[:300]

print(f"{'kernel':46} {'status':6} {'max rel err':>11} {'gpu us':>8} {'sim FLOPs':>10} {'global B':>9}")
for r in results:
    print(f"{r['name'][:46]:46} {r['status']:6} {r.get('maxRelErr', float('nan')):11.2e} {r.get('gpuTimeUs', float('nan')):8.2f} "
          f"{str(r.get('simFlops', '-')):>10} {str(r.get('simGlobalBytes', '-')):>9}" + (f"   <- {r['detail']}" if r['detail'] else ""))

passed = sum(1 for r in results if r["status"] == "PASS")
print("-" * 78)
print(f"  PARITY: {passed}/{len(results)} kernels match the physical {dev_name}")
out = {"device": dev_name, "computeCapability": f"{major}.{minor}", "kernelsTested": len(results), "passed": passed, "results": results}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), "test-results-gpu-parity.json"), "w"), indent=2)
sys.exit(0 if passed == len(results) else 1)
