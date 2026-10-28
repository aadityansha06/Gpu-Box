import ctypes

print("Checking NVIDIA Driver API...")
try:
    cuda = ctypes.windll.nvcuda
    print("CUDA Driver DLL loaded successfully:", cuda)
    res = cuda.cuInit(0)
    print("cuInit(0) result code:", res)
    if res == 0:
        count = ctypes.c_int()
        cuda.cuDeviceGetCount(ctypes.byref(count))
        print("Device count:", count.value)
        dev = ctypes.c_int()
        cuda.cuDeviceGet(ctypes.byref(dev), 0)
        name = (ctypes.c_char * 128)()
        cuda.cuDeviceGetName(name, 128, dev)
        print("Device 0 Name:", name.value.decode('utf-8'))
        
        # Check compute capability
        major = ctypes.c_int()
        minor = ctypes.c_int()
        cuda.cuDeviceGetAttribute(ctypes.byref(major), 75, dev) # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
        cuda.cuDeviceGetAttribute(ctypes.byref(minor), 76, dev) # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
        print(f"Compute Capability: {major.value}.{minor.value}")
        
        # Check total memory
        totalMem = ctypes.c_size_t()
        cuda.cuDeviceTotalMem(ctypes.byref(totalMem), dev)
        print(f"Total VRAM: {totalMem.value / (1024**2):.1f} MiB")
except Exception as e:
    print("Failed to query CUDA Driver:", e)
