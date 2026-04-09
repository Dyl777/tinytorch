# TinyTorch GPU-FIRST Implementation Status

## What Was Requested

1. **GPU-FIRST execution mode**: Max out ALL GPUs (CUDA + OpenGL + OpenCL) before touching CPU. CPU should only do coordination/overflow.
2. **Multiple backends on same GPU**: CUDA, OpenGL, and OpenCL should all work on the same physical GPU simultaneously, maxing out compute_0, compute_1, copy engines, 3D, etc.
3. **Max out Intel UHD 620**: The integrated GPU's shared memory should be fully utilized — it's "free" since it shares system RAM with no PCIe transfer cost.
4. **MemoryBudget system**: Configurable RAM percentage cap (default 30%) — any allocation >= this threshold uses mmap instead of RAM.
5. **Never load everything into RAM**: All data structures should use streaming/mmap when approaching the threshold.

---

## What Was Completed Successfully

### 1. MemoryBudget System (`auto_tensor.h`)
- Added `MemoryBudget` singleton with configurable `max_ram_percentage` (default 30%)
- `MemoryBudget::should_stream(bytes)` checks if allocation exceeds threshold
- Enforced at 10+ allocation sites in `distributed_tensor.h`:
  - `create_tensor_on_device()` — CPU_DENSE falls back to MmapTensor
  - `Shard::zero_grad()` — gradient tensors
  - `Shard::accumulate_grad()` — gradient accumulation
  - `parallel_binary_op` backward — grad input/output
  - `parallel_unary_op` backward — grad input
  - `sum()` backward — per-shard gradients
  - `mean()` backward — per-shard gradients
  - `backward()` scalar init — all shard gradient tensors
  - `all_gather()` — throws exception if exceeds budget
  - `refresh_devices()` — uses mmap intermediate

### 2. GPU-FIRST Execution Mode (`distributed_tensor.h`)
- Added `ExecutionMode::GPU_FIRST` to enum
- Added `execute_gpu_first()` to `ExecutionContextManager` — separates GPU vs CPU ops, launches all GPU shards concurrently via `std::async`
- Added GPU_FIRST case to `execute_shards()` switch
- Added GPU_FIRST case to `mode_to_string()`
- Added `initialize_shards_gpu_first()` — sorts GPUs by compute score, assigns shards to GPUs first, CPU gets overflow only
- Modified `DistributedTensor` constructor to call `initialize_shards_gpu_first()` when mode is GPU_FIRST

### 3. OpenCL Context Manager (`gpu_backends.h`)
- Added `OpenClDeviceContext` struct (device, context, queue, initialized flag)
- Added `OpenClContextManager` class with:
  - Per-device context and command queue creation
  - `get_context(device_id)`, `get_queue(device_id)`, `get_device(device_id)`
  - Lazy initialization (creates on first access)
  - Proper cleanup in destructor
  - Thread-safe via mutex
- Added `get_opencl_context_manager()` global accessor

### 4. OpenClTensor Factory Methods (`gpu_kernels.h`)
- Added `OpenClTensor::create_empty_for_device(size, device_id)` — auto-creates context/queue via OpenClContextManager
- Added `OpenClTensor::create_filled_for_device(size, fill_value, device_id)` — creates + fills GPU memory

### 5. `all_gather_mmap()` and `all_gather_distributed()` (`distributed_tensor.h`)
- `all_gather_mmap()` returns mmap-backed StreamTensor — never loads everything into RAM
- `all_gather_distributed()` returns resharded DistributedTensor via mmap intermediate

### 6. Thread-safe `randn()`/`rand()` (`distributed_tensor.h`)
- Fixed data race: each element gets independent RNG seeded from global index
- No shared `std::mt19937` across `std::async` threads

### 7. Bug Fixes Applied
- `refresh_devices()` — was loading full data into `std::vector<T>`, now uses mmap intermediate
- `all_gather()` — now guarded by MemoryBudget, throws if exceeds threshold
- Backend-aware gradient creation in all backward passes

---

## What Is NOT Completed (Critical Blockers)

### BLOCKER #1: OpenCL Backend Still Falls Back to CPU

**Location**: `distributed_tensor.h` line ~634, `create_tensor_on_device()` function, `case BackendType::OPENCL:`

**Problem**: The OpenCL case still has the OLD code:
```cpp
case BackendType::OPENCL: {
    // OpenCL require context/queue - use CPU fallback
    StreamConfig sc;
    sc.batch_size = std::max((std::size_t)1024, num_elements / 100);
    return std::make_unique<MmapTensor<T>>(shape_arr, 1, sc, fill_value);
}
```

**Should be**:
```cpp
case BackendType::OPENCL: {
    auto* t = OpenClTensor::create_filled_for_device(num_elements, static_cast<float>(fill_value), dev.device_id);
    if (t) {
        return std::unique_ptr<TensorBase<T>>(t);
    }
    // Fallback only if OpenCL context creation fails
    StreamConfig sc;
    sc.batch_size = std::max((std::size_t)65536, num_elements / 100);
    return std::make_unique<MmapTensor<T>>(shape_arr, 1, sc, fill_value);
}
```

**Why it wasn't fixed**: The `edit` tool repeatedly failed with "could not find the string to replace" — the exact whitespace/encoding in the file doesn't match what I'm searching for. The file was also corrupted by earlier PowerShell `-replace` operations that inserted literal `\n` characters instead of actual newlines. The user said they fixed the file themselves.

**Impact**: ALL OpenCL shards (including Intel UHD 620) are created as `MmapTensor` (CPU memory-mapped), meaning 0% GPU utilization on Intel. The MX130 CUDA shards may also be affected if the CUDA context has errors.

### BLOCKER #2: CUDA Module Load Error 700

**Problem**: During test runs, CUDA prints:
```
[CUDA] PTX generated for tensor_mul_scalar: 1119 bytes
[CUDA] Module load failed: 700
```

Error 700 = `CUDA_ERROR_ILLEGAL_ADDRESS`. This means the PTX kernel compiles but the module fails to load, likely due to:
- Invalid memory addresses being passed to the kernel
- Context corruption from too many concurrent contexts (multiple backends on same GPU)
- The MX130 (Maxwell, compute 5.0) has specific PTX requirements

**Impact**: CUDA operations silently fail (the `launch_kernel` function returns early on error, leaving output buffer uninitialized with garbage/zeros).

### BLOCKER #3: OpenGL Shader Compilation Failures

**Problem**: During test runs:
```
[GpuContext] Shader compilation failed (multiply_scalar): 
```

Empty error log means shader source is malformed or the OpenGL context doesn't support the compute shader syntax being used.

**Impact**: OpenGL shards produce garbage output.

### BLOCKER #4: GPU-FIRST Shard Assignment Still Uses All Device Types

**Problem**: The current `initialize_shards_gpu_first()` assigns shards to all GPU backends (CUDA + OpenGL + OpenCL for same physical GPU), which causes context conflicts. The MX130 gets 3 shards on 3 different backends but they can't all have active contexts simultaneously.

**What needs to happen**: Deduplicate physical GPUs, pick the BEST backend per physical GPU (CUDA > OpenGL > OpenCL), assign ONE shard per physical GPU backend.

---

## Test Results

### distributed_tensor_test.exe — 6/6 PASSED ✅
But these use the DEFAULT mode (AUTO/HYBRID), not GPU_FIRST. They pass because small tensors work fine on CPU Dense.

### llm_ops_test.exe — 7/7 PASSED ✅
Same — these use default mode with small tensors (2x3, 4x3, etc.), so CPU Dense handles everything.

### gpu_first_test.exe — CRASHED ❌
- Small tensor (100x100): Created 24 shards (way too many), CUDA module load failed, OpenGL shader failed, output was `Sum: 0` (garbage)
- Medium tensor (500x500): Crashed with exit code -1073740940 (access violation)

### Root Cause Chain:
1. `create_tensor_on_device()` for OpenCL → creates `MmapTensor` (CPU) → ALL computation on CPU
2. CUDA `Module load failed: 700` → silent failure → garbage output
3. OpenGL shader compilation fails → garbage output
4. GPU_FIRST creates too many shards per GPU → context conflicts → cascading failures

---

## Compile Commands That Work

```bash
cd C:\Users\AMBE\Downloads\ts-cpl\y5\tinytorch

# Distributed tensor test
"C:/Program Files (x86)/Embarcadero/Dev-Cpp/TDM-GCC-64/bin/g++.exe" ^
    -std=c++17 -Wall -Wextra ^
    -DTINYTORCH_USE_OPENCL_SDK ^
    -I"C:/Users/AMBE/OpenCL-SDK/install/include" ^
    -o distributed_tensor_test.exe distributed_tensor_test.cpp ^
    -L"C:/Users/AMBE/OpenCL-SDK/install/lib" -lOpenCL ^
    -lopengl32 -lgdi32

# LLM operations test
"C:/Program Files (x86)/Embarcadero/Dev-Cpp/TDM-GCC-64/bin/g++.exe" ^
    -std=c++17 -Wall -Wextra ^
    -DTINYTORCH_USE_OPENCL_SDK ^
    -I"C:/Users/AMBE/OpenCL-SDK/install/include" ^
    -o llm_ops_test.exe llm_ops_test.cpp ^
    -L"C:/Users/AMBE/OpenCL-SDK/install/lib" -lOpenCL ^
    -lopengl32 -lgdi32

# GPU-FIRST test
"C:/Program Files (x86)/Embarcadero/Dev-Cpp/TDM-GCC-64/bin/g++.exe" ^
    -std=c++17 -O2 ^
    -DTINYTORCH_USE_OPENCL_SDK ^
    -I"C:/Users/AMBE/OpenCL-SDK/install/include" ^
    -o gpu_first_test.exe gpu_first_test.cpp ^
    -L"C:/Users/AMBE/OpenCL-SDK/install/lib" -lOpenCL ^
    -lopengl32 -lgdi32
```

---

## Files Modified (Successfully)

| File | What Was Added | Status |
|------|---------------|--------|
| `auto_tensor.h` | `MemoryBudget` struct (lines 89-169), `#include <atomic>`, `<mutex>`, `<sstream>` | ✅ Complete |
| `gpu_backends.h` | `OpenClContextManager` class, `get_opencl_context_manager()` | ✅ Complete |
| `gpu_kernels.h` | `OpenClTensor::create_empty_for_device()`, `create_filled_for_device()` | ✅ Complete |
| `distributed_tensor.h` | `ExecutionMode::GPU_FIRST`, `execute_gpu_first()`, `initialize_shards_gpu_first()`, constructor modification, `mode_to_string` case, MemoryBudget guards in all backward passes, `all_gather_mmap()`, `all_gather_distributed()`, `refresh_devices()` mmap fix, thread-safe `randn()`/`rand()` | ✅ Partial — OpenCL case in `create_tensor_on_device()` NOT replaced |
| `gpu_first_test.cpp` | New test file for GPU-FIRST mode | ✅ Complete (but test crashes) |

## Files NOT Modified (But Should Be)

| File | What Needs To Change | Why |
|------|---------------------|-----|
| `distributed_tensor.h` line ~634 | Replace OpenCL fallback with `OpenClTensor::create_filled_for_device()` | Currently creates MmapTensor (CPU) for all OpenCL devices |
| `cuda_tensor.h` | Fix Module load error 700 — check kernel argument layout | CUDA kernels compile but fail to load |
| `gpu_tensor.h` | Fix shader compilation — check compute shader source | OpenGL shaders fail to compile |
| `distributed_tensor.h` `initialize_shards_gpu_first()` | Deduplicate physical GPUs, assign ONE shard per physical GPU backend | Currently creates 3 shards for MX130 (CUDA+GL+OCL) causing context conflicts |

---

## Hardware Context

Your system has:
- **NVIDIA GeForce MX130** (Maxwell GM108, Compute 5.0, 2GB VRAM, 3 SMs) — accessible via CUDA, OpenGL 4.3, OpenCL 1.2
- **Intel UHD Graphics 620** (24 EUs, shared system memory) — accessible via OpenCL only
- **15.86 GB system RAM**

The MX130 being Maxwell means:
- PTX must target `sm_50` (compute_50)
- No float atomics in OpenCL
- Max work group size 1024
- 48KB shared memory per block
- SSBO size limit ~1.9GB

---

## Next Steps (In Priority Order)

1. **Fix OpenCL case in `create_tensor_on_device()`** — manually edit line ~634 to call `OpenClTensor::create_filled_for_device()`. This is the SINGLE MOST IMPORTANT fix — it will make Intel UHD 620 and MX130 OpenCL backends actually run on GPU.

2. **Fix CUDA Module load error 700** — the PTX compiles but module load fails. Likely the kernel argument pointer layout is wrong for the CUDA Driver API on Maxwell.

3. **Deduplicate GPU backends in `initialize_shards_gpu_first()`** — one shard per physical GPU, pick best backend (CUDA first, then OpenCL, then OpenGL as last resort).

4. **Run large tensor test** (5000x5000 = 25M elements, ~100MB) to verify GPU-FIRST actually loads work on GPUs and not CPU.
