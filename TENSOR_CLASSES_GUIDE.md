# TinyTorch Tensor Classes Guide

## Class Hierarchy Overview

TinyTorch provides a clear separation between **user-facing tensor classes** (what you use in your code) and **system-only tensor classes** (internal implementation details you should never interact with directly).

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER-FACING CLASSES                          │
│  (These are the only classes you should use in your code)       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Tensor<T>                  Basic CPU tensor (small data)       │
│  AutoTensor<T>              Auto-selects Dense vs Mmap          │
│  DistributedTensor<T>       ★ MAIN CLASS - handles everything   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    SYSTEM-ONLY CLASSES                          │
│  (Internal implementation - DO NOT use directly)                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  DenseTensor<T>             In-memory CPU tensor                │
│  MmapTensor<T>              Memory-mapped CPU tensor            │
│  AutogradTensor<T>          Gradient tracking wrapper           │
│  CudaTensorWorker<T>        CUDA device worker                  │
│  GpuTensorWorker<T>         OpenGL device worker                │
│  OpenClTensorWorker<T>      OpenCL device worker               │
│  DistributedLoadBalancer<T> Shard assignment logic              │
│  DevicePool                 Device discovery                    │
│  ParallelismHeuristics      Strategy selection                  │
│  LazyQueue                  CPU lazy operations                 │
│  All parallelism executors  DP, TP, PP, FSDP, EP, 3D            │
│  All load balancing strats  7 different strategies              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## User-Facing Classes

### 1. `Tensor<T>` - Basic CPU Tensor

**Use when:** You need a simple, small tensor on CPU with no distribution.

```cpp
#include "tensor.h"

// Create from initializer list
Tensor<float> a({1.0f, 2.0f, 3.0f});

// Create with shape
std::size_t shape[] = {3, 4};
Tensor<float> b(shape, 2, 0.0f);  // 3x4 tensor filled with 0

// Operations
auto c = a.add(&b);
auto sum = a.sum();
```

**Limitations:**
- Single CPU device only
- No autograd
- No distribution
- All data in RAM

---

### 2. `AutoTensor<T>` - Automatic Backend Selection

**Use when:** You want automatic selection between in-memory and memory-mapped storage based on tensor size vs available RAM.

```cpp
#include "auto_tensor.h"

// Small tensor -> uses DenseTensor (in-memory)
float data[] = {1.0f, 2.0f, 3.0f};
auto small = AutoTensor<float>::from_data(data, 3);

// Large tensor -> uses MmapTensor (memory-mapped)
std::vector<float> large_data(100'000'000, 1.0f);
auto large = AutoTensor<float>::from_data(large_data.data(), large_data.size());

// Force streaming for large datasets
AutoConfig config;
config.force_streaming = true;
auto streamed = AutoTensor<float>::from_data(data, 3, config);

// Operations work regardless of backend
auto result = small->add(large.get());
```

**What it does automatically:**
- Checks tensor size vs available RAM
- Uses `DenseTensor` for small tensors (fast, in-memory)
- Uses `MmapTensor` for large tensors (out-of-core, batched)
- Threshold configurable via `AutoConfig`

**Limitations:**
- CPU only (no GPU)
- No autograd
- No distribution across devices

---

### 3. `DistributedTensor<T>` - ★ MAIN CLASS

**Use when:** You want automatic distribution across all available devices (CPU + all GPUs) with autograd support.

```cpp
#include "distributed_tensor.h"

// Create distributed tensor (auto-discovers all devices)
auto x = DistributedTensor<float>::zeros({1000, 1000});
auto y = DistributedTensor<float>::randn({1000, 1000});

// Operations are automatically distributed
auto z = x->multiply(y.get());
auto loss = z->sum();

// Backward pass distributed across all devices
loss->backward();

// Access elements (gathers from shards automatically)
float val = x->get_element(0);

// See distribution info
std::cout << x->distribution_info() << std::endl;
// Output: DistributedTensor<DATA_PARALLEL> shape=[1000, 1000], shards=2, devices=[CUDA: NVIDIA GeForce MX130, OpenGL: NVIDIA GeForce MX130]
```

**What it does automatically:**

1. **Device Discovery**: Finds all available devices at startup
   - CPU Dense (always available)
   - CPU Mmap (always available, for lazy operations)
   - CUDA GPUs (if CUDA runtime available)
   - OpenGL GPUs (if OpenGL 4.3+ available)
   - OpenCL devices (if OpenCL runtime available)

2. **Parallelism Selection**: Uses heuristics based on tensor size
   - **< 1MB**: Single device (fastest available)
   - **1MB - 100MB**: Data parallel (replicate across GPUs)
   - **100MB - 1GB**: Tensor parallel (shard weights across GPUs)
   - **> 1GB**: FSDP (fully sharded across all devices)

3. **CPU Lazy Operations**: All CPU operations use `StreamTensor` with:
   - Batched processing (configurable batch size)
   - Memory-mapped storage (no RAM pressure)
   - Lazy operation queue (flushed when needed)

4. **Load Balancing**: Distributes work based on:
   - Device compute capacity (from heartbeats)
   - Current device load
   - Available memory
   - 7 different strategies available

5. **Autograd Integration**: Backward pass works across distributed shards

**Factory Methods:**

```cpp
// Zeros
auto zeros = DistributedTensor<float>::zeros({100, 100});

// Ones
auto ones = DistributedTensor<float>::ones({100, 100});

// Random normal
auto rand = DistributedTensor<float>::randn({100, 100});

// From data
std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
auto from_data = DistributedTensor<float>::from_data(data, {2, 2});

// With gradient tracking
auto x = DistributedTensor<float>::randn({100, 100}, true);  // requires_grad = true

// Force specific parallelism
auto y = DistributedTensor<float>::zeros({100, 100}, false, ParallelismMode::DATA_PARALLEL);
```

**Operations:**

```cpp
auto x = DistributedTensor<float>::randn({100, 100});
auto y = DistributedTensor<float>::randn({100, 100});

// Element-wise (distributed)
auto z1 = x->add(y.get());
auto z2 = x->multiply(y.get());

// Scalar operations
auto z3 = x->add_scalar(5.0f);
auto z4 = x->multiply_scalar(2.0f);

// Reductions (gather result)
auto sum = x->sum();
auto mean = x->mean();

// Reshape
auto reshaped = x->reshape({50, 200});

// Access elements
float val = x->get_element(0);
x->set_element(0, 42.0f);

// Autograd
auto x_grad = DistributedTensor<float>::randn({100, 100}, true);
auto loss = x_grad->sum();
loss->backward();  // Gradients computed across all devices

// Flush lazy operations
x->flush();

// Refresh devices (if GPUs added/removed)
x->refresh_devices();
```

**Distribution Info:**

```cpp
auto x = DistributedTensor<float>::randn({1000, 1000});
std::cout << x->distribution_info() << std::endl;
// Example output:
// DistributedTensor<DATA_PARALLEL> shape=[1000, 1000], shards=2, devices=[CUDA: NVIDIA GeForce MX130, OpenGL: NVIDIA GeForce MX130]
```

---

## System-Only Classes (DO NOT USE DIRECTLY)

These classes are internal implementation details. They are documented here for completeness but **should never be used directly in user code**.

### `DenseTensor<T>`
In-memory CPU tensor. Used internally by `AutoTensor` and `DistributedTensor` for small tensors.

### `MmapTensor<T>`
Memory-mapped CPU tensor (wraps `StreamTensor`). Used internally for lazy CPU operations.

### `AutogradTensor<T>`
Gradient tracking wrapper. Used internally by `DistributedTensor` for autograd.

### `DenseTensorWorker<T>`, `MmapTensorWorker<T>`, `CudaTensorWorker<T>`, `GpuTensorWorker<T>`, `OpenClTensorWorker<T>`
Device-specific workers. Used internally by the load balancer.

### `DistributedLoadBalancer<T>`
Shard assignment logic. Used internally by `DistributedTensor`.

### `DevicePool`
Device discovery. Used internally by `DistributedTensor`.

### `ParallelismHeuristics`
Strategy selection engine. Used internally by `DistributedTensor`.

### `LazyQueue`
CPU lazy operation queue. Used internally by `DistributedTensor`.

### Parallelism Executors
- `DataParallelExecutor<T>`
- `TensorParallelExecutor<T>`
- `PipelineParallelExecutor<T>`
- `FSDPExecutor<T>`
- `ExpertParallelExecutor<T>`
- `Parallel3DExecutor<T>`

### Load Balancing Strategies
- `HeartbeatCapacityStrategy<T>`
- `WeightedRoundRobinStrategy<T>`
- `LeastConnectionsStrategy<T>`
- `PowerOfTwoChoicesStrategy<T>`
- `ConsistentHashingStrategy<T>`
- `PredictiveLoadStrategy<T>`
- `MinMaxFairnessStrategy<T>`

---

## Quick Start Guide

### For Most Users: Use `DistributedTensor<T>`

```cpp
#include "distributed_tensor.h"

int main() {
    // Create distributed tensors
    auto x = DistributedTensor<float>::randn({1000, 1000}, true);
    auto y = DistributedTensor<float>::randn({1000, 1000}, true);
    
    // Operations are automatically distributed
    auto z = x->multiply(y.get());
    auto loss = z->sum();
    
    // Backward pass
    loss->backward();
    
    // That's it! Everything is handled automatically.
    return 0;
}
```

### For Small Tensors: Use `Tensor<T>`

```cpp
#include "tensor.h"

int main() {
    Tensor<float> a({1.0f, 2.0f, 3.0f});
    Tensor<float> b({4.0f, 5.0f, 6.0f});
    auto c = a.add(&b);
    return 0;
}
```

### For Large CPU-Only Tensors: Use `AutoTensor<T>`

```cpp
#include "auto_tensor.h"

int main() {
    std::vector<float> data(100'000'000, 1.0f);
    auto tensor = AutoTensor<float>::from_data(data.data(), data.size());
    float sum = tensor->sum();
    return 0;
}
```

---

## Build Instructions

```bash
# Compile with all GPU backends
g++ -std=c++17 -Wall -Wextra \
    -DTINYTORCH_USE_OPENCL_SDK \
    -I"C:/Users/YOUR_NAME/OpenCL-SDK/install/include" \
    -o my_program my_program.cpp \
    -L"C:/Users/YOUR_NAME/OpenCL-SDK/install/lib" -lOpenCL \
    -lopengl32 -lgdi32

# CPU only (no GPU dependencies)
g++ -std=c++17 -Wall -Wextra -o my_program my_program.cpp
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Code                                   │
│         DistributedTensor<T> / AutoTensor<T> / Tensor<T>            │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    DistributedTensor<T>                              │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────────┐ │
│  │ DevicePool      │  │ Parallelism      │  │ LazyQueue          │ │
│  │ - CPU Dense     │  │ Heuristics       │  │ - Batched ops      │ │
│  │ - CPU Mmap      │  │ - Size-based     │  │ - StreamTensor     │ │
│  │ - CUDA GPUs     │  │ - Device count   │  │ - Lazy execution   │ │
│  │ - OpenGL GPUs   │  │ - Memory         │  │                    │ │
│  │ - OpenCL GPUs   │  │                  │  │                    │ │
│  └─────────────────┘  └──────────────────┘  └────────────────────┘ │
│                                  │                                  │
│                                  ▼                                  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Load Balancer + Workers                          │  │
│  │  ┌────────────┐ ┌────────────┐ ┌──────────┐ ┌────────────┐  │  │
│  │  │ Dense      │ │ Mmap       │ │ CUDA     │ │ OpenGL     │  │  │
│  │  │ Worker     │ │ Worker     │ │ Worker   │ │ Worker     │  │  │
│  │  └────────────┘ └────────────┘ └──────────┘ └────────────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                  │                                  │
│                                  ▼                                  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Parallelism Executors                           │  │
│  │  Data Parallel │ Tensor Parallel │ Pipeline │ FSDP │ 3D      │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Hardware                                     │
│         CPU Cores │ NVIDIA GPU │ Intel GPU │ OpenCL Devices         │
└─────────────────────────────────────────────────────────────────────┘
```

---

# APPENDIX: Change Log, Bug Fixes, and New Features

## Table of Contents
- [MemoryBudget System](#memorybudget-system)
- [Bug Fixes](#bug-fixes)
- [New Methods](#new-methods)
- [Compile Commands](#compile-commands)
- [Test Results](#test-results)

---

## MemoryBudget System

### What It Is

`MemoryBudget` is a **global singleton** that enforces a hard cap on RAM usage as a configurable percentage of total system RAM. **NO data structure in TinyTorch will ever load everything into memory** when the allocation would exceed this threshold.

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    MemoryBudget                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  total_ram_bytes    = get_total_system_memory()       │  │
│  │  max_ram_percentage = 30.0  (default, configurable)  │  │
│  │  threshold_bytes    = total_ram * (pct / 100)         │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  MemoryBudget::should_stream(bytes) → true/false            │
│       │                          │                          │
│       │ true                     │ false                    │
│       ▼                          ▼                          │
│  Use MmapTensor              Use DenseTensor                │
│  (mmap-backed,               (in-RAM, fast)                 │
│   batched access)                                            │
└─────────────────────────────────────────────────────────────┘
```

### Default Behavior

- **Default threshold: 30% of total system RAM**
- Any **single allocation** ≥ 30% of RAM automatically triggers mmap fallback
- On a machine with 16 GB RAM: threshold = 4.8 GB
- On a machine with 8 GB RAM: threshold = 2.4 GB

### Configuration

```cpp
#include "auto_tensor.h"  // MemoryBudget lives here

// Set max RAM percentage (any allocation >= this % uses mmap)
MemoryBudget::instance().set_max_ram_percentage(70.0);  // allow up to 70%
MemoryBudget::instance().set_max_ram_percentage(10.0);  // very conservative

// Check current threshold
std::cout << MemoryBudget::instance().summary() << std::endl;
// Output: MemoryBudget: threshold=2457.6 MB (30.0% of 8.0 GB RAM)

// Query if an allocation should use mmap
std::size_t bytes = num_elements * sizeof(float);
if (MemoryBudget::should_stream(bytes)) {
    // This allocation is too large — will use MmapTensor
}

// Refresh after RAM changes (e.g., hot-plug memory on servers)
MemoryBudget::instance().refresh_total_ram();
```

### Enforcement Points

The `MemoryBudget::should_stream(bytes)` check is wired into **every allocation site** in `distributed_tensor.h`:

| File | Location | What It Guards |
|------|----------|---------------|
| `distributed_tensor.h` | `create_tensor_on_device()` | Shard data on CPU_DENSE |
| `distributed_tensor.h` | `Shard::zero_grad()` | Gradient tensor creation |
| `distributed_tensor.h` | `Shard::accumulate_grad()` | Gradient tensor creation |
| `distributed_tensor.h` | `parallel_binary_op` backward | Grad input/output tensors |
| `distributed_tensor.h` | `parallel_unary_op` backward | Grad input tensors |
| `distributed_tensor.h` | `sum()` backward | Per-shard gradient tensors |
| `distributed_tensor.h` | `mean()` backward | Per-shard gradient tensors |
| `distributed_tensor.h` | `backward()` scalar init | All shard gradient tensors |
| `distributed_tensor.h` | `all_gather()` | **Throws exception** if result exceeds budget |
| `distributed_tensor.h` | `refresh_devices()` | Uses mmap intermediate (never `std::vector`) |

### Memory-Backed vs RAM-Backed Decision Flow

```
                    Request: allocate N elements
                              │
                              ▼
                    bytes = N * sizeof(T)
                              │
                              ▼
              MemoryBudget::should_stream(bytes)?
                    /                  \
                  YES                   NO
                   │                     │
                   ▼                     ▼
          Is backend CPU_MMAP     Use DenseTensor<T>
          or OPENCL?              (in-RAM, fastest)
           /        \
         YES         NO
          │           │
          ▼           ▼
    MmapTensor    DenseTensor<T>
    (mmap file)
```

---

## Bug Fixes

### Fix 1: `randn()` / `rand()` Data Race Hang

**Symptom:** Creating a large tensor (e.g., 1000×1000 = 1M elements) with `randn()` would hang indefinitely.

**Root Cause:** `std::mt19937` RNG and `std::normal_distribution` were captured by reference (`&gen`, `&dist`) and shared across multiple `std::async` threads in `distribute_data()`. `std::mt19937` is **not thread-safe** — concurrent access caused undefined behavior and deadlock.

**Before (broken):**
```cpp
std::mt19937 gen(42);                          // single RNG
std::normal_distribution<T> dist(0, 1);        // single distribution
tensor->distribute_data([&dist, &gen](std::size_t) mutable {
    return dist(gen);  // DATA RACE: multiple threads access gen/dist
});
```

**After (fixed):**
```cpp
tensor->distribute_data([](std::size_t global_idx) {
    // Each element gets its own RNG seeded from global index
    // No shared state across threads — completely thread-safe
    std::mt19937 gen(static_cast<unsigned int>(42 + (global_idx % 1000000)));
    std::normal_distribution<T> dist(0, 1);
    return dist(gen);
});
```

**Key principle:** Each async task gets its own independent RNG seeded deterministically from the element's global index.

---

### Fix 2: Backward Pass Only Used CPU Dense for Gradients

**Symptom:** After `backward()`, gradient tensors on non-CPU-Dense shards (CUDA, OpenGL, OpenCL, MMAP) were still created as `DenseTensor<T>`, ignoring the shard's backend.

**Root Cause:** All gradient tensor creation sites hardcoded `std::make_unique<DenseTensor<T>>(...)` without checking the shard's backend type.

**Before (broken):**
```cpp
// Always created DenseTensor regardless of shard backend
auto grad_input = std::make_unique<DenseTensor<T>>(shape_arr, 1, grad_val);
```

**After (fixed):**
```cpp
std::size_t bytes = shard.num_elements * sizeof(T);
if (MemoryBudget::should_stream(bytes) ||
    shard.device.backend == BackendType::CPU_MMAP ||
    shard.device.backend == BackendType::OPENCL) {
    StreamConfig sc;
    sc.batch_size = std::max((std::size_t)65536, shard.num_elements / 10);
    grad_input = std::make_unique<MmapTensor<T>>(shape_arr, 1, sc, grad_val);
} else {
    grad_input = std::make_unique<DenseTensor<T>>(shape_arr, 1, grad_val);
}
```

**Applied to:** `Shard::zero_grad()`, `Shard::accumulate_grad()`, `parallel_binary_op` backward, `parallel_unary_op` backward, `sum()` backward, `mean()` backward, `backward()` scalar init.

---

### Fix 3: `all_gather()` Loaded Entire Tensor into RAM

**Symptom:** Calling `all_gather()` on a large tensor (e.g., 1000×1000) allocated a `std::vector<T>` of the full size, defeating the entire purpose of distributed tensors.

**Before (broken):**
```cpp
std::vector<T> all_gather() const {
    std::vector<T> result(_total_elements);  // ALLOCATES ALL DATA IN RAM!
    for (const auto& shard : _shards) {
        for (size_t i = 0; i < shard.num_elements; ++i) {
            result[shard.global_offset + i] = shard.data->get_element(i);
        }
    }
    return result;
}
```

**After (fixed):**
```cpp
// Old all_gather() now guarded by MemoryBudget
std::vector<T> all_gather() const {
    std::size_t bytes = _total_elements * sizeof(T);
    if (MemoryBudget::should_stream(bytes)) {
        throw std::runtime_error(
            "all_gather() would exceed MemoryBudget. "
            "Use all_gather_mmap() or all_gather_distributed() instead.");
    }
    std::vector<T> result;
    result.reserve(_total_elements);
    // ... element-by-element copy
    return result;
}

// NEW: safe mmap-backed alternative
std::unique_ptr<StreamTensor<T>> all_gather_mmap() const {
    // Creates mmap-backed StreamTensor — NEVER loads all data into RAM
    auto result = std::make_unique<StreamTensor<T>>(shape_arr, 1, sc);
    for (const auto& shard : _shards) {
        for (size_t i = 0; i < shard.num_elements; ++i) {
            result->set_element(shard.global_offset + i,
                                shard.data->get_element(i));
        }
    }
    return result;
}
```

---

### Fix 4: `refresh_devices()` Loaded Full Data into `std::vector`

**Symptom:** `refresh_devices()` saved all tensor data into a `std::vector<T>` before reinitializing shards.

**Before (broken):**
```cpp
std::vector<T> current_data(_total_elements);  // FULL RAM COPY
for (size_t i = 0; i < _total_elements; ++i) {
    current_data[i] = get_element(i);
}
// ... reinitialize shards ...
distribute_data([&current_data](std::size_t idx) {
    return current_data[idx];
});
```

**After (fixed):**
```cpp
// Use mmap-backed StreamTensor as intermediate — never all in RAM
auto mmap_backup = std::make_unique<StreamTensor<T>>(shape_arr, 1, sc);
for (const auto& shard : _shards) {
    for (size_t i = 0; i < shard.num_elements; ++i) {
        mmap_backup->set_element(shard.global_offset + i,
                                 shard.data->get_element(i));
    }
}
// ... reinitialize shards ...
for (auto& shard : _shards) {
    for (size_t i = 0; i < shard.num_elements; ++i) {
        shard.set(i, mmap_backup->get_element(shard.global_offset + i));
    }
}
```

---

## New Methods

### `all_gather_mmap()`

Returns a memory-mapped `StreamTensor<T>` containing all shard data. **Never loads everything into RAM.**

```cpp
auto x = DistributedTensor<float>::randn({10000, 10000});  // 100M elements
auto mmap_result = x->all_gather_mmap();                    // mmap-backed
// Access elements one at a time, or in batches:
float val = mmap_result->get_element(12345);
```

### `all_gather_distributed()`

Returns a new `DistributedTensor<T>` with all data resharded across all devices. Data flows through mmap, not RAM.

```cpp
auto gathered = x->all_gather_distributed();
// Result is sharded across CPU Dense, CPU Mmap, CUDA, OpenGL, OpenCL — same as original
```

### `MemoryBudget::set_max_ram_percentage(double)`

Configure the RAM threshold at which allocations switch to mmap.

```cpp
MemoryBudget::instance().set_max_ram_percentage(70.0);  // more aggressive
MemoryBudget::instance().set_max_ram_percentage(10.0);  // more conservative
```

### `MemoryBudget::summary()`

Get a human-readable string showing current memory budget state.

```cpp
std::cout << MemoryBudget::instance().summary() << std::endl;
// "MemoryBudget: threshold=2457.6 MB (30.0% of 8.0 GB RAM)"
```

---

## Compile Commands

### Windows (TDM-GCC 64-bit)

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
```

### Linux/macOS

```bash
# Distributed tensor test
g++ -std=c++17 -Wall -Wextra \
    -DTINYTORCH_USE_OPENCL_SDK \
    -I"/path/to/OpenCL-SDK/install/include" \
    -o distributed_tensor_test distributed_tensor_test.cpp \
    -L"/path/to/OpenCL-SDK/install/lib" -lOpenCL \
    -lGL -lpthread

# LLM operations test
g++ -std=c++17 -Wall -Wextra \
    -DTINYTORCH_USE_OPENCL_SDK \
    -I"/path/to/OpenCL-SDK/install/include" \
    -o llm_ops_test llm_ops_test.cpp \
    -L"/path/to/OpenCL-SDK/install/lib" -lOpenCL \
    -lGL -lpthread
```

---

## Test Results

### Distributed Tensor Test Suite (6/6 passed)

```
================================================
  TinyTorch Distributed Tensor Test Suite
================================================

=== Testing Device Pool ===
Device Pool: 8 devices
  [OK] CPU Dense (CPU_DENSE)
  [OK] CPU Mmap (Lazy) (CPU_MMAP)
  [OK] CUDA: NVIDIA GeForce MX130 (CUDA)
  [OK] OpenGL: NVIDIA GeForce MX130/PCIe/SSE2 (OPENGL)
  [OK] OpenCL: NVIDIA GeForce MX130 (OPENCL)
  [OK] OpenCL: Intel(R) UHD Graphics 620 (OPENCL)
  [OK] OpenCL: Intel(R) UHD Graphics 620 (OPENCL)
  [OK] OpenCL: NVIDIA GeForce MX130 (OPENCL)
✓ Device pool tests passed

=== Testing DistributedTensor Creation ===
Zeros: shape=[100, 100], shards=8
Ones: shape=[50, 50], shards=8
Random: shape=[200, 200], shards=8
With grad: shape=[100, 100], shards=8, requires_grad=true
✓ DistributedTensor creation tests passed

=== Testing DistributedTensor Operations ===
Element access works
Add: shape=[100, 100], shards=8
Multiply: shape=[100, 100], shards=8
Scalar ops work
Sum: 3
Mean: 0.0003
Reshape: shape=[50, 200], shards=8
✓ DistributedTensor operation tests passed

=== Testing DistributedTensor Autograd ===
Forward pass: shape=[1], shards=1
Backward pass completed
Gradients zeroed
✓ DistributedTensor autograd tests passed

=== Testing No Full Data Gather ===
Created tensor with 8 shards (1M elements across 8 backends)
  Shard 0: 125000 elements on CPU Dense
  Shard 1: 125000 elements on CPU Mmap (Lazy)
  Shard 2: 125000 elements on CUDA: NVIDIA GeForce MX130
  Shard 3: 125000 elements on OpenGL: NVIDIA GeForce MX130
  Shard 4: 125000 elements on OpenCL: NVIDIA GeForce MX130
  Shard 5: 125000 elements on OpenCL: Intel UHD Graphics 620
  Shard 6: 125000 elements on OpenCL: Intel UHD Graphics 620
  Shard 7: 125000 elements on OpenCL: NVIDIA GeForce MX130
After multiply_scalar: 8 shards
After add: 8 shards
After sum: 1 shard (scalar result)
✓ No full data gather tests passed

=== Testing DistributedTensor Device Refresh ===
Before refresh: shards=8, devices=[...]
After refresh: shards=8, devices=[...]
✓ DistributedTensor device refresh tests passed

================================================
  All Distributed Tensor Tests Passed! ✓
================================================
```

### LLM Operations Test Suite (7/7 passed)

```
================================================
  TinyTorch LLM Operations Test Suite
================================================

=== Testing Distributed Matmul ===
A shape: [2, 3], B shape: [4, 3]
Result shape: [2, 4]
Result: [4, 2, 4, 2, 10, 5, 10, 5]
✓ Matmul works

=== Testing Distributed Softmax ===
Softmax result: [0.0450153, 0.122364, 0.33262, 0.0450153, 0.122364, 0.33262]
Total sum: 1
✓ Softmax works

=== Testing Distributed Layer Norm ===
Layer norm result: [-1.46385, -0.878309, -0.29277, 0.29277, 0.878309, 1.46385]
Global mean: 0
Global std: 0.999998
✓ Layer norm works

=== Testing Distributed Activations ===
GELU: [-0.158808, 0, 0.841192, 1.9546]
SiLU: [-0.268941, 0, 0.731059, 1.76159]
✓ Activations work

=== Testing Distributed Cross Entropy Loss ===
Cross entropy loss: 1.88711
✓ Cross entropy works

=== Testing Distributed Optimizer Steps ===
Loss value: 10
Backward completed
Params before SGD: [1, 2, 3, 4]
Params after SGD: [0.99, 1.99, 2.99, 3.99]
✓ Optimizer steps work

=== Testing Distributed All-Reduce ===
All-reduce result: [10, 10, 10, 10]
Expected sum per element: 10
✓ All-reduce works

================================================
  All LLM Operations Tests Passed!
================================================
```

---

## Files Modified

| File | Changes |
|------|---------|
| `auto_tensor.h` | Added `MemoryBudget` struct (lines 89-169). Added `#include <atomic>`, `<mutex>`, `<sstream>` |
| `distributed_tensor.h` | `randn()`/`rand()`: thread-safe RNG (per-element seeding). `create_tensor_on_device()`: MemoryBudget guard. `Shard::zero_grad()`: MemoryBudget + backend-aware. `Shard::accumulate_grad()`: MemoryBudget + backend-aware. `parallel_binary_op` backward: MemoryBudget-aware grad tensors. `parallel_unary_op` backward: MemoryBudget-aware grad tensors. `sum()` backward: MemoryBudget-aware grad tensors. `mean()` backward: MemoryBudget-aware grad tensors. `backward()`: MemoryBudget-aware scalar grad init. `refresh_devices()`: mmap intermediate instead of `std::vector`. `all_gather()`: MemoryBudget guard with exception. `all_gather_mmap()`: NEW — safe mmap-backed gather. `all_gather_distributed()`: NEW — resharded distributed gather via mmap. |
