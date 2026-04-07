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
