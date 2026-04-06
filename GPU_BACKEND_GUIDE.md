# TinyTorch GPU Backend Guide

## Overview

TinyTorch supports **three GPU backends** for hardware-accelerated tensor operations:

| Backend | GPU Support | Performance | Setup Complexity |
|---------|-------------|-------------|------------------|
| **OpenCL** | All GPUs (NVIDIA, Intel, AMD) | Good | Medium |
| **CUDA** | NVIDIA only | Best | High |
| **OpenGL** | All GPUs (one at a time) | Moderate | Low |

The system automatically detects and selects the best available GPU, but you can override this behavior.

---

## Quick Start

### 1. Install OpenCL Runtime (Recommended)

OpenCL provides the broadest GPU support and is the **recommended backend** for most users.

#### For NVIDIA GPUs:
- Install the latest [NVIDIA Driver](https://www.nvidia.com/Download/index.aspx)
- OpenCL is included with the driver (no separate installation needed)
- Verify: Run `clinfo` or the test suite

#### For Intel GPUs:
- Install [Intel Graphics Driver](https://www.intel.com/content/www/us/en/download-center/home.html)
- For better performance, install [Intel OpenCL Runtime](https://www.intel.com/content/www/us/en/developer/articles/tool/opencl-drivers.html)
- **Intel HD/UHD/Iris Xe**: Supported via Intel OpenCL HD Graphics driver
- **Intel Arc**: Supported via Intel OpenCL runtime

#### For AMD GPUs:
- Install [AMD Adrenalin Driver](https://www.amd.com/en/support)
- OpenCL is included with the driver

### 2. Install CUDA (Optional, NVIDIA Only)

CUDA provides the best performance for NVIDIA GPUs but requires the CUDA Toolkit.

1. Download [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
2. Install with default options
3. Add to PATH: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\bin`
4. Verify: `nvcc --version`

### 3. Build OpenCL SDK (Optional, for Development)

If you want to use the official Khronos OpenCL SDK headers:

```bash
git clone --recursive https://github.com/KhronosGroup/OpenCL-SDK.git
cd OpenCL-SDK
cmake -G "Visual Studio 17 2022" -A x64 -T v143 -D CMAKE_INSTALL_PREFIX=./OpenCL-SDK/install -B ./OpenCL-SDK/build -S ./OpenCL-SDK
cmake --build OpenCL-SDK/build --config Release --target install -- /m /v:minimal
```

Compile with:
```bash
g++ -std=c++17 -DTINYTORCH_USE_OPENCL_SDK \
    -I"C:/Users/YOUR_NAME/OpenCL-SDK/install/include" \
    -L"C:/Users/YOUR_NAME/OpenCL-SDK/install/lib" -lOpenCL \
    your_code.cpp -lopengl32 -lgdi32
```

---

## GPU Selection Behavior

### Auto-Selection Logic

The `UnifiedGpuSelector::auto_select()` method uses this priority:

1. **NVIDIA discrete GPUs** (via OpenCL native driver)
2. **AMD discrete GPUs**
3. **Highest scoring GPU** (based on compute units, memory, driver type)

### Scoring System

Each GPU receives a score based on:
- **Compute units**: `units * 10.0`
- **Memory**: `GB * 5.0`
- **Native NVIDIA driver**: `+100.0`
- **Native Intel driver**: `+50.0`
- **Discrete GPU (NVIDIA/AMD)**: `+50.0`
- **Microsoft OpenCLOn12 wrapper**: `-20.0`

**Example scores for a typical laptop:**
```
#0: Intel UHD 620 (Intel driver)     → 321.7 points
#1: NVIDIA MX130 (NVIDIA CUDA)       → 189.9 points ← Selected (discrete)
#2: Intel UHD 620 (Microsoft D3D12)  →  29.6 points
#3: NVIDIA MX130 (Microsoft D3D12)   →  -0.2 points
```

---

## Making Specific GPUs Appear

### Problem: Only Intel GPU Shows Up

**Cause**: Windows defaults to the integrated GPU for OpenGL applications.

**Solution**: The `NvOptimusEnablement` export forces NVIDIA GPU usage for OpenGL:

```cpp
// In gpu_backends.h (already included)
extern "C" {
    __declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
    __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}
```

This is **already enabled by default** in TinyTorch.

### Problem: Only One GPU Detected

**Cause**: OpenGL can only expose one GPU at a time. OpenCL can see all GPUs.

**Solution**: Use OpenCL backend instead of OpenGL. OpenCL enumerates all GPUs simultaneously.

### Problem: Microsoft OpenCLOn12 Instead of Native Driver

**Cause**: Windows 10/11 includes a D3D12-to-OpenCL translation layer that appears as "OpenCLOn12".

**Solution**: Install the native driver for your GPU:
- **NVIDIA**: Install latest Game Ready driver from nvidia.com
- **Intel**: Install Intel OpenCL HD Graphics driver
- **AMD**: Install Adrenalin driver

The scoring system automatically penalizes OpenCLOn12 devices (`-20.0` points).

---

## Switching Backends in Code

### Method 1: Compile-Time Flag

```bash
# Use OpenCL SDK headers
g++ -DTINYTORCH_USE_OPENCL_SDK -I"path/to/OpenCL-SDK/install/include" ...

# Use dynamic loading (default, no flag needed)
g++ your_code.cpp -lopengl32 -lgdi32
```

### Method 2: Runtime Selection

```cpp
#include "gpu_backends.h"

auto& selector = get_unified_gpu_selector();
selector.enumerate();

// Auto-select (prefers discrete GPUs)
selector.auto_select();

// Select by name
selector.select_by_name("MX130");
selector.select_by_name("NVIDIA");
selector.select_by_name("Intel");

// Select discrete GPU only (skip Intel iGPU)
selector.select_discrete_only();

// Select specific device by index
selector.select_device(0);  // First GPU in list

// Get selected GPU info
const auto* gpu = selector.selected_gpu();
std::cout << "Using: " << gpu->name << " (" << gpu->vendor << ")\n";
std::cout << "Backend: " << (gpu->backend == GpuBackend::OpenCL ? "OpenCL" : 
                             gpu->backend == GpuBackend::CUDA ? "CUDA" : "OpenGL") << "\n";
```

### Method 3: Force Specific Backend

```cpp
// Force OpenCL
auto& cl_mgr = get_opencl_manager();
cl_mgr.initialize();
cl_mgr.select_by_name("NVIDIA");  // Select NVIDIA GPU via OpenCL

// Force CUDA (if available)
auto& cuda_mgr = get_cuda_manager();
cuda_mgr.initialize();
cuda_mgr.auto_select();

// Force OpenGL (only sees one GPU)
auto& gl_sel = get_gpu_selector();
gl_sel.enumerate();
gl_sel.auto_select();
```

---

## Code Locations

| File | Purpose |
|------|---------|
| `gpu_backends.h` | Main GPU backend abstraction layer |
| `gpu_tensor.h` | OpenGL compute shader implementation |
| `tensor.h` | Base CPU tensor implementation |
| `stream_tensor.h` | Memory-mapped tensor for large datasets |
| `auto_tensor.h` | Automatic CPU/GPU selection |

### Key Functions to Modify

| Function | Location | What it does |
|----------|----------|--------------|
| `UnifiedGpuSelector::auto_select()` | `gpu_backends.h:665` | GPU selection priority |
| `OpenClManager::initialize()` | `gpu_backends.h:335` | OpenCL device enumeration |
| `GpuContext::initialize()` | `gpu_tensor.h:920` | OpenGL context creation |
| `CudaManager::initialize()` | `gpu_backends.h:105` | CUDA runtime loading |

### Changing GPU Selection Priority

Edit `UnifiedGpuSelector::auto_select()` in `gpu_backends.h`:

```cpp
void auto_select() {
    if (_gpus.empty()) { _selected_index = -1; return; }
    
    // Current: Prefer NVIDIA discrete GPUs
    for (int i = 0; i < static_cast<int>(_gpus.size()); ++i) {
        if (!_gpus[i].is_intel && _gpus[i].is_nvidia) {
            _selected_index = i;
            return;
        }
    }
    
    // Add: Prefer AMD discrete GPUs
    for (int i = 0; i < static_cast<int>(_gpus.size()); ++i) {
        if (!_gpus[i].is_intel && _gpus[i].is_amd) {
            _selected_index = i;
            return;
        }
    }
    
    // Add: Prefer Intel (uncomment to use Intel GPU by default)
    // for (int i = 0; i < static_cast<int>(_gpus.size()); ++i) {
    //     if (_gpus[i].is_intel) {
    //         _selected_index = i;
    //         return;
    //     }
    // }
    
    // Fall back to highest score
    _selected_index = _gpus[0].device_id;
}
```

### Changing Scoring Weights

Edit the scoring section in `OpenClManager::initialize()`:

```cpp
// Current scoring weights
info.compute_score = info.compute_units * 10.0 + 
                     (info.global_mem_size / (1024.0 * 1024.0 * 1024.0)) * 5.0;

// Prefer native drivers over Microsoft's OpenCLOn12 wrapper
if (platform_lower.find("nvidia cuda") != std::string::npos) {
    info.compute_score += 100.0;  // Native NVIDIA driver
} else if (platform_lower.find("intel") != std::string::npos && 
           platform_lower.find("openclon12") == std::string::npos) {
    info.compute_score += 50.0;   // Native Intel driver
} else if (platform_lower.find("openclon12") != std::string::npos) {
    info.compute_score -= 20.0;   // Microsoft D3D12 wrapper (slower)
}

// Prefer discrete GPUs
if (vendor_lower.find("nvidia") != std::string::npos) {
    info.compute_score += 50.0;
    info.is_nvidia = true;
}
```

---

## Troubleshooting

### "No GPUs detected across any backend"

1. **Check drivers**: Ensure GPU drivers are installed and up to date
2. **OpenCL runtime**: Install OpenCL runtime for your GPU vendor
3. **Verify with clinfo**: Run `clinfo` to see if OpenCL detects your GPU

### "CUDA runtime not found"

1. Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
2. Add to PATH: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\bin`
3. Or copy `cudart64_*.dll` to your executable directory

### "OpenCL runtime not found"

1. Install GPU vendor's OpenCL runtime:
   - **NVIDIA**: Latest Game Ready driver
   - **Intel**: [Intel OpenCL Runtime](https://www.intel.com/content/www/us/en/developer/articles/tool/opencl-drivers.html)
   - **AMD**: Adrenalin driver
2. Or copy `OpenCL.dll` to your executable directory

### Wrong GPU Selected

```cpp
// Force specific GPU by name
selector.select_by_name("MX130");

// Or force discrete GPU only
selector.select_discrete_only();

// Or select by index (check output of selector.print_all())
selector.select_device(1);  // Second GPU in list
```

### Performance Issues with OpenCLOn12

Microsoft's OpenCLOn12 is a D3D12-to-OpenCL translation layer that's significantly slower than native drivers.

**Fix**: Install native GPU driver:
- **NVIDIA**: https://www.nvidia.com/Download/index.aspx
- **Intel**: https://www.intel.com/content/www/us/en/download-center/home.html
- **AMD**: https://www.amd.com/en/support

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    UnifiedGpuSelector                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ OpenCL      │  │ CUDA        │  │ OpenGL              │  │
│  │ Manager     │  │ Manager     │  │ Selector            │  │
│  │             │  │             │  │                     │  │
│  │ • All GPUs  │  │ • NVIDIA    │  │ • One GPU           │  │
│  │ • Native    │  │ • Best perf │  │ • Compute shaders   │  │
│  │ • Fallback  │  │ • Optional  │  │ • NvOptimus export  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    Auto-selection logic:
                    1. NVIDIA discrete
                    2. AMD discrete
                    3. Highest score
```

---

## Quick Reference

| Task | Command/Code |
|------|--------------|
| List all GPUs | `selector.print_all(std::cout);` |
| Auto-select best | `selector.auto_select();` |
| Force NVIDIA | `selector.select_by_name("NVIDIA");` |
| Force discrete only | `selector.select_discrete_only();` |
| Check backend | `gpu->backend == GpuBackend::OpenCL` |
| Compile with SDK | `-DTINYTORCH_USE_OPENCL_SDK -I"path/to/include"` |
| Run test suite | `.\gpu_backends_test.exe` |
