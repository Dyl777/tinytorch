# TinyTorch GPU Backend - Technical Guide

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


## Architecture Support Matrix

| GPU Architecture | Compute Capability | OpenGL 4.3+ | OpenCL 1.2+ | CUDA | Notes |
|-----------------|-------------------|-------------|-------------|------|-------|
| **NVIDIA Maxwell** (GTX 750-980, MX130) | 5.0-5.2 | Yes | Yes | Yes | See Maxwell-specific issues below |
| **NVIDIA Pascal** (GTX 10xx) | 6.0-6.1 | Yes | Yes | Yes | Full support |
| **NVIDIA Turing** (RTX 20xx, MX450) | 7.5 | Yes | Yes | Yes | Full support |
| **NVIDIA Ampere** (RTX 30xx) | 8.0-8.6 | Yes | Yes | Yes | Full support |
| **Intel HD 500-600** | N/A | Yes | Yes | No | Limited OpenCL performance |
| **Intel UHD 600-700** | N/A | Yes | Yes | No | Better OpenCL via Intel driver |
| **Intel Iris Xe** | N/A | Yes | Yes | No | Good OpenCL performance |
| **AMD GCN 1.0-3.0** | N/A | Yes | Yes | No | OpenCL via AMD driver |
| **AMD RDNA 1-3** | N/A | Yes | Yes | No | Best OpenCL support |

---

## Maxwell Architecture (Compute 5.0) Specific Issues

### OpenGL Compute Shader Limitations

The NVIDIA GeForce MX130 uses the **Maxwell GM108** chip (Compute Capability 5.0). While it supports OpenGL 4.6, there are several quirks:

#### 1. SSBO Size Limitations
- **Max SSBO size**: ~1.9GB (limited by 2GB VRAM)
- **Max work group size**: 1024 threads (not 1536 like Pascal+)
- **Shared memory**: 48KB per block (less than Pascal's 64KB)

**Workaround**: Use smaller batch sizes for large tensors:
```cpp
// In gpu_kernels.h
size_t global = ((n + 255) / 256) * 256;  // Use 256, not 512
size_t local = 256;  // Maxwell max is 1024, but 256 is safer
```

#### 2. Atomic Operations
- `atomic_add` for floats is **not supported** in OpenCL 1.2 on Maxwell
- `atomic_cmpxchg` works but is slow

**Workaround**: Use multi-pass reduction instead of atomics:
```c
// Instead of atomic_add(out, value):
// Pass 1: Each work group writes partial sum to buffer
// Pass 2: CPU sums the partial results
```

#### 3. OpenCL Driver Issues
- NVIDIA's OpenCL driver for Maxwell reports **CL_DEVICE_MAX_COMPUTE_UNITS = 3** (actual SM count)
- Intel's OpenCL driver reports **24 CUs** for UHD 620 (EU count)
- **Don't compare CU counts across vendors** - they mean different things

#### 4. NvOptimusEnablement
On laptops with dual GPUs (Intel + NVIDIA), OpenGL defaults to Intel. The export:
```cpp
extern "C" {
    __declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
}
```
**Must be in the main executable**, not a DLL. If placed in a shared library, it won't work.

#### 5. Memory Alignment
Maxwell requires **16-byte alignment** for SSBOs. Unaligned access causes silent corruption.

**Fix**: Always allocate buffers with padding:
```cpp
size_t aligned_size = ((n * sizeof(float) + 15) / 16) * 16;
cl_mem buf = clCreateBuffer(context, CL_MEM_READ_WRITE, aligned_size, nullptr, &err);
```

---

## Known Issues and Workarounds

### Issue 1: `clCreateCommandQueue` Deprecated in OpenCL 2.0+

**Error**: `warning: '_cl_command_queue* clCreateCommandQueue(...)' is deprecated`

**Fix**: Use `clCreateCommandQueueWithProperties` for OpenCL 2.0+:
```cpp
#ifdef CL_VERSION_2_0
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
#else
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
#endif
```

### Issue 2: CUDA DLL Not Found

**Error**: `[CudaManager] CUDA runtime not found`

**Cause**: Code was looking for `cudart64_110.dll` but CUDA 12.8 uses `cudart64_12.dll`.

**Fix**: Updated DLL search order in `gpu_backends.h`:
```cpp
_cuda_lib = LoadLibraryA("cudart64_12.dll");  // CUDA 12.x
if (!_cuda_lib) _cuda_lib = LoadLibraryA("cudart64_11.dll");  // CUDA 11.x
if (!_cuda_lib) _cuda_lib = LoadLibraryA("cudart64_10.dll");  // CUDA 10.x
// ... etc
```

### Issue 3: Microsoft OpenCLOn12 Performance

**Symptom**: Intel/AMD GPUs via Microsoft's D3D12 translation layer are **10-50x slower** than native drivers.

**Cause**: OpenCLOn12 translates OpenCL calls to Direct3D 12, adding overhead.

**Fix**: Install native drivers:
- **Intel**: [Intel OpenCL HD Graphics Driver](https://www.intel.com/content/www/us/en/download/785597/intel-opencl-graphics-driver.html)
- **AMD**: [AMD Adrenalin](https://www.amd.com/en/support)

The scoring system automatically penalizes OpenCLOn12:
```cpp
if (platform_lower.find("openclon12") != std::string::npos) {
    info.compute_score -= 20.0;  // Penalty
}
```

### Issue 4: `std::vector` Size Limit on 32-bit

**Error**: `terminate called after throwing an instance of 'std::length_error'`
**what()**: `cannot create std::vector larger than max_size()`

**Cause**: On 32-bit systems, `std::vector` is limited to ~2GB.

**Fix**: Use streaming/chunked processing for large tensors:
```cpp
// Process in chunks of 100M elements
size_t chunk_size = 100000000;
for (size_t offset = 0; offset < n; offset += chunk_size) {
    size_t count = std::min(chunk_size, n - offset);
    // Process chunk...
}
```

### Issue 5: OpenGL Function Pointer Loading on Windows

**Warning**: `cast between incompatible function types from 'PROC' to 'PFNGLGENBUFFERSPROC'`

**Cause**: `wglGetProcAddress` returns `PROC` (generic function pointer), but we cast to specific types.

**Status**: **Harmless warning**. This is standard practice for OpenGL extension loading. The cast is safe because all OpenGL function pointers have the same calling convention on Windows.

**Suppress**: Add `-Wno-cast-function-type` to compiler flags.

---

## Installation Scripts

### Linux: Auto-Detect and Install GPU Drivers

```bash
#!/bin/bash
# gpu_setup_linux.sh - Detect and install GPU drivers on Linux

set -e

echo "=== TinyTorch GPU Setup (Linux) ==="

# Detect GPU
GPU_VENDOR=$(lspci | grep -i -E 'vga|3d|display' | grep -oiE 'nvidia|amd|intel' | head -1 | tr '[:upper:]' '[:lower:]')

if [ -z "$GPU_VENDOR" ]; then
    echo "ERROR: No GPU detected"
    exit 1
fi

echo "Detected GPU vendor: $GPU_VENDOR"

case $GPU_VENDOR in
    nvidia)
        echo "Installing NVIDIA drivers..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y nvidia-driver-535 nvidia-cuda-toolkit opencl-headers
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y akmod-nvidia xorg-x11-drv-nvidia-cuda
        elif command -v pacman &> /dev/null; then
            sudo pacman -S nvidia nvidia-utils cuda opencl-headers
        fi
        
        # Verify
        nvidia-smi
        nvcc --version
        ;;
    amd)
        echo "Installing AMD drivers..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y mesa-opencl-icd ocl-icd-opencl-dev opencl-headers
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y ocl-icd ocl-icd-devel
        fi
        
        # Verify
        clinfo | head -20
        ;;
    intel)
        echo "Installing Intel OpenCL runtime..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y intel-opencl-icd ocl-icd-opencl-dev opencl-headers
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y intel-opencl
        fi
        
        # Verify
        clinfo | head -20
        ;;
esac

# Install build dependencies
echo "Installing build dependencies..."
if command -v apt-get &> /dev/null; then
    sudo apt-get install -y build-essential cmake git libgl1-mesa-dev
elif command -v dnf &> /dev/null; then
    sudo dnf groupinstall -y "Development Tools"
    sudo dnf install -y cmake git mesa-libGL-devel
fi

echo "=== Setup Complete ==="
echo "Run: cd tinytorch && ./build.sh"
```

### Windows: Auto-Detect and Install GPU Drivers (PowerShell)

```powershell
# gpu_setup_windows.ps1 - Detect and guide GPU driver installation on Windows

Write-Host "=== TinyTorch GPU Setup (Windows) ===" -ForegroundColor Cyan

# Detect GPU
$gpu = Get-PnpDevice -Class Display | Where-Object { $_.Status -eq 'OK' } | Select-Object -First 1

if (-not $gpu) {
    Write-Host "ERROR: No GPU detected" -ForegroundColor Red
    exit 1
}

$gpuName = $gpu.FriendlyName
Write-Host "Detected GPU: $gpuName" -ForegroundColor Green

# Check for NVIDIA
if ($gpuName -match 'NVIDIA|GeForce|Quadro|Tesla') {
    Write-Host "NVIDIA GPU detected" -ForegroundColor Yellow
    
    # Check if CUDA is installed
    $cudaPath = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -ErrorAction SilentlyContinue
    if ($cudaPath) {
        Write-Host "CUDA found: $($cudaPath.Name)" -ForegroundColor Green
        nvcc --version
    } else {
        Write-Host "CUDA not found. Download from:" -ForegroundColor Yellow
        Write-Host "  https://developer.nvidia.com/cuda-downloads" -ForegroundColor White
    }
    
    # Check OpenCL
    $openclDll = Get-ChildItem "C:\Windows\System32\nvcuda.dll" -ErrorAction SilentlyContinue
    if ($openclDll) {
        Write-Host "OpenCL (NVIDIA) found" -ForegroundColor Green
    } else {
        Write-Host "OpenCL not found. Update NVIDIA driver:" -ForegroundColor Yellow
        Write-Host "  https://www.nvidia.com/Download/index.aspx" -ForegroundColor White
    }
}
# Check for Intel
elseif ($gpuName -match 'Intel|UHD|Iris|HD Graphics') {
    Write-Host "Intel GPU detected" -ForegroundColor Yellow
    
    # Check OpenCL
    $intelOpenCL = Get-ChildItem "C:\Windows\System32\IntelOpenCL64.dll" -ErrorAction SilentlyContinue
    if ($intelOpenCL) {
        Write-Host "Intel OpenCL found" -ForegroundColor Green
    } else {
        Write-Host "Intel OpenCL not found. Install from:" -ForegroundColor Yellow
        Write-Host "  https://www.intel.com/content/www/us/en/download/785597/intel-opencl-graphics-driver.html" -ForegroundColor White
    }
}
# Check for AMD
elseif ($gpuName -match 'AMD|Radeon') {
    Write-Host "AMD GPU detected" -ForegroundColor Yellow
    
    # Check OpenCL
    $amdOpenCL = Get-ChildItem "C:\Windows\System32\amdocl64.dll" -ErrorAction SilentlyContinue
    if ($amdOpenCL) {
        Write-Host "AMD OpenCL found" -ForegroundColor Green
    } else {
        Write-Host "AMD OpenCL not found. Install Adrenalin driver:" -ForegroundColor Yellow
        Write-Host "  https://www.amd.com/en/support" -ForegroundColor White
    }
}

# Check for OpenCL SDK
$openclSdk = Test-Path "$env:USERPROFILE\OpenCL-SDK\install\bin\OpenCL.dll"
if ($openclSdk) {
    Write-Host "OpenCL SDK found" -ForegroundColor Green
} else {
    Write-Host "OpenCL SDK not found. Build from source:" -ForegroundColor Yellow
    Write-Host "  git clone --recursive https://github.com/KhronosGroup/OpenCL-SDK.git" -ForegroundColor White
    Write-Host "  cd OpenCL-SDK" -ForegroundColor White
    Write-Host "  cmake -G 'Visual Studio 17 2022' -A x64 -B build -S ." -ForegroundColor White
    Write-Host "  cmake --build build --config Release --target install" -ForegroundColor White
}

Write-Host "`n=== Setup Complete ===" -ForegroundColor Cyan
```

### Cross-Platform: Verify GPU Setup

```bash
#!/bin/bash
# verify_gpu.sh - Verify all GPU backends are working

echo "=== GPU Backend Verification ==="

# Check OpenGL
echo -n "OpenGL: "
if command -v glxinfo &> /dev/null; then
    glxinfo | grep "OpenGL version" | head -1
elif command -v nvidia-smi &> /dev/null; then
    echo "Available (via nvidia-smi)"
else
    echo "Not found"
fi

# Check OpenCL
echo -n "OpenCL: "
if command -v clinfo &> /dev/null; then
    clinfo | grep "Device Name" | head -5
else
    echo "clinfo not installed"
fi

# Check CUDA
echo -n "CUDA: "
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release"
else
    echo "Not found"
fi

# Check NVIDIA driver
echo -n "NVIDIA Driver: "
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "Not found"
fi

echo "=== Done ==="
```

### Build Script for TinyTorch with GPU Support

```bash
#!/bin/bash
# build.sh - Build TinyTorch with GPU support

set -e

echo "=== Building TinyTorch ==="

# Detect OpenCL SDK
OPENCL_SDK="$HOME/OpenCL-SDK/install"
USE_OPENCL_SDK=""
OPENCL_INCLUDE=""
OPENCL_LIB=""

if [ -d "$OPENCL_SDK" ]; then
    echo "Found OpenCL SDK at: $OPENCL_SDK"
    USE_OPENCL_SDK="-DTINYTORCH_USE_OPENCL_SDK"
    OPENCL_INCLUDE="-I$OPENCL_SDK/include"
    OPENCL_LIB="-L$OPENCL_SDK/lib -lOpenCL"
else
    echo "OpenCL SDK not found, using dynamic loading"
    OPENCL_LIB="-lOpenCL"
fi

# Compiler
CXX="g++"
CXXFLAGS="-std=c++17 -Wall -Wextra -O2"
LDFLAGS="-lopengl32 -lgdi32 $OPENCL_LIB"

# Build GPU backend tests
echo "Building GPU backends test..."
$CXX $CXXFLAGS $USE_OPENCL_SDK $OPENCL_INCLUDE \
    -o gpu_backends_test tinytorch/gpu_backends_test.cpp \
    $LDFLAGS

echo "Building GPU kernel tests..."
$CXX $CXXFLAGS $USE_OPENCL_SDK $OPENCL_INCLUDE \
    -o gpu_kernel_test tinytorch/gpu_kernel_test.cpp \
    $LDFLAGS

echo "=== Build Complete ==="
echo "Run tests:"
echo "  ./gpu_backends_test"
echo "  ./gpu_kernel_test"
```

---

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

## Performance Benchmarks

### Element-wise Operations (100K elements, float32)

| Operation | MX130 (OpenCL) | UHD 620 (OpenCL) | MX130 (OpenCLOn12) | Speedup |
|-----------|---------------|------------------|-------------------|---------|
| add | 13ms | 1078ms | 64ms | **83x** |
| sub | 4ms | 1372ms | 28ms | **343x** |
| mul | 0ms | 655ms | 17ms | **∞** |
| div | 15ms | 903ms | 32ms | **60x** |
| add_scalar | 5ms | 1001ms | 32ms | **200x** |
| negate | 5ms | 897ms | 27ms | **179x** |
| abs | 5ms | 1080ms | 24ms | **216x** |
| clamp | 7ms | 1152ms | 30ms | **164x** |

**Key insight**: Native NVIDIA driver is **50-300x faster** than Intel iGPU and **2-5x faster** than Microsoft's OpenCLOn12 wrapper.

### Reduction Operations (100K elements)

| Operation | MX130 | UHD 620 | Notes |
|-----------|-------|---------|-------|
| sum | 4.6ms | 1.8ms | CPU fallback (fast enough) |
| mean | 1.2ms | 0ms | CPU fallback |
| max | 2.9ms | 1.5ms | CPU fallback |
| min | 0ms | 0ms | CPU fallback |
| dot | 2.2ms | 0.9ms | CPU fallback |

**Note**: Reductions currently use CPU fallback for accuracy. GPU reduction kernels exist but require multi-pass implementation for Maxwell due to lack of float atomics.

---

## Troubleshooting Flowchart

```
GPU not detected?
├── Check driver installed
│   ├── NVIDIA: nvidia-smi
│   ├── Intel: clinfo | grep Intel
│   └── AMD: clinfo | grep AMD
├── Check OpenCL runtime
│   ├── Windows: C:\Windows\System32\OpenCL.dll
│   └── Linux: /etc/OpenCL/vendors/
└── Check NvOptimusEnablement export
    └── Must be in main executable, not DLL

CUDA not found?
├── Check CUDA Toolkit installed
│   └── nvcc --version
├── Check PATH includes CUDA bin directory
│   └── C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin
└── Check cudart64_*.dll exists
    └── Try all versions: 12, 11, 10, 9, 8

OpenCL kernel compilation fails?
├── Check OpenCL version: clGetDeviceInfo(CL_DEVICE_OPENCL_C_VERSION)
├── Maxwell: Use OpenCL 1.2 features only
├── Check for float atomics (not supported on Maxwell)
└── Check work group size (max 1024 on Maxwell)
```

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