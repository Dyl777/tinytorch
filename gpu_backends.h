#pragma once
#include "tensor.h"
#include "stream_tensor.h"
#include "auto_tensor.h"
#include "gpu_tensor.h"
#include <memory>
#include <functional>
#include <vector>
#include <string>
#include <stdexcept>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <map>

// ============================================================================
// NVIDIA Optimus: Force discrete GPU for OpenGL
// ============================================================================
#if defined(_WIN32) || defined(_WIN64)
extern "C" {
    __declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
    __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}
#endif

// ============================================================================
// CUDA Backend (dynamic loading)
// ============================================================================
#ifdef _WIN32
#include <windows.h>
#endif

typedef int cudaError_t;
enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3
};

typedef cudaError_t (*cudaMalloc_t)(void**, size_t);
typedef cudaError_t (*cudaFree_t)(void*);
typedef cudaError_t (*cudaMemcpy_t)(void*, const void*, size_t, cudaMemcpyKind);
typedef cudaError_t (*cudaGetDeviceCount_t)(int*);
typedef cudaError_t (*cudaSetDevice_t)(int);
typedef cudaError_t (*cudaGetLastError_t)(void);
typedef cudaError_t (*cudaDeviceSynchronize_t)(void);
typedef const char* (*cudaGetErrorString_t)(cudaError_t);

struct CudaFunctions {
    cudaMalloc_t cudaMalloc = nullptr;
    cudaFree_t cudaFree = nullptr;
    cudaMemcpy_t cudaMemcpy = nullptr;
    cudaGetDeviceCount_t cudaGetDeviceCount = nullptr;
    cudaSetDevice_t cudaSetDevice = nullptr;
    cudaGetLastError_t cudaGetLastError = nullptr;
    cudaDeviceSynchronize_t cudaDeviceSynchronize = nullptr;
    cudaGetErrorString_t cudaGetErrorString = nullptr;

    bool is_available() const {
        return cudaMalloc != nullptr && cudaGetDeviceCount != nullptr;
    }
};

struct CudaDeviceInfo {
    int device_id;
    std::string name;
    int compute_capability_major;
    int compute_capability_minor;
    size_t total_memory;
    int multiprocessor_count;
    int max_threads_per_block;
    double compute_score;

    CudaDeviceInfo()
        : device_id(-1), compute_capability_major(0), compute_capability_minor(0),
          total_memory(0), multiprocessor_count(0), max_threads_per_block(0),
          compute_score(0.0)
    {}
};

class CudaManager {
private:
    bool _initialized;
    CudaFunctions _funcs;
    std::vector<CudaDeviceInfo> _devices;
    int _selected_device;

#ifdef _WIN32
    HMODULE _cuda_lib;
#else
    void* _cuda_lib;
#endif

    void* load_symbol(const char* name) {
#ifdef _WIN32
        return (void*)GetProcAddress(_cuda_lib, name);
#else
        return dlsym(_cuda_lib, name);
#endif
    }

public:
    CudaManager() : _initialized(false), _selected_device(-1)
#ifdef _WIN32
    , _cuda_lib(nullptr)
#else
    , _cuda_lib(nullptr)
#endif
    {}

    bool initialize() {
        if (_initialized) return true;

#ifdef _WIN32
        // Try CUDA 12.x first, then 11.x, then generic names
        _cuda_lib = LoadLibraryA("cudart64_12.dll");
        if (!_cuda_lib) _cuda_lib = LoadLibraryA("cudart64_11.dll");
        if (!_cuda_lib) _cuda_lib = LoadLibraryA("cudart64_10.dll");
        if (!_cuda_lib) _cuda_lib = LoadLibraryA("cudart64_9.dll");
        if (!_cuda_lib) _cuda_lib = LoadLibraryA("cudart64_8.dll");
        if (!_cuda_lib) _cuda_lib = LoadLibraryA("cudart64_7.dll");
        if (!_cuda_lib) _cuda_lib = LoadLibraryA("cudart64.dll");
#else
        _cuda_lib = dlopen("libcudart.so", RTLD_LAZY);
        if (!_cuda_lib) _cuda_lib = dlopen("libcudart.so.12", RTLD_LAZY);
        if (!_cuda_lib) _cuda_lib = dlopen("libcudart.so.11", RTLD_LAZY);
#endif

        if (!_cuda_lib) {
            std::cout << "[CudaManager] CUDA runtime not found" << std::endl;
            return false;
        }

        _funcs.cudaMalloc = (cudaMalloc_t)load_symbol("cudaMalloc");
        _funcs.cudaFree = (cudaFree_t)load_symbol("cudaFree");
        _funcs.cudaMemcpy = (cudaMemcpy_t)load_symbol("cudaMemcpy");
        _funcs.cudaGetDeviceCount = (cudaGetDeviceCount_t)load_symbol("cudaGetDeviceCount");
        _funcs.cudaSetDevice = (cudaSetDevice_t)load_symbol("cudaSetDevice");
        _funcs.cudaGetLastError = (cudaGetLastError_t)load_symbol("cudaGetLastError");
        _funcs.cudaDeviceSynchronize = (cudaDeviceSynchronize_t)load_symbol("cudaDeviceSynchronize");
        _funcs.cudaGetErrorString = (cudaGetErrorString_t)load_symbol("cudaGetErrorString");

        if (!_funcs.is_available()) {
            std::cerr << "[CudaManager] Failed to load CUDA functions" << std::endl;
            return false;
        }

        int count = 0;
        cudaError_t err = _funcs.cudaGetDeviceCount(&count);
        if (err != 0 || count == 0) {
            std::cout << "[CudaManager] No CUDA devices found" << std::endl;
            return false;
        }

        std::cout << "[CudaManager] Found " << count << " CUDA device(s)" << std::endl;

        for (int i = 0; i < count; ++i) {
            CudaDeviceInfo info;
            info.device_id = i;
            info.name = "CUDA Device #" + std::to_string(i);
            info.compute_capability_major = 5;
            info.compute_capability_minor = 0;
            info.total_memory = 2ULL * 1024 * 1024 * 1024;
            info.multiprocessor_count = 3;
            info.max_threads_per_block = 1024;
            info.compute_score = 100.0 + info.multiprocessor_count * 10.0;
            _devices.push_back(info);
        }

        _initialized = true;
        return true;
    }

    bool is_available() const { return _initialized && _funcs.is_available(); }
    const std::vector<CudaDeviceInfo>& devices() const { return _devices; }
    const CudaFunctions& functions() const { return _funcs; }
    int selected_device() const { return _selected_device; }

    void select_device(int device_id) {
        if (device_id < 0 || device_id >= static_cast<int>(_devices.size())) {
            throw std::invalid_argument("Invalid CUDA device ID");
        }
        _selected_device = device_id;
        _funcs.cudaSetDevice(device_id);
    }

    void auto_select() {
        if (_devices.empty()) { _selected_device = -1; return; }
        int best = 0;
        for (int i = 1; i < static_cast<int>(_devices.size()); ++i) {
            if (_devices[i].compute_score > _devices[best].compute_score) best = i;
        }
        select_device(best);
    }

    void print_devices(std::ostream& os) const {
        os << "CUDA Devices:\n";
        for (const auto& dev : _devices) {
            os << "  #" << dev.device_id << ": " << dev.name
               << " (CC " << dev.compute_capability_major << "." << dev.compute_capability_minor
               << ", " << (dev.total_memory / (1024.0 * 1024.0 * 1024.0)) << " GB"
               << ", score: " << dev.compute_score << ")\n";
        }
    }

    ~CudaManager() {
        if (_cuda_lib) {
#ifdef _WIN32
            FreeLibrary(_cuda_lib);
#else
            dlclose(_cuda_lib);
#endif
        }
    }
};

inline CudaManager& get_cuda_manager() {
    static CudaManager mgr;
    return mgr;
}

// ============================================================================
// OpenCL Backend
// ============================================================================
// Uses the official Khronos OpenCL SDK headers and links against OpenCL.lib
// SDK built from: https://github.com/KhronosGroup/OpenCL-SDK.git
// Install path: C:\Users\AMBE\OpenCL-SDK\install
//
// To use the SDK, compile with:
//   -DTINYTORCH_USE_OPENCL_SDK
//   -I"C:\Users\AMBE\OpenCL-SDK\install\include"
//   -L"C:\Users\AMBE\OpenCL-SDK\install\lib" -lOpenCL
//
// Or copy OpenCL.dll to your executable directory for dynamic loading fallback.

#ifdef _WIN32
#include <windows.h>
#endif

// Use official OpenCL headers if available
#ifdef TINYTORCH_USE_OPENCL_SDK
    #include <CL/cl.h>
    // SDK provides all types - use them directly
    typedef cl_int (*clGetPlatformIDs_t)(cl_uint, cl_platform_id*, cl_uint*);
    typedef cl_int (*clGetDeviceIDs_t)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
    typedef cl_int (*clGetDeviceInfo_t)(cl_device_id, cl_device_info, size_t, void*, size_t*);
    typedef cl_int (*clGetPlatformInfo_t)(cl_platform_id, cl_platform_info, size_t, void*, size_t*);
#else
    // Fallback: minimal OpenCL types for dynamic loading
    typedef unsigned int cl_uint;
    typedef unsigned long long cl_ulong;
    typedef int cl_int;
    typedef size_t cl_size_t;
    typedef struct _cl_platform_id* cl_platform_id;
    typedef struct _cl_device_id* cl_device_id;
    typedef cl_uint cl_device_type;
    typedef cl_uint cl_device_info;
    typedef cl_uint cl_platform_info;
    #define CL_DEVICE_NAME 0x102B
    #define CL_DEVICE_VENDOR 0x102C
    #define CL_DEVICE_MAX_COMPUTE_UNITS 0x102E
    #define CL_DEVICE_GLOBAL_MEM_SIZE 0x1030
    #define CL_DEVICE_TYPE 0x1028
    #define CL_DEVICE_TYPE_GPU 0x00000002
    #define CL_PLATFORM_NAME 0x0902
    #define CL_SUCCESS 0
    typedef cl_int (*clGetPlatformIDs_t)(cl_uint, cl_platform_id*, cl_uint*);
    typedef cl_int (*clGetDeviceIDs_t)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
    typedef cl_int (*clGetDeviceInfo_t)(cl_device_id, cl_device_info, cl_size_t, void*, cl_size_t*);
    typedef cl_int (*clGetPlatformInfo_t)(cl_platform_id, cl_platform_info, cl_size_t, void*, cl_size_t*);
#endif

struct OpenClFunctions {
#ifdef TINYTORCH_USE_OPENCL_SDK
    // SDK: use function pointers directly
    clGetPlatformIDs_t clGetPlatformIDs = ::clGetPlatformIDs;
    clGetDeviceIDs_t clGetDeviceIDs = ::clGetDeviceIDs;
    clGetDeviceInfo_t clGetDeviceInfo = ::clGetDeviceInfo;
    clGetPlatformInfo_t clGetPlatformInfo = ::clGetPlatformInfo;
#else
    clGetPlatformIDs_t clGetPlatformIDs = nullptr;
    clGetDeviceIDs_t clGetDeviceIDs = nullptr;
    clGetDeviceInfo_t clGetDeviceInfo = nullptr;
    clGetPlatformInfo_t clGetPlatformInfo = nullptr;
#endif

    bool is_available() const {
        return clGetPlatformIDs != nullptr && clGetDeviceIDs != nullptr;
    }
};

struct OpenClDeviceInfo {
    int device_id;
    std::string name;
    std::string vendor;
    std::string platform_name;
    cl_device_id device;
    cl_uint compute_units;
    cl_ulong global_mem_size;
    double compute_score;
    bool is_nvidia;
    bool is_intel;
    bool is_amd;

    OpenClDeviceInfo()
        : device_id(-1), device(nullptr), compute_units(0),
          global_mem_size(0), compute_score(0.0),
          is_nvidia(false), is_intel(false), is_amd(false)
    {}
};

class OpenClManager {
private:
    bool _initialized;
    OpenClFunctions _funcs;
    std::vector<OpenClDeviceInfo> _devices;
    int _selected_device;

#ifdef _WIN32
    HMODULE _cl_lib;
#else
    void* _cl_lib;
#endif

    void* load_symbol(const char* name) {
#ifdef _WIN32
        return (void*)GetProcAddress(_cl_lib, name);
#else
        return dlsym(_cl_lib, name);
#endif
    }

public:
    OpenClManager() : _initialized(false), _selected_device(-1)
#ifdef _WIN32
    , _cl_lib(nullptr)
#else
    , _cl_lib(nullptr)
#endif
    {}

    bool initialize() {
        if (_initialized) return true;

#ifdef TINYTORCH_USE_OPENCL_SDK
        // SDK: functions are already assigned in OpenClFunctions struct
        _initialized = _funcs.is_available();
        if (!_initialized) {
            std::cout << "[OpenClManager] OpenCL functions not available" << std::endl;
            return false;
        }
#else
        // Dynamic loading fallback
#ifdef _WIN32
        _cl_lib = LoadLibraryA("OpenCL.dll");
#else
        _cl_lib = dlopen("libOpenCL.so", RTLD_LAZY);
        if (!_cl_lib) _cl_lib = dlopen("libOpenCL.so.1", RTLD_LAZY);
#endif

        if (!_cl_lib) {
            std::cout << "[OpenClManager] OpenCL runtime not found" << std::endl;
            return false;
        }

        _funcs.clGetPlatformIDs = (clGetPlatformIDs_t)load_symbol("clGetPlatformIDs");
        _funcs.clGetDeviceIDs = (clGetDeviceIDs_t)load_symbol("clGetDeviceIDs");
        _funcs.clGetDeviceInfo = (clGetDeviceInfo_t)load_symbol("clGetDeviceInfo");
        _funcs.clGetPlatformInfo = (clGetPlatformInfo_t)load_symbol("clGetPlatformInfo");

        if (!_funcs.is_available()) {
            std::cerr << "[OpenClManager] Failed to load OpenCL functions" << std::endl;
            return false;
        }
#endif

        cl_uint num_platforms = 0;
        _funcs.clGetPlatformIDs(0, nullptr, &num_platforms);
        if (num_platforms == 0) {
            std::cout << "[OpenClManager] No OpenCL platforms found" << std::endl;
            return false;
        }

        std::vector<cl_platform_id> platforms(num_platforms);
        _funcs.clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

        int device_idx = 0;
        for (cl_uint p = 0; p < num_platforms; ++p) {
            char plat_name[256] = {0};
            _funcs.clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, sizeof(plat_name), plat_name, nullptr);

            cl_uint num_devices = 0;
            _funcs.clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
            if (num_devices == 0) continue;

            std::vector<cl_device_id> devices(num_devices);
            _funcs.clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);

            for (cl_uint d = 0; d < num_devices; ++d) {
                OpenClDeviceInfo info;
                info.device_id = device_idx++;
                info.device = devices[d];
                info.platform_name = plat_name;

                char dev_name[256] = {0};
                char dev_vendor[256] = {0};
                _funcs.clGetDeviceInfo(devices[d], CL_DEVICE_NAME, sizeof(dev_name), dev_name, nullptr);
                _funcs.clGetDeviceInfo(devices[d], CL_DEVICE_VENDOR, sizeof(dev_vendor), dev_vendor, nullptr);
                _funcs.clGetDeviceInfo(devices[d], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(info.compute_units), &info.compute_units, nullptr);
                _funcs.clGetDeviceInfo(devices[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(info.global_mem_size), &info.global_mem_size, nullptr);

                info.name = dev_name;
                info.vendor = dev_vendor;
                info.compute_score = info.compute_units * 10.0 + (info.global_mem_size / (1024.0 * 1024.0 * 1024.0)) * 5.0;

                std::string vendor_lower = info.vendor;
                std::transform(vendor_lower.begin(), vendor_lower.end(), vendor_lower.begin(), ::tolower);
                std::string platform_lower = info.platform_name;
                std::transform(platform_lower.begin(), platform_lower.end(), platform_lower.begin(), ::tolower);
                
                // Prefer native drivers over Microsoft's OpenCLOn12 wrapper
                if (platform_lower.find("nvidia cuda") != std::string::npos) {
                    info.compute_score += 100.0;  // Native NVIDIA driver
                } else if (platform_lower.find("intel") != std::string::npos && platform_lower.find("openclon12") == std::string::npos) {
                    info.compute_score += 50.0;   // Native Intel driver
                } else if (platform_lower.find("openclon12") != std::string::npos) {
                    info.compute_score -= 20.0;   // Microsoft D3D12 wrapper (slower)
                }
                
                // Prefer discrete GPUs
                if (vendor_lower.find("nvidia") != std::string::npos) {
                    info.compute_score += 50.0;
                    info.is_nvidia = true;
                }
                if (vendor_lower.find("intel") != std::string::npos) {
                    info.is_intel = true;
                }
                if (vendor_lower.find("amd") != std::string::npos) {
                    info.compute_score += 50.0;
                    info.is_amd = true;
                }

                _devices.push_back(info);
            }
        }

        if (_devices.empty()) {
            std::cout << "[OpenClManager] No OpenCL GPU devices found" << std::endl;
            return false;
        }

        std::cout << "[OpenClManager] Found " << _devices.size() << " OpenCL GPU device(s)" << std::endl;
        _initialized = true;
        return true;
    }

    bool is_available() const { return _initialized && _funcs.is_available(); }
    const std::vector<OpenClDeviceInfo>& devices() const { return _devices; }
    int selected_device() const { return _selected_device; }

    void select_device(int device_id) {
        if (device_id < 0 || device_id >= static_cast<int>(_devices.size())) {
            throw std::invalid_argument("Invalid OpenCL device ID");
        }
        _selected_device = device_id;
    }

    void auto_select() {
        if (_devices.empty()) { _selected_device = -1; return; }
        int best = 0;
        for (int i = 1; i < static_cast<int>(_devices.size()); ++i) {
            if (_devices[i].compute_score > _devices[best].compute_score) best = i;
        }
        select_device(best);
    }

    void select_by_name(const std::string& pattern) {
        std::string pat = pattern;
        std::transform(pat.begin(), pat.end(), pat.begin(), ::tolower);
        for (const auto& dev : _devices) {
            std::string name = dev.name;
            std::transform(name.begin(), name.end(), name.begin(), ::tolower);
            if (name.find(pat) != std::string::npos) {
                select_device(dev.device_id);
                return;
            }
        }
    }

    void print_devices(std::ostream& os) const {
        os << "OpenCL Devices:\n";
        for (const auto& dev : _devices) {
            os << "  #" << dev.device_id << ": " << dev.name << " (" << dev.vendor << ")"
               << " [" << dev.platform_name << "]"
               << " (" << dev.compute_units << " CUs"
               << ", " << (dev.global_mem_size / (1024.0 * 1024.0 * 1024.0)) << " GB"
               << ", score: " << dev.compute_score << ")\n";
        }
    }

    ~OpenClManager() {
#ifndef TINYTORCH_USE_OPENCL_SDK
        if (_cl_lib) {
#ifdef _WIN32
            FreeLibrary(_cl_lib);
#else
            dlclose(_cl_lib);
#endif
        }
#endif
    }
};

inline OpenClManager& get_opencl_manager() {
    static OpenClManager mgr;
    return mgr;
}

// ============================================================================
// Unified GpuSelector: Combines OpenGL, CUDA, and OpenCL
// ============================================================================
enum class GpuBackend {
    OpenGL,
    CUDA,
    OpenCL
};

struct UnifiedGpuInfo {
    GpuBackend backend;
    int device_id;
    std::string name;
    std::string vendor;
    double compute_score;
    bool is_nvidia;
    bool is_intel;
    bool is_amd;

    UnifiedGpuInfo()
        : backend(GpuBackend::OpenGL), device_id(-1),
          compute_score(0.0), is_nvidia(false), is_intel(false), is_amd(false)
    {}
};

class UnifiedGpuSelector {
private:
    std::vector<UnifiedGpuInfo> _gpus;
    int _selected_index;

public:
    UnifiedGpuSelector() : _selected_index(-1) {}

    bool enumerate() {
        _gpus.clear();

        // 1. OpenCL (can see ALL GPUs)
        auto& cl_mgr = get_opencl_manager();
        if (cl_mgr.initialize()) {
            for (const auto& dev : cl_mgr.devices()) {
                UnifiedGpuInfo info;
                info.backend = GpuBackend::OpenCL;
                info.device_id = dev.device_id;
                info.name = dev.name;
                info.vendor = dev.vendor;
                info.compute_score = dev.compute_score;
                info.is_nvidia = dev.vendor.find("NVIDIA") != std::string::npos ||
                                 dev.vendor.find("nvidia") != std::string::npos;
                info.is_intel = dev.vendor.find("Intel") != std::string::npos ||
                                dev.vendor.find("intel") != std::string::npos;
                info.is_amd = dev.vendor.find("AMD") != std::string::npos ||
                              dev.vendor.find("amd") != std::string::npos;
                _gpus.push_back(info);
            }
        }

        // 2. CUDA (NVIDIA only)
        auto& cuda_mgr = get_cuda_manager();
        if (cuda_mgr.initialize()) {
            for (const auto& dev : cuda_mgr.devices()) {
                bool found = false;
                for (const auto& existing : _gpus) {
                    if (existing.is_nvidia && existing.name.find(dev.name) != std::string::npos) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    UnifiedGpuInfo info;
                    info.backend = GpuBackend::CUDA;
                    info.device_id = dev.device_id;
                    info.name = dev.name;
                    info.vendor = "NVIDIA Corporation";
                    info.compute_score = dev.compute_score;
                    info.is_nvidia = true;
                    _gpus.push_back(info);
                }
            }
        }

        // 3. OpenGL (may only see one GPU)
        auto& gl_sel = get_gpu_selector();
        if (gl_sel.get_gpus().empty()) gl_sel.enumerate();
        for (const auto& dev : gl_sel.get_gpus()) {
            bool found = false;
            for (const auto& existing : _gpus) {
                if (existing.name.find(dev.name) != std::string::npos ||
                    dev.name.find(existing.name) != std::string::npos) {
                    found = true;
                    break;
                }
            }
            if (!found && dev.supports_compute_shaders()) {
                UnifiedGpuInfo info;
                info.backend = GpuBackend::OpenGL;
                info.device_id = dev.device_id;
                info.name = dev.name;
                info.vendor = dev.vendor;
                info.compute_score = dev.compute_score;
                info.is_nvidia = dev.vendor.find("NVIDIA") != std::string::npos;
                info.is_intel = dev.vendor.find("Intel") != std::string::npos;
                info.is_amd = dev.vendor.find("AMD") != std::string::npos;
                _gpus.push_back(info);
            }
        }

        std::sort(_gpus.begin(), _gpus.end(),
                  [](const UnifiedGpuInfo& a, const UnifiedGpuInfo& b) {
                      return a.compute_score > b.compute_score;
                  });

        for (int i = 0; i < static_cast<int>(_gpus.size()); ++i) {
            _gpus[i].device_id = i;
        }

        return !_gpus.empty();
    }

    const std::vector<UnifiedGpuInfo>& gpus() const { return _gpus; }

    void select_device(int device_id) {
        if (device_id < 0 || device_id >= static_cast<int>(_gpus.size())) {
            throw std::invalid_argument("Invalid device ID");
        }
        _selected_index = device_id;
    }

    bool select_by_name(const std::string& pattern) {
        std::string pat = pattern;
        std::transform(pat.begin(), pat.end(), pat.begin(), ::tolower);
        for (const auto& gpu : _gpus) {
            std::string name = gpu.name;
            std::transform(name.begin(), name.end(), name.begin(), ::tolower);
            if (name.find(pat) != std::string::npos) {
                select_device(gpu.device_id);
                return true;
            }
        }
        return false;
    }

    void select_discrete_only() {
        for (const auto& gpu : _gpus) {
            if (!gpu.is_intel) {
                select_device(gpu.device_id);
                return;
            }
        }
        auto_select();
    }

    void auto_select() {
        if (_gpus.empty()) { _selected_index = -1; return; }
        
        // Prefer discrete GPUs (NVIDIA/AMD) over integrated (Intel)
        for (int i = 0; i < static_cast<int>(_gpus.size()); ++i) {
            if (!_gpus[i].is_intel && _gpus[i].is_nvidia) {
                _selected_index = i;
                return;
            }
        }
        for (int i = 0; i < static_cast<int>(_gpus.size()); ++i) {
            if (!_gpus[i].is_intel && _gpus[i].is_amd) {
                _selected_index = i;
                return;
            }
        }
        // Fall back to highest score
        _selected_index = _gpus[0].device_id;
    }

    int selected_device() const { return _selected_index; }
    const UnifiedGpuInfo* selected_gpu() const {
        if (_selected_index < 0) return nullptr;
        return &_gpus[_selected_index];
    }

    void print_all(std::ostream& os) const {
        os << "Detected " << _gpus.size() << " GPU(s) across all backends:\n";
        for (const auto& gpu : _gpus) {
            const char* backend_str = gpu.backend == GpuBackend::CUDA ? "CUDA" :
                                      gpu.backend == GpuBackend::OpenCL ? "OpenCL" : "OpenGL";
            os << "  #" << gpu.device_id << " [" << backend_str << "]: " << gpu.name
               << " (" << gpu.vendor << ") - score: " << gpu.compute_score << "\n";
        }
        if (_selected_index >= 0) {
            os << "Selected: GPU #" << _selected_index << "\n";
        }
    }
};

inline UnifiedGpuSelector& get_unified_gpu_selector() {
    static UnifiedGpuSelector selector;
    return selector;
}
