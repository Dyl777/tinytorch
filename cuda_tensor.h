#pragma once
#include "gpu_backends.h"
#include <vector>
#include <string>
#include <functional>
#include <memory>
#include <cmath>
#include <algorithm>
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#endif

// ============================================================================
// CUDA Tensor Implementation
// ============================================================================
// Uses NVRTC for runtime compilation of CUDA kernels, similar to OpenCL's
// runtime compilation. This allows a header-only implementation without nvcc.

// NVRTC function types
typedef int (*nvrtcCreateProgram_t)(void**, const char*, const char*, int, const char**, const char**);
typedef int (*nvrtcDestroyProgram_t)(void**);
typedef int (*nvrtcCompileProgram_t)(void*, int, const char**);
typedef int (*nvrtcGetPTXSize_t)(void*, size_t*);
typedef int (*nvrtcGetPTX_t)(void*, char*);
typedef int (*nvrtcGetProgramLogSize_t)(void*, size_t*);
typedef int (*nvrtcGetProgramLog_t)(void*, char*);
typedef const char* (*nvrtcGetErrorString_t)(int);

// CUDA Driver API function types
typedef int (*cuInit_t)(unsigned int);
typedef int (*cuDeviceGetCount_t)(int*);
typedef int (*cuDeviceGet_t)(int*, int);
typedef int (*cuDeviceGetName_t)(char*, int, int);
typedef int (*cuDeviceComputeCapability_t)(int*, int*, int);
typedef int (*cuCtxCreate_v2_t)(void**, unsigned int, int);
typedef int (*cuCtxDestroy_v2_t)(void*);
typedef int (*cuCtxSetCurrent_t)(void*);
typedef int (*cuModuleLoadData_t)(void**, const void*);
typedef int (*cuModuleUnload_t)(void*);
typedef int (*cuModuleGetFunction_t)(void**, void*, const char*);
typedef int (*cuLaunchKernel_t)(void*, unsigned int, unsigned int, unsigned int,
                                 unsigned int, unsigned int, unsigned int,
                                 unsigned int, void*, void**, void**);
typedef int (*cuMemAlloc_v2_t)(void**, size_t);
typedef int (*cuMemFree_v2_t)(void*);
typedef int (*cuMemcpyHtoD_v2_t)(void*, const void*, size_t);
typedef int (*cuMemcpyDtoH_v2_t)(void*, void*, size_t);
typedef int (*cuMemcpyDtoD_v2_t)(void*, void*, size_t);
typedef int (*cuStreamCreate_t)(void**, unsigned int);
typedef int (*cuStreamDestroy_v2_t)(void*);
typedef int (*cuStreamSynchronize_t)(void*);

// ============================================================================
// Global CUDA Driver function pointers (loaded once)
// ============================================================================
struct CudaDriverFunctions {
    // NVRTC
    nvrtcCreateProgram_t nvrtcCreateProgram = nullptr;
    nvrtcDestroyProgram_t nvrtcDestroyProgram = nullptr;
    nvrtcCompileProgram_t nvrtcCompileProgram = nullptr;
    nvrtcGetPTXSize_t nvrtcGetPTXSize = nullptr;
    nvrtcGetPTX_t nvrtcGetPTX = nullptr;
    nvrtcGetProgramLogSize_t nvrtcGetProgramLogSize = nullptr;
    nvrtcGetProgramLog_t nvrtcGetProgramLog = nullptr;
    nvrtcGetErrorString_t nvrtcGetErrorString = nullptr;

    // CUDA Driver API
    cuInit_t cuInit = nullptr;
    cuDeviceGetCount_t cuDeviceGetCount = nullptr;
    cuDeviceGet_t cuDeviceGet = nullptr;
    cuDeviceGetName_t cuDeviceGetName = nullptr;
    cuDeviceComputeCapability_t cuDeviceComputeCapability = nullptr;
    cuCtxCreate_v2_t cuCtxCreate = nullptr;
    cuCtxDestroy_v2_t cuCtxDestroy = nullptr;
    cuCtxSetCurrent_t cuCtxSetCurrent = nullptr;
    cuModuleLoadData_t cuModuleLoadData = nullptr;
    cuModuleUnload_t cuModuleUnload = nullptr;
    cuModuleGetFunction_t cuModuleGetFunction = nullptr;
    cuLaunchKernel_t cuLaunchKernel = nullptr;
    cuMemAlloc_v2_t cuMemAlloc = nullptr;
    cuMemFree_v2_t cuMemFree = nullptr;
    cuMemcpyHtoD_v2_t cuMemcpyHtoD = nullptr;
    cuMemcpyDtoH_v2_t cuMemcpyDtoH = nullptr;
    cuMemcpyDtoD_v2_t cuMemcpyDtoD = nullptr;
    cuStreamCreate_t cuStreamCreate = nullptr;
    cuStreamDestroy_v2_t cuStreamDestroy = nullptr;
    cuStreamSynchronize_t cuStreamSynchronize = nullptr;

    bool is_loaded() const { return nvrtcCreateProgram && cuInit && cuMemAlloc && cuLaunchKernel; }
};

inline CudaDriverFunctions& get_cuda_driver_functions() {
    static CudaDriverFunctions funcs;
    static bool initialized = false;
    
    if (!initialized) {
#ifdef _WIN32
        HMODULE nvrtc_lib = LoadLibraryA("nvrtc64_120_0.dll");
        if (!nvrtc_lib) nvrtc_lib = LoadLibraryA("nvrtc64_110_0.dll");
        if (!nvrtc_lib) nvrtc_lib = LoadLibraryA("nvrtc64_100_0.dll");
        if (!nvrtc_lib) nvrtc_lib = LoadLibraryA("nvrtc64.dll");
        HMODULE cuda_driver_lib = LoadLibraryA("nvcuda.dll");
#else
        void* nvrtc_lib = dlopen("libnvrtc.so", RTLD_LAZY);
        void* cuda_driver_lib = dlopen("libcuda.so", RTLD_LAZY);
#endif

        auto load_sym = [](void* lib, const char* name) {
#ifdef _WIN32
            return (void*)GetProcAddress((HMODULE)lib, name);
#else
            return dlsym(lib, name);
#endif
        };

        if (nvrtc_lib) {
            funcs.nvrtcCreateProgram = (nvrtcCreateProgram_t)load_sym(nvrtc_lib, "nvrtcCreateProgram");
            funcs.nvrtcDestroyProgram = (nvrtcDestroyProgram_t)load_sym(nvrtc_lib, "nvrtcDestroyProgram");
            funcs.nvrtcCompileProgram = (nvrtcCompileProgram_t)load_sym(nvrtc_lib, "nvrtcCompileProgram");
            funcs.nvrtcGetPTXSize = (nvrtcGetPTXSize_t)load_sym(nvrtc_lib, "nvrtcGetPTXSize");
            funcs.nvrtcGetPTX = (nvrtcGetPTX_t)load_sym(nvrtc_lib, "nvrtcGetPTX");
            funcs.nvrtcGetProgramLogSize = (nvrtcGetProgramLogSize_t)load_sym(nvrtc_lib, "nvrtcGetProgramLogSize");
            funcs.nvrtcGetProgramLog = (nvrtcGetProgramLog_t)load_sym(nvrtc_lib, "nvrtcGetProgramLog");
            funcs.nvrtcGetErrorString = (nvrtcGetErrorString_t)load_sym(nvrtc_lib, "nvrtcGetErrorString");
        }

        if (cuda_driver_lib) {
            funcs.cuInit = (cuInit_t)load_sym(cuda_driver_lib, "cuInit");
            funcs.cuDeviceGetCount = (cuDeviceGetCount_t)load_sym(cuda_driver_lib, "cuDeviceGetCount");
            funcs.cuDeviceGet = (cuDeviceGet_t)load_sym(cuda_driver_lib, "cuDeviceGet");
            funcs.cuDeviceGetName = (cuDeviceGetName_t)load_sym(cuda_driver_lib, "cuDeviceGetName");
            funcs.cuDeviceComputeCapability = (cuDeviceComputeCapability_t)load_sym(cuda_driver_lib, "cuDeviceComputeCapability");
            funcs.cuCtxCreate = (cuCtxCreate_v2_t)load_sym(cuda_driver_lib, "cuCtxCreate_v2");
            funcs.cuCtxDestroy = (cuCtxDestroy_v2_t)load_sym(cuda_driver_lib, "cuCtxDestroy_v2");
            funcs.cuCtxSetCurrent = (cuCtxSetCurrent_t)load_sym(cuda_driver_lib, "cuCtxSetCurrent");
            funcs.cuModuleLoadData = (cuModuleLoadData_t)load_sym(cuda_driver_lib, "cuModuleLoadData");
            funcs.cuModuleUnload = (cuModuleUnload_t)load_sym(cuda_driver_lib, "cuModuleUnload");
            funcs.cuModuleGetFunction = (cuModuleGetFunction_t)load_sym(cuda_driver_lib, "cuModuleGetFunction");
            funcs.cuLaunchKernel = (cuLaunchKernel_t)load_sym(cuda_driver_lib, "cuLaunchKernel");
            funcs.cuMemAlloc = (cuMemAlloc_v2_t)load_sym(cuda_driver_lib, "cuMemAlloc_v2");
            funcs.cuMemFree = (cuMemFree_v2_t)load_sym(cuda_driver_lib, "cuMemFree_v2");
            funcs.cuMemcpyHtoD = (cuMemcpyHtoD_v2_t)load_sym(cuda_driver_lib, "cuMemcpyHtoD_v2");
            funcs.cuMemcpyDtoH = (cuMemcpyDtoH_v2_t)load_sym(cuda_driver_lib, "cuMemcpyDtoH_v2");
            funcs.cuMemcpyDtoD = (cuMemcpyDtoD_v2_t)load_sym(cuda_driver_lib, "cuMemcpyDtoD_v2");
            funcs.cuStreamCreate = (cuStreamCreate_t)load_sym(cuda_driver_lib, "cuStreamCreate");
            funcs.cuStreamDestroy = (cuStreamDestroy_v2_t)load_sym(cuda_driver_lib, "cuStreamDestroy_v2");
            funcs.cuStreamSynchronize = (cuStreamSynchronize_t)load_sym(cuda_driver_lib, "cuStreamSynchronize");
        }

        initialized = true;
    }
    
    return funcs;
}

// ============================================================================
// CUDA Kernel Sources
// ============================================================================
static const char* CUDA_KERNEL_ADD = R"(
extern "C" __global__ void tensor_add(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
)";

static const char* CUDA_KERNEL_SUB = R"(
extern "C" __global__ void tensor_sub(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] - b[i];
}
)";

static const char* CUDA_KERNEL_MUL = R"(
extern "C" __global__ void tensor_mul(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}
)";

static const char* CUDA_KERNEL_DIV = R"(
extern "C" __global__ void tensor_div(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] / b[i];
}
)";

static const char* CUDA_KERNEL_ADD_SCALAR = R"(
extern "C" __global__ void tensor_add_scalar(const float* a, float s, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + s;
}
)";

static const char* CUDA_KERNEL_SUB_SCALAR = R"(
extern "C" __global__ void tensor_sub_scalar(const float* a, float s, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] - s;
}
)";

static const char* CUDA_KERNEL_MUL_SCALAR = R"(
extern "C" __global__ void tensor_mul_scalar(const float* a, float s, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * s;
}
)";

static const char* CUDA_KERNEL_DIV_SCALAR = R"(
extern "C" __global__ void tensor_div_scalar(const float* a, float s, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] / s;
}
)";

static const char* CUDA_KERNEL_NEGATE = R"(
extern "C" __global__ void tensor_negate(const float* a, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = -a[i];
}
)";

static const char* CUDA_KERNEL_ABS = R"(
extern "C" __global__ void tensor_abs(const float* a, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fabsf(a[i]);
}
)";

static const char* CUDA_KERNEL_CLAMP = R"(
extern "C" __global__ void tensor_clamp(const float* a, float min_val, float max_val, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = a[i];
        out[i] = v < min_val ? min_val : (v > max_val ? max_val : v);
    }
}
)";

static const char* CUDA_KERNEL_GT = R"(
extern "C" __global__ void tensor_gt(const float* a, float threshold, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] > threshold ? 1 : 0;
}
)";

static const char* CUDA_KERNEL_LT = R"(
extern "C" __global__ void tensor_lt(const float* a, float threshold, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] < threshold ? 1 : 0;
}
)";

// ============================================================================
// CUDA Tensor Class
// ============================================================================
class CudaTensor : public TensorBase<float> {
private:
    int _device_id;
    void* _ctx;
    void* _stream;
    void* _buffer;
    size_t _size;
    std::string _device_name;
    bool _owns_context;
    
    // Multi-dimensional tensor metadata
    std::vector<std::size_t> _shape;
    std::vector<std::size_t> _stride;
    std::size_t _ndim;

    const CudaDriverFunctions& F;  // Reference to global functions
    
    // Compute strides from shape (row-major)
    static std::vector<std::size_t> compute_strides(const std::vector<std::size_t>& shape) {
        std::size_t ndim = shape.size();
        if (ndim == 0) return {};
        
        std::vector<std::size_t> stride(ndim);
        stride[ndim - 1] = 1;
        for (std::size_t i = ndim - 1; i > 0; --i) {
            stride[i - 1] = stride[i] * shape[i];
        }
        return stride;
    }
    
    // Compute total size from shape
    static std::size_t compute_total_size(const std::vector<std::size_t>& shape) {
        if (shape.empty()) return 0;
        std::size_t size = 1;
        for (auto dim : shape) size *= dim;
        return size;
    }

    std::string compile_to_ptx(const char* source, const char* kernel_name) const {
        void* prog = nullptr;
        int err = F.nvrtcCreateProgram(&prog, source, kernel_name, 0, nullptr, nullptr);
        if (err != 0) {
            std::cerr << "[CUDA] NVRTC create failed: " << F.nvrtcGetErrorString(err) << std::endl;
            return "";
        }

        const char* opts[] = {"--gpu-architecture=compute_50", "--ftz=true", "--prec-div=false", "--prec-sqrt=false"};
        err = F.nvrtcCompileProgram(prog, 4, opts);
        if (err != 0) {
            size_t log_size;
            F.nvrtcGetProgramLogSize(prog, &log_size);
            std::vector<char> log(log_size);
            F.nvrtcGetProgramLog(prog, log.data());
            std::cerr << "[CUDA] NVRTC compile failed for " << kernel_name << ":\n" << log.data() << std::endl;
            F.nvrtcDestroyProgram(&prog);
            return "";
        }

        size_t ptx_size;
        F.nvrtcGetPTXSize(prog, &ptx_size);
        std::string ptx(ptx_size, '\0');
        F.nvrtcGetPTX(prog, &ptx[0]);
        F.nvrtcDestroyProgram(&prog);

        return ptx;
    }

    void launch_kernel(const char* source, const char* kernel_name, void** args, int num_args, size_t n) const {
        // Set context current before any CUDA operations
        if (F.cuCtxSetCurrent) {
            int err = F.cuCtxSetCurrent(_ctx);
            if (err != 0) {
                std::cerr << "[CUDA] Failed to set context current: " << err << std::endl;
                return;
            }
        }
        
        std::string ptx = compile_to_ptx(source, kernel_name);
        if (ptx.empty()) {
            std::cerr << "[CUDA] PTX compilation failed for " << kernel_name << std::endl;
            return;
        }

        void* module = nullptr;
        int err = F.cuModuleLoadData(&module, ptx.c_str());
        if (err != 0) {
            std::cerr << "[CUDA] Module load failed: " << err << std::endl;
            return;
        }

        void* function = nullptr;
        err = F.cuModuleGetFunction(&function, module, kernel_name);
        if (err != 0) {
            std::cerr << "[CUDA] Get function failed: " << err << std::endl;
            F.cuModuleUnload(module);
            return;
        }

        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;

        err = F.cuLaunchKernel(function, grid_size, 1, 1, block_size, 1, 1, 0, _stream, args, nullptr);
        if (err != 0) {
            std::cerr << "[CUDA] Launch failed: " << err << " (grid=" << grid_size << ", block=" << block_size << ")" << std::endl;
            F.cuModuleUnload(module);
            return;
        }

        F.cuStreamSynchronize(_stream);
        F.cuModuleUnload(module);
    }
    
    // Launch a 2D kernel with customizable grid and block dimensions
    void launch_2d_kernel(const char* source, const char* kernel_name, void** args, int num_args, 
                          int width, int height) const {
        // Set context current before any CUDA operations
        if (F.cuCtxSetCurrent) {
            int err = F.cuCtxSetCurrent(_ctx);
            if (err != 0) {
                std::cerr << "[CUDA] Failed to set context current: " << err << std::endl;
                return;
            }
        }
        
        std::string ptx = compile_to_ptx(source, kernel_name);
        if (ptx.empty()) {
            std::cerr << "[CUDA] PTX compilation failed for " << kernel_name << std::endl;
            return;
        }

        void* module = nullptr;
        int err = F.cuModuleLoadData(&module, ptx.c_str());
        if (err != 0) {
            std::cerr << "[CUDA] Module load failed: " << err << std::endl;
            return;
        }

        void* function = nullptr;
        err = F.cuModuleGetFunction(&function, module, kernel_name);
        if (err != 0) {
            std::cerr << "[CUDA] Get function failed: " << err << std::endl;
            F.cuModuleUnload(module);
            return;
        }

        int block_x = 16, block_y = 16;
        int grid_x = (width + block_x - 1) / block_x;
        int grid_y = (height + block_y - 1) / block_y;

        err = F.cuLaunchKernel(function, grid_x, grid_y, 1, block_x, block_y, 1, 0, _stream, args, nullptr);
        if (err != 0) {
            std::cerr << "[CUDA] Launch failed: " << err << std::endl;
            F.cuModuleUnload(module);
            return;
        }

        F.cuStreamSynchronize(_stream);
        F.cuModuleUnload(module);
    }

    CudaTensor(int device_id, void* ctx, void* stream, const std::vector<std::size_t>& shape, bool owns_ctx = false)
        : _device_id(device_id), _ctx(ctx), _stream(stream), _shape(shape), _owns_context(owns_ctx),
          F(get_cuda_driver_functions()) {
        _ndim = shape.size();
        _stride = compute_strides(shape);
        _size = compute_total_size(shape);
        
        F.cuMemAlloc(&_buffer, _size * sizeof(float));

        char name[256] = {0};
        F.cuDeviceGetName(name, sizeof(name), device_id);
        _device_name = name;
    }

public:
    // Initialize CUDA and create tensor from data (1D)
    static CudaTensor* from_data(const std::vector<float>& data, int device_id) {
        auto& F = get_cuda_driver_functions();
        if (!F.is_loaded()) {
            std::cerr << "[CUDA] Failed to load CUDA functions" << std::endl;
            return nullptr;
        }

        F.cuInit(0);
        void* ctx = nullptr;
        F.cuCtxCreate(&ctx, 0, device_id);
        void* stream = nullptr;
        F.cuStreamCreate(&stream, 0);

        std::vector<std::size_t> shape = {data.size()};
        CudaTensor* tensor = new CudaTensor(device_id, ctx, stream, shape, true);
        
        // Set context current before memory operations
        F.cuCtxSetCurrent(ctx);
        F.cuMemcpyHtoD(tensor->_buffer, data.data(), data.size() * sizeof(float));
        return tensor;
    }
    
    // Initialize CUDA and create tensor from data with shape
    static CudaTensor* from_data(const std::vector<float>& data, const std::vector<std::size_t>& shape, int device_id) {
        auto& F = get_cuda_driver_functions();
        if (!F.is_loaded()) {
            std::cerr << "[CUDA] Failed to load CUDA functions" << std::endl;
            return nullptr;
        }
        
        std::size_t total = compute_total_size(shape);
        if (total != data.size()) {
            throw std::invalid_argument("Data size doesn't match shape");
        }

        F.cuInit(0);
        void* ctx = nullptr;
        F.cuCtxCreate(&ctx, 0, device_id);
        void* stream = nullptr;
        F.cuStreamCreate(&stream, 0);

        CudaTensor* tensor = new CudaTensor(device_id, ctx, stream, shape, true);
        
        // Set context current before memory operations
        F.cuCtxSetCurrent(ctx);
        F.cuMemcpyHtoD(tensor->_buffer, data.data(), data.size() * sizeof(float));
        return tensor;
    }

    // Create empty tensor (1D)
    static CudaTensor* create_empty(size_t size, int device_id) {
        std::vector<std::size_t> shape = {size};
        return create_empty_with_shape(shape, device_id);
    }
    
    // Create empty tensor with specific shape
    static CudaTensor* create_empty_with_shape(const std::vector<std::size_t>& shape, int device_id) {
        auto& F = get_cuda_driver_functions();
        if (!F.is_loaded()) return nullptr;

        F.cuInit(0);
        void* ctx = nullptr;
        F.cuCtxCreate(&ctx, 0, device_id);
        void* stream = nullptr;
        F.cuStreamCreate(&stream, 0);

        return new CudaTensor(device_id, ctx, stream, shape, true);
    }
    
    // Create result tensor sharing this tensor's context (for operations)
    CudaTensor* create_result_with_shape(const std::vector<std::size_t>& shape) const {
        auto& F = get_cuda_driver_functions();
        void* stream = nullptr;
        F.cuStreamCreate(&stream, 0);
        
        // Create tensor sharing context (not owning it)
        CudaTensor* result = new CudaTensor(_device_id, _ctx, stream, shape, false);
        return result;
    }
    
    CudaTensor* create_result(size_t size) const {
        return create_result_with_shape({size});
    }

    ~CudaTensor() {
        if (_buffer) F.cuMemFree(_buffer);
        if (_owns_context && _stream) F.cuStreamDestroy(_stream);
        if (_owns_context && _ctx) F.cuCtxDestroy(_ctx);
    }

    std::string device_name() const { return _device_name; }
    size_t size() const { return _size; }
    void* buffer() const { return _buffer; }
    void* stream() const { return _stream; }
    void* context() const { return _ctx; }
    int device_id() const { return _device_id; }

    std::vector<float> to_host() const {
        // Set context current before memory operations
        if (F.cuCtxSetCurrent) {
            F.cuCtxSetCurrent(_ctx);
        }
        std::vector<float> data(_size);
        F.cuMemcpyDtoH(data.data(), _buffer, _size * sizeof(float));
        return data;
    }

    // TensorBase interface
    std::size_t ndim() const override { return _ndim; }
    std::size_t total_size() const override { return _size; }
    const std::size_t* shape() const override { return _shape.data(); }
    const std::size_t* stride() const override { return _stride.data(); }

    float get_element(std::size_t index) const override {
        std::vector<float> data = to_host();
        return data[index];
    }

    void set_element(std::size_t index, float value) override {
        std::vector<float> data = to_host();
        data[index] = value;
        F.cuMemcpyHtoD(_buffer, data.data(), _size * sizeof(float));
    }

    float operator()(std::size_t i) const override {
        if (_ndim == 0) {
            throw std::invalid_argument("Can't index into scalar. Use get_element(0)");
        }
        if (i >= _shape[0]) {
            throw std::out_of_range("Index out of bounds");
        }
        std::size_t flat_idx = i * _stride[0];
        std::vector<float> data = to_host();
        return data[flat_idx];
    }
    
    float& operator()(std::size_t i) override {
        static float temp;
        temp = const_cast<CudaTensor*>(this)->operator()(i);
        return temp;
    }
    
    float operator()(std::size_t i, std::size_t j) const override {
        if (_ndim != 2) {
            throw std::invalid_argument("Can only double index into 2D tensors");
        }
        if (i >= _shape[0]) {
            throw std::out_of_range("Row index out of bounds");
        }
        if (j >= _shape[1]) {
            throw std::out_of_range("Column index out of bounds");
        }
        std::size_t flat_idx = i * _stride[0] + j * _stride[1];
        std::vector<float> data = to_host();
        return data[flat_idx];
    }
    
    float& operator()(std::size_t i, std::size_t j) override {
        static float temp;
        temp = const_cast<CudaTensor*>(this)->operator()(i, j);
        return temp;
    }

    bool is_streaming() const override { return false; }
    std::string backend_name() const override { return "CudaTensor (" + _device_name + ")"; }

    // Binary operations
    std::unique_ptr<TensorBase<float>> add(const TensorBase<float>* other) const override {
        const CudaTensor* cuda_other = dynamic_cast<const CudaTensor*>(other);
        if (!cuda_other || cuda_other->_size != _size) {
            throw std::invalid_argument("Incompatible tensors for CUDA add");
        }

        CudaTensor* result = create_result(_size);
        int n = static_cast<int>(_size);
        void* a_ptr = _buffer;
        void* b_ptr = cuda_other->_buffer;
        void* out_ptr = result->_buffer;
        void* args[] = {&a_ptr, &b_ptr, &out_ptr, &n};

        launch_kernel(CUDA_KERNEL_ADD, "tensor_add", args, 4, _size);
        return std::unique_ptr<TensorBase<float>>(result);
    }

    std::unique_ptr<TensorBase<float>> subtract(const TensorBase<float>* other) const override {
        const CudaTensor* cuda_other = dynamic_cast<const CudaTensor*>(other);
        if (!cuda_other || cuda_other->_size != _size) {
            throw std::invalid_argument("Incompatible tensors for CUDA sub");
        }

        CudaTensor* result = create_result(_size);
        int n = static_cast<int>(_size);
        void* a_ptr = _buffer;
        void* b_ptr = cuda_other->_buffer;
        void* out_ptr = result->_buffer;
        void* args[] = {&a_ptr, &b_ptr, &out_ptr, &n};

        launch_kernel(CUDA_KERNEL_SUB, "tensor_sub", args, 4, _size);
        return std::unique_ptr<TensorBase<float>>(result);
    }

    std::unique_ptr<TensorBase<float>> multiply(const TensorBase<float>* other) const override {
        const CudaTensor* cuda_other = dynamic_cast<const CudaTensor*>(other);
        if (!cuda_other || cuda_other->_size != _size) {
            throw std::invalid_argument("Incompatible tensors for CUDA mul");
        }

        CudaTensor* result = create_result(_size);
        int n = static_cast<int>(_size);
        void* a_ptr = _buffer;
        void* b_ptr = cuda_other->_buffer;
        void* out_ptr = result->_buffer;
        void* args[] = {&a_ptr, &b_ptr, &out_ptr, &n};

        launch_kernel(CUDA_KERNEL_MUL, "tensor_mul", args, 4, _size);
        return std::unique_ptr<TensorBase<float>>(result);
    }

    std::unique_ptr<TensorBase<float>> divide(const TensorBase<float>* other) const override {
        const CudaTensor* cuda_other = dynamic_cast<const CudaTensor*>(other);
        if (!cuda_other || cuda_other->_size != _size) {
            throw std::invalid_argument("Incompatible tensors for CUDA div");
        }

        CudaTensor* result = create_result(_size);
        int n = static_cast<int>(_size);
        void* a_ptr = _buffer;
        void* b_ptr = cuda_other->_buffer;
        void* out_ptr = result->_buffer;
        void* args[] = {&a_ptr, &b_ptr, &out_ptr, &n};

        launch_kernel(CUDA_KERNEL_DIV, "tensor_div", args, 4, _size);
        return std::unique_ptr<TensorBase<float>>(result);
    }

    std::unique_ptr<TensorBase<float>> add_scalar(float scalar) const override {
        return scalar_op(CUDA_KERNEL_ADD_SCALAR, "tensor_add_scalar", scalar);
    }

    std::unique_ptr<TensorBase<float>> subtract_scalar(float scalar) const override {
        return scalar_op(CUDA_KERNEL_SUB_SCALAR, "tensor_sub_scalar", scalar);
    }

    std::unique_ptr<TensorBase<float>> multiply_scalar(float scalar) const override {
        return scalar_op(CUDA_KERNEL_MUL_SCALAR, "tensor_mul_scalar", scalar);
    }

    std::unique_ptr<TensorBase<float>> divide_scalar(float scalar) const override {
        return scalar_op(CUDA_KERNEL_DIV_SCALAR, "tensor_div_scalar", scalar);
    }

    std::unique_ptr<TensorBase<float>> negate() const override {
        return unary_op(CUDA_KERNEL_NEGATE, "tensor_negate");
    }

    std::unique_ptr<TensorBase<float>> abs() const override {
        return unary_op(CUDA_KERNEL_ABS, "tensor_abs");
    }

    float sum() const override {
        std::vector<float> data = to_host();
        float result = 0.0f;
        for (float v : data) result += v;
        return result;
    }

    float mean() const override { return sum() / static_cast<float>(_size); }

    float max() const override {
        std::vector<float> data = to_host();
        float result = data[0];
        for (size_t i = 1; i < _size; ++i) {
            if (data[i] > result) result = data[i];
        }
        return result;
    }

    float min() const override {
        std::vector<float> data = to_host();
        float result = data[0];
        for (size_t i = 1; i < _size; ++i) {
            if (data[i] < result) result = data[i];
        }
        return result;
    }

    float dot(const TensorBase<float>* other) const override {
        const CudaTensor* cuda_other = dynamic_cast<const CudaTensor*>(other);
        if (!cuda_other || cuda_other->_size != _size) {
            throw std::invalid_argument("Incompatible tensors for CUDA dot");
        }

        std::vector<float> a = to_host();
        std::vector<float> b = cuda_other->to_host();
        float result = 0.0f;
        for (size_t i = 0; i < _size; ++i) {
            result += a[i] * b[i];
        }
        return result;
    }

    std::unique_ptr<TensorBase<float>> reshape(const std::size_t* new_shape, std::size_t new_ndim) const override {
        // Validate total size
        std::size_t new_total = 1;
        for (std::size_t i = 0; i < new_ndim; ++i) {
            new_total *= new_shape[i];
        }
        if (new_total != _size) {
            throw std::invalid_argument("Reshape must preserve total number of elements");
        }
        
        // Create new shape vector
        std::vector<std::size_t> shape_vec(new_shape, new_shape + new_ndim);
        
        // Reshape doesn't change the data, just the interpretation
        // We can share the same GPU buffer
        CudaTensor* result = create_result_with_shape(shape_vec);
        
        // Copy data (since create_result allocates new memory)
        F.cuCtxSetCurrent(_ctx);
        F.cuMemcpyDtoD(result->_buffer, _buffer, _size * sizeof(float));
        
        return std::unique_ptr<TensorBase<float>>(result);
    }

    std::unique_ptr<TensorBase<float>> transpose() const override {
        if (_ndim != 2) {
            throw std::invalid_argument("Transpose is only defined for 2D tensors");
        }
        
        // Create output tensor with swapped shape
        std::vector<std::size_t> new_shape = {_shape[1], _shape[0]};
        CudaTensor* result = create_result_with_shape(new_shape);
        
        // Launch transpose kernel
        int rows = static_cast<int>(_shape[0]);
        int cols = static_cast<int>(_shape[1]);
        void* a_ptr = _buffer;
        void* out_ptr = result->_buffer;
        void* args[] = {&a_ptr, &out_ptr, &rows, &cols};
        
        // Transpose kernel
        static const char* TRANSPOSE_KERNEL = R"(
extern "C" __global__ void tensor_transpose_2d(const float* a, float* out, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        out[col * rows + row] = a[row * cols + col];
    }
}
)";
        
        launch_2d_kernel(TRANSPOSE_KERNEL, "tensor_transpose_2d", args, 4, cols, rows);
        
        return std::unique_ptr<TensorBase<float>>(result);
    }

    std::unique_ptr<TensorBase<float>> clamp(float min_val, float max_val) const override {
        CudaTensor* result = create_result(_size);
        int n = static_cast<int>(_size);
        void* a_ptr = _buffer;
        void* out_ptr = result->_buffer;
        void* args[] = {&a_ptr, &min_val, &max_val, &out_ptr, &n};

        launch_kernel(CUDA_KERNEL_CLAMP, "tensor_clamp", args, 5, _size);
        return std::unique_ptr<TensorBase<float>>(result);
    }

    std::unique_ptr<TensorBase<bool>> greater_than(float threshold) const override {
        void* out_buf = nullptr;
        F.cuMemAlloc(&out_buf, _size * sizeof(int));

        int n = static_cast<int>(_size);
        void* a_ptr = _buffer;
        void* out_ptr = out_buf;
        void* args[] = {&a_ptr, &threshold, &out_ptr, &n};

        launch_kernel(CUDA_KERNEL_GT, "tensor_gt", args, 4, _size);

        std::vector<int> int_result(_size);
        F.cuMemcpyDtoH(int_result.data(), out_buf, _size * sizeof(int));
        F.cuMemFree(out_buf);

        std::vector<bool> bool_result(_size);
        for (size_t i = 0; i < _size; ++i) {
            bool_result[i] = int_result[i] != 0;
        }

        return std::unique_ptr<TensorBase<bool>>(new BoolTensorResult(bool_result));
    }

    std::unique_ptr<TensorBase<bool>> less_than(float threshold) const override {
        void* out_buf = nullptr;
        F.cuMemAlloc(&out_buf, _size * sizeof(int));

        int n = static_cast<int>(_size);
        void* a_ptr = _buffer;
        void* out_ptr = out_buf;
        void* args[] = {&a_ptr, &threshold, &out_ptr, &n};

        launch_kernel(CUDA_KERNEL_LT, "tensor_lt", args, 4, _size);

        std::vector<int> int_result(_size);
        F.cuMemcpyDtoH(int_result.data(), out_buf, _size * sizeof(int));
        F.cuMemFree(out_buf);

        std::vector<bool> bool_result(_size);
        for (size_t i = 0; i < _size; ++i) {
            bool_result[i] = int_result[i] != 0;
        }

        return std::unique_ptr<TensorBase<bool>>(new BoolTensorResult(bool_result));
    }

    std::unique_ptr<float[]> to_array() const override {
        std::vector<float> data = to_host();
        std::unique_ptr<float[]> result(new float[_size]);
        for (size_t i = 0; i < _size; ++i) result[i] = data[i];
        return result;
    }

    void print(std::ostream& os) const override {
        std::vector<float> data = to_host();
        
        if (_ndim == 0) {
            os << data[0];
        } else if (_ndim == 1) {
            os << "[";
            size_t print_limit = std::min(_size, (size_t)20);
            for (size_t i = 0; i < print_limit; ++i) {
                os << data[i];
                if (i != print_limit - 1) os << ", ";
            }
            if (_size > print_limit) os << ", ...";
            os << "]";
        } else if (_ndim == 2) {
            os << "[";
            for (size_t i = 0; i < _shape[0]; ++i) {
                os << "[";
                for (size_t j = 0; j < _shape[1]; ++j) {
                    os << data[i * _stride[0] + j * _stride[1]];
                    if (j != _shape[1] - 1) os << ", ";
                }
                os << "]";
                if (i != _shape[0] - 1) os << ", ";
            }
            os << "]";
        } else {
            os << "CudaTensor(ndim=" << _ndim << ", size=" << _size << ")";
        }
    }

    void batch_print(std::ostream& os, std::size_t batch_size) const override {
        std::vector<float> data = to_host();
        os << "[";
        for (size_t batch_start = 0; batch_start < _size; batch_start += batch_size) {
            size_t count = std::min(batch_size, _size - batch_start);
            for (size_t i = 0; i < count; ++i) {
                os << data[batch_start + i];
                if (batch_start + i != _size - 1) os << ", ";
            }
            os.flush();
            if (batch_start + count < _size) {
                os << "\n  ... (batch " << (batch_start / batch_size + 1) << " done) ";
            }
        }
        os << "]";
    }

private:
    std::unique_ptr<TensorBase<float>> unary_op(const char* source, const char* kernel_name) const {
        CudaTensor* result = create_result(_size);
        int n = static_cast<int>(_size);
        void* a_ptr = _buffer;
        void* out_ptr = result->_buffer;
        void* args[] = {&a_ptr, &out_ptr, &n};

        launch_kernel(source, kernel_name, args, 3, _size);
        return std::unique_ptr<TensorBase<float>>(result);
    }

    std::unique_ptr<TensorBase<float>> scalar_op(const char* source, const char* kernel_name, float scalar) const {
        CudaTensor* result = create_result(_size);
        int n = static_cast<int>(_size);
        void* a_ptr = _buffer;
        void* out_ptr = result->_buffer;
        void* args[] = {&a_ptr, &scalar, &out_ptr, &n};

        launch_kernel(source, kernel_name, args, 4, _size);
        return std::unique_ptr<TensorBase<float>>(result);
    }

    // ========================================================================
    // Distributed Parallelism Primitives (GPU-native CUDA implementations)
    // ========================================================================

    // Allgather: each rank contributes its shard, result is the full tensor
    // shards[i] is the shard from rank i (must be on same device), result is concatenated
    CudaTensor* allgather(const CudaTensor* const* shards, int world_size) const {
        std::size_t shard_size = _size;
        std::size_t total_size = shard_size * world_size;
        std::vector<std::size_t> shape = {total_size};
        CudaTensor* result = create_result_with_shape(shape);

        F.cuCtxSetCurrent(_ctx);
        for (int r = 0; r < world_size; ++r) {
            std::size_t offset = r * shard_size;
            F.cuMemcpyDtoD(static_cast<char*>(result->_buffer) + offset * sizeof(float),
                           shards[r]->_buffer, shard_size * sizeof(float));
        }

        return result;
    }

    // Reduce-scatter: extract this rank's shard from full tensor
    CudaTensor* reducescatter(const CudaTensor* full_tensor, int rank, int world_size) const {
        std::size_t shard_size = _size;
        std::size_t offset = rank * shard_size;
        std::vector<std::size_t> shape = {shard_size};
        CudaTensor* result = create_result_with_shape(shape);

        F.cuCtxSetCurrent(_ctx);
        F.cuMemcpyDtoD(result->_buffer,
                       static_cast<char*>(full_tensor->_buffer) + offset * sizeof(float),
                       shard_size * sizeof(float));

        return result;
    }

    // All-reduce sum: element-wise sum of tensors from all ranks
    CudaTensor* allreduce_sum(const CudaTensor* const* shards, int world_size) const {
        CudaTensor* result = create_result(_size);

        // Initialize result to zero
        std::vector<float> zeros(_size, 0.0f);
        F.cuCtxSetCurrent(_ctx);
        F.cuMemcpyHtoD(result->_buffer, zeros.data(), _size * sizeof(float));

        // Add each shard
        for (int r = 0; r < world_size; ++r) {
            int n = static_cast<int>(_size);
            void* a_ptr = result->_buffer;
            void* b_ptr = shards[r]->_buffer;
            void* out_ptr = result->_buffer;
            void* args[] = {&a_ptr, &b_ptr, &out_ptr, &n};
            launch_kernel(CUDA_KERNEL_ADD, "tensor_add", args, 4, _size);
        }

        return result;
    }

    // All-reduce mean: element-wise average of tensors from all ranks
    CudaTensor* allreduce_mean(const CudaTensor* const* shards, int world_size) const {
        CudaTensor* sum = allreduce_sum(shards, world_size);

        // Divide by world_size
        float inv_ws = 1.0f / static_cast<float>(world_size);
        auto result = sum->multiply_scalar(inv_ws);
        delete sum;
        return static_cast<CudaTensor*>(result.release());
    }

    // Broadcast: copy root rank's tensor to all other ranks
    CudaTensor* broadcast(const CudaTensor* root_tensor) const {
        CudaTensor* result = create_result(_size);

        F.cuCtxSetCurrent(_ctx);
        F.cuMemcpyDtoD(result->_buffer, root_tensor->_buffer, _size * sizeof(float));

        return result;
    }
};
