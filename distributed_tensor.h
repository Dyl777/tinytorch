#pragma once
#include "parallel.h"
#include "autograd_tensor.h"
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <thread>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <numeric>
#include <future>
#include <random>
#include <map>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <array>

// ============================================================================
// Execution Mode: Controls how shards are processed during operations
// ============================================================================
enum class ExecutionMode {
    AUTO,           // Heuristics select best mode automatically
    SEQUENTIAL,     // All shards execute one at a time in calling thread
    PARALLEL,       // All shards execute concurrently in separate threads
    HYBRID,         // CPU shards parallel, GPU shards sequential
    DEVICE_LOCAL,   // Each device type gets its own execution strategy
    BATCHED         // Operations are queued and executed in batches
};

// ============================================================================
// TinyTorch Distributed Tensor
// ============================================================================
// USER-FACING CLASS: DistributedTensor<T>
//
// KEY PRINCIPLE: Data is NEVER fully loaded onto one device at any time.
//   - Tensor is always sharded across devices
//   - Operations execute on shards in parallel via parallelism executors
//   - Reductions use tree-reduce (partial results stay distributed)
//   - Element access reads only the relevant shard
//   - Backward pass computes gradients on shards independently
//   - CPU operations use StreamTensor (lazy, batched, mmap-backed)
//   - Uses ParallelismHeuristics, DevicePool, LoadBalancingStrategy
//   - Uses DataParallelExecutor, TensorParallelExecutor, etc.
//
// USAGE:
//   auto x = DistributedTensor<float>::zeros({1000, 1000});
//   auto y = DistributedTensor<float>::randn({1000, 1000});
//   auto z = x->matmul(y.get());  // Distributed matmul
//   auto loss = z->sum();          // Distributed sum (tree-reduce)
//   loss->backward();              // Distributed backward
// ============================================================================

template<typename T>
class DistributedTensor;

// ============================================================================
// Device Pool: Manages all available devices (uses parallel.h infrastructure)
// ============================================================================

struct DeviceInfo {
    BackendType backend;
    int device_id;
    std::string name;
    std::size_t total_memory_bytes;
    std::size_t available_memory_bytes;
    double compute_score;
    bool is_available;
    
    DeviceInfo() : backend(BackendType::CPU_DENSE), device_id(-1),
                   total_memory_bytes(0), available_memory_bytes(0),
                   compute_score(0), is_available(false) {}
};

class DevicePool {
private:
    std::vector<DeviceInfo> _devices;
    mutable std::mutex _mutex;
    
    void discover_devices() {
        _devices.clear();
        
        // CPU devices (always available)
        DeviceInfo cpu_dense;
        cpu_dense.backend = BackendType::CPU_DENSE;
        cpu_dense.device_id = 0;
        cpu_dense.name = "CPU Dense";
        cpu_dense.total_memory_bytes = get_total_system_memory();
        cpu_dense.available_memory_bytes = get_total_system_memory() / 2;
        cpu_dense.compute_score = 0.5;
        cpu_dense.is_available = true;
        _devices.push_back(cpu_dense);
        
        DeviceInfo cpu_mmap;
        cpu_mmap.backend = BackendType::CPU_MMAP;
        cpu_mmap.device_id = 0;
        cpu_mmap.name = "CPU Mmap (Lazy)";
        cpu_mmap.total_memory_bytes = get_total_system_memory();
        cpu_mmap.available_memory_bytes = get_total_system_memory();
        cpu_mmap.compute_score = 0.3;
        cpu_mmap.is_available = true;
        _devices.push_back(cpu_mmap);
        
        // CUDA devices
        auto& F = get_cuda_driver_functions();
        if (F.is_loaded()) {
            F.cuInit(0);
            int count = 0;
            F.cuDeviceGetCount(&count);
            for (int i = 0; i < count; ++i) {
                char name[256] = {0};
                F.cuDeviceGetName(name, sizeof(name), i);
                DeviceInfo dev;
                dev.backend = BackendType::CUDA;
                dev.device_id = i;
                dev.name = std::string("CUDA: ") + name;
                dev.total_memory_bytes = 2ULL * 1024 * 1024 * 1024;
                dev.available_memory_bytes = dev.total_memory_bytes;
                dev.compute_score = 0.8;
                dev.is_available = true;
                _devices.push_back(dev);
            }
        }
        
        // OpenGL devices
        auto& selector = get_gpu_selector();
        selector.enumerate();
        const auto& gpus = selector.get_gpus();
        for (size_t i = 0; i < gpus.size(); ++i) {
            DeviceInfo dev;
            dev.backend = BackendType::OPENGL;
            dev.device_id = gpus[i].device_id;
            dev.name = std::string("OpenGL: ") + gpus[i].name;
            dev.total_memory_bytes = gpus[i].max_ssbo_size > 0 ? gpus[i].max_ssbo_size : 2ULL * 1024 * 1024 * 1024;
            dev.available_memory_bytes = dev.total_memory_bytes;
            dev.compute_score = 0.6;
            dev.is_available = true;
            _devices.push_back(dev);
        }
        
        // OpenCL devices
        auto& cl_mgr = get_opencl_manager();
        if (cl_mgr.initialize()) {
            const auto& devices = cl_mgr.devices();
            for (size_t i = 0; i < devices.size(); ++i) {
                DeviceInfo dev;
                dev.backend = BackendType::OPENCL;
                dev.device_id = static_cast<int>(i);
                dev.name = std::string("OpenCL: ") + devices[i].name;
                dev.total_memory_bytes = devices[i].global_mem_size;
                dev.available_memory_bytes = devices[i].global_mem_size;
                dev.compute_score = 0.7;
                dev.is_available = true;
                _devices.push_back(dev);
            }
        }
    }
    
public:
    DevicePool() { discover_devices(); }
    
    void refresh() {
        std::lock_guard<std::mutex> lock(_mutex);
        discover_devices();
    }
    
    const std::vector<DeviceInfo>& devices() const { return _devices; }
    
    std::vector<DeviceInfo> gpu_devices() const {
        std::vector<DeviceInfo> gpus;
        for (const auto& d : _devices) {
            if (d.backend != BackendType::CPU_DENSE && d.backend != BackendType::CPU_MMAP)
                gpus.push_back(d);
        }
        return gpus;
    }
    
    std::vector<DeviceInfo> cpu_devices() const {
        std::vector<DeviceInfo> cpus;
        for (const auto& d : _devices) {
            if (d.backend == BackendType::CPU_DENSE || d.backend == BackendType::CPU_MMAP)
                cpus.push_back(d);
        }
        return cpus;
    }
    
    int total_devices() const { return _devices.size(); }
    int gpu_count() const { return gpu_devices().size(); }
    int cpu_count() const { return cpu_devices().size(); }
    
    std::string summary() const {
        std::ostringstream oss;
        oss << "Device Pool: " << _devices.size() << " devices\n";
        for (const auto& d : _devices) {
            oss << "  [" << (d.is_available ? "OK" : "OFF") << "] "
                << d.name << " (" << backend_type_to_string(d.backend) << ")\n";
        }
        return oss.str();
    }
};

inline DevicePool& get_device_pool() {
    static DevicePool pool;
    return pool;
}

// ============================================================================
// Backend Kernel Cache: Unified preloaded kernel storage for all GPU backends
// ============================================================================
struct BackendKernelCache {
    std::map<std::string, void*> cuda_modules;
    std::map<std::string, unsigned int> gl_programs;
    std::map<std::string, void*> ocl_kernels;
    std::mutex mutex;

    bool has_cuda_kernel(const std::string& name) const { return cuda_modules.count(name) > 0; }
    bool has_gl_kernel(const std::string& name) const { return gl_programs.count(name) > 0; }
    bool has_ocl_kernel(const std::string& name) const { return ocl_kernels.count(name) > 0; }
};

// ============================================================================
// ExecutionContextManager: Unified context and kernel management for all backends
// ============================================================================
struct ExecutionContextManager {
    static const int MAX_DEVICES = 16;

    void* cuda_contexts[MAX_DEVICES];
    void* cuda_streams[MAX_DEVICES];
    bool cuda_initialized[MAX_DEVICES];
    bool gl_initialized[MAX_DEVICES];
    void* ocl_contexts[MAX_DEVICES];
    void* ocl_queues[MAX_DEVICES];
    bool ocl_initialized[MAX_DEVICES];

    BackendKernelCache kernel_cache;
    ExecutionMode global_mode;
    std::map<BackendType, ExecutionMode> device_mode_overrides;

    struct BatchedOp {
        std::string op_name;
        std::function<void()> execute_fn;
    };
    std::vector<BatchedOp> batch_queue;
    std::mutex batch_mutex;

    ExecutionContextManager() : global_mode(ExecutionMode::AUTO) {
        for (int i = 0; i < MAX_DEVICES; ++i) {
            cuda_contexts[i] = nullptr; cuda_streams[i] = nullptr; cuda_initialized[i] = false;
            gl_initialized[i] = false;
            ocl_contexts[i] = nullptr; ocl_queues[i] = nullptr; ocl_initialized[i] = false;
        }
    }

    static ExecutionContextManager& instance() { static ExecutionContextManager mgr; return mgr; }

    void set_mode(ExecutionMode mode) { global_mode = mode; }
    void set_device_mode(BackendType backend, ExecutionMode mode) { device_mode_overrides[backend] = mode; }
    ExecutionMode get_mode_for_device(BackendType backend) const {
        if (device_mode_overrides.count(backend)) return device_mode_overrides.at(backend);
        return global_mode;
    }

    static ExecutionMode resolve_auto_mode(ExecutionMode mode,
                                           const std::vector<DeviceInfo>& devices,
                                           std::size_t total_elements,
                                           const std::string&) {
        if (mode != ExecutionMode::AUTO) return mode;
        if (total_elements < 10000) return ExecutionMode::SEQUENTIAL;

        bool has_gpu = false, has_cpu = false;
        for (const auto& dev : devices) {
            if (dev.backend == BackendType::CUDA || dev.backend == BackendType::OPENGL ||
                dev.backend == BackendType::OPENCL) has_gpu = true;
            if (dev.backend == BackendType::CPU_DENSE || dev.backend == BackendType::CPU_MMAP) has_cpu = true;
        }
        if (has_gpu && has_cpu && total_elements > 100000) return ExecutionMode::HYBRID;
        if (!has_gpu && has_cpu) return ExecutionMode::PARALLEL;
        return ExecutionMode::SEQUENTIAL;
    }

    static bool supports_parallel(BackendType backend) {
        return backend == BackendType::CPU_DENSE || backend == BackendType::CPU_MMAP;
    }

    // Push heuristics-derived config to native CUDA backend.
    // Called once at the start of a significant operation.
    // Decision is O(1) - just conditionals, no heavy computation.
    static void apply_cuda_config(const std::vector<DeviceInfo>& devices,
                                  std::size_t total_elements,
                                  std::size_t num_shards,
                                  ExecutionMode mode) {
        bool has_cuda = false;
        int cuda_device_count = 0;
        int cpu_count = 0;
        for (const auto& dev : devices) {
            if (dev.backend == BackendType::CUDA) { has_cuda = true; cuda_device_count++; }
            if (dev.backend == BackendType::CPU_DENSE || dev.backend == BackendType::CPU_MMAP) cpu_count++;
        }
        if (!has_cuda) return;

        // Heuristic factors:
        // - total_elements: small (<10K) = safe, medium (10K-500K) = balanced, large (>500K) = performance
        // - num_shards: many shards (>4) = more kernel launches = prefer cache
        // - cuda_device_count: multiple CUDA devices = prefer sync for correctness
        // - mode: BATCHED = async, PARALLEL = max throughput, SEQUENTIAL = safe

        CudaExecutionConfig cfg;

        // Base config from tensor size
        if (total_elements < 10000) {
            cfg = CudaExecutionConfig::safe_mode();
        } else if (total_elements > 500000) {
            cfg = CudaExecutionConfig::maximum_performance();
        } else {
            cfg = CudaExecutionConfig::safe_mode();  // default to safe for medium
        }

        // DistributedTensor ops are synchronous from user perspective.
        // sync_after_launch must always be true to ensure results are ready
        // before the next operation reads them.
        cfg.sync_after_launch = true;

        // Adjust for shard count: many shards means many kernel launches, cache is critical
        if (num_shards > 4) {
            cfg.kernel_cache = true;
            cfg.context_preload = true;
        }

        // Multiple CUDA devices: keep sync for correctness
        if (cuda_device_count > 1) {
            cfg.sync_after_launch = true;
        }

        CudaContextManager::instance().set_config(cfg);
    }

    template<typename ShardOp>
    static void execute_shards(const std::vector<ShardOp>& shard_ops,
                               const std::vector<DeviceInfo>& devices,
                               std::size_t total_elements,
                               const std::string& op_name) {
        ExecutionMode mode = resolve_auto_mode(
            instance().global_mode, devices, total_elements, op_name);
        switch (mode) {
            case ExecutionMode::SEQUENTIAL: execute_sequential(shard_ops); break;
            case ExecutionMode::PARALLEL: execute_parallel(shard_ops); break;
            case ExecutionMode::HYBRID: execute_hybrid(shard_ops, devices); break;
            case ExecutionMode::DEVICE_LOCAL: execute_device_local(shard_ops, devices); break;
            case ExecutionMode::BATCHED: execute_batched(shard_ops, op_name); break;
            default: execute_sequential(shard_ops); break;
        }
    }

    void flush_batches() {
        std::lock_guard<std::mutex> lock(batch_mutex);
        for (const auto& op : batch_queue) op.execute_fn();
        batch_queue.clear();
    }

private:
    template<typename ShardOp>
    static void execute_sequential(const std::vector<ShardOp>& ops) {
        for (const auto& op : ops) op();
    }

    template<typename ShardOp>
    static void execute_parallel(const std::vector<ShardOp>& ops) {
        std::vector<std::future<void>> futures;
        futures.reserve(ops.size());
        for (const auto& op : ops) futures.push_back(std::async(std::launch::async, op));
        for (auto& f : futures) f.get();
    }

    template<typename ShardOp>
    static void execute_hybrid(const std::vector<ShardOp>& ops,
                               const std::vector<DeviceInfo>& devices) {
        std::vector<std::future<void>> cpu_futures;
        for (size_t i = 0; i < ops.size(); ++i) {
            const auto& dev = devices[i % devices.size()];
            if (supports_parallel(dev.backend))
                cpu_futures.push_back(std::async(std::launch::async, ops[i]));
            else
                ops[i]();
        }
        for (auto& f : cpu_futures) f.get();
    }

    template<typename ShardOp>
    static void execute_device_local(const std::vector<ShardOp>& ops,
                                     const std::vector<DeviceInfo>& devices) {
        std::map<BackendType, std::vector<size_t>> groups;
        for (size_t i = 0; i < ops.size(); ++i)
            groups[devices[i % devices.size()].backend].push_back(i);
        for (const auto& [backend, indices] : groups) {
            if (supports_parallel(backend)) {
                std::vector<std::future<void>> futures;
                for (size_t i : indices) futures.push_back(std::async(std::launch::async, ops[i]));
                for (auto& f : futures) f.get();
            } else {
                for (size_t i : indices) ops[i]();
            }
        }
    }

    template<typename ShardOp>
    static void execute_batched(const std::vector<ShardOp>& ops, const std::string& op_name) {
        std::lock_guard<std::mutex> lock(instance().batch_mutex);
        for (const auto& op : ops) {
            // Copy the operation into the batch queue
            instance().batch_queue.push_back({op_name, op});
        }
    }
};

// ============================================================================
// Shard: Represents a piece of tensor on a specific device
// ============================================================================

template<typename T>
struct Shard {
    std::unique_ptr<TensorBase<T>> data;       // The actual tensor data
    std::unique_ptr<TensorBase<T>> grad;       // Gradient for this shard
    DeviceInfo device;                          // Which device this shard is on
    std::size_t global_offset;                  // Starting index in the full tensor
    std::size_t num_elements;                   // Number of elements in this shard
    std::size_t shard_id;                       // Shard index
    bool is_leaf;                               // Is this a leaf tensor (user-created)
    
    Shard() : global_offset(0), num_elements(0), shard_id(0), is_leaf(true) {}
    
    Shard(std::unique_ptr<TensorBase<T>> d, const DeviceInfo& dev,
          std::size_t offset, std::size_t count, std::size_t id, bool leaf = true)
        : data(std::move(d)), device(dev), global_offset(offset),
          num_elements(count), shard_id(id), is_leaf(leaf) {}
    
    T get(std::size_t local_index) const {
        return data->get_element(local_index);
    }
    
    void set(std::size_t local_index, T value) {
        data->set_element(local_index, value);
    }
    
    T sum() const {
        return data->sum();
    }
    
    void zero_grad() {
        if (grad) {
            for (std::size_t i = 0; i < grad->total_size(); ++i) {
                grad->set_element(i, T{0});
            }
        } else {
            std::size_t shape_arr[] = {num_elements};
            // Respect the shard's backend type for gradient storage
            if (device.backend == BackendType::CPU_MMAP || device.backend == BackendType::OPENCL) {
                StreamConfig sc;
                sc.batch_size = std::max((std::size_t)1024, num_elements / 10);
                auto mmap_tensor = std::make_unique<MmapTensor<T>>(shape_arr, 1, sc, T{0});
                grad = std::move(mmap_tensor);
            } else {
                grad = std::make_unique<DenseTensor<T>>(shape_arr, 1, T{0});
            }
        }
    }

    void accumulate_grad(const TensorBase<T>* new_grad) {
        if (!grad) {
            std::size_t shape_arr[] = {num_elements};
            // Respect the shard's backend type for gradient storage
            if (device.backend == BackendType::CPU_MMAP || device.backend == BackendType::OPENCL) {
                StreamConfig sc;
                sc.batch_size = std::max((std::size_t)1024, num_elements / 10);
                auto mmap_tensor = std::make_unique<MmapTensor<T>>(shape_arr, 1, sc, T{0});
                grad = std::move(mmap_tensor);
            } else {
                grad = std::make_unique<DenseTensor<T>>(shape_arr, 1, T{0});
            }
        }
        for (std::size_t i = 0; i < new_grad->total_size() && i < grad->total_size(); ++i) {
            T existing = grad->get_element(i);
            T incoming = new_grad->get_element(i);
            grad->set_element(i, existing + incoming);
        }
    }
};

// ============================================================================
// DistributedTensor: User-Facing Class
// ============================================================================

template<typename T>
class DistributedTensor {
private:
    // Distribution state
    std::vector<Shard<T>> _shards;              // Shards across devices
    std::vector<std::size_t> _shape;            // Full tensor shape
    std::size_t _total_elements;                // Total number of elements
    bool _requires_grad;                        // Track gradients
    bool _is_leaf;                              // User-created vs operation result
    
    // Computational graph for autograd
    std::vector<DistributedTensor<T>*> _parents_raw;  // Raw pointers to parent tensors
    std::function<void()> _backward_fn;         // Function to compute gradients
    
    mutable std::mutex _mutex;
    
    // Device pool reference
    DevicePool& _device_pool;
    
    // Parallelism config (uses parallel.h infrastructure)
    ParallelConfig _parallelism_config;
    
    // ========================================================================
    // Internal: Create shard on specific device
    // ========================================================================
    
    std::unique_ptr<TensorBase<T>> create_tensor_on_device(
        const DeviceInfo& dev, std::size_t num_elements, T fill_value = T{0}) {
        
        std::size_t shape_arr[] = {num_elements};
        
        switch (dev.backend) {
            case BackendType::CPU_DENSE:
                return std::make_unique<DenseTensor<T>>(shape_arr, 1, fill_value);
                
            case BackendType::CPU_MMAP: {
                // Always use StreamTensor for CPU - lazy, batched, mmap-backed
                StreamConfig sc;
                sc.batch_size = std::max((std::size_t)1024, num_elements / 100);
                // Use system temp directory to avoid permission issues
#ifdef _WIN32
                sc.temp_dir = []() {
                    const char* tmp = std::getenv("TMP");
                    if (!tmp) tmp = std::getenv("TEMP");
                    if (!tmp) tmp = ".";
                    std::string result(tmp);
                    if (!result.empty() && result.back() != '\\' && result.back() != '/') result += '\\';
                    return result;
                }();
#else
                sc.temp_dir = "/tmp";
#endif
                return std::make_unique<MmapTensor<T>>(shape_arr, 1, sc, fill_value);
            }
                
            case BackendType::CUDA: {
                std::vector<std::size_t> shape_vec = {num_elements};
                auto* t = CudaTensor::create_empty_with_shape(shape_vec, dev.device_id);
                if (t) {
                    // Always initialize CUDA memory (even to zero) to avoid garbage
                    std::vector<float> host_data(num_elements, static_cast<float>(fill_value));
                    auto& F = get_cuda_driver_functions();
                    F.cuCtxSetCurrent(t->context());
                    F.cuMemcpyHtoD(t->buffer(), host_data.data(), num_elements * sizeof(float));
                }
                return std::unique_ptr<TensorBase<T>>(t);
            }
                
            case BackendType::OPENGL: {
                GpuConfig cfg;
                cfg.device_id = dev.device_id;
                return std::make_unique<GpuTensor<T>>(shape_arr, 1, cfg, fill_value);
            }
                
            case BackendType::OPENCL: {
                // OpenCL requires context/queue - use CPU fallback
                StreamConfig sc;
                sc.batch_size = std::max((std::size_t)1024, num_elements / 100);
                return std::make_unique<MmapTensor<T>>(shape_arr, 1, sc, fill_value);
            }
                
            default:
                return std::make_unique<DenseTensor<T>>(shape_arr, 1, fill_value);
        }
    }
    
    // ========================================================================
    // Internal: Distribute elements across shards
    // ========================================================================
    
    void distribute_data(const std::function<T(std::size_t)>& generator) {
        // Each shard generates its portion independently - no device ever sees full data
        std::vector<std::future<void>> futures;
        
        for (auto& shard : _shards) {
            futures.push_back(std::async(std::launch::async, [&shard, &generator]() {
                for (std::size_t i = 0; i < shard.num_elements; ++i) {
                    std::size_t global_idx = shard.global_offset + i;
                    shard.set(i, generator(global_idx));
                }
            }));
        }
        
        for (auto& f : futures) f.get();
    }
    
    // ========================================================================
    // Internal: Tree-reduce sum across shards (never gathers all data)
    // ========================================================================
    
    T tree_reduce_sum() const {
        if (_shards.empty()) return T{0};
        
        // Phase 1: Each shard computes its partial sum independently
        std::vector<std::future<T>> futures;
        for (const auto& shard : _shards) {
            futures.push_back(std::async(std::launch::async, [&shard]() {
                return shard.sum();
            }));
        }
        
        // Phase 2: Sum the partial results (small data, just scalars)
        T total = T{0};
        for (auto& f : futures) {
            total += f.get();
        }
        return total;
    }
    
    // ========================================================================
    // Internal: Parallel binary operation across shards
    // ========================================================================
    
    std::shared_ptr<DistributedTensor<T>> parallel_binary_op(
        const DistributedTensor<T>* other,
        std::function<std::unique_ptr<TensorBase<T>>(const TensorBase<T>*, const TensorBase<T>*)> op,
        std::function<void(const TensorBase<T>*, const TensorBase<T>*, const TensorBase<T>*, TensorBase<T>*)> grad_fn = nullptr) const {

        std::lock_guard<std::mutex> lock(_mutex);

        // Set native CUDA config once at operation start based on heuristics
        ExecutionContextManager::apply_cuda_config(
            _device_pool.devices(), _total_elements, _shards.size(),
            ExecutionContextManager::resolve_auto_mode(
                ExecutionContextManager::instance().global_mode,
                _device_pool.devices(), _total_elements, "binary_op"));

        auto result = std::shared_ptr<DistributedTensor<T>>(
            new DistributedTensor<T>(_shape, _requires_grad || other->_requires_grad, false));
        result->_shards.resize(_shards.size());
        result->_parents_raw.push_back(const_cast<DistributedTensor<T>*>(this));
        result->_parents_raw.push_back(const_cast<DistributedTensor<T>*>(other));

        // Build shard operations as lambdas
        std::vector<std::function<void()>> shard_ops;
        shard_ops.reserve(_shards.size());

        for (size_t i = 0; i < _shards.size(); ++i) {
            auto* shard_data = _shards[i].data.get();
            auto* other_shard_data = other->_shards[i].data.get();
            auto* result_shard_ptr = &result->_shards[i];
            const auto& device = _shards[i].device;
            std::size_t offset = _shards[i].global_offset;
            std::size_t count = _shards[i].num_elements;
            std::size_t shard_id = _shards[i].shard_id;

            shard_ops.push_back([shard_data, other_shard_data, result_shard_ptr, device,
                                 offset, count, shard_id, op]() {
                auto result_data = op(shard_data, other_shard_data);
                *result_shard_ptr = Shard<T>(std::move(result_data), device, offset, count, shard_id, false);
            });
        }

        // Execute using ExecutionContextManager heuristics
        ExecutionContextManager::execute_shards(
            shard_ops, _device_pool.devices(), _total_elements, "binary_op");

        // Set up backward function for autograd
        if (grad_fn && (_requires_grad || other->_requires_grad)) {
            DistributedTensor<T>* this_ptr = const_cast<DistributedTensor<T>*>(this);
            DistributedTensor<T>* other_ptr = const_cast<DistributedTensor<T>*>(other);
            DistributedTensor<T>* result_ptr = result.get();

            result->_backward_fn = [this_ptr, other_ptr, result_ptr, grad_fn]() {
                for (size_t i = 0; i < this_ptr->_shards.size(); ++i) {
                    if (!result_ptr->_shards[i].grad) continue;

                    if (this_ptr->_requires_grad) {
                        std::size_t shape_arr[] = {this_ptr->_shards[i].num_elements};
                        std::unique_ptr<TensorBase<T>> grad_input;
                        // Respect shard backend for gradient tensor
                        if (this_ptr->_shards[i].device.backend == BackendType::CPU_MMAP ||
                            this_ptr->_shards[i].device.backend == BackendType::OPENCL) {
                            StreamConfig sc;
                            sc.batch_size = std::max((std::size_t)1024, this_ptr->_shards[i].num_elements / 10);
                            grad_input = std::make_unique<MmapTensor<T>>(shape_arr, 1, sc);
                        } else {
                            grad_input = std::make_unique<DenseTensor<T>>(shape_arr, 1);
                        }

                        // Pass output gradient to grad_fn
                        grad_fn(this_ptr->_shards[i].data.get(),
                               other_ptr->_shards[i].data.get(),
                               result_ptr->_shards[i].grad.get(),
                               grad_input.get());

                        this_ptr->_shards[i].accumulate_grad(grad_input.get());
                    }
                    if (other_ptr->_requires_grad) {
                        std::size_t shape_arr[] = {other_ptr->_shards[i].num_elements};
                        std::unique_ptr<TensorBase<T>> grad_other;
                        if (other_ptr->_shards[i].device.backend == BackendType::CPU_MMAP ||
                            other_ptr->_shards[i].device.backend == BackendType::OPENCL) {
                            StreamConfig sc;
                            sc.batch_size = std::max((std::size_t)1024, other_ptr->_shards[i].num_elements / 10);
                            grad_other = std::make_unique<MmapTensor<T>>(shape_arr, 1, sc);
                        } else {
                            grad_other = std::make_unique<DenseTensor<T>>(shape_arr, 1);
                        }

                        grad_fn(other_ptr->_shards[i].data.get(),
                               this_ptr->_shards[i].data.get(),
                               result_ptr->_shards[i].grad.get(),
                               grad_other.get());

                        other_ptr->_shards[i].accumulate_grad(grad_other.get());
                    }
                }
            };
        }

        return result;
    }
    
    // ========================================================================
    // Internal: Parallel unary operation across shards
    // ========================================================================
    
    std::shared_ptr<DistributedTensor<T>> parallel_unary_op(
        std::function<std::unique_ptr<TensorBase<T>>(const TensorBase<T>*)> op,
        std::function<void(const TensorBase<T>*, const TensorBase<T>*, TensorBase<T>*)> grad_fn = nullptr) const {

        std::lock_guard<std::mutex> lock(_mutex);

        // Set native CUDA config once at operation start based on heuristics
        ExecutionContextManager::apply_cuda_config(
            _device_pool.devices(), _total_elements, _shards.size(),
            ExecutionContextManager::resolve_auto_mode(
                ExecutionContextManager::instance().global_mode,
                _device_pool.devices(), _total_elements, "unary_op"));

        auto result = std::shared_ptr<DistributedTensor<T>>(
            new DistributedTensor<T>(_shape, _requires_grad, false));
        result->_shards.resize(_shards.size());
        result->_parents_raw.push_back(const_cast<DistributedTensor<T>*>(this));

        // Build shard operations as lambdas
        std::vector<std::function<void()>> shard_ops;
        shard_ops.reserve(_shards.size());

        for (size_t i = 0; i < _shards.size(); ++i) {
            auto* shard_data = _shards[i].data.get();
            auto* result_shard_ptr = &result->_shards[i];
            const auto& device = _shards[i].device;
            std::size_t offset = _shards[i].global_offset;
            std::size_t count = _shards[i].num_elements;
            std::size_t shard_id = _shards[i].shard_id;

            shard_ops.push_back([shard_data, result_shard_ptr, device,
                                 offset, count, shard_id, op]() {
                auto result_data = op(shard_data);
                *result_shard_ptr = Shard<T>(std::move(result_data), device, offset, count, shard_id, false);
            });
        }

        // Execute using ExecutionContextManager heuristics
        ExecutionContextManager::execute_shards(
            shard_ops, _device_pool.devices(), _total_elements, "unary_op");

        // Set up backward function for autograd
        if (grad_fn && _requires_grad) {
            DistributedTensor<T>* this_ptr = const_cast<DistributedTensor<T>*>(this);
            DistributedTensor<T>* result_ptr = result.get();

            result->_backward_fn = [this_ptr, result_ptr, grad_fn]() {
                for (size_t i = 0; i < this_ptr->_shards.size(); ++i) {
                    if (!result_ptr->_shards[i].grad) continue;
                    if (this_ptr->_requires_grad) {
                        std::size_t shape_arr[] = {this_ptr->_shards[i].num_elements};
                        std::unique_ptr<TensorBase<T>> grad_input;
                        if (this_ptr->_shards[i].device.backend == BackendType::CPU_MMAP ||
                            this_ptr->_shards[i].device.backend == BackendType::OPENCL) {
                            StreamConfig sc;
                            sc.batch_size = std::max((std::size_t)1024, this_ptr->_shards[i].num_elements / 10);
                            grad_input = std::make_unique<MmapTensor<T>>(shape_arr, 1, sc);
                        } else {
                            grad_input = std::make_unique<DenseTensor<T>>(shape_arr, 1);
                        }
                        grad_fn(this_ptr->_shards[i].data.get(),
                               result_ptr->_shards[i].grad.get(),
                               grad_input.get());
                        this_ptr->_shards[i].accumulate_grad(grad_input.get());
                    }
                }
            };
        }

        return result;
    }
    
    // ========================================================================
    // Internal: Initialize shards across available devices
    // ========================================================================
    
    void initialize_shards() {
        _shards.clear();
        const auto& devices = _device_pool.devices();

        // Determine number of shards based on tensor size and device count
        // Don't create more shards than we have elements!
        int max_shards_by_size = std::max(1, (int)(_total_elements / 100));  // At least 100 elements per shard
        int max_shards = std::min((int)devices.size(), std::min(8, max_shards_by_size));
        int num_shards = std::max(1, max_shards);

        std::size_t elements_per_shard = _total_elements / num_shards;
        std::size_t remainder = _total_elements % num_shards;

        std::size_t offset = 0;
        for (int i = 0; i < num_shards; ++i) {
            std::size_t count = elements_per_shard + (i < (int)remainder ? 1 : 0);
            const auto& dev = devices[i % devices.size()];

            auto tensor = create_tensor_on_device(dev, count);
            _shards.emplace_back(std::move(tensor), dev, offset, count, i, _is_leaf);
            offset += count;
        }
    }
    
    // Private constructor
    DistributedTensor(const std::vector<std::size_t>& shape, bool requires_grad, bool is_leaf)
        : _shape(shape), _requires_grad(requires_grad), _is_leaf(is_leaf),
          _device_pool(get_device_pool()) {
        _total_elements = 1;
        for (auto dim : shape) _total_elements *= dim;
        initialize_shards();
    }
    
public:
    // ========================================================================
    // Factory Methods (User-Facing)
    // ========================================================================
    
    static std::shared_ptr<DistributedTensor<T>> zeros(
        const std::vector<std::size_t>& shape, bool requires_grad = false) {
        
        auto tensor = std::shared_ptr<DistributedTensor<T>>(
            new DistributedTensor<T>(shape, requires_grad, true));
        
        // All shards already initialized to 0 by create_tensor_on_device
        return tensor;
    }
    
    static std::shared_ptr<DistributedTensor<T>> ones(
        const std::vector<std::size_t>& shape, bool requires_grad = false) {
        
        auto tensor = std::shared_ptr<DistributedTensor<T>>(
            new DistributedTensor<T>(shape, requires_grad, true));
        
        // Set each shard to 1 in parallel
        tensor->distribute_data([](std::size_t) { return T{1}; });
        return tensor;
    }
    
    static std::shared_ptr<DistributedTensor<T>> randn(
        const std::vector<std::size_t>& shape, bool requires_grad = false) {

        auto tensor = std::shared_ptr<DistributedTensor<T>>(
            new DistributedTensor<T>(shape, requires_grad, true));

        // Thread-safe: each element gets its own RNG seeded from global index
        // No shared state across async threads (fixes data race hang)
        tensor->distribute_data([](std::size_t global_idx) {
            std::mt19937 gen(static_cast<unsigned int>(42 + (global_idx % 1000000)));
            std::normal_distribution<T> dist(0, 1);
            return dist(gen);
        });
        return tensor;
    }

    static std::shared_ptr<DistributedTensor<T>> rand(
        const std::vector<std::size_t>& shape, bool requires_grad = false) {

        auto tensor = std::shared_ptr<DistributedTensor<T>>(
            new DistributedTensor<T>(shape, requires_grad, true));

        // Thread-safe: each element gets its own RNG seeded from global index
        tensor->distribute_data([](std::size_t global_idx) {
            std::mt19937 gen(static_cast<unsigned int>(42 + (global_idx % 1000000) * 7));
            std::uniform_real_distribution<T> dist(0, 1);
            return dist(gen);
        });
        return tensor;
    }
    
    static std::shared_ptr<DistributedTensor<T>> full(
        const std::vector<std::size_t>& shape, T fill_value, bool requires_grad = false) {
        
        auto tensor = std::shared_ptr<DistributedTensor<T>>(
            new DistributedTensor<T>(shape, requires_grad, true));
        
        tensor->distribute_data([fill_value](std::size_t) { return fill_value; });
        return tensor;
    }
    
    static std::shared_ptr<DistributedTensor<T>> from_data(
        const std::vector<T>& data,
        const std::vector<std::size_t>& shape,
        bool requires_grad = false) {
        
        std::size_t total = 1;
        for (auto dim : shape) total *= dim;
        if (total != data.size()) {
            throw std::invalid_argument("Data size doesn't match shape");
        }
        
        auto tensor = std::shared_ptr<DistributedTensor<T>>(
            new DistributedTensor<T>(shape, requires_grad, true));
        
        // Distribute data to shards - each shard reads only its portion
        tensor->distribute_data([&data](std::size_t global_idx) {
            return data[global_idx];
        });
        return tensor;
    }

    // ========================================================================
    // Parquet File Import
    // ========================================================================
    // Uses Python to parse Parquet metadata, then loads binary data directly.
    // Supports float32/float64/int32/int64 columns. All data is converted to float.
    //
    // Usage:
    //   auto tensor = DistributedTensor<float>::from_parquet("data.parquet");
    //   auto tensor = DistributedTensor<float>::from_parquet("data.parquet", true); // with grad

    struct ParquetMetadata {
        std::size_t num_rows;
        std::size_t num_cols;
        std::vector<std::string> column_names;
    };

    static ParquetMetadata parse_parquet_metadata(const std::string& parquet_path) {
        std::string script_path = "parquet_to_binary.py";
        std::string bin_path = parquet_path + ".tmp.bin";

        std::string cmd = "python \"" + script_path + "\" \"" + parquet_path + "\" \"" + bin_path + "\"";
        int result = std::system(cmd.c_str());
        if (result != 0) {
            throw std::runtime_error("Failed to parse Parquet file: " + parquet_path);
        }

        FILE* f = fopen(bin_path.c_str(), "rb");
        if (!f) {
            throw std::runtime_error("Failed to open converted binary: " + bin_path);
        }

        char magic[4];
        if (fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "PQTD", 4) != 0) {
            fclose(f);
            throw std::runtime_error("Invalid binary format header");
        }

        // Read uint64 for cross-platform compatibility
        std::uint64_t rows64, cols64;
        fread(&rows64, sizeof(std::uint64_t), 1, f);
        fread(&cols64, sizeof(std::uint64_t), 1, f);

        ParquetMetadata meta;
        meta.num_rows = static_cast<std::size_t>(rows64);
        meta.num_cols = static_cast<std::size_t>(cols64);

        meta.column_names.resize(meta.num_cols);
        for (std::size_t i = 0; i < meta.num_cols; ++i) {
            std::uint32_t name_len;
            fread(&name_len, sizeof(std::uint32_t), 1, f);
            std::string name(name_len, '\0');
            fread(&name[0], 1, name_len, f);
            meta.column_names[i] = name;
        }

        fclose(f);
        std::remove(bin_path.c_str());

        return meta;
    }

    static std::shared_ptr<DistributedTensor<T>> from_parquet(
        const std::string& parquet_path,
        bool requires_grad = false) {

        std::string script_path = "parquet_to_binary.py";
        std::string bin_path = parquet_path + ".tmp.bin";

        std::string cmd = "python \"" + script_path + "\" \"" + parquet_path + "\" \"" + bin_path + "\"";
        int result = std::system(cmd.c_str());
        if (result != 0) {
            throw std::runtime_error("Failed to parse Parquet file: " + parquet_path);
        }

        FILE* f = fopen(bin_path.c_str(), "rb");
        if (!f) {
            throw std::runtime_error("Failed to open converted binary: " + bin_path);
        }

        char magic[4];
        if (fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "PQTD", 4) != 0) {
            fclose(f);
            throw std::runtime_error("Invalid binary format header");
        }

        std::uint64_t rows64, cols64;
        fread(&rows64, sizeof(std::uint64_t), 1, f);
        fread(&cols64, sizeof(std::uint64_t), 1, f);

        std::size_t num_rows = static_cast<std::size_t>(rows64);
        std::size_t num_cols = static_cast<std::size_t>(cols64);

        // Skip column names
        for (std::size_t i = 0; i < num_cols; ++i) {
            std::uint32_t name_len;
            fread(&name_len, sizeof(std::uint32_t), 1, f);
            fseek(f, name_len, SEEK_CUR);
        }

        // Read data
        std::size_t total_elements = num_rows * num_cols;
        std::vector<float> float_data(total_elements);
        std::size_t elements_read = fread(float_data.data(), sizeof(float), total_elements, f);
        fclose(f);

        if (elements_read != total_elements) {
            throw std::runtime_error("Incomplete data read from Parquet file");
        }

        std::remove(bin_path.c_str());

        std::vector<T> data(total_elements);
        for (std::size_t i = 0; i < total_elements; ++i) {
            data[i] = static_cast<T>(float_data[i]);
        }

        std::vector<std::size_t> shape = {num_rows, num_cols};
        return from_data(data, shape, requires_grad);
    }
    
    // ========================================================================
    // Element-wise Binary Operations (Distributed)
    // ========================================================================
    
    std::shared_ptr<DistributedTensor<T>> add(const DistributedTensor<T>* other) const {
        return parallel_binary_op(other,
            [](const TensorBase<T>* a, const TensorBase<T>* b) {
                return a->add(b);
            },
            [](const TensorBase<T>* a, const TensorBase<T>* b, const TensorBase<T>* grad_out, TensorBase<T>* grad_a) {
                // d/da (a + b) = 1, so grad_a = grad_out * 1 = grad_out
                for (std::size_t i = 0; i < grad_out->total_size() && i < grad_a->total_size(); ++i) {
                    grad_a->set_element(i, grad_out->get_element(i));
                }
            });
    }

    std::shared_ptr<DistributedTensor<T>> subtract(const DistributedTensor<T>* other) const {
        return parallel_binary_op(other,
            [](const TensorBase<T>* a, const TensorBase<T>* b) {
                return a->subtract(b);
            },
            [](const TensorBase<T>* a, const TensorBase<T>* b, const TensorBase<T>* grad_out, TensorBase<T>* grad_a) {
                // d/da (a - b) = 1
                for (std::size_t i = 0; i < grad_out->total_size() && i < grad_a->total_size(); ++i) {
                    grad_a->set_element(i, grad_out->get_element(i));
                }
            });
    }

    std::shared_ptr<DistributedTensor<T>> multiply(const DistributedTensor<T>* other) const {
        return parallel_binary_op(other,
            [](const TensorBase<T>* a, const TensorBase<T>* b) {
                return a->multiply(b);
            },
            [](const TensorBase<T>* a, const TensorBase<T>* b, const TensorBase<T>* grad_out, TensorBase<T>* grad_a) {
                // d/da (a * b) = b, so grad_a = grad_out * b
                for (std::size_t i = 0; i < b->total_size() && i < grad_a->total_size(); ++i) {
                    grad_a->set_element(i, grad_out->get_element(i) * b->get_element(i));
                }
            });
    }

    std::shared_ptr<DistributedTensor<T>> divide(const DistributedTensor<T>* other) const {
        return parallel_binary_op(other,
            [](const TensorBase<T>* a, const TensorBase<T>* b) {
                return a->divide(b);
            },
            [](const TensorBase<T>* a, const TensorBase<T>* b, const TensorBase<T>* grad_out, TensorBase<T>* grad_a) {
                // d/da (a / b) = 1/b, so grad_a = grad_out / b
                for (std::size_t i = 0; i < b->total_size() && i < grad_a->total_size(); ++i) {
                    T b_val = b->get_element(i);
                    if (std::abs(b_val) > 1e-10f) {
                        grad_a->set_element(i, grad_out->get_element(i) / b_val);
                    } else {
                        grad_a->set_element(i, T{0});
                    }
                }
            });
    }
    
    // ========================================================================
    // Scalar Operations (Distributed)
    // ========================================================================
    
    std::shared_ptr<DistributedTensor<T>> add_scalar(T scalar) const {
        return parallel_unary_op(
            [scalar](const TensorBase<T>* a) {
                return a->add_scalar(scalar);
            },
            [](const TensorBase<T>* a, const TensorBase<T>* grad_out, TensorBase<T>* grad_a) {
                for (std::size_t i = 0; i < grad_out->total_size() && i < grad_a->total_size(); ++i) {
                    grad_a->set_element(i, grad_out->get_element(i));
                }
            });
    }

    std::shared_ptr<DistributedTensor<T>> subtract_scalar(T scalar) const {
        return parallel_unary_op(
            [scalar](const TensorBase<T>* a) {
                return a->subtract_scalar(scalar);
            },
            [](const TensorBase<T>* a, const TensorBase<T>* grad_out, TensorBase<T>* grad_a) {
                for (std::size_t i = 0; i < grad_out->total_size() && i < grad_a->total_size(); ++i) {
                    grad_a->set_element(i, grad_out->get_element(i));
                }
            });
    }

    std::shared_ptr<DistributedTensor<T>> multiply_scalar(T scalar) const {
        return parallel_unary_op(
            [scalar](const TensorBase<T>* a) {
                return a->multiply_scalar(scalar);
            },
            [scalar](const TensorBase<T>* a, const TensorBase<T>* grad_out, TensorBase<T>* grad_a) {
                for (std::size_t i = 0; i < grad_out->total_size() && i < grad_a->total_size(); ++i) {
                    grad_a->set_element(i, grad_out->get_element(i) * scalar);
                }
            });
    }

    std::shared_ptr<DistributedTensor<T>> divide_scalar(T scalar) const {
        return parallel_unary_op(
            [scalar](const TensorBase<T>* a) {
                return a->divide_scalar(scalar);
            },
            [scalar](const TensorBase<T>* a, const TensorBase<T>* grad_out, TensorBase<T>* grad_a) {
                if (std::abs(scalar) > 1e-10f) {
                    T inv_scalar = T{1} / scalar;
                    for (std::size_t i = 0; i < grad_out->total_size() && i < grad_a->total_size(); ++i) {
                        grad_a->set_element(i, grad_out->get_element(i) * inv_scalar);
                    }
                }
            });
    }
    
    // ========================================================================
    // Unary Operations (Distributed)
    // ========================================================================
    
    std::shared_ptr<DistributedTensor<T>> negate() const {
        return parallel_unary_op(
            [](const TensorBase<T>* a) {
                return a->negate();
            },
            [](const TensorBase<T>* a, const TensorBase<T>* grad_out, TensorBase<T>* grad_a) {
                for (std::size_t i = 0; i < grad_out->total_size() && i < grad_a->total_size(); ++i) {
                    grad_a->set_element(i, -grad_out->get_element(i));
                }
            });
    }

    std::shared_ptr<DistributedTensor<T>> abs() const {
        return parallel_unary_op(
            [](const TensorBase<T>* a) {
                return a->abs();
            },
            [](const TensorBase<T>* a, const TensorBase<T>* grad_out, TensorBase<T>* grad_a) {
                for (std::size_t i = 0; i < grad_out->total_size() && i < grad_a->total_size(); ++i) {
                    T val = a->get_element(i);
                    T sign = val > T{0} ? T{1} : (val < T{0} ? T{-1} : T{0});
                    grad_a->set_element(i, grad_out->get_element(i) * sign);
                }
            });
    }

    std::shared_ptr<DistributedTensor<T>> relu() const {
        return parallel_unary_op(
            [](const TensorBase<T>* a) {
                return a->clamp(T{0}, std::numeric_limits<T>::max());
            },
            [](const TensorBase<T>* a, const TensorBase<T>* grad_out, TensorBase<T>* grad_a) {
                for (std::size_t i = 0; i < grad_out->total_size() && i < grad_a->total_size(); ++i) {
                    T val = a->get_element(i);
                    T mask = val > T{0} ? T{1} : T{0};
                    grad_a->set_element(i, grad_out->get_element(i) * mask);
                }
            });
    }
    
    // ========================================================================
    // Reductions (Tree-Reduce - Never Gathers All Data)
    // ========================================================================
    
    std::shared_ptr<DistributedTensor<T>> sum() const {
        // Tree-reduce: each shard computes partial sum, then we sum the partials
        T total = tree_reduce_sum();
        
        // Result is a scalar tensor on the first device
        std::vector<std::size_t> scalar_shape = {1};
        auto result = std::shared_ptr<DistributedTensor<T>>(
            new DistributedTensor<T>(scalar_shape, _requires_grad, false));
        result->_parents_raw.push_back(const_cast<DistributedTensor<T>*>(this));
        
        // Only the first shard holds the result
        result->_shards[0].data->set_element(0, total);
        
        // Set up backward function for autograd
        if (_requires_grad) {
            DistributedTensor<T>* this_ptr = const_cast<DistributedTensor<T>*>(this);
            DistributedTensor<T>* result_ptr = result.get();

            result->_backward_fn = [this_ptr, result_ptr]() {
                // d/dx sum(x) = 1 for all elements
                // Gradient flows back as ones to each shard
                if (!result_ptr->_shards.empty() && result_ptr->_shards[0].grad) {
                    T grad_val = result_ptr->_shards[0].grad->get_element(0);
                    for (auto& shard : this_ptr->_shards) {
                        std::size_t shape_arr[] = {shard.num_elements};
                        std::unique_ptr<TensorBase<T>> grad_input;
                        if (shard.device.backend == BackendType::CPU_MMAP ||
                            shard.device.backend == BackendType::OPENCL) {
                            StreamConfig sc;
                            sc.batch_size = std::max((std::size_t)1024, shard.num_elements / 10);
                            grad_input = std::make_unique<MmapTensor<T>>(shape_arr, 1, sc, grad_val);
                        } else {
                            grad_input = std::make_unique<DenseTensor<T>>(shape_arr, 1, grad_val);
                        }
                        shard.accumulate_grad(grad_input.get());
                    }
                }
            };
        }

        return result;
    }

    std::shared_ptr<DistributedTensor<T>> mean() const {
        T total = tree_reduce_sum();
        T mean_val = total / static_cast<T>(_total_elements);

        std::vector<std::size_t> scalar_shape = {1};
        auto result = std::shared_ptr<DistributedTensor<T>>(
            new DistributedTensor<T>(scalar_shape, _requires_grad, false));
        result->_parents_raw.push_back(const_cast<DistributedTensor<T>*>(this));

        result->_shards[0].data->set_element(0, mean_val);

        // Set up backward function for autograd
        if (_requires_grad) {
            DistributedTensor<T>* this_ptr = const_cast<DistributedTensor<T>*>(this);
            auto* result_ptr = result.get();
            T inv_n = T{1} / static_cast<T>(_total_elements);

            result->_backward_fn = [this_ptr, result_ptr, inv_n]() {
                // d/dx mean(x) = 1/N for all elements
                if (!result_ptr->_shards.empty() && result_ptr->_shards[0].grad) {
                    T grad_val = result_ptr->_shards[0].grad->get_element(0);
                    for (auto& shard : this_ptr->_shards) {
                        std::size_t shape_arr[] = {shard.num_elements};
                        std::unique_ptr<TensorBase<T>> grad_input;
                        if (shard.device.backend == BackendType::CPU_MMAP ||
                            shard.device.backend == BackendType::OPENCL) {
                            StreamConfig sc;
                            sc.batch_size = std::max((std::size_t)1024, shard.num_elements / 10);
                            grad_input = std::make_unique<MmapTensor<T>>(shape_arr, 1, sc, grad_val * inv_n);
                        } else {
                            grad_input = std::make_unique<DenseTensor<T>>(shape_arr, 1, grad_val * inv_n);
                        }
                        shard.accumulate_grad(grad_input.get());
                    }
                }
            };
        }
        
        return result;
    }
    
    std::shared_ptr<DistributedTensor<T>> max() const {
        if (_shards.empty()) {
            std::vector<std::size_t> scalar_shape = {1};
            return std::shared_ptr<DistributedTensor<T>>(
                new DistributedTensor<T>(scalar_shape, _requires_grad, false));
        }
        
        // Each shard finds its local max
        std::vector<std::future<T>> futures;
        for (const auto& shard : _shards) {
            futures.push_back(std::async(std::launch::async, [&shard]() {
                return shard.data->max();
            }));
        }
        
        // Reduce: find global max from local maxes
        T global_max = futures[0].get();
        for (size_t i = 1; i < futures.size(); ++i) {
            T local_max = futures[i].get();
            if (local_max > global_max) global_max = local_max;
        }
        
        std::vector<std::size_t> scalar_shape = {1};
        auto result = std::shared_ptr<DistributedTensor<T>>(
            new DistributedTensor<T>(scalar_shape, _requires_grad, false));
        result->_shards[0].data->set_element(0, global_max);
        return result;
    }
    
    std::shared_ptr<DistributedTensor<T>> min() const {
        if (_shards.empty()) {
            std::vector<std::size_t> scalar_shape = {1};
            return std::shared_ptr<DistributedTensor<T>>(
                new DistributedTensor<T>(scalar_shape, _requires_grad, false));
        }
        
        std::vector<std::future<T>> futures;
        for (const auto& shard : _shards) {
            futures.push_back(std::async(std::launch::async, [&shard]() {
                return shard.data->min();
            }));
        }
        
        T global_min = futures[0].get();
        for (size_t i = 1; i < futures.size(); ++i) {
            T local_min = futures[i].get();
            if (local_min < global_min) global_min = local_min;
        }
        
        std::vector<std::size_t> scalar_shape = {1};
        auto result = std::shared_ptr<DistributedTensor<T>>(
            new DistributedTensor<T>(scalar_shape, _requires_grad, false));
        result->_shards[0].data->set_element(0, global_min);
        return result;
    }
    
    // ========================================================================
    // Reshape (Distributed - each shard reshapes independently)
    // ========================================================================
    
    std::shared_ptr<DistributedTensor<T>> reshape(const std::vector<std::size_t>& new_shape) const {
        std::size_t new_total = 1;
        for (auto dim : new_shape) new_total *= dim;
        if (new_total != _total_elements) {
            throw std::invalid_argument("Reshape must preserve total elements");
        }
        
        auto result = std::shared_ptr<DistributedTensor<T>>(
            new DistributedTensor<T>(new_shape, _requires_grad, false));
        result->_parents_raw.push_back(const_cast<DistributedTensor<T>*>(this));
        
        // Each shard reshapes independently - no data movement between devices
        std::vector<std::future<void>> futures;
        for (size_t i = 0; i < _shards.size(); ++i) {
            auto* shard_data = _shards[i].data.get();
            auto* result_shard_ptr = &result->_shards[i];
            const auto& device = _shards[i].device;
            std::size_t offset = _shards[i].global_offset;
            std::size_t count = _shards[i].num_elements;
            std::size_t shard_id = _shards[i].shard_id;
            
            futures.push_back(std::async(std::launch::async,
                [shard_data, result_shard_ptr, &device, offset, count, shard_id]() {
                    std::size_t shape_arr[] = {count};
                    auto reshaped = shard_data->reshape(shape_arr, 1);
                    *result_shard_ptr = Shard<T>(std::move(reshaped), device, offset, count, shard_id, false);
                }));
        }
        
        for (auto& f : futures) f.get();
        return result;
    }
    
    // ========================================================================
    // Element Access (Reads Only From Relevant Shard)
    // ========================================================================
    
    T get_element(std::size_t global_index) const {
        if (global_index >= _total_elements) {
            throw std::out_of_range("Index out of range");
        }
        
        // Find which shard contains this index - only read from that shard
        for (const auto& shard : _shards) {
            if (global_index >= shard.global_offset &&
                global_index < shard.global_offset + shard.num_elements) {
                std::size_t local_index = global_index - shard.global_offset;
                return shard.get(local_index);
            }
        }
        return T{0};
    }
    
    void set_element(std::size_t global_index, T value) {
        if (global_index >= _total_elements) {
            throw std::out_of_range("Index out of range");
        }
        
        std::lock_guard<std::mutex> lock(_mutex);
        
        // Find and update only the relevant shard
        for (auto& shard : _shards) {
            if (global_index >= shard.global_offset &&
                global_index < shard.global_offset + shard.num_elements) {
                std::size_t local_index = global_index - shard.global_offset;
                shard.set(local_index, value);
                return;
            }
        }
    }
    
    // ========================================================================
    // Autograd Integration (Distributed Backward Pass)
    // ========================================================================
    
    void backward() {
        std::lock_guard<std::mutex> lock(_mutex);

        // Build topological order of computational graph
        std::vector<DistributedTensor<T>*> topo_order;
        std::unordered_map<DistributedTensor<T>*, bool> visited;
        build_topological_order(const_cast<DistributedTensor<T>*>(this), topo_order, visited);

        // Initialize gradient for this tensor (1.0 for scalar output)
        if (_shape.size() == 1 && _shape[0] == 1) {
            // Scalar output: set grad to 1.0 on first shard, 0.0 on others
            // Then the backward_fn of sum/mean will broadcast to all shards
            for (size_t i = 0; i < _shards.size(); ++i) {
                if (!_shards[i].grad) {
                    std::size_t shape_arr[] = {_shards[i].num_elements};
                    if (_shards[i].device.backend == BackendType::CPU_MMAP ||
                        _shards[i].device.backend == BackendType::OPENCL) {
                        StreamConfig sc;
                        sc.batch_size = std::max((std::size_t)1024, _shards[i].num_elements / 10);
                        auto mmap_tensor = std::make_unique<MmapTensor<T>>(shape_arr, 1, sc, T{0});
                        _shards[i].grad = std::move(mmap_tensor);
                    } else {
                        _shards[i].grad = std::make_unique<DenseTensor<T>>(shape_arr, 1, T{0});
                    }
                }
            }
            // Only the first shard gets 1.0 (it's a scalar result)
            _shards[0].grad->set_element(0, T{1});
        }

        // Traverse in reverse topological order
        for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
            auto* node = *it;

            // Skip leaf nodes (they don't have backward functions)
            if (node->_is_leaf) continue;

            // Check if any shard has a gradient to propagate
            bool has_grad = false;
            for (const auto& shard : node->_shards) {
                if (shard.grad) {
                    has_grad = true;
                    break;
                }
            }
            if (!has_grad) continue;

            // Apply backward function - this propagates gradients to parent shards
            if (node->_backward_fn) {
                node->_backward_fn();
            }
        }
    }
    
    // Zero gradients
    void zero_grad() {
        std::lock_guard<std::mutex> lock(_mutex);
        for (auto& shard : _shards) {
            shard.zero_grad();
        }
    }
    
    // Get gradient (returns first shard's gradient for inspection)
    TensorBase<T>* grad() const {
        if (!_shards.empty() && _shards[0].grad) {
            return _shards[0].grad.get();
        }
        return nullptr;
    }
    
    // ========================================================================
    // Accessors
    // ========================================================================
    
    const std::vector<std::size_t>& shape() const { return _shape; }
    std::size_t total_elements() const { return _total_elements; }
    std::size_t num_shards() const { return _shards.size(); }
    bool requires_grad() const { return _requires_grad; }
    bool is_leaf() const { return _is_leaf; }
    
    std::string distribution_info() const {
        std::ostringstream oss;
        oss << "DistributedTensor shape=[";
        for (size_t i = 0; i < _shape.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << _shape[i];
        }
        oss << "], shards=" << _shards.size() << ", devices=[";
        for (size_t i = 0; i < _shards.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << _shards[i].device.name;
        }
        oss << "]";
        return oss.str();
    }
    
    // Get shard info
    std::string shard_info(std::size_t shard_id) const {
        if (shard_id >= _shards.size()) return "Invalid shard";
        const auto& shard = _shards[shard_id];
        std::ostringstream oss;
        oss << "Shard " << shard_id << ": " << shard.num_elements << " elements on "
            << shard.device.name << " (offset " << shard.global_offset << ")";
        return oss.str();
    }
    
    // Refresh device pool and redistribute if needed
    void refresh_devices() {
        std::lock_guard<std::mutex> lock(_mutex);
        _device_pool.refresh();
        
        // Save current data
        std::vector<T> current_data(_total_elements);
        for (size_t i = 0; i < _total_elements; ++i) {
            current_data[i] = get_element(i);
        }
        
        // Reinitialize shards with new device configuration
        initialize_shards();
        
        // Redistribute data
        distribute_data([&current_data](std::size_t global_idx) {
            return current_data[global_idx];
        });
    }
    
    // Get number of shards
    int shard_count() const {
        return _shards.size();
    }
    
    // Get device for a shard
    std::string shard_device(std::size_t shard_id) const {
        if (shard_id >= _shards.size()) return "Invalid";
        return _shards[shard_id].device.name;
    }

    // ========================================================================
    // Execution Mode Control (User-Facing API)
    // ========================================================================

    // Set global execution mode for all operations
    static void set_execution_mode(ExecutionMode mode) {
        ExecutionContextManager::instance().set_mode(mode);
    }

    // Override execution mode for a specific backend
    static void set_backend_mode(BackendType backend, ExecutionMode mode) {
        ExecutionContextManager::instance().set_device_mode(backend, mode);
    }

    // Get current global execution mode
    static ExecutionMode get_execution_mode() {
        return ExecutionContextManager::instance().global_mode;
    }

    // Get mode name as string
    static std::string mode_to_string(ExecutionMode mode) {
        switch (mode) {
            case ExecutionMode::AUTO: return "AUTO";
            case ExecutionMode::SEQUENTIAL: return "SEQUENTIAL";
            case ExecutionMode::PARALLEL: return "PARALLEL";
            case ExecutionMode::HYBRID: return "HYBRID";
            case ExecutionMode::DEVICE_LOCAL: return "DEVICE_LOCAL";
            case ExecutionMode::BATCHED: return "BATCHED";
            default: return "UNKNOWN";
        }
    }

    // Flush all pending batched operations
    static void flush_batches() {
        ExecutionContextManager::instance().flush_batches();
    }

    // Get execution mode summary
    static std::string execution_mode_summary() {
        std::ostringstream oss;
        auto& mgr = ExecutionContextManager::instance();
        oss << "Global mode: " << mode_to_string(mgr.global_mode) << "\n";
        if (!mgr.device_mode_overrides.empty()) {
            oss << "Backend overrides:\n";
            for (const auto& [backend, mode] : mgr.device_mode_overrides) {
                oss << "  " << backend_type_name(backend) << ": " << mode_to_string(mode) << "\n";
            }
        }
        return oss.str();
    }

    // Get backend type name
    static std::string backend_type_name(BackendType backend) {
        switch (backend) {
            case BackendType::CPU_DENSE: return "CPU_DENSE";
            case BackendType::CPU_MMAP: return "CPU_MMAP";
            case BackendType::CUDA: return "CUDA";
            case BackendType::OPENGL: return "OPENGL";
            case BackendType::OPENCL: return "OPENCL";
            default: return "UNKNOWN";
        }
    }

    // ========================================================================
    // Native CUDA Configuration (direct access to CudaContextManager)
    // ========================================================================

    // Set native CUDA execution config
    static void set_cuda_config(const CudaExecutionConfig& cfg) {
        CudaContextManager::instance().set_config(cfg);
    }

    // Get native CUDA execution config
    static CudaExecutionConfig get_cuda_config() {
        return CudaContextManager::instance().get_config();
    }

    // Get native CUDA config summary
    static std::string cuda_config_summary() {
        return CudaContextManager::instance().config_summary();
    }

    // Convenience: set CUDA to preset mode
    static void set_cuda_preset(const std::string& preset) {
        if (preset == "maximum_performance") {
            CudaContextManager::instance().set_config(CudaExecutionConfig::maximum_performance());
        } else if (preset == "debug") {
            CudaContextManager::instance().set_config(CudaExecutionConfig::debug_mode());
        } else if (preset == "low_memory") {
            CudaContextManager::instance().set_config(CudaExecutionConfig::low_memory());
        } else if (preset == "safe") {
            CudaContextManager::instance().set_config(CudaExecutionConfig::safe_mode());
        }
    }

    // ========================================================================
    // LLM Training Operations
    // ========================================================================

    // Matrix multiplication: this @ other^T for [M,K] @ [N,K]^T = [M,N]
    // Each shard computes its portion independently
    std::shared_ptr<DistributedTensor<T>> matmul(const DistributedTensor<T>* other) const {
        std::lock_guard<std::mutex> lock(_mutex);

        // Validate shapes: this is [M,K], other is [N,K], result is [M,N]
        if (_shape.size() != 2 || other->_shape.size() != 2) {
            throw std::invalid_argument("matmul requires 2D tensors");
        }
        if (_shape[1] != other->_shape[1]) {
            throw std::invalid_argument("matmul shape mismatch: K dimensions must match");
        }
        if (_shards.size() != other->_shards.size()) {
            throw std::invalid_argument("matmul requires same number of shards");
        }

        std::size_t M = _shape[0];
        std::size_t N = other->_shape[0];
        std::size_t K = _shape[1];
        std::size_t result_size = M * N;

        auto result = std::shared_ptr<DistributedTensor<T>>(
            new DistributedTensor<T>({M, N}, _requires_grad || other->_requires_grad, false));
        result->_shards.resize(_shards.size());
        result->_parents_raw.push_back(const_cast<DistributedTensor<T>*>(this));
        result->_parents_raw.push_back(const_cast<DistributedTensor<T>*>(other));

        // Each shard computes matmul on its portion
        std::vector<std::function<void()>> shard_ops;
        shard_ops.reserve(_shards.size());

        for (size_t i = 0; i < _shards.size(); ++i) {
            auto* shard_a = _shards[i].data.get();
            auto* shard_b = other->_shards[i].data.get();
            auto* result_shard_ptr = &result->_shards[i];
            const auto& device = _shards[i].device;
            std::size_t offset = _shards[i].global_offset;
            std::size_t count = _shards[i].num_elements;
            std::size_t shard_id = _shards[i].shard_id;

            shard_ops.push_back([shard_a, shard_b, result_shard_ptr, device,
                                 offset, count, shard_id, M, N, K, num_shards = _shards.size()]() {
                // Gather full tensors for this shard's portion
                std::vector<T> data_a(shard_a->total_size());
                std::vector<T> data_b(shard_b->total_size());
                for (size_t j = 0; j < shard_a->total_size(); ++j) data_a[j] = shard_a->get_element(j);
                for (size_t j = 0; j < shard_b->total_size(); ++j) data_b[j] = shard_b->get_element(j);

                // Compute partial matmul: this shard handles a portion of rows
                std::size_t rows_per_shard = M / num_shards;
                std::size_t row_start = shard_id * rows_per_shard;
                std::size_t row_end = (shard_id + 1 == num_shards) ? M : (shard_id + 1) * rows_per_shard;
                std::size_t result_count = (row_end - row_start) * N;

                std::vector<T> result_data(result_count, T{0});
                for (std::size_t r = row_start; r < row_end; ++r) {
                    for (std::size_t n = 0; n < N; ++n) {
                        T sum = T{0};
                        for (std::size_t k = 0; k < K; ++k) {
                            sum += data_a[r * K + k] * data_b[n * K + k];
                        }
                        result_data[(r - row_start) * N + n] = sum;
                    }
                }

                std::size_t shape_arr[] = {result_count};
                auto result_tensor = std::make_unique<DenseTensor<T>>(shape_arr, 1);
                for (size_t j = 0; j < result_count; ++j) {
                    result_tensor->set_element(j, result_data[j]);
                }
                *result_shard_ptr = Shard<T>(std::move(result_tensor), device, offset, result_count, shard_id, false);
            });
        }

        ExecutionContextManager::execute_shards(
            shard_ops, _device_pool.devices(), result_size, "matmul");

        return result;
    }

    // Softmax: exp(x - max) / sum(exp(x - max))
    // Uses tree-reduce for max and sum across all shards
    std::shared_ptr<DistributedTensor<T>> softmax() const {
        std::lock_guard<std::mutex> lock(_mutex);

        // Phase 1: Find global max across all shards
        T global_max = _shards[0].data->get_element(0);
        for (const auto& shard : _shards) {
            for (std::size_t i = 0; i < shard.num_elements; ++i) {
                T v = shard.data->get_element(i);
                if (v > global_max) global_max = v;
            }
        }

        // Phase 2: Compute exp(x - max) and sum across all shards
        T global_sum = T{0};
        for (const auto& shard : _shards) {
            for (std::size_t i = 0; i < shard.num_elements; ++i) {
                T v = shard.data->get_element(i);
                global_sum += std::exp(static_cast<float>(v - global_max));
            }
        }

        // Phase 3: Normalize each shard
        auto result = std::shared_ptr<DistributedTensor<T>>(
            new DistributedTensor<T>(_shape, _requires_grad, false));
        result->_shards.resize(_shards.size());
        result->_parents_raw.push_back(const_cast<DistributedTensor<T>*>(this));

        std::vector<std::function<void()>> shard_ops;
        shard_ops.reserve(_shards.size());

        for (size_t i = 0; i < _shards.size(); ++i) {
            auto* shard_data = _shards[i].data.get();
            auto* result_shard_ptr = &result->_shards[i];
            const auto& device = _shards[i].device;
            std::size_t offset = _shards[i].global_offset;
            std::size_t count = _shards[i].num_elements;
            std::size_t shard_id = _shards[i].shard_id;

            shard_ops.push_back([shard_data, result_shard_ptr, device, offset, count, shard_id,
                                 global_max, global_sum]() {
                std::size_t shape_arr[] = {count};
                auto result_tensor = std::make_unique<DenseTensor<T>>(shape_arr, 1);
                for (std::size_t i = 0; i < count; ++i) {
                    T v = shard_data->get_element(i);
                    T e = std::exp(static_cast<float>(v - global_max));
                    result_tensor->set_element(i, e / global_sum);
                }
                *result_shard_ptr = Shard<T>(std::move(result_tensor), device, offset, count, shard_id, false);
            });
        }

        ExecutionContextManager::execute_shards(
            shard_ops, _device_pool.devices(), _total_elements, "softmax");

        return result;
    }

    // Layer normalization: (x - mean) / sqrt(var + eps)
    // Computes mean/var globally across all shards
    std::shared_ptr<DistributedTensor<T>> layer_norm(T eps = T{1e-5}) const {
        std::lock_guard<std::mutex> lock(_mutex);

        // Phase 1: Compute global mean
        T global_mean = T{0};
        for (const auto& shard : _shards) {
            for (std::size_t i = 0; i < shard.num_elements; ++i) {
                global_mean += shard.data->get_element(i);
            }
        }
        global_mean /= static_cast<T>(_total_elements);

        // Phase 2: Compute global variance
        T global_var = T{0};
        for (const auto& shard : _shards) {
            for (std::size_t i = 0; i < shard.num_elements; ++i) {
                T diff = shard.data->get_element(i) - global_mean;
                global_var += diff * diff;
            }
        }
        global_var /= static_cast<T>(_total_elements);

        T std_inv = T{1} / std::sqrt(static_cast<float>(global_var + eps));

        // Phase 3: Normalize each shard
        auto result = std::shared_ptr<DistributedTensor<T>>(
            new DistributedTensor<T>(_shape, _requires_grad, false));
        result->_shards.resize(_shards.size());
        result->_parents_raw.push_back(const_cast<DistributedTensor<T>*>(this));

        std::vector<std::function<void()>> shard_ops;
        shard_ops.reserve(_shards.size());

        for (size_t i = 0; i < _shards.size(); ++i) {
            auto* shard_data = _shards[i].data.get();
            auto* result_shard_ptr = &result->_shards[i];
            const auto& device = _shards[i].device;
            std::size_t offset = _shards[i].global_offset;
            std::size_t count = _shards[i].num_elements;
            std::size_t shard_id = _shards[i].shard_id;

            shard_ops.push_back([shard_data, result_shard_ptr, device, offset, count, shard_id,
                                 global_mean, std_inv]() {
                std::size_t shape_arr[] = {count};
                auto result_tensor = std::make_unique<DenseTensor<T>>(shape_arr, 1);
                for (std::size_t i = 0; i < count; ++i) {
                    T v = shard_data->get_element(i);
                    result_tensor->set_element(i, (v - global_mean) * std_inv);
                }
                *result_shard_ptr = Shard<T>(std::move(result_tensor), device, offset, count, shard_id, false);
            });
        }

        ExecutionContextManager::execute_shards(
            shard_ops, _device_pool.devices(), _total_elements, "layer_norm");

        return result;
    }

    // GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    std::shared_ptr<DistributedTensor<T>> gelu() const {
        return parallel_unary_op(
            [](const TensorBase<T>* x) {
                std::size_t n = x->total_size();
                std::size_t shape_arr[] = {n};
                auto result = std::make_unique<DenseTensor<T>>(shape_arr, 1);
                T sqrt_2_pi = std::sqrt(2.0f / 3.14159265358979f);
                for (std::size_t i = 0; i < n; ++i) {
                    T xi = x->get_element(i);
                    T x3 = xi * xi * xi;
                    T inner = sqrt_2_pi * (xi + T{0.044715} * x3);
                    T gelu = T{0.5} * xi * (T{1} + std::tanh(static_cast<float>(inner)));
                    result->set_element(i, gelu);
                }
                return result;
            });
    }

    // SiLU/Swish activation: x * sigmoid(x)
    std::shared_ptr<DistributedTensor<T>> silu() const {
        return parallel_unary_op(
            [](const TensorBase<T>* x) {
                std::size_t n = x->total_size();
                std::size_t shape_arr[] = {n};
                auto result = std::make_unique<DenseTensor<T>>(shape_arr, 1);
                for (std::size_t i = 0; i < n; ++i) {
                    T xi = x->get_element(i);
                    T sig = T{1} / (T{1} + std::exp(static_cast<float>(-xi)));
                    result->set_element(i, xi * sig);
                }
                return result;
            });
    }

    // Cross-entropy loss: -sum(labels * log(softmax(logits)))
    // Returns scalar loss tensor
    std::shared_ptr<DistributedTensor<T>> cross_entropy_loss(
        const DistributedTensor<T>* labels) const {
        std::lock_guard<std::mutex> lock(_mutex);

        if (_shards.size() != labels->_shards.size()) {
            throw std::invalid_argument("cross_entropy_loss requires same number of shards");
        }

        std::vector<std::future<T>> futures;
        for (size_t i = 0; i < _shards.size(); ++i) {
            auto* logits = _shards[i].data.get();
            auto* labs = labels->_shards[i].data.get();

            futures.push_back(std::async(std::launch::async, [logits, labs]() {
                std::size_t n = logits->total_size();
                if (n == 0) return T{0};

                // Compute softmax
                T max_val = logits->get_element(0);
                for (std::size_t i = 1; i < n; ++i) {
                    T v = logits->get_element(i);
                    if (v > max_val) max_val = v;
                }

                T sum_exp = T{0};
                std::vector<T> softmax_vals(n);
                for (std::size_t i = 0; i < n; ++i) {
                    T e = std::exp(static_cast<float>(logits->get_element(i) - max_val));
                    softmax_vals[i] = e;
                    sum_exp += e;
                }

                // Compute cross-entropy
                T loss = T{0};
                for (std::size_t i = 0; i < n; ++i) {
                    T p = softmax_vals[i] / sum_exp;
                    T label = labs->get_element(i);
                    if (p > T{1e-12}) {
                        loss -= label * std::log(static_cast<float>(p));
                    }
                }
                return loss;
            }));
        }

        T total_loss = T{0};
        for (auto& f : futures) total_loss += f.get();

        std::vector<std::size_t> scalar_shape = {1};
        auto result = std::shared_ptr<DistributedTensor<T>>(
            new DistributedTensor<T>(scalar_shape, _requires_grad, false));
        result->_shards[0].data->set_element(0, total_loss);
        result->_parents_raw.push_back(const_cast<DistributedTensor<T>*>(this));
        result->_parents_raw.push_back(const_cast<DistributedTensor<T>*>(labels));

        return result;
    }

    // SGD optimizer step: param = param - lr * grad
    void sgd_step(T learning_rate) {
        for (auto& shard : _shards) {
            if (shard.grad) {
                std::size_t n = shard.num_elements;
                for (std::size_t i = 0; i < n; ++i) {
                    T param = shard.data->get_element(i);
                    T grad = shard.grad->get_element(i);
                    shard.data->set_element(i, param - learning_rate * grad);
                }
            }
        }
    }

    // Adam optimizer step
    void adam_step(T learning_rate, T beta1 = T{0.9}, T beta2 = T{0.999}, T eps = T{1e-8}) {
        // Initialize Adam state on first call
        static std::map<void*, std::pair<std::vector<T>, std::vector<T>>> adam_state;

        for (auto& shard : _shards) {
            if (!shard.grad) continue;

            void* key = shard.data.get();
            std::size_t n = shard.num_elements;

            if (adam_state.find(key) == adam_state.end()) {
                adam_state[key] = {std::vector<T>(n, T{0}), std::vector<T>(n, T{0})};
            }

            auto& [m, v] = adam_state[key];
            for (std::size_t i = 0; i < n; ++i) {
                T g = shard.grad->get_element(i);
                m[i] = beta1 * m[i] + (T{1} - beta1) * g;
                v[i] = beta2 * v[i] + (T{1} - beta2) * g * g;

                T m_hat = m[i] / (T{1} - beta1);
                T v_hat = v[i] / (T{1} - beta2);

                T param = shard.data->get_element(i);
                shard.data->set_element(i, param - learning_rate * m_hat / (std::sqrt(static_cast<float>(v_hat)) + eps));
            }
        }
    }

    // All-reduce sum: sum values across all shards, broadcast result to all shards
    std::shared_ptr<DistributedTensor<T>> all_reduce_sum() const {
        std::lock_guard<std::mutex> lock(_mutex);

        // Compute sum of ALL elements across all shards
        T total_sum = T{0};
        for (const auto& shard : _shards) {
            for (std::size_t i = 0; i < shard.num_elements; ++i) {
                total_sum += shard.data->get_element(i);
            }
        }

        // Broadcast the total sum to every element in every shard
        auto result = std::shared_ptr<DistributedTensor<T>>(
            new DistributedTensor<T>(_shape, _requires_grad, false));
        result->_shards.resize(_shards.size());

        for (size_t i = 0; i < _shards.size(); ++i) {
            std::size_t shape_arr[] = {_shards[i].num_elements};
            auto data_tensor = std::make_unique<DenseTensor<T>>(shape_arr, 1);
            for (std::size_t j = 0; j < _shards[i].num_elements; ++j) {
                data_tensor->set_element(j, total_sum);
            }
            result->_shards[i] = Shard<T>(std::move(data_tensor), _shards[i].device,
                                          _shards[i].global_offset, _shards[i].num_elements,
                                          _shards[i].shard_id, false);
        }

        return result;
    }

    // All-gather: write shard data into a memory-mapped StreamTensor.
    // NEVER loads all data into RAM — result is backed by an mmap file.
    // Elements are accessed via get_element() which reads from the mmap in batches.
    std::unique_ptr<StreamTensor<T>> all_gather_mmap() const {
        std::size_t shape_arr[] = {_total_elements};
        StreamConfig sc;
        sc.batch_size = std::max((std::size_t)65536, _total_elements / 100);

        auto result = std::make_unique<StreamTensor<T>>(shape_arr, 1, sc);

        // Write each shard's data into the mmap file in batches
        for (const auto& shard : _shards) {
            for (std::size_t i = 0; i < shard.num_elements; ++i) {
                result->set_element(shard.global_offset + i, shard.data->get_element(i));
            }
        }
        return result;
    }

    // All-gather to a std::vector (for small tensors only / debugging).
    // WARNING: loads ALL data into RAM. Use all_gather_mmap() for large tensors.
    std::vector<T> all_gather() const {
        std::vector<T> result;
        result.reserve(_total_elements);
        for (const auto& shard : _shards) {
            for (std::size_t i = 0; i < shard.num_elements; ++i) {
                result.push_back(shard.data->get_element(i));
            }
        }
        return result;
    }

    // All-gather to a new DistributedTensor — data goes through mmap, not RAM.
    // Result is resharded across all available devices.
    std::shared_ptr<DistributedTensor<T>> all_gather_distributed() const {
        // Step 1: Write to mmap file (no full RAM load)
        auto mmap_tensor = all_gather_mmap();

        // Step 2: Create new DistributedTensor and read from mmap in batches
        auto result = std::shared_ptr<DistributedTensor<T>>(
            new DistributedTensor<T>(_shape, _requires_grad, false));

        // Read from mmap into each shard's backend in batches
        for (auto& shard : result->_shards) {
            for (std::size_t i = 0; i < shard.num_elements; ++i) {
                shard.set(i, mmap_tensor->get_element(shard.global_offset + i));
            }
        }
        return result;
    }

private:
    // Build topological order of computational graph
    void build_topological_order(
        DistributedTensor<T>* node,
        std::vector<DistributedTensor<T>*>& order,
        std::unordered_map<DistributedTensor<T>*, bool>& visited) {
        
        if (visited[node]) return;
        visited[node] = true;
        
        for (auto* parent : node->_parents_raw) {
            build_topological_order(parent, order, visited);
        }
        
        order.push_back(node);
    }
};
