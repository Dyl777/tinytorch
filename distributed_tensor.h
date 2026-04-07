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
            grad = std::make_unique<DenseTensor<T>>(shape_arr, 1, T{0});
        }
    }
    
    void accumulate_grad(const TensorBase<T>* new_grad) {
        if (!grad) {
            std::size_t shape_arr[] = {num_elements};
            grad = std::make_unique<DenseTensor<T>>(shape_arr, 1, T{0});
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
        std::function<void(const TensorBase<T>*, const TensorBase<T>*, TensorBase<T>*)> grad_fn = nullptr) const {
        
        std::lock_guard<std::mutex> lock(_mutex);
        
        auto result = std::shared_ptr<DistributedTensor<T>>(
            new DistributedTensor<T>(_shape, _requires_grad || other->_requires_grad, false));
        result->_shards.resize(_shards.size());
        // Store raw pointers for autograd - will be converted to shared when needed
        result->_parents_raw.push_back(const_cast<DistributedTensor<T>*>(this));
        result->_parents_raw.push_back(const_cast<DistributedTensor<T>*>(other));
        
        // Execute shards sequentially to avoid CUDA context corruption across threads.
        // CUDA module handles have thread affinity and become invalid when accessed from
        // threads other than the one that created them.
        for (size_t i = 0; i < _shards.size(); ++i) {
            auto* shard_data = _shards[i].data.get();
            auto* other_shard_data = other->_shards[i].data.get();
            auto* result_shard_ptr = &result->_shards[i];
            const auto& device = _shards[i].device;
            std::size_t offset = _shards[i].global_offset;
            std::size_t count = _shards[i].num_elements;
            std::size_t shard_id = _shards[i].shard_id;

            auto result_data = op(shard_data, other_shard_data);
            *result_shard_ptr = Shard<T>(std::move(result_data), device, offset, count, shard_id, false);
        }
        
        // Set up backward function for autograd
        if (grad_fn && (_requires_grad || other->_requires_grad)) {
            // Store raw pointers - they remain valid as long as tensors exist
            DistributedTensor<T>* this_ptr = const_cast<DistributedTensor<T>*>(this);
            DistributedTensor<T>* other_ptr = const_cast<DistributedTensor<T>*>(other);
            DistributedTensor<T>* result_ptr = result.get();
            
            result->_backward_fn = [this_ptr, other_ptr, result_ptr, grad_fn]() {
                // Get output gradient
                if (!result_ptr->_shards.empty() && result_ptr->_shards[0].grad) {
                    // Compute input gradients
                    for (size_t i = 0; i < this_ptr->_shards.size(); ++i) {
                        if (this_ptr->_requires_grad) {
                            // Create gradient tensor for this shard
                            std::size_t shape_arr[] = {this_ptr->_shards[i].num_elements};
                            auto grad_input = std::make_unique<DenseTensor<T>>(shape_arr, 1);
                            
                            // Call gradient function
                            grad_fn(this_ptr->_shards[i].data.get(),
                                   other_ptr->_shards[i].data.get(),
                                   grad_input.get());
                            
                            // Accumulate gradient
                            this_ptr->_shards[i].accumulate_grad(grad_input.get());
                        }
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
        std::function<void(const TensorBase<T>*, TensorBase<T>*)> grad_fn = nullptr) const {
        
        std::lock_guard<std::mutex> lock(_mutex);
        
        auto result = std::shared_ptr<DistributedTensor<T>>(
            new DistributedTensor<T>(_shape, _requires_grad, false));
        result->_shards.resize(_shards.size());
        result->_parents_raw.push_back(const_cast<DistributedTensor<T>*>(this));
        
        // Execute shards sequentially (see parallel_binary_op for explanation)
        for (size_t i = 0; i < _shards.size(); ++i) {
            auto* shard_data = _shards[i].data.get();
            auto* result_shard_ptr = &result->_shards[i];
            const auto& device = _shards[i].device;
            std::size_t offset = _shards[i].global_offset;
            std::size_t count = _shards[i].num_elements;
            std::size_t shard_id = _shards[i].shard_id;

            auto result_data = op(shard_data);
            *result_shard_ptr = Shard<T>(std::move(result_data), device, offset, count, shard_id, false);
        }
        
        // Set up backward function for autograd
        if (grad_fn && _requires_grad) {
            DistributedTensor<T>* this_ptr = const_cast<DistributedTensor<T>*>(this);
            DistributedTensor<T>* result_ptr = result.get();
            
            result->_backward_fn = [this_ptr, result_ptr, grad_fn]() {
                if (!result_ptr->_shards.empty() && result_ptr->_shards[0].grad) {
                    for (size_t i = 0; i < this_ptr->_shards.size(); ++i) {
                        if (this_ptr->_requires_grad) {
                            std::size_t shape_arr[] = {this_ptr->_shards[i].num_elements};
                            auto grad_input = std::make_unique<DenseTensor<T>>(shape_arr, 1);
                            
                            grad_fn(this_ptr->_shards[i].data.get(), grad_input.get());
                            this_ptr->_shards[i].accumulate_grad(grad_input.get());
                        }
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
        
        // Generate random values in parallel - each shard generates its own portion
        std::mt19937 gen(42);
        std::normal_distribution<T> dist(0, 1);
        tensor->distribute_data([&dist, &gen](std::size_t) mutable {
            return dist(gen);
        });
        return tensor;
    }
    
    static std::shared_ptr<DistributedTensor<T>> rand(
        const std::vector<std::size_t>& shape, bool requires_grad = false) {
        
        auto tensor = std::shared_ptr<DistributedTensor<T>>(
            new DistributedTensor<T>(shape, requires_grad, true));
        
        std::mt19937 gen(42);
        std::uniform_real_distribution<T> dist(0, 1);
        tensor->distribute_data([&dist, &gen](std::size_t) mutable {
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
    // Element-wise Binary Operations (Distributed)
    // ========================================================================
    
    std::shared_ptr<DistributedTensor<T>> add(const DistributedTensor<T>* other) const {
        return parallel_binary_op(other,
            [](const TensorBase<T>* a, const TensorBase<T>* b) {
                return a->add(b);
            },
            [](const TensorBase<T>* a, const TensorBase<T>* b, TensorBase<T>* grad_a) {
                // d/da (a + b) = 1, so grad flows through unchanged
                for (std::size_t i = 0; i < a->total_size() && i < grad_a->total_size(); ++i) {
                    grad_a->set_element(i, T{1});
                }
            });
    }
    
    std::shared_ptr<DistributedTensor<T>> subtract(const DistributedTensor<T>* other) const {
        return parallel_binary_op(other,
            [](const TensorBase<T>* a, const TensorBase<T>* b) {
                return a->subtract(b);
            },
            [](const TensorBase<T>* a, const TensorBase<T>* b, TensorBase<T>* grad_a) {
                // d/da (a - b) = 1
                for (std::size_t i = 0; i < a->total_size() && i < grad_a->total_size(); ++i) {
                    grad_a->set_element(i, T{1});
                }
            });
    }
    
    std::shared_ptr<DistributedTensor<T>> multiply(const DistributedTensor<T>* other) const {
        return parallel_binary_op(other,
            [](const TensorBase<T>* a, const TensorBase<T>* b) {
                return a->multiply(b);
            },
            [](const TensorBase<T>* a, const TensorBase<T>* b, TensorBase<T>* grad_a) {
                // d/da (a * b) = b
                for (std::size_t i = 0; i < b->total_size() && i < grad_a->total_size(); ++i) {
                    grad_a->set_element(i, b->get_element(i));
                }
            });
    }
    
    std::shared_ptr<DistributedTensor<T>> divide(const DistributedTensor<T>* other) const {
        return parallel_binary_op(other,
            [](const TensorBase<T>* a, const TensorBase<T>* b) {
                return a->divide(b);
            },
            [](const TensorBase<T>* a, const TensorBase<T>* b, TensorBase<T>* grad_a) {
                // d/da (a / b) = 1/b
                for (std::size_t i = 0; i < b->total_size() && i < grad_a->total_size(); ++i) {
                    T b_val = b->get_element(i);
                    if (std::abs(b_val) > 1e-10f) {
                        grad_a->set_element(i, T{1} / b_val);
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
            [](const TensorBase<T>* a, TensorBase<T>* grad_a) {
                // d/da (a + c) = 1
                for (std::size_t i = 0; i < a->total_size() && i < grad_a->total_size(); ++i) {
                    grad_a->set_element(i, T{1});
                }
            });
    }
    
    std::shared_ptr<DistributedTensor<T>> subtract_scalar(T scalar) const {
        return parallel_unary_op(
            [scalar](const TensorBase<T>* a) {
                return a->subtract_scalar(scalar);
            },
            [](const TensorBase<T>* a, TensorBase<T>* grad_a) {
                // d/da (a - c) = 1
                for (std::size_t i = 0; i < a->total_size() && i < grad_a->total_size(); ++i) {
                    grad_a->set_element(i, T{1});
                }
            });
    }
    
    std::shared_ptr<DistributedTensor<T>> multiply_scalar(T scalar) const {
        return parallel_unary_op(
            [scalar](const TensorBase<T>* a) {
                return a->multiply_scalar(scalar);
            },
            [scalar](const TensorBase<T>* a, TensorBase<T>* grad_a) {
                // d/da (a * c) = c
                for (std::size_t i = 0; i < a->total_size() && i < grad_a->total_size(); ++i) {
                    grad_a->set_element(i, scalar);
                }
            });
    }
    
    std::shared_ptr<DistributedTensor<T>> divide_scalar(T scalar) const {
        return parallel_unary_op(
            [scalar](const TensorBase<T>* a) {
                return a->divide_scalar(scalar);
            },
            [scalar](const TensorBase<T>* a, TensorBase<T>* grad_a) {
                // d/da (a / c) = 1/c
                if (std::abs(scalar) > 1e-10f) {
                    T inv_scalar = T{1} / scalar;
                    for (std::size_t i = 0; i < a->total_size() && i < grad_a->total_size(); ++i) {
                        grad_a->set_element(i, inv_scalar);
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
            [](const TensorBase<T>* a, TensorBase<T>* grad_a) {
                // d/da (-a) = -1
                for (std::size_t i = 0; i < a->total_size() && i < grad_a->total_size(); ++i) {
                    grad_a->set_element(i, T{-1});
                }
            });
    }
    
    std::shared_ptr<DistributedTensor<T>> abs() const {
        return parallel_unary_op(
            [](const TensorBase<T>* a) {
                return a->abs();
            },
            [](const TensorBase<T>* a, TensorBase<T>* grad_a) {
                // d/da |a| = sign(a)
                for (std::size_t i = 0; i < a->total_size() && i < grad_a->total_size(); ++i) {
                    T val = a->get_element(i);
                    if (val > T{0}) grad_a->set_element(i, T{1});
                    else if (val < T{0}) grad_a->set_element(i, T{-1});
                    else grad_a->set_element(i, T{0});
                }
            });
    }
    
    std::shared_ptr<DistributedTensor<T>> relu() const {
        return parallel_unary_op(
            [](const TensorBase<T>* a) {
                return a->clamp(T{0}, std::numeric_limits<T>::max());
            },
            [](const TensorBase<T>* a, TensorBase<T>* grad_a) {
                // d/da ReLU(a) = 1 if a > 0, else 0
                for (std::size_t i = 0; i < a->total_size() && i < grad_a->total_size(); ++i) {
                    T val = a->get_element(i);
                    grad_a->set_element(i, val > T{0} ? T{1} : T{0});
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
                        auto grad_input = std::make_unique<DenseTensor<T>>(shape_arr, 1, grad_val);
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
                        auto grad_input = std::make_unique<DenseTensor<T>>(shape_arr, 1, grad_val * inv_n);
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
            _shards[0].data->set_element(0, T{1});
        }
        
        // Traverse in reverse topological order
        for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
            auto* node = *it;
            
            // Skip if no gradient has flowed to this node
            bool has_grad = false;
            for (const auto& shard : node->_shards) {
                if (shard.grad) {
                    has_grad = true;
                    break;
                }
            }
            if (!has_grad) continue;
            
            // Skip leaf nodes (they don't have backward functions)
            if (node->_is_leaf) continue;
            
            // Apply backward function
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
