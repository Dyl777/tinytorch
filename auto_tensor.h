#pragma once
#include "tensor.h"
#include "stream_tensor.h"
#include <memory>
#include <functional>
#include <limits>
#include <algorithm>
#include <cstdio>

#if defined(_WIN32) || defined(_WIN64)
    #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#elif defined(__APPLE__) || defined(__MACH__)
    #include <sys/types.h>
    #include <sys/sysctl.h>
#endif

// ============================================================================
// AutoTensor: Automatic In-Memory vs Memory-Mapped Tensor Selection
// ============================================================================
// What is AutoTensor?
// -------------------
// AutoTensor is an abstract factory pattern that automatically chooses between
// in-memory Tensor and memory-mapped StreamTensor based on data size relative
// to available system memory.
//
// Why is this useful?
// --------------------
// 1. LARGE FILES (e.g., 300MB parquet): If data exceeds a configurable
//    threshold (e.g., 50% of RAM), StreamTensor is used to avoid OOM.
//
// 2. SMALL FILES: Regular Tensor is used for maximum performance.
//
// 3. UNIFIED INTERFACE: Regardless of backend, you use the same API.
//    The abstraction handles the routing transparently.
//
// How it works:
// -------------
// 1. Estimate data size (rows * columns * sizeof(T))
// 2. Query available system memory
// 3. If data_size / total_memory > threshold → StreamTensor
// 4. Otherwise → Tensor
//
// The threshold is configurable via AutoConfig.

// ============================================================================
// Platform-specific memory query
// ============================================================================
inline std::size_t get_total_system_memory() {
#if defined(_WIN32) || defined(_WIN64)
    // Windows: GlobalMemoryStatusEx
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        return static_cast<std::size_t>(status.ullTotalPhys);
    }
    return 8ULL * 1024 * 1024 * 1024; // Fallback: 8GB
#elif defined(__linux__)
    // Linux: read /proc/meminfo
    std::size_t total_kb = 0;
    FILE* f = fopen("/proc/meminfo", "r");
    if (f) {
        char buf[256];
        while (fgets(buf, sizeof(buf), f)) {
            if (sscanf(buf, "MemTotal: %zu kB", &total_kb) == 1) break;
        }
        fclose(f);
    }
    return total_kb * 1024;
#elif defined(__APPLE__) || defined(__MACH__)
    // macOS: sysctl
    int mib[2] = {CTL_HW, HW_MEMSIZE};
    uint64_t memory = 0;
    std::size_t length = sizeof(memory);
    if (sysctl(mib, 2, &memory, &length, nullptr, 0) == 0) {
        return static_cast<std::size_t>(memory);
    }
    return 8ULL * 1024 * 1024 * 1024; // Fallback: 8GB
#else
    return 8ULL * 1024 * 1024 * 1024; // Fallback: 8GB
#endif
}

// ============================================================================
// AutoConfig: Configuration for automatic backend selection
// ============================================================================
// memory_threshold: Fraction of total RAM at which to switch to StreamTensor.
//   0.5 = use StreamTensor if data > 50% of RAM (default)
//   0.1 = use StreamTensor if data > 10% of RAM (more conservative)
//   0.9 = only use StreamTensor if data > 90% of RAM (aggressive)
//
// stream_config: Passed to StreamTensor if that backend is chosen.
//
// force_streaming: If true, ALWAYS use StreamTensor regardless of data size.
//   This is the primary knob for large datasets that can't fit in memory.
//   When force_streaming=true, all data is stored in mmap-backed files and
//   processed in batches, so even multi-gigabyte tensors can be handled.
//
// force_dense: If true, ALWAYS use Tensor regardless of data size.

struct AutoConfig {
    double memory_threshold;       // Fraction of RAM (0.0 - 1.0)
    StreamConfig stream_config;    // Config for StreamTensor backend
    bool force_streaming;          // ALWAYS use StreamTensor
    bool force_dense;              // ALWAYS use Tensor
    std::size_t print_batch_size;  // Elements per batch when printing large tensors

    AutoConfig()
        : memory_threshold(0.5),
          force_streaming(false),
          force_dense(false),
          print_batch_size(100)
    {}
};

// ============================================================================
// Abstract Tensor Interface
// ============================================================================
// All operations are virtual, implemented by both DenseTensor and MmapTensor.
// This provides a unified API regardless of which backend is chosen.

template<typename T>
class TensorBase {
public:
    virtual ~TensorBase() = default;

    // Shape and size
    virtual std::size_t ndim() const = 0;
    virtual std::size_t total_size() const = 0;
    virtual const std::size_t* shape() const = 0;
    virtual const std::size_t* stride() const = 0;

    // Element access
    virtual T get_element(std::size_t index) const = 0;
    virtual void set_element(std::size_t index, T value) = 0;
    virtual T operator()(std::size_t i) const = 0;
    virtual T& operator()(std::size_t i) = 0;
    virtual T operator()(std::size_t i, std::size_t j) const = 0;
    virtual T& operator()(std::size_t i, std::size_t j) = 0;

    // Backend info
    virtual bool is_streaming() const = 0;
    virtual std::string backend_name() const = 0;

    // Element-wise operations (returns new TensorBase)
    virtual std::unique_ptr<TensorBase<T>> add(const TensorBase<T>* other) const = 0;
    virtual std::unique_ptr<TensorBase<T>> subtract(const TensorBase<T>* other) const = 0;
    virtual std::unique_ptr<TensorBase<T>> multiply(const TensorBase<T>* other) const = 0;
    virtual std::unique_ptr<TensorBase<T>> divide(const TensorBase<T>* other) const = 0;
    virtual std::unique_ptr<TensorBase<T>> add_scalar(T scalar) const = 0;
    virtual std::unique_ptr<TensorBase<T>> subtract_scalar(T scalar) const = 0;
    virtual std::unique_ptr<TensorBase<T>> multiply_scalar(T scalar) const = 0;
    virtual std::unique_ptr<TensorBase<T>> divide_scalar(T scalar) const = 0;

    // Unary operations
    virtual std::unique_ptr<TensorBase<T>> negate() const = 0;
    virtual std::unique_ptr<TensorBase<T>> abs() const = 0;

    // Reductions
    virtual T sum() const = 0;
    virtual T mean() const = 0;
    virtual T max() const = 0;
    virtual T min() const = 0;
    virtual T dot(const TensorBase<T>* other) const = 0;

    // Transforms
    virtual std::unique_ptr<TensorBase<T>> reshape(const std::size_t* new_shape, std::size_t new_ndim) const = 0;
    virtual std::unique_ptr<TensorBase<T>> transpose() const = 0;
    virtual std::unique_ptr<TensorBase<T>> clamp(T min_val, T max_val) const = 0;

    // Comparisons
    virtual std::unique_ptr<TensorBase<bool>> greater_than(T threshold) const = 0;
    virtual std::unique_ptr<TensorBase<bool>> less_than(T threshold) const = 0;

    // Export
    virtual std::unique_ptr<T[]> to_array() const = 0;

    // Print
    virtual void print(std::ostream& os) const = 0;

    // Batch-print all elements in chunks.
    // This is essential for tensors too large to fit in memory.
    // Instead of loading everything into RAM, it prints `batch_size` elements
    // at a time, flushing each batch before loading the next.
    // For small tensors this just prints normally.
    virtual void batch_print(std::ostream& os, std::size_t batch_size = 100) const = 0;
};

// ============================================================================
// DenseTensor: Wraps regular in-memory Tensor
// ============================================================================
template<typename T>
class DenseTensor : public TensorBase<T> {
private:
    std::unique_ptr<Tensor<T>> _tensor;

    // Helper to create Tensor from shape
    static Tensor<T>* create_from_shape(const std::size_t* shape, std::size_t ndim) {
        return new Tensor<T>(shape, ndim);
    }

public:
    // Construct from existing Tensor (takes ownership)
    explicit DenseTensor(Tensor<T>* tensor) : _tensor(tensor) {}

    // Construct from raw data
    DenseTensor(const T* data, std::size_t size)
        : _tensor(new Tensor<T>(data, size)) {}

    // Construct from shape with fill value
    DenseTensor(const std::size_t* shape, std::size_t ndim, T fill_value = T{})
        : _tensor(new Tensor<T>(shape, ndim, fill_value)) {}

    // Construct from 1D initializer list
    DenseTensor(std::initializer_list<T> data)
        : _tensor(new Tensor<T>(data)) {}

    // Shape and size
    std::size_t ndim() const override { return _tensor->ndim(); }
    std::size_t total_size() const override { return _tensor->total_size(); }
    const std::size_t* shape() const override { return _tensor->shape(); }
    const std::size_t* stride() const override { return _tensor->stride(); }

    // Element access
    T get_element(std::size_t index) const override {
        if (_tensor->ndim() == 1) return (*_tensor)(index);
        return _tensor->data_non_volatile()[index];
    }

    void set_element(std::size_t index, T value) override {
        if (_tensor->ndim() == 1) {
            (*_tensor)(index) = value;
        } else {
            _tensor->data_non_volatile()[index] = value;
        }
    }

    T operator()(std::size_t i) const override { return (*_tensor)(i); }
    T& operator()(std::size_t i) override { return (*_tensor)(i); }
    T operator()(std::size_t i, std::size_t j) const override { return (*_tensor)(i, j); }
    T& operator()(std::size_t i, std::size_t j) override { return (*_tensor)(i, j); }

    bool is_streaming() const override { return false; }
    std::string backend_name() const override { return "DenseTensor (in-memory)"; }

    // Element-wise operations
    std::unique_ptr<TensorBase<T>> add(const TensorBase<T>* other) const override {
        const DenseTensor<T>* denseOther = dynamic_cast<const DenseTensor<T>*>(other);
        if (!denseOther) throw std::invalid_argument("Mixed backend operations not supported");
        Tensor<T>* result = _tensor->add(denseOther->_tensor.get());
        return std::make_unique<DenseTensor<T>>(result);
    }

    std::unique_ptr<TensorBase<T>> subtract(const TensorBase<T>* other) const override {
        const DenseTensor<T>* denseOther = dynamic_cast<const DenseTensor<T>*>(other);
        if (!denseOther) throw std::invalid_argument("Mixed backend operations not supported");
        Tensor<T>* result = _tensor->subtract(denseOther->_tensor.get());
        return std::make_unique<DenseTensor<T>>(result);
    }

    std::unique_ptr<TensorBase<T>> multiply(const TensorBase<T>* other) const override {
        const DenseTensor<T>* denseOther = dynamic_cast<const DenseTensor<T>*>(other);
        if (!denseOther) throw std::invalid_argument("Mixed backend operations not supported");
        Tensor<T>* result = _tensor->multiply(denseOther->_tensor.get());
        return std::make_unique<DenseTensor<T>>(result);
    }

    std::unique_ptr<TensorBase<T>> divide(const TensorBase<T>* other) const override {
        const DenseTensor<T>* denseOther = dynamic_cast<const DenseTensor<T>*>(other);
        if (!denseOther) throw std::invalid_argument("Mixed backend operations not supported");
        Tensor<T>* result = _tensor->divide(denseOther->_tensor.get());
        return std::make_unique<DenseTensor<T>>(result);
    }

    std::unique_ptr<TensorBase<T>> add_scalar(T scalar) const override {
        Tensor<T>* result = _tensor->add_scalar(scalar);
        return std::make_unique<DenseTensor<T>>(result);
    }

    std::unique_ptr<TensorBase<T>> subtract_scalar(T scalar) const override {
        Tensor<T>* result = _tensor->subtract_scalar(scalar);
        return std::make_unique<DenseTensor<T>>(result);
    }

    std::unique_ptr<TensorBase<T>> multiply_scalar(T scalar) const override {
        Tensor<T>* result = _tensor->multiply_scalar(scalar);
        return std::make_unique<DenseTensor<T>>(result);
    }

    std::unique_ptr<TensorBase<T>> divide_scalar(T scalar) const override {
        Tensor<T>* result = _tensor->divide_scalar(scalar);
        return std::make_unique<DenseTensor<T>>(result);
    }

    // Unary operations
    std::unique_ptr<TensorBase<T>> negate() const override {
        Tensor<T>* result = _tensor->negate();
        return std::make_unique<DenseTensor<T>>(result);
    }

    std::unique_ptr<TensorBase<T>> abs() const override {
        Tensor<T>* result = _tensor->abs();
        return std::make_unique<DenseTensor<T>>(result);
    }

    // Reductions
    T sum() const override { return _tensor->sum(); }
    T max() const override { return _tensor->max(); }
    T min() const override { return _tensor->min(); }

    // mean() and dot() are only meaningful for floating-point types.
    // For bool, we provide a fallback to avoid compilation errors.
    T mean() const override {
        if constexpr (std::is_floating_point<T>::value) {
            return _tensor->mean();
        } else {
            return T{}; // Fallback for non-floating-point types (e.g., bool)
        }
    }

    T dot(const TensorBase<T>* other) const override {
        if constexpr (std::is_floating_point<T>::value) {
            const DenseTensor<T>* denseOther = dynamic_cast<const DenseTensor<T>*>(other);
            if (!denseOther) throw std::invalid_argument("Mixed backend operations not supported");
            return _tensor->dot(denseOther->_tensor.get());
        } else {
            return T{}; // Fallback for non-floating-point types
        }
    }

    // Transforms
    std::unique_ptr<TensorBase<T>> reshape(const std::size_t* new_shape, std::size_t new_ndim) const override {
        Tensor<T>* result = _tensor->reshape(new_shape, new_ndim);
        return std::make_unique<DenseTensor<T>>(result);
    }

    std::unique_ptr<TensorBase<T>> transpose() const override {
        Tensor<T>* result = _tensor->transpose();
        return std::make_unique<DenseTensor<T>>(result);
    }

    std::unique_ptr<TensorBase<T>> clamp(T min_val, T max_val) const override {
        Tensor<T>* result = _tensor->clamp(min_val, max_val);
        return std::make_unique<DenseTensor<T>>(result);
    }

    // Comparisons
    std::unique_ptr<TensorBase<bool>> greater_than(T threshold) const override {
        Tensor<bool>* result = _tensor->greater_than(threshold);
        return std::make_unique<DenseTensor<bool>>(result);
    }

    std::unique_ptr<TensorBase<bool>> less_than(T threshold) const override {
        Tensor<bool>* result = _tensor->less_than(threshold);
        return std::make_unique<DenseTensor<bool>>(result);
    }

    // Export
    std::unique_ptr<T[]> to_array() const override {
        std::unique_ptr<T[]> result(new T[_tensor->total_size()]);
        const T* src = _tensor->data_non_volatile();
        for (std::size_t i = 0; i < _tensor->total_size(); ++i) {
            result[i] = src[i];
        }
        return result;
    }

    // Print
    void print(std::ostream& os) const override {
        os << *_tensor;
    }

    // Batch print for DenseTensor - processes in chunks to avoid
    // building a huge string in memory for very large tensors
    void batch_print(std::ostream& os, std::size_t batch_size) const override {
        std::size_t total = _tensor->total_size();
        std::size_t ndim = _tensor->ndim();

        if (ndim == 0) {
            os << _tensor->item();
            return;
        }

        os << "[";
        for (std::size_t batch_start = 0; batch_start < total; batch_start += batch_size) {
            std::size_t batch_end = std::min(batch_start + batch_size, total);

            for (std::size_t i = batch_start; i < batch_end; ++i) {
                if (ndim == 1) {
                    os << (*_tensor)(i);
                } else {
                    os << _tensor->data_non_volatile()[i];
                }
                if (i != total - 1) os << ", ";
            }

            // Flush after each batch so we don't buffer everything
            os.flush();

            if (batch_end < total) {
                os << "\n  ... (batch " << (batch_start / batch_size + 1) << " done, continuing) ";
            }
        }
        os << "]";
    }

    // ========================================================================
    // Distributed Parallelism Primitives
    // ========================================================================

    Tensor<T>* allgather(const TensorBase<T>* const* shards, int world_size) const {
        std::size_t shard_size = _tensor->total_size();
        std::size_t total_size = shard_size * world_size;
        std::size_t new_shape[] = {total_size};
        Tensor<T>* result = new Tensor<T>(new_shape, 1);

        for (int r = 0; r < world_size; ++r) {
            std::size_t offset = r * shard_size;
            for (std::size_t i = 0; i < shard_size; ++i) {
                result->data_non_volatile()[offset + i] = shards[r]->get_element(i);
            }
        }
        return result;
    }

    Tensor<T>* reducescatter(const TensorBase<T>* full_tensor, int rank, int world_size) const {
        std::size_t shard_size = _tensor->total_size();
        std::size_t offset = rank * shard_size;
        Tensor<T>* result = new Tensor<T>(_tensor->shape(), _tensor->ndim());

        for (std::size_t i = 0; i < shard_size; ++i) {
            result->data_non_volatile()[i] = full_tensor->get_element(offset + i);
        }
        return result;
    }

    Tensor<T>* allreduce_sum(const TensorBase<T>* const* shards, int world_size) const {
        Tensor<T>* result = new Tensor<T>(_tensor->shape(), _tensor->ndim());

        for (std::size_t i = 0; i < _tensor->total_size(); ++i) {
            T sum = T{};
            for (int r = 0; r < world_size; ++r) {
                sum += shards[r]->get_element(i);
            }
            result->data_non_volatile()[i] = sum;
        }
        return result;
    }

    Tensor<T>* allreduce_mean(const TensorBase<T>* const* shards, int world_size) const {
        Tensor<T>* result = new Tensor<T>(_tensor->shape(), _tensor->ndim());
        T inv_world_size = T{1} / static_cast<T>(world_size);

        for (std::size_t i = 0; i < _tensor->total_size(); ++i) {
            T sum = T{};
            for (int r = 0; r < world_size; ++r) {
                sum += shards[r]->get_element(i);
            }
            result->data_non_volatile()[i] = sum * inv_world_size;
        }
        return result;
    }

    Tensor<T>* broadcast(const TensorBase<T>* root_tensor) const {
        Tensor<T>* result = new Tensor<T>(_tensor->shape(), _tensor->ndim());
        for (std::size_t i = 0; i < _tensor->total_size(); ++i) {
            result->data_non_volatile()[i] = root_tensor->get_element(i);
        }
        return result;
    }

    // Allow all DenseTensor instantiations to access each other's private members
    template<typename U>
    friend class DenseTensor;
};

// ============================================================================
// MmapTensor: Wraps memory-mapped StreamTensor
// ============================================================================
template<typename T>
class MmapTensor : public TensorBase<T> {
private:
    std::unique_ptr<StreamTensor<T>> _stream;
    StreamConfig _config;

public:
    // Construct from existing StreamTensor (takes ownership)
    explicit MmapTensor(StreamTensor<T>* stream, const StreamConfig& config = StreamConfig{})
        : _stream(stream), _config(config) {}

    // Construct from raw data (copies to mmap)
    MmapTensor(const T* data, std::size_t size, const StreamConfig& config = StreamConfig{})
        : _stream(new StreamTensor<T>(data, size, config)), _config(config) {}

    // Construct from shape with fill value
    MmapTensor(const std::size_t* shape, std::size_t ndim,
               const StreamConfig& config = StreamConfig{}, T fill_value = T{})
        : _stream(new StreamTensor<T>(shape, ndim, config, fill_value)), _config(config) {}

    // Shape and size
    std::size_t ndim() const override { return _stream->ndim(); }
    std::size_t total_size() const override { return _stream->total_size(); }
    const std::size_t* shape() const override { return _stream->shape(); }
    const std::size_t* stride() const override { return _stream->stride(); }

    // Element access
    T get_element(std::size_t index) const override { return _stream->get_element(index); }
    void set_element(std::size_t index, T value) override { _stream->set_element(index, value); }
    T operator()(std::size_t i) const override { return (*_stream)(i); }
    T& operator()(std::size_t i) override { return (*_stream)(i); }
    T operator()(std::size_t i, std::size_t j) const override { return (*_stream)(i, j); }
    T& operator()(std::size_t i, std::size_t j) override { return (*_stream)(i, j); }

    bool is_streaming() const override { return true; }
    std::string backend_name() const override { return "MmapTensor (memory-mapped)"; }

    // Element-wise operations
    std::unique_ptr<TensorBase<T>> add(const TensorBase<T>* other) const override {
        const MmapTensor<T>* mmapOther = dynamic_cast<const MmapTensor<T>*>(other);
        if (!mmapOther) throw std::invalid_argument("Mixed backend operations not supported");
        StreamTensor<T>* result = _stream->add(mmapOther->_stream.get(), _config);
        return std::make_unique<MmapTensor<T>>(result, _config);
    }

    std::unique_ptr<TensorBase<T>> subtract(const TensorBase<T>* other) const override {
        const MmapTensor<T>* mmapOther = dynamic_cast<const MmapTensor<T>*>(other);
        if (!mmapOther) throw std::invalid_argument("Mixed backend operations not supported");
        StreamTensor<T>* result = _stream->subtract(mmapOther->_stream.get(), _config);
        return std::make_unique<MmapTensor<T>>(result, _config);
    }

    std::unique_ptr<TensorBase<T>> multiply(const TensorBase<T>* other) const override {
        const MmapTensor<T>* mmapOther = dynamic_cast<const MmapTensor<T>*>(other);
        if (!mmapOther) throw std::invalid_argument("Mixed backend operations not supported");
        StreamTensor<T>* result = _stream->multiply(mmapOther->_stream.get(), _config);
        return std::make_unique<MmapTensor<T>>(result, _config);
    }

    std::unique_ptr<TensorBase<T>> divide(const TensorBase<T>* other) const override {
        const MmapTensor<T>* mmapOther = dynamic_cast<const MmapTensor<T>*>(other);
        if (!mmapOther) throw std::invalid_argument("Mixed backend operations not supported");
        StreamTensor<T>* result = _stream->divide(mmapOther->_stream.get(), _config);
        return std::make_unique<MmapTensor<T>>(result, _config);
    }

    std::unique_ptr<TensorBase<T>> add_scalar(T scalar) const override {
        StreamTensor<T>* result = _stream->add_scalar(scalar, _config);
        return std::make_unique<MmapTensor<T>>(result, _config);
    }

    std::unique_ptr<TensorBase<T>> subtract_scalar(T scalar) const override {
        StreamTensor<T>* result = _stream->subtract_scalar(scalar, _config);
        return std::make_unique<MmapTensor<T>>(result, _config);
    }

    std::unique_ptr<TensorBase<T>> multiply_scalar(T scalar) const override {
        StreamTensor<T>* result = _stream->multiply_scalar(scalar, _config);
        return std::make_unique<MmapTensor<T>>(result, _config);
    }

    std::unique_ptr<TensorBase<T>> divide_scalar(T scalar) const override {
        StreamTensor<T>* result = _stream->divide_scalar(scalar, _config);
        return std::make_unique<MmapTensor<T>>(result, _config);
    }

    // Unary operations
    std::unique_ptr<TensorBase<T>> negate() const override {
        StreamTensor<T>* result = _stream->negate(_config);
        return std::make_unique<MmapTensor<T>>(result, _config);
    }

    std::unique_ptr<TensorBase<T>> abs() const override {
        StreamTensor<T>* result = _stream->abs(_config);
        return std::make_unique<MmapTensor<T>>(result, _config);
    }

    // Reductions
    T sum() const override { return _stream->batched_sum(_config); }
    T mean() const override { return _stream->batched_mean(_config); }
    T max() const override { return _stream->batched_max(_config); }
    T min() const override { return _stream->batched_min(_config); }

    T dot(const TensorBase<T>* other) const override {
        const MmapTensor<T>* mmapOther = dynamic_cast<const MmapTensor<T>*>(other);
        if (!mmapOther) throw std::invalid_argument("Mixed backend operations not supported");
        return _stream->batched_dot(mmapOther->_stream.get(), _config);
    }

    // Transforms
    std::unique_ptr<TensorBase<T>> reshape(const std::size_t* new_shape, std::size_t new_ndim) const override {
        StreamTensor<T>* result = _stream->reshape(new_shape, new_ndim, _config);
        return std::make_unique<MmapTensor<T>>(result, _config);
    }

    std::unique_ptr<TensorBase<T>> transpose() const override {
        StreamTensor<T>* result = _stream->transpose(_config);
        return std::make_unique<MmapTensor<T>>(result, _config);
    }

    std::unique_ptr<TensorBase<T>> clamp(T min_val, T max_val) const override {
        StreamTensor<T>* result = _stream->clamp(min_val, max_val, _config);
        return std::make_unique<MmapTensor<T>>(result, _config);
    }

    // Comparisons
    std::unique_ptr<TensorBase<bool>> greater_than(T threshold) const override {
        StreamTensor<bool>* result = _stream->greater_than(threshold, _config);
        return std::make_unique<MmapTensor<bool>>(result, _config);
    }

    std::unique_ptr<TensorBase<bool>> less_than(T threshold) const override {
        StreamTensor<bool>* result = _stream->less_than(threshold, _config);
        return std::make_unique<MmapTensor<bool>>(result, _config);
    }

    // Export
    std::unique_ptr<T[]> to_array() const override {
        return std::unique_ptr<T[]>(_stream->to_flat_array());
    }

    // Print
    void print(std::ostream& os) const override {
        os << *_stream;
    }

    // Batch print for MmapTensor - reads from mmap file in chunks.
    // This is the key method for printing tensors larger than RAM.
    // Only `batch_size` elements are in RAM at any time.
    void batch_print(std::ostream& os, std::size_t batch_size) const override {
        std::size_t total = _stream->total_size();
        std::size_t ndim = _stream->ndim();

        if (ndim == 0) {
            os << _stream->get_element(0);
            return;
        }

        os << "[";
        for (std::size_t batch_start = 0; batch_start < total; batch_start += batch_size) {
            std::size_t count = std::min(batch_size, total - batch_start);

            for (std::size_t i = 0; i < count; ++i) {
                std::size_t idx = batch_start + i;
                os << _stream->get_element(idx);
                if (idx != total - 1) os << ", ";
            }

            // Flush after each batch - only `count` elements were in RAM
            os.flush();

            if (batch_start + count < total) {
                os << "\n  ... (batch " << (batch_start / batch_size + 1)
                   << " done, continuing) ";
            }
        }
        os << "]";
    }

    // Allow all MmapTensor instantiations to access each other's private members
    template<typename U>
    friend class MmapTensor;
};

// ============================================================================
// AutoTensor Factory: Chooses backend based on data size
// ============================================================================
template<typename T>
class AutoTensor {
public:
    // Load data and automatically choose backend
    static std::unique_ptr<TensorBase<T>> from_data(
        const T* data,
        std::size_t size,
        const AutoConfig& config = AutoConfig{}) {

        std::size_t data_bytes = size * sizeof(T);
        std::size_t total_memory = get_total_system_memory();
        double usage_ratio = static_cast<double>(data_bytes) / static_cast<double>(total_memory);

        bool use_streaming = config.force_streaming ||
                             (!config.force_dense && usage_ratio > config.memory_threshold);

        if (use_streaming) {
            std::cout << "[AutoTensor] Data size: " << (data_bytes / (1024.0 * 1024.0)) << " MB, "
                      << "Memory usage: " << (usage_ratio * 100.0) << "% of system RAM"
                      << " → Using StreamTensor (memory-mapped)" << std::endl;
            return std::make_unique<MmapTensor<T>>(data, size, config.stream_config);
        } else {
            std::cout << "[AutoTensor] Data size: " << (data_bytes / (1024.0 * 1024.0)) << " MB, "
                      << "Memory usage: " << (usage_ratio * 100.0) << "% of system RAM"
                      << " → Using Tensor (in-memory)" << std::endl;
            return std::make_unique<DenseTensor<T>>(data, size);
        }
    }

    // Create from shape with fill value
    static std::unique_ptr<TensorBase<T>> from_shape(
        const std::size_t* shape,
        std::size_t ndim,
        const AutoConfig& config = AutoConfig{},
        T fill_value = T{}) {

        // Compute total size
        std::size_t total = 1;
        for (std::size_t i = 0; i < ndim; ++i) total *= shape[i];

        std::size_t data_bytes = total * sizeof(T);
        std::size_t total_memory = get_total_system_memory();
        double usage_ratio = static_cast<double>(data_bytes) / static_cast<double>(total_memory);

        bool use_streaming = config.force_streaming ||
                             (!config.force_dense && usage_ratio > config.memory_threshold);

        if (use_streaming) {
            std::cout << "[AutoTensor] Tensor size: " << (data_bytes / (1024.0 * 1024.0)) << " MB, "
                      << "Memory usage: " << (usage_ratio * 100.0) << "% → StreamTensor" << std::endl;
            return std::make_unique<MmapTensor<T>>(shape, ndim, config.stream_config, fill_value);
        } else {
            std::cout << "[AutoTensor] Tensor size: " << (data_bytes / (1024.0 * 1024.0)) << " MB, "
                      << "Memory usage: " << (usage_ratio * 100.0) << "% → Tensor" << std::endl;
            return std::make_unique<DenseTensor<T>>(shape, ndim, fill_value);
        }
    }

    // Query system info
    static void print_system_info() {
        std::size_t total_mem = get_total_system_memory();
        std::cout << "Total system memory: " << (total_mem / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
        std::cout << "AutoTensor threshold: data > " << (AutoConfig{}.memory_threshold * 100)
                  << "% of RAM triggers streaming backend" << std::endl;
    }
};

// ============================================================================
// Stream output for TensorBase
// ============================================================================
template<typename T>
std::ostream& operator<<(std::ostream& os, const TensorBase<T>& tensor) {
    tensor.print(os);
    return os;
}
