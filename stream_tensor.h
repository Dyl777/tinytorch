#pragma once
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <cstring>
#include <functional>
#include <type_traits>
#include <algorithm>

// ============================================================================
// Cross-Platform mmap Abstraction
// ============================================================================
// What is mmap?
// -------------
// `mmap` (memory-mapped file) maps a file or device directly into the process's
// virtual address space. This allows treating file contents as if they were in
// RAM, with the OS handling paging data in/out automatically.
//
// Why is it here?
// ---------------
// 1. LARGE DATASETS: When tensors exceed available RAM, mmap lets the OS
//    manage which portions reside in physical memory at any time.
//
// 2. BATCHED PROCESSING: By mapping a file and accessing it in chunks (batches),
//    we can process tensors larger than physical memory. The OS swaps pages
//    in/out transparently.
//
// 3. ZERO-COPY I/O: No need to read() into a buffer then process. The file
//    IS the buffer.
//
// Platform differences:
// ---------------------
// POSIX (Linux/macOS): Uses mmap(), munmap(), msync()
// Windows: Uses CreateFileMapping(), MapViewOfFile(), UnmapViewOfFile()
//
// The MmapFile class below abstracts these differences.

#if defined(_WIN32) || defined(_WIN64)
    #define TINYTORCH_WINDOWS
    #include <windows.h>
#else
    #define TINYTORCH_POSIX
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <unistd.h>
#endif

// ============================================================================
// MmapFile: Cross-platform memory-mapped file wrapper
// ============================================================================
class MmapFile {
private:
    void* _mapping;          // Base address of mapped region
    std::size_t _size;       // Size of mapped region in bytes
    std::string _filepath;   // Path to backing file
    bool _is_mapped;         // Whether mapping is currently active

#if defined(TINYTORCH_WINDOWS)
    HANDLE _file_handle;
    HANDLE _mapping_handle;
#else
    int _fd;                 // File descriptor
#endif

public:
    MmapFile()
        : _mapping(nullptr), _size(0), _is_mapped(false)
#if defined(TINYTORCH_WINDOWS)
        , _file_handle(INVALID_HANDLE_VALUE), _mapping_handle(nullptr)
#else
        , _fd(-1)
#endif
    {}

    // Create or open a memory-mapped file
    bool open(const std::string& filepath, std::size_t size) {
        close();
        _filepath = filepath;
        _size = size;

#if defined(TINYTORCH_WINDOWS)
        // Windows: CreateFile + CreateFileMapping + MapViewOfFile
        _file_handle = CreateFileA(
            filepath.c_str(),
            GENERIC_READ | GENERIC_WRITE,
            0,  // no sharing
            nullptr,
            OPEN_ALWAYS,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS,
            nullptr
        );
        if (_file_handle == INVALID_HANDLE_VALUE) {
            return false;
        }

        // Set file size if newly created
        LARGE_INTEGER file_size;
        if (GetFileSizeEx(_file_handle, &file_size) && file_size.QuadPart == 0) {
            LARGE_INTEGER new_size;
            new_size.QuadPart = static_cast<LONGLONG>(size);
            SetFilePointerEx(_file_handle, new_size, nullptr, FILE_BEGIN);
            SetEndOfFile(_file_handle);
        }

        _mapping_handle = CreateFileMappingA(
            _file_handle,
            nullptr,
            PAGE_READWRITE,
            static_cast<DWORD>(size >> 32),  // high 32 bits
            static_cast<DWORD>(size & 0xFFFFFFFF),  // low 32 bits
            nullptr
        );
        if (_mapping_handle == nullptr) {
            CloseHandle(_file_handle);
            return false;
        }

        _mapping = MapViewOfFile(
            _mapping_handle,
            FILE_MAP_ALL_ACCESS,
            0, 0,  // offset (full 64-bit)
            0      // map entire file
        );
        if (_mapping == nullptr) {
            CloseHandle(_mapping_handle);
            CloseHandle(_file_handle);
            return false;
        }

#else
        // POSIX: open + mmap
        _fd = ::open(filepath.c_str(), O_RDWR | O_CREAT, 0666);
        if (_fd < 0) {
            return false;
        }

        // Set file size
        struct stat st;
        if (fstat(_fd, &st) == 0 && st.st_size == 0) {
            if (ftruncate(_fd, static_cast<off_t>(size)) != 0) {
                ::close(_fd);
                return false;
            }
        }

        _mapping = mmap(
            nullptr,           // let kernel choose address
            size,
            PROT_READ | PROT_WRITE,
            MAP_SHARED,        // changes visible to other processes and file
            _fd,
            0                  // offset
        );
        if (_mapping == MAP_FAILED) {
            ::close(_fd);
            _mapping = nullptr;
            return false;
        }
#endif

        _is_mapped = true;
        return true;
    }

    // Flush changes to disk
    void sync() {
        if (!_is_mapped || !_mapping) return;

#if defined(TINYTORCH_WINDOWS)
        FlushViewOfFile(_mapping, _size);
#else
        msync(_mapping, _size, MS_SYNC);
#endif
    }

    // Advise the OS about access pattern (optional optimization)
    void advise_sequential() {
#if defined(TINYTORCH_POSIX)
        if (_is_mapped && _mapping) {
            madvise(_mapping, _size, MADV_SEQUENTIAL);
        }
#endif
    }

    void advise_random() {
#if defined(TINYTORCH_POSIX)
        if (_is_mapped && _mapping) {
            madvise(_mapping, _size, MADV_RANDOM);
        }
#endif
    }

    // Get pointer to data at byte offset
    void* data_at(std::size_t byte_offset) {
        if (!_is_mapped || !_mapping) return nullptr;
        if (byte_offset >= _size) return nullptr;
        return static_cast<char*>(_mapping) + byte_offset;
    }

    const void* data_at(std::size_t byte_offset) const {
        if (!_is_mapped || !_mapping) return nullptr;
        if (byte_offset >= _size) return nullptr;
        return static_cast<const char*>(_mapping) + byte_offset;
    }

    // Get raw pointer to entire mapping
    void* data() { return _mapping; }
    const void* data() const { return _mapping; }

    std::size_t size() const { return _size; }
    bool is_mapped() const { return _is_mapped; }
    const std::string& filepath() const { return _filepath; }

    // Unmap and close
    void close() {
        if (_mapping && _is_mapped) {
#if defined(TINYTORCH_WINDOWS)
            UnmapViewOfFile(_mapping);
            if (_mapping_handle) CloseHandle(_mapping_handle);
            if (_file_handle != INVALID_HANDLE_VALUE) CloseHandle(_file_handle);
#else
            munmap(_mapping, _size);
            if (_fd >= 0) ::close(_fd);
#endif
        }
        _mapping = nullptr;
        _size = 0;
        _is_mapped = false;
        _filepath.clear();
#if defined(TINYTORCH_WINDOWS)
        _file_handle = INVALID_HANDLE_VALUE;
        _mapping_handle = nullptr;
#else
        _fd = -1;
#endif
    }

    ~MmapFile() {
        close();
    }

    // Non-copyable
    MmapFile(const MmapFile&) = delete;
    MmapFile& operator=(const MmapFile&) = delete;

    // Movable
    MmapFile(MmapFile&& other) noexcept
        : _mapping(other._mapping), _size(other._size),
          _filepath(std::move(other._filepath)), _is_mapped(other._is_mapped)
#if defined(TINYTORCH_WINDOWS)
        , _file_handle(other._file_handle), _mapping_handle(other._mapping_handle)
#else
        , _fd(other._fd)
#endif
    {
        other._mapping = nullptr;
        other._size = 0;
        other._is_mapped = false;
#if defined(TINYTORCH_WINDOWS)
        other._file_handle = INVALID_HANDLE_VALUE;
        other._mapping_handle = nullptr;
#else
        other._fd = -1;
#endif
    }

    MmapFile& operator=(MmapFile&& other) noexcept {
        if (this != &other) {
            close();
            _mapping = other._mapping;
            _size = other._size;
            _filepath = std::move(other._filepath);
            _is_mapped = other._is_mapped;
#if defined(TINYTORCH_WINDOWS)
            _file_handle = other._file_handle;
            _mapping_handle = other._mapping_handle;
            other._file_handle = INVALID_HANDLE_VALUE;
            other._mapping_handle = nullptr;
#else
            _fd = other._fd;
            other._fd = -1;
#endif
            other._mapping = nullptr;
            other._size = 0;
            other._is_mapped = false;
        }
        return *this;
    }
};

// ============================================================================
// StreamConfig: Configuration for streaming/batched tensor operations
// ============================================================================
// What is this?
// -------------
// StreamConfig controls how large tensors are processed in batches when they
// exceed available memory. Instead of loading the entire tensor into RAM, we:
//
// 1. Store tensor data in a memory-mapped file (via MmapFile)
// 2. Process it in chunks of `batch_size` elements at a time
// 3. Each batch is loaded into a small in-memory buffer, operated on, then
//    written back to the mmap file
//
// Why batch?
// ----------
// - Memory efficiency: Only `batch_size` elements need to be in RAM at once
// - Cache friendliness: Small batches fit in CPU cache
// - Out-of-core processing: Tensors larger than RAM can still be processed
//
// Tuning:
// -------
// - batch_size: Larger = fewer I/O ops but more RAM. Smaller = less RAM but
//   more I/O overhead. A good starting point is 1M-10M elements.
// - max_memory_bytes: Upper bound on RAM usage. If batch_size * sizeof(T)
//   exceeds this, batch_size is reduced automatically.

struct StreamConfig {
    std::size_t batch_size;       // Number of elements per batch (default: 1M)
    std::size_t max_memory_bytes; // Max RAM to use for batch buffers (default: 256MB)
    std::string temp_dir;         // Directory for mmap backing files
    bool auto_cleanup;            // Delete temp files on destruction

    StreamConfig()
        : batch_size(1024 * 1024),           // 1M elements
          max_memory_bytes(256 * 1024 * 1024), // 256MB
          temp_dir("."),
          auto_cleanup(true)
    {}

    // Adjust batch size based on element size to stay within memory budget
    std::size_t effective_batch_size(std::size_t element_size) const {
        std::size_t max_elements = max_memory_bytes / element_size;
        return std::min(batch_size, max_elements);
    }
};

// ============================================================================
// Forward declarations
// ============================================================================
template<typename T>
class StreamTensor;

// Free function operators for StreamTensor
template<typename T>
StreamTensor<T>* stream_add(const StreamTensor<T>* a, const StreamTensor<T>* b,
                            const StreamConfig& config = StreamConfig{});

template<typename T>
StreamTensor<T>* stream_subtract(const StreamTensor<T>* a, const StreamTensor<T>* b,
                                 const StreamConfig& config = StreamConfig{});

template<typename T>
StreamTensor<T>* stream_multiply(const StreamTensor<T>* a, const StreamTensor<T>* b,
                                 const StreamConfig& config = StreamConfig{});

template<typename T>
StreamTensor<T>* stream_divide(const StreamTensor<T>* a, const StreamTensor<T>* b,
                               const StreamConfig& config = StreamConfig{});

// ============================================================================
// StreamTensor: Memory-mapped tensor with batched streaming operations
// ============================================================================
// What is StreamTensor?
// ---------------------
// StreamTensor is a tensor whose data lives in a memory-mapped file rather
// than RAM. Operations on StreamTensors are performed in batches: a small
// chunk is read into a RAM buffer, the operation is performed, and the
// result is written back to the mmap file.
//
// This allows processing tensors that are much larger than available RAM,
// at the cost of I/O latency.
//
// When to use StreamTensor vs Tensor:
// ------------------------------------
// - Use Tensor when data fits comfortably in RAM (fast, low overhead)
// - Use StreamTensor when data exceeds RAM or you need persistence
//
// The `stream` parameter in operations controls the batch size and memory
// usage. A larger batch size means fewer I/O operations but more RAM usage.

template<typename T>
class StreamTensor {
private:
    MmapFile _mmap;            // Memory-mapped file backing storage
    std::size_t* _shape;       // Shape array
    std::size_t* _stride;      // Stride array
    std::size_t _ndim;         // Number of dimensions
    std::size_t _total_size;   // Total number of elements
    StreamConfig _config;      // Streaming configuration

    void compute_strides() {
        if (_ndim == 0) {
            _stride = nullptr;
            return;
        }
        _stride = new std::size_t[_ndim];
        _stride[_ndim - 1] = 1;
        for (std::size_t i = _ndim - 1; i > 0; --i) {
            _stride[i - 1] = _stride[i] * _shape[i];
        }
    }

    std::size_t compute_total_size() const {
        if (_ndim == 0) return 1;
        std::size_t size = 1;
        for (std::size_t i = 0; i < _ndim; ++i) {
            size *= _shape[i];
        }
        return size;
    }

    // Get byte offset for element index
    std::size_t byte_offset(std::size_t element_index) const {
        return element_index * sizeof(T);
    }

public:
    // Default constructor
    StreamTensor() : _shape(nullptr), _stride(nullptr),
                     _ndim(0), _total_size(0) {}

    // Create a StreamTensor with given shape, backed by mmap file
    StreamTensor(const std::size_t* shape, std::size_t ndim,
                 const StreamConfig& config = StreamConfig{},
                 T fill_value = T{})
        : _ndim(ndim), _config(config) {

        _shape = new std::size_t[_ndim];
        std::memcpy(_shape, shape, _ndim * sizeof(std::size_t));
        _total_size = compute_total_size();

        // Create mmap backing file
        std::string filepath = _config.temp_dir + "/stream_tensor_" +
                               std::to_string(reinterpret_cast<std::uintptr_t>(this)) + ".bin";

        if (!_mmap.open(filepath, _total_size * sizeof(T))) {
            delete[] _shape;
            throw std::runtime_error("Failed to create memory-mapped file: " + filepath);
        }

        compute_strides();

        // Initialize with fill value
        fill_batched(fill_value);
    }

    // Create StreamTensor from regular Tensor (copies data to mmap)
    StreamTensor(const T* data, std::size_t size,
                 const StreamConfig& config = StreamConfig{})
        : _ndim(1), _total_size(size), _config(config) {

        _shape = new std::size_t[1];
        _shape[0] = size;

        std::string filepath = _config.temp_dir + "/stream_tensor_" +
                               std::to_string(reinterpret_cast<std::uintptr_t>(this)) + ".bin";

        if (!_mmap.open(filepath, _total_size * sizeof(T))) {
            delete[] _shape;
            throw std::runtime_error("Failed to create memory-mapped file: " + filepath);
        }

        compute_strides();

        // Copy data in batches
        std::size_t batch = effective_batch_size();
        for (std::size_t offset = 0; offset < _total_size; offset += batch) {
            std::size_t count = std::min(batch, _total_size - offset);
            std::size_t byte_off = byte_offset(offset);
            void* dest = _mmap.data_at(byte_off);
            if (dest) {
                std::memcpy(dest, data + offset, count * sizeof(T));
            }
        }
    }

    // Create StreamTensor from 1D initializer list
    StreamTensor(std::initializer_list<T> data,
                 const StreamConfig& config = StreamConfig{})
        : StreamTensor(data.begin(), data.size(), config) {}

    // Copy constructor
    StreamTensor(const StreamTensor& other)
        : _ndim(other._ndim), _total_size(other._total_size),
          _config(other._config) {

        _shape = new std::size_t[_ndim];
        std::memcpy(_shape, other._shape, _ndim * sizeof(std::size_t));

        std::string filepath = _config.temp_dir + "/stream_tensor_copy_" +
                               std::to_string(reinterpret_cast<std::uintptr_t>(this)) + ".bin";

        if (!_mmap.open(filepath, _total_size * sizeof(T))) {
            delete[] _shape;
            throw std::runtime_error("Failed to create memory-mapped file for copy");
        }

        compute_strides();

        // Copy data in batches
        std::size_t batch = effective_batch_size();
        for (std::size_t offset = 0; offset < _total_size; offset += batch) {
            std::size_t count = std::min(batch, _total_size - offset);
            const void* src = other._mmap.data_at(byte_offset(offset));
            void* dest = _mmap.data_at(byte_offset(offset));
            if (src && dest) {
                std::memcpy(dest, src, count * sizeof(T));
            }
        }
    }

    // Move constructor
    StreamTensor(StreamTensor&& other) noexcept
        : _mmap(std::move(other._mmap)), _shape(other._shape),
          _stride(other._stride), _ndim(other._ndim),
          _total_size(other._total_size), _config(other._config) {
        other._shape = nullptr;
        other._stride = nullptr;
        other._ndim = 0;
        other._total_size = 0;
    }

    // Copy assignment
    StreamTensor& operator=(const StreamTensor& other) {
        if (this != &other) {
            _mmap.close();
            delete[] _shape;
            delete[] _stride;

            _ndim = other._ndim;
            _total_size = other._total_size;
            _config = other._config;

            _shape = new std::size_t[_ndim];
            std::memcpy(_shape, other._shape, _ndim * sizeof(std::size_t));

            std::string filepath = _config.temp_dir + "/stream_tensor_assign_" +
                                   std::to_string(reinterpret_cast<std::uintptr_t>(this)) + ".bin";

            if (!_mmap.open(filepath, _total_size * sizeof(T))) {
                delete[] _shape;
                throw std::runtime_error("Failed to create memory-mapped file for assignment");
            }

            compute_strides();

            std::size_t batch = effective_batch_size();
            for (std::size_t offset = 0; offset < _total_size; offset += batch) {
                std::size_t count = std::min(batch, _total_size - offset);
                const void* src = other._mmap.data_at(byte_offset(offset));
                void* dest = _mmap.data_at(byte_offset(offset));
                if (src && dest) {
                    std::memcpy(dest, src, count * sizeof(T));
                }
            }
        }
        return *this;
    }

    // Move assignment
    StreamTensor& operator=(StreamTensor&& other) noexcept {
        if (this != &other) {
            _mmap.close();
            delete[] _shape;
            delete[] _stride;

            _mmap = std::move(other._mmap);
            _shape = other._shape;
            _stride = other._stride;
            _ndim = other._ndim;
            _total_size = other._total_size;
            _config = other._config;

            other._shape = nullptr;
            other._stride = nullptr;
            other._ndim = 0;
            other._total_size = 0;
        }
        return *this;
    }

    ~StreamTensor() {
        delete[] _shape;
        delete[] _stride;
        // _mmap closes automatically, flushing if needed
    }

    // Accessors
    const std::size_t* shape() const { return _shape; }
    const std::size_t* stride() const { return _stride; }
    std::size_t ndim() const { return _ndim; }
    std::size_t total_size() const { return _total_size; }
    const StreamConfig& config() const { return _config; }
    StreamConfig& config() { return _config; }

    // Get effective batch size for this type (public for testing/inspection)
    std::size_t effective_batch_size() const {
        return _config.effective_batch_size(sizeof(T));
    }

    // Get pointer to element (direct mmap access - no batching)
    const T* data() const { return static_cast<const T*>(_mmap.data()); }
    T* data() { return static_cast<T*>(_mmap.data()); }

    // Flush pending writes to disk
    void sync() { _mmap.sync(); }

    // Fill entire tensor with a value (batched)
    void fill_batched(T value) {
        std::size_t batch = effective_batch_size();
        T* buffer = new T[batch];
        for (std::size_t i = 0; i < batch; ++i) buffer[i] = value;

        for (std::size_t offset = 0; offset < _total_size; offset += batch) {
            std::size_t count = std::min(batch, _total_size - offset);
            void* dest = _mmap.data_at(byte_offset(offset));
            if (dest) {
                std::memcpy(dest, buffer, count * sizeof(T));
            }
        }
        delete[] buffer;
    }

    // Read a single element (direct mmap access)
    T get_element(std::size_t index) const {
        if (index >= _total_size) {
            throw std::invalid_argument("Index out of bounds");
        }
        const T* ptr = static_cast<const T*>(_mmap.data_at(byte_offset(index)));
        if (!ptr) throw std::runtime_error("Failed to access mmap data");
        return *ptr;
    }

    // Write a single element (direct mmap access)
    void set_element(std::size_t index, T value) {
        if (index >= _total_size) {
            throw std::invalid_argument("Index out of bounds");
        }
        T* ptr = static_cast<T*>(_mmap.data_at(byte_offset(index)));
        if (!ptr) throw std::runtime_error("Failed to access mmap data");
        *ptr = value;
    }

    // 1D indexing (direct mmap access)
    T operator()(std::size_t i) const {
        if (_ndim != 1) {
            throw std::invalid_argument("StreamTensor operator() requires 1D tensor");
        }
        if (i >= _shape[0]) {
            throw std::invalid_argument("Index out of bounds");
        }
        return get_element(i * _stride[0]);
    }

    T& operator()(std::size_t i) {
        if (_ndim != 1) {
            throw std::invalid_argument("StreamTensor operator() requires 1D tensor");
        }
        if (i >= _shape[0]) {
            throw std::invalid_argument("Index out of bounds");
        }
        T* ptr = static_cast<T*>(_mmap.data_at(byte_offset(i * _stride[0])));
        if (!ptr) throw std::runtime_error("Failed to access mmap data");
        return *ptr;
    }

    // 2D indexing (direct mmap access)
    T operator()(std::size_t i, std::size_t j) const {
        if (_ndim != 2) {
            throw std::invalid_argument("StreamTensor operator() requires 2D tensor");
        }
        if (i >= _shape[0]) {
            throw std::invalid_argument("Row index out of bounds");
        }
        if (j >= _shape[1]) {
            throw std::invalid_argument("Column index out of bounds");
        }
        return get_element(i * _stride[0] + j * _stride[1]);
    }

    T& operator()(std::size_t i, std::size_t j) {
        if (_ndim != 2) {
            throw std::invalid_argument("StreamTensor operator() requires 2D tensor");
        }
        if (i >= _shape[0]) {
            throw std::invalid_argument("Row index out of bounds");
        }
        if (j >= _shape[1]) {
            throw std::invalid_argument("Column index out of bounds");
        }
        T* ptr = static_cast<T*>(_mmap.data_at(byte_offset(i * _stride[0] + j * _stride[1])));
        if (!ptr) throw std::runtime_error("Failed to access mmap data");
        return *ptr;
    }

    // ========================================================================
    // Batched Element-wise Operations
    // ========================================================================
    // These operations process the tensor in chunks (batches). For each batch:
    // 1. Read batch from mmap file into RAM buffer
    // 2. Perform the operation on the buffer
    // 3. Write result back to mmap file
    //
    // This allows processing tensors larger than available RAM, since only
    // `batch_size` elements need to be in memory at any time.

    StreamTensor<T>* batched_unary_op(std::function<T(T)> op,
                                       const StreamConfig& config = StreamConfig{}) const {
        StreamTensor<T>* result = new StreamTensor<T>(_shape, _ndim, config);
        std::size_t batch = std::min(effective_batch_size(), config.effective_batch_size(sizeof(T)));
        T* in_buffer = new T[batch];
        T* out_buffer = new T[batch];

        for (std::size_t offset = 0; offset < _total_size; offset += batch) {
            std::size_t count = std::min(batch, _total_size - offset);

            // Swap IN: read batch from mmap
            const void* src = _mmap.data_at(byte_offset(offset));
            if (src) std::memcpy(in_buffer, src, count * sizeof(T));

            // Process batch in RAM
            for (std::size_t i = 0; i < count; ++i) {
                out_buffer[i] = op(in_buffer[i]);
            }

            // Swap OUT: write batch to mmap
            void* dest = result->_mmap.data_at(byte_offset(offset));
            if (dest) std::memcpy(dest, out_buffer, count * sizeof(T));
        }

        delete[] in_buffer;
        delete[] out_buffer;
        return result;
    }

    StreamTensor<T>* batched_binary_op(const StreamTensor<T>* other,
                                        std::function<T(T, T)> op,
                                        const StreamConfig& config = StreamConfig{}) const {
        if (_total_size != other->_total_size) {
            throw std::invalid_argument("StreamTensor shapes must match for binary operation");
        }

        StreamTensor<T>* result = new StreamTensor<T>(_shape, _ndim, config);
        std::size_t batch = std::min(effective_batch_size(), config.effective_batch_size(sizeof(T)));
        T* a_buffer = new T[batch];
        T* b_buffer = new T[batch];
        T* out_buffer = new T[batch];

        for (std::size_t offset = 0; offset < _total_size; offset += batch) {
            std::size_t count = std::min(batch, _total_size - offset);

            // Swap IN: read batches from both mmap files
            const void* src_a = _mmap.data_at(byte_offset(offset));
            const void* src_b = other->_mmap.data_at(byte_offset(offset));
            if (src_a) std::memcpy(a_buffer, src_a, count * sizeof(T));
            if (src_b) std::memcpy(b_buffer, src_b, count * sizeof(T));

            // Process batch in RAM
            for (std::size_t i = 0; i < count; ++i) {
                out_buffer[i] = op(a_buffer[i], b_buffer[i]);
            }

            // Swap OUT: write batch to result mmap file
            void* dest = result->_mmap.data_at(byte_offset(offset));
            if (dest) std::memcpy(dest, out_buffer, count * sizeof(T));
        }

        delete[] a_buffer;
        delete[] b_buffer;
        delete[] out_buffer;
        return result;
    }

    StreamTensor<T>* batched_scalar_op(T scalar, std::function<T(T, T)> op,
                                        const StreamConfig& config = StreamConfig{}) const {
        StreamTensor<T>* result = new StreamTensor<T>(_shape, _ndim, config);
        std::size_t batch = std::min(effective_batch_size(), config.effective_batch_size(sizeof(T)));
        T* in_buffer = new T[batch];
        T* out_buffer = new T[batch];

        for (std::size_t offset = 0; offset < _total_size; offset += batch) {
            std::size_t count = std::min(batch, _total_size - offset);

            // Swap IN
            const void* src = _mmap.data_at(byte_offset(offset));
            if (src) std::memcpy(in_buffer, src, count * sizeof(T));

            // Process
            for (std::size_t i = 0; i < count; ++i) {
                out_buffer[i] = op(in_buffer[i], scalar);
            }

            // Swap OUT
            void* dest = result->_mmap.data_at(byte_offset(offset));
            if (dest) std::memcpy(dest, out_buffer, count * sizeof(T));
        }

        delete[] in_buffer;
        delete[] out_buffer;
        return result;
    }

    // Public batched operation interfaces
    StreamTensor<T>* add(const StreamTensor<T>* other,
                         const StreamConfig& config = StreamConfig{}) const {
        return batched_binary_op(other, std::plus<T>(), config);
    }

    StreamTensor<T>* subtract(const StreamTensor<T>* other,
                              const StreamConfig& config = StreamConfig{}) const {
        return batched_binary_op(other, std::minus<T>(), config);
    }

    StreamTensor<T>* multiply(const StreamTensor<T>* other,
                              const StreamConfig& config = StreamConfig{}) const {
        return batched_binary_op(other, std::multiplies<T>(), config);
    }

    StreamTensor<T>* divide(const StreamTensor<T>* other,
                            const StreamConfig& config = StreamConfig{}) const {
        return batched_binary_op(other, std::divides<T>(), config);
    }

    StreamTensor<T>* add_scalar(T scalar,
                                const StreamConfig& config = StreamConfig{}) const {
        return batched_scalar_op(scalar, std::plus<T>(), config);
    }

    StreamTensor<T>* subtract_scalar(T scalar,
                                     const StreamConfig& config = StreamConfig{}) const {
        return batched_scalar_op(scalar, std::minus<T>(), config);
    }

    StreamTensor<T>* multiply_scalar(T scalar,
                                     const StreamConfig& config = StreamConfig{}) const {
        return batched_scalar_op(scalar, std::multiplies<T>(), config);
    }

    StreamTensor<T>* divide_scalar(T scalar,
                                   const StreamConfig& config = StreamConfig{}) const {
        return batched_scalar_op(scalar, std::divides<T>(), config);
    }

    StreamTensor<T>* negate(const StreamConfig& config = StreamConfig{}) const {
        return batched_unary_op(std::negate<T>(), config);
    }

    StreamTensor<T>* abs(const StreamConfig& config = StreamConfig{}) const {
        return batched_unary_op([](T x) { return x < T{} ? -x : x; }, config);
    }

    // ========================================================================
    // Batched Reduction Operations
    // ========================================================================
    // Reductions accumulate results across batches. Each batch is processed
    // independently and the partial result is carried forward.

    T batched_sum(const StreamConfig& config = StreamConfig{}) const {
        std::size_t batch = std::min(effective_batch_size(), config.effective_batch_size(sizeof(T)));
        T* buffer = new T[batch];
        T result = T{};

        for (std::size_t offset = 0; offset < _total_size; offset += batch) {
            std::size_t count = std::min(batch, _total_size - offset);

            // Swap IN
            const void* src = _mmap.data_at(byte_offset(offset));
            if (src) std::memcpy(buffer, src, count * sizeof(T));

            // Accumulate
            for (std::size_t i = 0; i < count; ++i) {
                result += buffer[i];
            }
        }

        delete[] buffer;
        return result;
    }

    T batched_mean(const StreamConfig& config = StreamConfig{}) const {
        return batched_sum(config) / static_cast<T>(_total_size);
    }

    T batched_max(const StreamConfig& config = StreamConfig{}) const {
        if (_total_size == 0) {
            throw std::runtime_error("Cannot find max of empty StreamTensor");
        }

        std::size_t batch = std::min(effective_batch_size(), config.effective_batch_size(sizeof(T)));
        T* buffer = new T[batch];
        T result = static_cast<T>(-1e308); // Very small initial value
        bool first = true;

        for (std::size_t offset = 0; offset < _total_size; offset += batch) {
            std::size_t count = std::min(batch, _total_size - offset);

            const void* src = _mmap.data_at(byte_offset(offset));
            if (src) std::memcpy(buffer, src, count * sizeof(T));

            for (std::size_t i = 0; i < count; ++i) {
                if (first || buffer[i] > result) {
                    result = buffer[i];
                    first = false;
                }
            }
        }

        delete[] buffer;
        return result;
    }

    T batched_min(const StreamConfig& config = StreamConfig{}) const {
        if (_total_size == 0) {
            throw std::runtime_error("Cannot find min of empty StreamTensor");
        }

        std::size_t batch = std::min(effective_batch_size(), config.effective_batch_size(sizeof(T)));
        T* buffer = new T[batch];
        T result = static_cast<T>(1e308); // Very large initial value
        bool first = true;

        for (std::size_t offset = 0; offset < _total_size; offset += batch) {
            std::size_t count = std::min(batch, _total_size - offset);

            const void* src = _mmap.data_at(byte_offset(offset));
            if (src) std::memcpy(buffer, src, count * sizeof(T));

            for (std::size_t i = 0; i < count; ++i) {
                if (first || buffer[i] < result) {
                    result = buffer[i];
                    first = false;
                }
            }
        }

        delete[] buffer;
        return result;
    }

    // ========================================================================
    // Batched Dot Product
    // ========================================================================
    T batched_dot(const StreamTensor<T>* other,
                  const StreamConfig& config = StreamConfig{}) const {
        if (_ndim != 1 || other->_ndim != 1) {
            throw std::invalid_argument("Dot product requires 1D StreamTensors");
        }
        if (_total_size != other->_total_size) {
            throw std::invalid_argument("StreamTensor shapes must match for dot product");
        }

        std::size_t batch = std::min(effective_batch_size(), config.effective_batch_size(sizeof(T)));
        T* a_buffer = new T[batch];
        T* b_buffer = new T[batch];
        T result = T{};

        for (std::size_t offset = 0; offset < _total_size; offset += batch) {
            std::size_t count = std::min(batch, _total_size - offset);

            // Swap IN: read batches from both tensors
            const void* src_a = _mmap.data_at(byte_offset(offset));
            const void* src_b = other->_mmap.data_at(byte_offset(offset));
            if (src_a) std::memcpy(a_buffer, src_a, count * sizeof(T));
            if (src_b) std::memcpy(b_buffer, src_b, count * sizeof(T));

            // Accumulate dot product
            for (std::size_t i = 0; i < count; ++i) {
                result += a_buffer[i] * b_buffer[i];
            }
        }

        delete[] a_buffer;
        delete[] b_buffer;
        return result;
    }

    // ========================================================================
    // Batched Reshape
    // ========================================================================
    StreamTensor<T>* reshape(const std::size_t* new_shape, std::size_t new_ndim,
                             const StreamConfig& config = StreamConfig{}) const {
        std::size_t new_total = 1;
        for (std::size_t i = 0; i < new_ndim; ++i) {
            new_total *= new_shape[i];
        }
        if (new_total != _total_size) {
            throw std::invalid_argument("Reshape must preserve total number of elements");
        }

        StreamTensor<T>* result = new StreamTensor<T>(new_shape, new_ndim, config);

        // Data layout is identical, just copy raw bytes in batches
        std::size_t batch = std::min(effective_batch_size(), config.effective_batch_size(sizeof(T)));
        T* buffer = new T[batch];

        for (std::size_t offset = 0; offset < _total_size; offset += batch) {
            std::size_t count = std::min(batch, _total_size - offset);
            const void* src = _mmap.data_at(byte_offset(offset));
            void* dest = result->_mmap.data_at(byte_offset(offset));
            if (src && dest) {
                std::memcpy(dest, src, count * sizeof(T));
            }
        }

        delete[] buffer;
        return result;
    }

    // ========================================================================
    // Batched Transpose (2D only)
    // ========================================================================
    // Transpose requires strided access, so we process row-by-row in batches
    StreamTensor<T>* transpose(const StreamConfig& config = StreamConfig{}) const {
        if (_ndim != 2) {
            throw std::invalid_argument("Transpose is only defined for 2D StreamTensors");
        }

        std::size_t new_shape[] = {_shape[1], _shape[0]};
        StreamTensor<T>* result = new StreamTensor<T>(new_shape, 2, config);

        // Process in batches of rows
        std::size_t batch_rows = std::max(static_cast<std::size_t>(1),
                                          effective_batch_size() / _shape[1]);

        T* row_buffer = new T[_shape[1]];

        for (std::size_t row_start = 0; row_start < _shape[0]; row_start += batch_rows) {
            std::size_t row_end = std::min(row_start + batch_rows, _shape[0]);

            for (std::size_t i = row_start; i < row_end; ++i) {
                // Read row i from source
                for (std::size_t j = 0; j < _shape[1]; ++j) {
                    row_buffer[j] = get_element(i * _shape[1] + j);
                }

                // Write as column j in result (row j of transposed)
                for (std::size_t j = 0; j < _shape[1]; ++j) {
                    result->set_element(j * _shape[0] + i, row_buffer[j]);
                }
            }
        }

        delete[] row_buffer;
        return result;
    }

    // ========================================================================
    // Batched Clamp
    // ========================================================================
    StreamTensor<T>* clamp(T min_val, T max_val,
                           const StreamConfig& config = StreamConfig{}) const {
        if (min_val > max_val) {
            throw std::invalid_argument("min_val must be <= max_val");
        }

        return batched_unary_op([min_val, max_val](T x) {
            return x < min_val ? min_val : (x > max_val ? max_val : x);
        }, config);
    }

    // ========================================================================
    // Batched Comparison Operations
    // ========================================================================
    StreamTensor<bool>* greater_than(T threshold,
                                     const StreamConfig& config = StreamConfig{}) const {
        StreamTensor<bool>* result = new StreamTensor<bool>(_shape, _ndim, config);
        std::size_t batch = std::min(effective_batch_size(), config.effective_batch_size(sizeof(T)));
        T* in_buffer = new T[batch];
        bool* out_buffer = new bool[batch];

        for (std::size_t offset = 0; offset < _total_size; offset += batch) {
            std::size_t count = std::min(batch, _total_size - offset);

            const void* src = _mmap.data_at(byte_offset(offset));
            if (src) std::memcpy(in_buffer, src, count * sizeof(T));

            for (std::size_t i = 0; i < count; ++i) {
                out_buffer[i] = in_buffer[i] > threshold;
            }

            void* dest = result->_mmap.data_at(offset * sizeof(bool));
            if (dest) std::memcpy(dest, out_buffer, count * sizeof(bool));
        }

        delete[] in_buffer;
        delete[] out_buffer;
        return result;
    }

    StreamTensor<bool>* less_than(T threshold,
                                  const StreamConfig& config = StreamConfig{}) const {
        StreamTensor<bool>* result = new StreamTensor<bool>(_shape, _ndim, config);
        std::size_t batch = std::min(effective_batch_size(), config.effective_batch_size(sizeof(T)));
        T* in_buffer = new T[batch];
        bool* out_buffer = new bool[batch];

        for (std::size_t offset = 0; offset < _total_size; offset += batch) {
            std::size_t count = std::min(batch, _total_size - offset);

            const void* src = _mmap.data_at(byte_offset(offset));
            if (src) std::memcpy(in_buffer, src, count * sizeof(T));

            for (std::size_t i = 0; i < count; ++i) {
                out_buffer[i] = in_buffer[i] < threshold;
            }

            void* dest = result->_mmap.data_at(offset * sizeof(bool));
            if (dest) std::memcpy(dest, out_buffer, count * sizeof(bool));
        }

        delete[] in_buffer;
        delete[] out_buffer;
        return result;
    }

    // ========================================================================
    // Stream to regular Tensor (load entire tensor into RAM)
    // ========================================================================
    T* to_flat_array() const {
        T* result = new T[_total_size];
        std::size_t batch = effective_batch_size();

        for (std::size_t offset = 0; offset < _total_size; offset += batch) {
            std::size_t count = std::min(batch, _total_size - offset);
            const void* src = _mmap.data_at(byte_offset(offset));
            if (src) std::memcpy(result + offset, src, count * sizeof(T));
        }

        return result;
    }

    // Friend declarations
    template<typename U>
    friend std::ostream& operator<<(std::ostream& os, const StreamTensor<U>& tensor);

    template<typename U>
    friend StreamTensor<U>* stream_add(const StreamTensor<U>* a, const StreamTensor<U>* b,
                                       const StreamConfig& config);
    template<typename U>
    friend StreamTensor<U>* stream_subtract(const StreamTensor<U>* a, const StreamTensor<U>* b,
                                            const StreamConfig& config);
    template<typename U>
    friend StreamTensor<U>* stream_multiply(const StreamTensor<U>* a, const StreamTensor<U>* b,
                                            const StreamConfig& config);
    template<typename U>
    friend StreamTensor<U>* stream_divide(const StreamTensor<U>* a, const StreamTensor<U>* b,
                                          const StreamConfig& config);

    // Allow all StreamTensor instantiations to access each other's private members.
    // This is needed for operations like greater_than() that return StreamTensor<bool>
    // from a StreamTensor<float>, which are different template instantiations.
    template<typename U>
    friend class StreamTensor;
};

// ============================================================================
// Free function operators for StreamTensor
// ============================================================================
template<typename T>
StreamTensor<T>* stream_add(const StreamTensor<T>* a, const StreamTensor<T>* b,
                            const StreamConfig& config) {
    return a->add(b, config);
}

template<typename T>
StreamTensor<T>* stream_subtract(const StreamTensor<T>* a, const StreamTensor<T>* b,
                                 const StreamConfig& config) {
    return a->subtract(b, config);
}

template<typename T>
StreamTensor<T>* stream_multiply(const StreamTensor<T>* a, const StreamTensor<T>* b,
                                 const StreamConfig& config) {
    return a->multiply(b, config);
}

template<typename T>
StreamTensor<T>* stream_divide(const StreamTensor<T>* a, const StreamTensor<T>* b,
                               const StreamConfig& config) {
    return a->divide(b, config);
}

// ============================================================================
// Stream output operator (prints first/last elements for large tensors)
// ============================================================================
template<typename T>
std::ostream& operator<<(std::ostream& os, const StreamTensor<T>& tensor) {
    if (tensor._ndim == 0) {
        os << tensor.get_element(0);
    } else if (tensor._total_size <= 20) {
        // Small tensor: print all elements
        if (tensor._ndim == 1) {
            os << "[";
            for (std::size_t i = 0; i < tensor._shape[0]; ++i) {
                os << tensor(i);
                if (i != tensor._shape[0] - 1) os << ", ";
            }
            os << "]";
        } else if (tensor._ndim == 2) {
            os << "[";
            for (std::size_t i = 0; i < tensor._shape[0]; ++i) {
                os << "[";
                for (std::size_t j = 0; j < tensor._shape[1]; ++j) {
                    os << tensor(i, j);
                    if (j != tensor._shape[1] - 1) os << ", ";
                }
                os << "]";
                if (i != tensor._shape[0] - 1) os << ", ";
            }
            os << "]";
        }
    } else {
        // Large tensor: print summary
        os << "StreamTensor(ndim=" << tensor._ndim
           << ", shape=[";
        for (std::size_t i = 0; i < tensor._ndim; ++i) {
            os << tensor._shape[i];
            if (i != tensor._ndim - 1) os << ", ";
        }
        os << "], total_size=" << tensor._total_size
           << ", batch_size=" << tensor.effective_batch_size()
           << ", mmap_file=" << tensor._mmap.filepath()
           << ")";
    }
    return os;
}
