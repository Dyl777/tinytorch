#pragma once
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <cstring>
#include <initializer_list>
#include <functional>
#include <type_traits>

// ============================================================================
// SFINAE (Substitution Failure Is Not An Error) Type Traits
// ============================================================================
// SFINAE allows the compiler to discard invalid template specializations during
// overload resolution without producing a hard error. We use std::enable_if to
// conditionally enable function templates based on type properties.

// Trait to check if T is an arithmetic type (int, float, double, etc.)
template<typename T>
using EnableIfArithmetic = typename std::enable_if<std::is_arithmetic<T>::value, T>::type;

// Trait to check if T is an integral type (int, long, etc.)
template<typename T>
using EnableIfIntegral = typename std::enable_if<std::is_integral<T>::value, T>::type;

// Trait to check if T is a floating-point type (float, double, long double)
template<typename T>
using EnableIfFloatingPoint = typename std::enable_if<std::is_floating_point<T>::value, T>::type;

// Trait to check if T is a pointer type
template<typename T>
using EnableIfPointer = typename std::enable_if<std::is_pointer<T>::value, T>::type;

// Trait to check if T is NOT a pointer type
template<typename T>
using EnableIfNotPointer = typename std::enable_if<!std::is_pointer<T>::value, T>::type;

// Trait to check if T is trivially copyable (can be safely memcpy'd)
template<typename T>
using EnableIfTriviallyCopyable = typename std::enable_if<std::is_trivially_copyable<T>::value, T>::type;

// ============================================================================
// VOLATILE Keyword Explanation
// ============================================================================
// What is volatile?
// -----------------
// The `volatile` keyword is a type qualifier in C/C++ that tells the compiler
// that a variable's value may change at ANY time, even without any apparent
// modification in the code. This prevents the compiler from applying certain
// optimizations on that variable.
//
// Why is it there?
// ----------------
// 1. MEMORY-MAPPED I/O: When working with hardware registers (e.g., GPU memory,
//    device controllers), the hardware can modify these locations independently.
//    Without volatile, the compiler might cache the value in a register and
//    never re-read from memory, missing hardware updates.
//
// 2. SIGNAL HANDLERS: Variables modified inside signal handlers (e.g., SIGINT,
//    SIGSEGV) must be volatile because the signal can interrupt normal flow
//    at any point.
//
// 3. MULTI-THREADED CODE (legacy): Before C++11's atomic types, volatile was
//    sometimes used for shared variables. NOTE: volatile is NOT a replacement
//    for proper synchronization (std::atomic, mutexes). It only prevents
//    compiler optimizations, not CPU reordering.
//
// 4. SETJMP/LONGJMP: Variables that change between setjmp() and longjmp()
//    calls should be volatile to ensure correct values after the jump.
//
// What volatile PREVENTS the compiler from doing:
// ------------------------------------------------
// - Caching the variable in a CPU register across accesses
// - Reordering reads/writes to the variable
// - Eliminating "redundant" reads (the compiler can't assume two consecutive
//   reads return the same value)
// - Eliminating "dead" writes (the compiler can't assume a write is useless
//   just because the value isn't read afterward)
//
// What volatile does NOT do:
// --------------------------
// - It does NOT provide atomicity (use std::atomic for that)
// - It does NOT provide memory barriers (use std::atomic_thread_fence)
// - It does NOT make code thread-safe by itself
//
// In this Tensor class, volatile is used on data pointers and member functions
// to demonstrate correct handling of hardware-mapped tensor data scenarios,
// such as when tensor memory is backed by GPU memory or DMA buffers.

// ============================================================================
// __asm__ Inline Assembly Explanation
// ============================================================================
// What is __asm__?
// ----------------
// `__asm__` (or `asm`) is a GCC/Clang extension that allows embedding raw
// assembly code directly in C/C++ source. It's used here for low-level
// operations that cannot be expressed in standard C++.
//
// Syntax:
//   __asm__ volatile (
//       "assembly template"
//       : output operands      (optional)
//       : input operands       (optional)
//       : clobbered registers  (optional)
//   );
//
// The `volatile` after __asm__ tells the compiler not to optimize away or
// reorder the assembly block, even if it appears to have no side effects.
//
// Why is it here?
// ---------------
// 1. Memory barriers: Ensuring memory operations complete in order
// 2. Prefetch hints: Telling the CPU to preload data into cache
// 3. CPU-specific instructions: Using SIMD, cache control, etc.
// 4. Debugging/Profiling: Inserting breakpoints or performance counters
//
// NOTE: __asm__ is compiler-specific (GCC/Clang). MSVC uses __asm instead.
// This code uses the GNU extended asm syntax.

// Memory barrier using inline assembly - prevents CPU from reordering memory ops
inline void memory_barrier() {
#if defined(__GNUC__) || defined(__clang__)
    __asm__ volatile (
        "mfence"          // Memory Fence - ensures all memory ops before this
                          // instruction complete before any ops after it
        :                 // no outputs
        :                 // no inputs
        : "memory"        // clobber: tells compiler memory may have changed
    );
#else
    // Fallback for non-GCC compilers: use compiler barrier only
    __asm__ volatile ("" ::: "memory");
#endif
}

// Cache prefetch using inline assembly - hints the CPU to load data into cache
inline void prefetch_read(const volatile void* addr) {
#if defined(__GNUC__) || defined(__clang__)
    __asm__ volatile (
        "prefetcht0 (%0)" // Prefetch to L1 cache with T0 (temporal) hint
        :                 // no outputs
        : "r"(addr)       // input: address in any register
        :                 // no clobbers
    );
#endif
}

// ============================================================================
// Tensor Class Template
// ============================================================================
template<typename T>
class Tensor {
private:
    // Raw C-style pointer for tensor data.
    // volatile is used here to support scenarios where tensor data resides in
    // memory-mapped I/O regions (e.g., GPU VRAM, DMA buffers) where the
    // hardware may modify the data independently of the CPU.
    volatile T* _data;

    // Raw C-style pointers for shape and stride arrays
    std::size_t* _shape;
    std::size_t* _stride;
    std::size_t _ndim;
    std::size_t _total_size;

    // Helper to cast away volatile for memcpy (safe for byte-copying)
    static void* volatile_cast(volatile void* ptr) {
        return const_cast<void*>(ptr);
    }

    static const void* volatile_cast_const(volatile const void* ptr) {
        return const_cast<const void*>(ptr);
    }

    // Compute strides from shape using row-major (C-style) layout
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

    // Compute total number of elements from shape
    std::size_t compute_total_size() const {
        if (_ndim == 0) return 1;
        std::size_t size = 1;
        for (std::size_t i = 0; i < _ndim; ++i) {
            size *= _shape[i];
        }
        return size;
    }

public:
    // Default constructor
    Tensor() : _data(nullptr), _shape(nullptr), _stride(nullptr),
               _ndim(0), _total_size(0) {}

    // Scalar constructor (non-template, always available)
    explicit Tensor(T value) : _ndim(0), _total_size(1) {
        _data = new T[1];
        const_cast<volatile T&>(_data[0]) = value;
        _shape = nullptr;
        _stride = nullptr;
    }

    // 1D tensor constructor from initializer list
    // SFINAE: Only enabled for arithmetic types (int, float, double, etc.)
    Tensor(std::initializer_list<EnableIfArithmetic<T>> data) : _ndim(1) {
        _total_size = data.size();
        _data = new T[_total_size];
        std::size_t i = 0;
        for (const auto& val : data) {
            const_cast<volatile T&>(_data[i++]) = val;
        }
        _shape = new std::size_t[1];
        _shape[0] = _total_size;
        compute_strides();
    }

    // 1D tensor constructor from raw C pointer
    // SFINAE: Only enabled for trivially copyable types
    template<typename U>
    Tensor(const U* data, std::size_t size,
           typename std::enable_if<std::is_trivially_copyable<U>::value>::type* = nullptr)
        : _ndim(1) {
        _total_size = size;
        _data = new T[_total_size];
        std::memcpy(volatile_cast(_data), data, _total_size * sizeof(T));
        _shape = new std::size_t[1];
        _shape[0] = _total_size;
        compute_strides();
    }

    // 2D tensor constructor from nested initializer lists
    // SFINAE: Only enabled for arithmetic types
    Tensor(std::initializer_list<std::initializer_list<EnableIfArithmetic<T>>> data) {
        auto outer_it = data.begin();
        _shape = new std::size_t[2];
        _shape[0] = data.size();
        _shape[1] = outer_it->size();
        _ndim = 2;
        _total_size = _shape[0] * _shape[1];
        _data = new T[_total_size];

        // Validate dimensions and copy data
        std::size_t row = 0;
        for (const auto& row_list : data) {
            if (row_list.size() != _shape[1]) {
                delete[] _data;
                delete[] _shape;
                throw std::invalid_argument("Inconsistent row sizes in 2D tensor");
            }
            std::size_t col = 0;
            for (const auto& val : row_list) {
                const_cast<volatile T&>(_data[row * _shape[1] + col]) = val;
                ++col;
            }
            ++row;
        }
        compute_strides();
    }

    // N-dimensional tensor constructor from raw C pointer arrays
    Tensor(const std::size_t* shape, std::size_t ndim, T fill_value = T{})
        : _ndim(ndim) {
        _shape = new std::size_t[_ndim];
        std::memcpy(_shape, shape, _ndim * sizeof(std::size_t));
        _total_size = compute_total_size();
        _data = new T[_total_size];
        for (std::size_t i = 0; i < _total_size; ++i) {
            const_cast<volatile T&>(_data[i]) = fill_value;
        }
        compute_strides();
    }

    // Copy constructor
    Tensor(const Tensor& other) : _ndim(other._ndim), _total_size(other._total_size) {
        _data = new T[_total_size];
        std::memcpy(volatile_cast(_data), volatile_cast_const(other._data), _total_size * sizeof(T));

        if (_ndim > 0) {
            _shape = new std::size_t[_ndim];
            std::memcpy(_shape, other._shape, _ndim * sizeof(std::size_t));
            compute_strides();
        } else {
            _shape = nullptr;
            _stride = nullptr;
        }
    }

    // Move constructor
    Tensor(Tensor&& other) noexcept
        : _data(other._data), _shape(other._shape), _stride(other._stride),
          _ndim(other._ndim), _total_size(other._total_size) {
        other._data = nullptr;
        other._shape = nullptr;
        other._stride = nullptr;
        other._ndim = 0;
        other._total_size = 0;
    }

    // Copy assignment
    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            delete[] _data;
            delete[] _shape;
            delete[] _stride;

            _ndim = other._ndim;
            _total_size = other._total_size;

            _data = new T[_total_size];
            std::memcpy(volatile_cast(_data), volatile_cast_const(other._data), _total_size * sizeof(T));

            if (_ndim > 0) {
                _shape = new std::size_t[_ndim];
                std::memcpy(_shape, other._shape, _ndim * sizeof(std::size_t));
                compute_strides();
            } else {
                _shape = nullptr;
                _stride = nullptr;
            }
        }
        return *this;
    }

    // Move assignment
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            delete[] _data;
            delete[] _shape;
            delete[] _stride;

            _data = other._data;
            _shape = other._shape;
            _stride = other._stride;
            _ndim = other._ndim;
            _total_size = other._total_size;

            other._data = nullptr;
            other._shape = nullptr;
            other._stride = nullptr;
            other._ndim = 0;
            other._total_size = 0;
        }
        return *this;
    }

    // Destructor
    ~Tensor() {
        delete[] _data;
        delete[] _shape;
        delete[] _stride;
    }

    // Accessors returning raw C pointers
    const std::size_t* shape() const { return _shape; }
    const std::size_t* stride() const { return _stride; }
    std::size_t ndim() const { return _ndim; }
    std::size_t total_size() const { return _total_size; }

    // Const data accessor - returns volatile pointer for hardware-mapped memory
    const volatile T* data() const { return _data; }

    // Mutable data accessor - returns volatile pointer
    volatile T* data() { return _data; }

    // SFINAE: Non-volatile accessor for normal (non-hardware) use cases
    // Only enabled when T is NOT volatile-qualified
    template<typename U = T>
    typename std::enable_if<!std::is_volatile<U>::value, const T*>::type
    data_non_volatile() const {
        return const_cast<const T*>(_data);
    }

    template<typename U = T>
    typename std::enable_if<!std::is_volatile<U>::value, T*>::type
    data_non_volatile() {
        return const_cast<T*>(_data);
    }

    // Item access for scalars and single-element tensors
    const T& item() const {
        if (_total_size != 1) {
            throw std::runtime_error("item() can only be called on tensors with a single element");
        }
        return const_cast<const T&>(_data[0]);
    }

    T& item() {
        if (_total_size != 1) {
            throw std::runtime_error("item() can only be called on tensors with a single element");
        }
        return const_cast<T&>(_data[0]);
    }

    // ========================================================================
    // VOLATILE Member Functions
    // ========================================================================
    // These functions are specifically designed for volatile-qualified tensor
    // data, such as memory-mapped hardware buffers. The volatile qualifier
    // ensures that every read/write actually hits memory (not a cached register
    // value), which is critical when hardware can modify the data independently.

    // Volatile item access - for reading hardware-mapped single-element tensors
    // The volatile keyword ensures each read actually fetches from memory,
    // catching any hardware updates that may have occurred since the last read.
    volatile T& volatile_item() volatile {
        if (_total_size != 1) {
            throw std::runtime_error("volatile_item() can only be called on single-element tensors");
        }
        // Memory barrier before read - ensures we see the latest hardware state
        memory_barrier();
        return const_cast<volatile T&>(_data[0]);
    }

    // Volatile item access (const version)
    const volatile T& volatile_item() const volatile {
        if (_total_size != 1) {
            throw std::runtime_error("volatile_item() can only be called on single-element tensors");
        }
        memory_barrier();
        return _data[0];
    }

    // Volatile 1D indexing - for hardware-mapped 1D tensor data
    // Each access performs an actual memory read, not a cached value
    volatile T& volatile_at(std::size_t i) volatile {
        if (_ndim != 1) {
            throw std::invalid_argument("volatile_at() requires a 1D tensor");
        }
        if (i >= _shape[0]) {
            throw std::invalid_argument("Index " + std::to_string(i) +
                                       " is out of bounds for array of size " +
                                       std::to_string(_shape[0]));
        }
        memory_barrier();
        return const_cast<volatile T&>(_data[i * _stride[0]]);
    }

    const volatile T& volatile_at(std::size_t i) const volatile {
        if (_ndim != 1) {
            throw std::invalid_argument("volatile_at() requires a 1D tensor");
        }
        if (i >= _shape[0]) {
            throw std::invalid_argument("Index " + std::to_string(i) +
                                       " is out of bounds for array of size " +
                                       std::to_string(_shape[0]));
        }
        memory_barrier();
        return _data[i * _stride[0]];
    }

    // Volatile 2D indexing - for hardware-mapped 2D tensor data
    volatile T& volatile_at(std::size_t i, std::size_t j) volatile {
        if (_ndim != 2) {
            throw std::invalid_argument("volatile_at() requires a 2D tensor");
        }
        if (i >= _shape[0]) {
            throw std::invalid_argument("Row index " + std::to_string(i) +
                                       " is out of bounds for tensor with " +
                                       std::to_string(_shape[0]) + " rows");
        }
        if (j >= _shape[1]) {
            throw std::invalid_argument("Column index " + std::to_string(j) +
                                       " is out of bounds for tensor with " +
                                       std::to_string(_shape[1]) + " columns");
        }
        memory_barrier();
        return const_cast<volatile T&>(_data[i * _stride[0] + j * _stride[1]]);
    }

    const volatile T& volatile_at(std::size_t i, std::size_t j) const volatile {
        if (_ndim != 2) {
            throw std::invalid_argument("volatile_at() requires a 2D tensor");
        }
        if (i >= _shape[0]) {
            throw std::invalid_argument("Row index " + std::to_string(i) +
                                       " is out of bounds for tensor with " +
                                       std::to_string(_shape[0]) + " rows");
        }
        if (j >= _shape[1]) {
            throw std::invalid_argument("Column index " + std::to_string(j) +
                                       " is out of bounds for tensor with " +
                                       std::to_string(_shape[1]) + " columns");
        }
        memory_barrier();
        return _data[i * _stride[0] + j * _stride[1]];
    }

    // ========================================================================
    // Standard (non-volatile) Indexing
    // ========================================================================

    // 1D indexing
    const T& operator()(std::size_t i) const {
        if (_ndim == 0) {
            throw std::invalid_argument("Can't index into scalar. Use item() instead");
        }
        if (_ndim == 1) {
            if (i >= _shape[0]) {
                throw std::invalid_argument("Index " + std::to_string(i) +
                                           " is out of bounds for array of size " +
                                           std::to_string(_shape[0]));
            }
            return const_cast<const T&>(_data[i * _stride[0]]);
        }
        throw std::invalid_argument("This is not a 1D tensor. Use appropriate indexing.");
    }

    T& operator()(std::size_t i) {
        if (_ndim == 0) {
            throw std::invalid_argument("Can't index into scalar. Use item() instead");
        }
        if (_ndim == 1) {
            if (i >= _shape[0]) {
                throw std::invalid_argument("Index " + std::to_string(i) +
                                           " is out of bounds for array of size " +
                                           std::to_string(_shape[0]));
            }
            return const_cast<T&>(_data[i * _stride[0]]);
        }
        throw std::invalid_argument("This is not a 1D tensor. Use appropriate indexing.");
    }

    // 2D indexing
    const T& operator()(std::size_t i, std::size_t j) const {
        if (_ndim != 2) {
            throw std::invalid_argument("Can only double index into 2D tensors");
        }
        if (i >= _shape[0]) {
            throw std::invalid_argument("Row index " + std::to_string(i) +
                                       " is out of bounds for tensor with " +
                                       std::to_string(_shape[0]) + " rows");
        }
        if (j >= _shape[1]) {
            throw std::invalid_argument("Column index " + std::to_string(j) +
                                       " is out of bounds for tensor with " +
                                       std::to_string(_shape[1]) + " columns");
        }
        return const_cast<const T&>(_data[i * _stride[0] + j * _stride[1]]);
    }

    T& operator()(std::size_t i, std::size_t j) {
        if (_ndim != 2) {
            throw std::invalid_argument("Can only double index into 2D tensors");
        }
        if (i >= _shape[0]) {
            throw std::invalid_argument("Row index " + std::to_string(i) +
                                       " is out of bounds for tensor with " +
                                       std::to_string(_shape[0]) + " rows");
        }
        if (j >= _shape[1]) {
            throw std::invalid_argument("Column index " + std::to_string(j) +
                                       " is out of bounds for tensor with " +
                                       std::to_string(_shape[1]) + " columns");
        }
        return const_cast<T&>(_data[i * _stride[0] + j * _stride[1]]);
    }

    // N-dimensional indexing using raw C pointer array
    const T& at(const std::size_t* indices) const {
        if (_ndim == 0) {
            throw std::invalid_argument("Can't index into scalar. Use item() instead");
        }

        std::size_t offset = 0;
        for (std::size_t i = 0; i < _ndim; ++i) {
            if (indices[i] >= _shape[i]) {
                throw std::invalid_argument("Index " + std::to_string(indices[i]) +
                                           " is out of bounds for dimension " +
                                           std::to_string(i) + " with size " +
                                           std::to_string(_shape[i]));
            }
            offset += indices[i] * _stride[i];
        }
        return const_cast<const T&>(_data[offset]);
    }

    T& at(const std::size_t* indices) {
        if (_ndim == 0) {
            throw std::invalid_argument("Can't index into scalar. Use item() instead");
        }

        std::size_t offset = 0;
        for (std::size_t i = 0; i < _ndim; ++i) {
            if (indices[i] >= _shape[i]) {
                throw std::invalid_argument("Index " + std::to_string(indices[i]) +
                                           " is out of bounds for dimension " +
                                           std::to_string(i) + " with size " +
                                           std::to_string(_shape[i]));
            }
            offset += indices[i] * _stride[i];
        }
        return const_cast<T&>(_data[offset]);
    }

    // ========================================================================
    // SFINAE-Constrained Element-wise Operations
    // ========================================================================
    // These operations use SFINAE to ensure they are only instantiated for
    // arithmetic types, preventing nonsensical operations on pointers or
    // user-defined types without proper operator overloads.

    // SFINAE: Only enabled for arithmetic types
    template<typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value, Tensor<T>*>::type
    add(const Tensor<T>* other) const {
        return elementwise_op(other, std::plus<T>());
    }

    template<typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value, Tensor<T>*>::type
    subtract(const Tensor<T>* other) const {
        return elementwise_op(other, std::minus<T>());
    }

    template<typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value, Tensor<T>*>::type
    multiply(const Tensor<T>* other) const {
        return elementwise_op(other, std::multiplies<T>());
    }

    template<typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value, Tensor<T>*>::type
    divide(const Tensor<T>* other) const {
        return elementwise_op(other, std::divides<T>());
    }

    // ========================================================================
    // SFINAE-Constrained Scalar Operations
    // ========================================================================

    template<typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value, Tensor<T>*>::type
    add_scalar(T scalar) const {
        return scalar_op(scalar, std::plus<T>());
    }

    template<typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value, Tensor<T>*>::type
    subtract_scalar(T scalar) const {
        return scalar_op(scalar, std::minus<T>());
    }

    template<typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value, Tensor<T>*>::type
    multiply_scalar(T scalar) const {
        return scalar_op(scalar, std::multiplies<T>());
    }

    template<typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value, Tensor<T>*>::type
    divide_scalar(T scalar) const {
        return scalar_op(scalar, std::divides<T>());
    }

    // ========================================================================
    // SFINAE: Dot product (only for floating-point types)
    // ========================================================================
    // Dot product is mathematically meaningful for floating-point types.
    // For integers, overflow behavior is implementation-defined, so we
    // constrain this to floating-point types only.

    template<typename U = T>
    typename std::enable_if<std::is_floating_point<U>::value, T>::type
    dot(const Tensor<T>* other) const {
        if (_ndim != 1 || other->_ndim != 1) {
            throw std::invalid_argument("Dot product requires 1D tensors");
        }
        if (_shape[0] != other->_shape[0]) {
            throw std::invalid_argument("Tensor shapes must match for dot product");
        }

        T result = T{};
        for (std::size_t i = 0; i < _total_size; ++i) {
            result += const_cast<const T&>(_data[i]) * const_cast<const T&>(other->_data[i]);
        }
        return result;
    }

    // ========================================================================
    // SFINAE: Element-wise absolute value (only for arithmetic types)
    // ========================================================================

    template<typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value, Tensor<T>*>::type
    abs() const {
        T* result_data = new T[_total_size];
        for (std::size_t i = 0; i < _total_size; ++i) {
            T val = const_cast<const T&>(_data[i]);
            result_data[i] = val < T{} ? -val : val;
        }
        Tensor<T>* result = new Tensor<T>(_shape, _ndim);
        delete[] result->_data;
        result->_data = result_data;
        return result;
    }

    // ========================================================================
    // SFINAE: Element-wise negation (only for arithmetic types)
    // ========================================================================

    template<typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value, Tensor<T>*>::type
    negate() const {
        T* result_data = new T[_total_size];
        for (std::size_t i = 0; i < _total_size; ++i) {
            result_data[i] = -const_cast<const T&>(_data[i]);
        }
        Tensor<T>* result = new Tensor<T>(_shape, _ndim);
        delete[] result->_data;
        result->_data = result_data;
        return result;
    }

    // ========================================================================
    // SFINAE: Sum reduction (only for arithmetic types)
    // ========================================================================

    template<typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value, T>::type
    sum() const {
        T result = T{};
        for (std::size_t i = 0; i < _total_size; ++i) {
            result += const_cast<const T&>(_data[i]);
        }
        return result;
    }

    // ========================================================================
    // SFINAE: Mean reduction (only for floating-point types)
    // ========================================================================

    template<typename U = T>
    typename std::enable_if<std::is_floating_point<U>::value, T>::type
    mean() const {
        T result = T{};
        for (std::size_t i = 0; i < _total_size; ++i) {
            result += const_cast<const T&>(_data[i]);
        }
        return result / static_cast<T>(_total_size);
    }

    // ========================================================================
    // SFINAE: Max element (only for arithmetic types)
    // ========================================================================

    template<typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value, T>::type
    max() const {
        if (_total_size == 0) {
            throw std::runtime_error("Cannot find max of empty tensor");
        }
        T result = const_cast<const T&>(_data[0]);
        for (std::size_t i = 1; i < _total_size; ++i) {
            T val = const_cast<const T&>(_data[i]);
            if (val > result) {
                result = val;
            }
        }
        return result;
    }

    // ========================================================================
    // SFINAE: Min element (only for arithmetic types)
    // ========================================================================

    template<typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value, T>::type
    min() const {
        if (_total_size == 0) {
            throw std::runtime_error("Cannot find min of empty tensor");
        }
        T result = const_cast<const T&>(_data[0]);
        for (std::size_t i = 1; i < _total_size; ++i) {
            T val = const_cast<const T&>(_data[i]);
            if (val < result) {
                result = val;
            }
        }
        return result;
    }

    // ========================================================================
    // SFINAE: Argmax (only for arithmetic types)
    // ========================================================================
    // Returns the flat index of the maximum element

    template<typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value, std::size_t>::type
    argmax() const {
        if (_total_size == 0) {
            throw std::runtime_error("Cannot find argmax of empty tensor");
        }
        std::size_t max_idx = 0;
        for (std::size_t i = 1; i < _total_size; ++i) {
            if (const_cast<const T&>(_data[i]) > const_cast<const T&>(_data[max_idx])) {
                max_idx = i;
            }
        }
        return max_idx;
    }

    // ========================================================================
    // SFINAE: Argmin (only for arithmetic types)
    // ========================================================================

    template<typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value, std::size_t>::type
    argmin() const {
        if (_total_size == 0) {
            throw std::runtime_error("Cannot find argmin of empty tensor");
        }
        std::size_t min_idx = 0;
        for (std::size_t i = 1; i < _total_size; ++i) {
            if (const_cast<const T&>(_data[i]) < const_cast<const T&>(_data[min_idx])) {
                min_idx = i;
            }
        }
        return min_idx;
    }

    // ========================================================================
    // SFINAE: Reshape (only for arithmetic types)
    // ========================================================================
    // Returns a new tensor with the same data but different shape.
    // The total number of elements must remain the same.

    template<typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value, Tensor<T>*>::type
    reshape(const std::size_t* new_shape, std::size_t new_ndim) const {
        // Validate total size
        std::size_t new_total = 1;
        for (std::size_t i = 0; i < new_ndim; ++i) {
            new_total *= new_shape[i];
        }
        if (new_total != _total_size) {
            throw std::invalid_argument("Reshape must preserve total number of elements");
        }

        Tensor<T>* result = new Tensor<T>(new_shape, new_ndim);
        std::memcpy(volatile_cast(result->_data), volatile_cast_const(_data), _total_size * sizeof(T));
        return result;
    }

    // ========================================================================
    // SFINAE: Transpose (only for 2D arithmetic tensors)
    // ========================================================================

    template<typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value, Tensor<T>*>::type
    transpose() const {
        if (_ndim != 2) {
            throw std::invalid_argument("Transpose is only defined for 2D tensors");
        }

        std::size_t new_shape[] = {_shape[1], _shape[0]};
        Tensor<T>* result = new Tensor<T>(new_shape, 2);

        for (std::size_t i = 0; i < _shape[0]; ++i) {
            for (std::size_t j = 0; j < _shape[1]; ++j) {
                const_cast<volatile T&>(result->_data[j * _shape[0] + i]) =
                    const_cast<const T&>(_data[i * _shape[1] + j]);
            }
        }
        return result;
    }

    // ========================================================================
    // SFINAE: Clamp (only for arithmetic types)
    // ========================================================================
    // Clamps all values to [min_val, max_val]

    template<typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value, Tensor<T>*>::type
    clamp(T min_val, T max_val) const {
        if (min_val > max_val) {
            throw std::invalid_argument("min_val must be <= max_val");
        }

        T* result_data = new T[_total_size];
        for (std::size_t i = 0; i < _total_size; ++i) {
            T val = const_cast<const T&>(_data[i]);
            result_data[i] = val < min_val ? min_val : (val > max_val ? max_val : val);
        }
        Tensor<T>* result = new Tensor<T>(_shape, _ndim);
        delete[] result->_data;
        result->_data = result_data;
        return result;
    }

    // ========================================================================
    // SFINAE: Element-wise comparison (returns Tensor<bool>)
    // ========================================================================

    Tensor<bool>* greater_than(T threshold) const {
        bool* result_data = new bool[_total_size];
        for (std::size_t i = 0; i < _total_size; ++i) {
            result_data[i] = const_cast<const T&>(_data[i]) > threshold;
        }
        Tensor<bool>* result = new Tensor<bool>(_shape, _ndim);
        delete[] result->_data;
        result->_data = result_data;
        return result;
    }

    Tensor<bool>* less_than(T threshold) const {
        bool* result_data = new bool[_total_size];
        for (std::size_t i = 0; i < _total_size; ++i) {
            result_data[i] = const_cast<const T&>(_data[i]) < threshold;
        }
        Tensor<bool>* result = new Tensor<bool>(_shape, _ndim);
        delete[] result->_data;
        result->_data = result_data;
        return result;
    }

    Tensor<bool>* equal_to(T threshold) const {
        bool* result_data = new bool[_total_size];
        for (std::size_t i = 0; i < _total_size; ++i) {
            result_data[i] = const_cast<const T&>(_data[i]) == threshold;
        }
        Tensor<bool>* result = new Tensor<bool>(_shape, _ndim);
        delete[] result->_data;
        result->_data = result_data;
        return result;
    }

private:
    // Generic element-wise operation using function object and raw C pointers
    // SFINAE: Only enabled for arithmetic types
    template<typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value, Tensor<T>*>::type
    elementwise_op(const Tensor<T>* other, std::function<T(T, T)> op) const {
        // Broadcasting logic
        if (_ndim == 0 && other->_ndim == 0) {
            return new Tensor<T>(op(this->item(), other->item()));
        }

        if (_ndim == 0 && other->_ndim == 1) {
            T* result_data = new T[other->_total_size];
            for (std::size_t i = 0; i < other->_total_size; ++i) {
                result_data[i] = op(this->item(), const_cast<const T&>(other->_data[i]));
            }
            Tensor<T>* result = new Tensor<T>(other->_shape, other->_ndim);
            delete[] result->_data;
            result->_data = result_data;
            return result;
        }

        if (_ndim == 1 && other->_ndim == 0) {
            T* result_data = new T[_total_size];
            for (std::size_t i = 0; i < _total_size; ++i) {
                result_data[i] = op(const_cast<const T&>(_data[i]), other->item());
            }
            Tensor<T>* result = new Tensor<T>(_shape, _ndim);
            delete[] result->_data;
            result->_data = result_data;
            return result;
        }

        if (_ndim == 1 && other->_ndim == 1) {
            if (_shape[0] != other->_shape[0]) {
                throw std::invalid_argument("Tensor shapes must match for element-wise operation");
            }
            T* result_data = new T[_total_size];
            for (std::size_t i = 0; i < _total_size; ++i) {
                result_data[i] = op(const_cast<const T&>(_data[i]), const_cast<const T&>(other->_data[i]));
            }
            Tensor<T>* result = new Tensor<T>(_shape, _ndim);
            delete[] result->_data;
            result->_data = result_data;
            return result;
        }

        if (_ndim == 2 && other->_ndim == 2) {
            if (_shape[0] != other->_shape[0] || _shape[1] != other->_shape[1]) {
                throw std::invalid_argument("Tensor shapes must match for element-wise operation");
            }
            T* result_data = new T[_total_size];
            for (std::size_t i = 0; i < _total_size; ++i) {
                result_data[i] = op(const_cast<const T&>(_data[i]), const_cast<const T&>(other->_data[i]));
            }
            Tensor<T>* result = new Tensor<T>(_shape, _ndim);
            delete[] result->_data;
            result->_data = result_data;
            return result;
        }

        throw std::invalid_argument("Unsupported tensor dimensions for element-wise operation");
    }

    // Scalar operation helper returning raw C pointer
    template<typename U = T>
    typename std::enable_if<std::is_arithmetic<U>::value, Tensor<T>*>::type
    scalar_op(T scalar, std::function<T(T, T)> op) const {
        T* result_data = new T[_total_size];
        for (std::size_t i = 0; i < _total_size; ++i) {
            result_data[i] = op(const_cast<const T&>(_data[i]), scalar);
        }
        Tensor<T>* result = new Tensor<T>(_shape, _ndim);
        delete[] result->_data;
        result->_data = result_data;
        return result;
    }

public:
    // Friend stream operator
    template<typename U>
    friend std::ostream& operator<<(std::ostream& os, const Tensor<U>& tensor);

    // Allow all Tensor instantiations to access each other's private members
    // This is needed for operations like greater_than() that return Tensor<bool>
    // from a Tensor<float>, which are different template instantiations.
    template<typename U>
    friend class Tensor;
};

// ============================================================================
// Stream output operator
// ============================================================================
template<typename T>
std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor) {
    if (tensor._ndim == 0) {
        os << tensor.item();
    } else if (tensor._ndim == 1) {
        os << "[";
        for (std::size_t i = 0; i < tensor._shape[0]; ++i) {
            os << tensor(i);
            if (i != tensor._shape[0] - 1) {
                os << ", ";
            }
        }
        os << "]";
    } else if (tensor._ndim == 2) {
        os << "[";
        for (std::size_t i = 0; i < tensor._shape[0]; ++i) {
            os << "[";
            for (std::size_t j = 0; j < tensor._shape[1]; ++j) {
                os << tensor(i, j);
                if (j != tensor._shape[1] - 1) {
                    os << ", ";
                }
            }
            os << "]";
            if (i != tensor._shape[0] - 1) {
                os << ", ";
            }
        }
        os << "]";
    } else {
        os << "Tensor(ndim=" << tensor._ndim << ", size=" << tensor._total_size << ")";
    }
    return os;
}
