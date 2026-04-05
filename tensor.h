#pragma once
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <cstring>
#include <initializer_list>
#include <functional>

template<typename T>
class Tensor {
private:
    T* _data;                    // Raw C-style pointer for data
    std::size_t* _shape;         // Raw C-style pointer for shape array
    std::size_t* _stride;        // Raw C-style pointer for stride array
    std::size_t _ndim;           // Number of dimensions
    std::size_t _total_size;     // Total number of elements

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

    void init_from_flat_array(const T* data, std::size_t size) {
        _data = new T[size];
        std::memcpy(_data, data, size * sizeof(T));
        _total_size = size;
    }

public:
    // Default constructor
    Tensor() : _data(nullptr), _shape(nullptr), _stride(nullptr), 
               _ndim(0), _total_size(0) {}

    // Scalar constructor
    explicit Tensor(T value) : _ndim(0), _total_size(1) {
        _data = new T[1];
        _data[0] = value;
        _shape = nullptr;
        _stride = nullptr;
    }

    // 1D tensor constructor from initializer list
    Tensor(std::initializer_list<T> data) : _ndim(1) {
        _total_size = data.size();
        _data = new T[_total_size];
        std::size_t i = 0;
        for (const auto& val : data) {
            _data[i++] = val;
        }
        _shape = new std::size_t[1];
        _shape[0] = _total_size;
        compute_strides();
    }

    // 1D tensor constructor from raw C pointer
    Tensor(const T* data, std::size_t size) : _ndim(1) {
        _total_size = size;
        _data = new T[_total_size];
        std::memcpy(_data, data, _total_size * sizeof(T));
        _shape = new std::size_t[1];
        _shape[0] = _total_size;
        compute_strides();
    }

    // 2D tensor constructor from nested initializer lists
    Tensor(std::initializer_list<std::initializer_list<T>> data) {
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
                _data[row * _shape[1] + col] = val;
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
            _data[i] = fill_value;
        }
        compute_strides();
    }

    // Copy constructor
    Tensor(const Tensor& other) : _ndim(other._ndim), _total_size(other._total_size) {
        _data = new T[_total_size];
        std::memcpy(_data, other._data, _total_size * sizeof(T));
        
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
            std::memcpy(_data, other._data, _total_size * sizeof(T));
            
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
    const T* data() const { return _data; }
    T* data() { return _data; }

    // Item access for scalars and single-element tensors
    const T& item() const {
        if (_total_size != 1) {
            throw std::runtime_error("item() can only be called on tensors with a single element");
        }
        return _data[0];
    }

    T& item() {
        if (_total_size != 1) {
            throw std::runtime_error("item() can only be called on tensors with a single element");
        }
        return _data[0];
    }

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
            return _data[i * _stride[0]];
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
            return _data[i * _stride[0]];
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
        return _data[i * _stride[0] + j * _stride[1]];
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
        return _data[i * _stride[0] + j * _stride[1]];
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
        return _data[offset];
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
        return _data[offset];
    }

    // Element-wise operations returning raw C pointer to result
    Tensor<T>* add(const Tensor<T>* other) const {
        return elementwise_op(other, std::plus<T>());
    }

    Tensor<T>* subtract(const Tensor<T>* other) const {
        return elementwise_op(other, std::minus<T>());
    }

    Tensor<T>* multiply(const Tensor<T>* other) const {
        return elementwise_op(other, std::multiplies<T>());
    }

    Tensor<T>* divide(const Tensor<T>* other) const {
        return elementwise_op(other, std::divides<T>());
    }

    // Scalar operations returning raw C pointer to result
    Tensor<T>* add_scalar(T scalar) const {
        return scalar_op(scalar, std::plus<T>());
    }

    Tensor<T>* subtract_scalar(T scalar) const {
        return scalar_op(scalar, std::minus<T>());
    }

    Tensor<T>* multiply_scalar(T scalar) const {
        return scalar_op(scalar, std::multiplies<T>());
    }

    Tensor<T>* divide_scalar(T scalar) const {
        return scalar_op(scalar, std::divides<T>());
    }

private:
    // Generic element-wise operation using function object and raw C pointers
    Tensor<T>* elementwise_op(const Tensor<T>* other, std::function<T(T, T)> op) const {
        // Broadcasting logic
        if (_ndim == 0 && other->_ndim == 0) {
            // Scalar + Scalar
            return new Tensor<T>(op(item(), other->item()));
        }
        
        if (_ndim == 0 && other->_ndim == 1) {
            // Scalar + 1D
            T* result_data = new T[other->_total_size];
            for (std::size_t i = 0; i < other->_total_size; ++i) {
                result_data[i] = op(item(), other->_data[i]);
            }
            Tensor<T>* result = new Tensor<T>(other->_shape, other->_ndim);
            delete[] result->_data;
            result->_data = result_data;
            return result;
        }
        
        if (_ndim == 1 && other->_ndim == 0) {
            // 1D + Scalar
            T* result_data = new T[_total_size];
            for (std::size_t i = 0; i < _total_size; ++i) {
                result_data[i] = op(_data[i], other->item());
            }
            Tensor<T>* result = new Tensor<T>(_shape, _ndim);
            delete[] result->_data;
            result->_data = result_data;
            return result;
        }
        
        if (_ndim == 1 && other->_ndim == 1) {
            // 1D + 1D
            if (_shape[0] != other->_shape[0]) {
                throw std::invalid_argument("Tensor shapes must match for element-wise operation");
            }
            T* result_data = new T[_total_size];
            for (std::size_t i = 0; i < _total_size; ++i) {
                result_data[i] = op(_data[i], other->_data[i]);
            }
            Tensor<T>* result = new Tensor<T>(_shape, _ndim);
            delete[] result->_data;
            result->_data = result_data;
            return result;
        }
        
        if (_ndim == 2 && other->_ndim == 2) {
            // 2D + 2D
            if (_shape[0] != other->_shape[0] || _shape[1] != other->_shape[1]) {
                throw std::invalid_argument("Tensor shapes must match for element-wise operation");
            }
            T* result_data = new T[_total_size];
            for (std::size_t i = 0; i < _total_size; ++i) {
                result_data[i] = op(_data[i], other->_data[i]);
            }
            Tensor<T>* result = new Tensor<T>(_shape, _ndim);
            delete[] result->_data;
            result->_data = result_data;
            return result;
        }
        
        throw std::invalid_argument("Unsupported tensor dimensions for element-wise operation");
    }

    // Scalar operation helper returning raw C pointer
    Tensor<T>* scalar_op(T scalar, std::function<T(T, T)> op) const {
        T* result_data = new T[_total_size];
        for (std::size_t i = 0; i < _total_size; ++i) {
            result_data[i] = op(_data[i], scalar);
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
};

// Stream output operator implementation
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
