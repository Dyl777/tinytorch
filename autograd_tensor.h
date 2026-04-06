#pragma once
#include "auto_tensor.h"
#include <memory>
#include <vector>
#include <functional>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <unordered_set>
#include <unordered_map>

// ============================================================================
// Autograd Engine for TinyTorch
// ============================================================================
// Implements automatic differentiation using a computational graph.
// 
// Design:
// -------
// 1. Each AutogradTensor wraps a TensorBase<T> (the data)
// 2. Operations create new AutogradTensors and record the computation graph
// 3. backward() traverses the graph in reverse topological order
// 4. Gradients are accumulated in leaf tensors
// ============================================================================

// Forward declaration
template<typename T>
class AutogradTensor;

// ============================================================================
// Computational Graph Node
// ============================================================================

template<typename T>
struct GradNode {
    AutogradTensor<T>* tensor;
    std::vector<AutogradTensor<T>*> parents;
    std::function<void(const TensorBase<T>*)> backward_fn;
    bool requires_grad;
    bool is_leaf;
    
    GradNode() : tensor(nullptr), requires_grad(false), is_leaf(true), backward_fn(nullptr) {}
    GradNode(AutogradTensor<T>* t, bool req_grad, bool leaf)
        : tensor(t), requires_grad(req_grad), is_leaf(leaf), backward_fn(nullptr) {}
};

// ============================================================================
// AutogradTensor: Wraps TensorBase with gradient tracking
// ============================================================================

template<typename T>
class AutogradTensor : public TensorBase<T> {
private:
    std::unique_ptr<TensorBase<T>> _data;  // The actual data tensor
    std::unique_ptr<TensorBase<T>> _grad;  // Accumulated gradient
    bool _requires_grad;
    bool _is_leaf;
    
    // Computational graph
    std::vector<AutogradTensor<T>*> _parents;
    std::function<void(const TensorBase<T>*)> _backward_fn;
    
    // Global tracking of all tensors for graph building
    static std::vector<AutogradTensor<T>*> _all_tensors;
    
    void register_self() {
        _all_tensors.push_back(this);
    }
    
    bool any_parent_requires_grad() const {
        for (auto* parent : _parents) {
            if (parent && parent->_requires_grad) {
                return true;
            }
        }
        return false;
    }
    
    bool should_track_grad() const {
        return _requires_grad || any_parent_requires_grad();
    }
    
public:
    // ========================================================================
    // Constructors
    // ========================================================================
    
    explicit AutogradTensor(std::unique_ptr<TensorBase<T>> data, bool requires_grad = false)
        : _data(std::move(data)), _requires_grad(requires_grad), _is_leaf(true) {
        register_self();
    }
    
    AutogradTensor(const std::size_t* shape, std::size_t ndim, T fill_value = T{}, bool requires_grad = false)
        : _requires_grad(requires_grad), _is_leaf(true) {
        _data = std::make_unique<DenseTensor<T>>(shape, ndim, fill_value);
        register_self();
    }
    
    AutogradTensor(std::initializer_list<T> data, bool requires_grad = false)
        : _requires_grad(requires_grad), _is_leaf(true) {
        _data = std::make_unique<DenseTensor<T>>(data);
        register_self();
    }
    
    ~AutogradTensor() {
        // Remove from global list
        _all_tensors.erase(
            std::remove(_all_tensors.begin(), _all_tensors.end(), this),
            _all_tensors.end()
        );
    }
    
    // ========================================================================
    // TensorBase interface
    // ========================================================================
    
    std::size_t ndim() const override { return _data->ndim(); }
    std::size_t total_size() const override { return _data->total_size(); }
    const std::size_t* shape() const override { return _data->shape(); }
    const std::size_t* stride() const override { return _data->stride(); }
    
    T get_element(std::size_t index) const override { return _data->get_element(index); }
    void set_element(std::size_t index, T value) override { _data->set_element(index, value); }
    T operator()(std::size_t i) const override { return _data->operator()(i); }
    T& operator()(std::size_t i) override { return _data->operator()(i); }
    T operator()(std::size_t i, std::size_t j) const override { return _data->operator()(i, j); }
    T& operator()(std::size_t i, std::size_t j) override { return _data->operator()(i, j); }
    
    bool is_streaming() const override { return _data->is_streaming(); }
    std::string backend_name() const override { return "Autograd<" + _data->backend_name() + ">"; }
    
    // ========================================================================
    // Gradient interface
    // ========================================================================
    
    bool requires_grad() const { return _requires_grad; }
    bool is_leaf() const { return _is_leaf; }
    
    void set_requires_grad(bool requires_grad) {
        if (!is_leaf()) {
            throw std::runtime_error("Can only set requires_grad on leaf tensors");
        }
        _requires_grad = requires_grad;
    }
    
    const TensorBase<T>* grad() const { return _grad.get(); }
    TensorBase<T>* grad() { return _grad.get(); }
    
    void zero_grad() {
        if (_requires_grad) {
            auto shape_vec = get_shape_vector();
            _grad = create_zeros(shape_vec.data(), shape_vec.size());
        }
    }
    
    void accumulate_grad(const TensorBase<T>* new_grad) {
        if (!_grad) {
            _grad = clone_tensor(new_grad);
        } else {
            auto sum_result = _grad->add(new_grad);
            _grad = std::move(sum_result);
        }
    }
    
    // ========================================================================
    // Element-wise operations
    // ========================================================================
    
    // Helper to extract underlying data tensor
    const TensorBase<T>* get_data_tensor() const { return _data.get(); }
    
    // Helper to extract data from any TensorBase (unwraps AutogradTensor if needed)
    static const TensorBase<T>* extract_data(const TensorBase<T>* tensor) {
        auto* auto_tensor = dynamic_cast<const AutogradTensor<T>*>(tensor);
        if (auto_tensor) {
            return auto_tensor->_data.get();
        }
        return tensor;
    }
    
    std::unique_ptr<TensorBase<T>> add(const TensorBase<T>* other) const override {
        auto* other_data = extract_data(other);
        auto result_data = _data->add(other_data);
        auto result = std::make_unique<AutogradTensor<T>>(std::move(result_data), false);
        
        // Set up gradient tracking if needed
        setup_binary_op_grad(
            result.get(), 
            const_cast<AutogradTensor<T>*>(this),
            extract_autograd(other),
            [](const TensorBase<T>* grad_output, const TensorBase<T>* input, const TensorBase<T>* /*other*/) {
                // d/dx (x + y) = 1, so grad flows through unchanged
                // Need to handle broadcasting
                return broadcast_gradient_back(grad_output, input);
            }
        );
        
        return result;
    }
    
    std::unique_ptr<TensorBase<T>> subtract(const TensorBase<T>* other) const override {
        auto* other_data = extract_data(other);
        auto result_data = _data->subtract(other_data);
        auto result = std::make_unique<AutogradTensor<T>>(std::move(result_data), false);
        
        setup_binary_op_grad(
            result.get(),
            const_cast<AutogradTensor<T>*>(this),
            extract_autograd(other),
            [](const TensorBase<T>* grad_output, const TensorBase<T>* input, const TensorBase<T>* /*other*/) {
                // d/dx (x - y) = 1
                return broadcast_gradient_back(grad_output, input);
            }
        );
        
        return result;
    }
    
    std::unique_ptr<TensorBase<T>> multiply(const TensorBase<T>* other) const override {
        auto* other_data = extract_data(other);
        auto result_data = _data->multiply(other_data);
        auto result = std::make_unique<AutogradTensor<T>>(std::move(result_data), false);
        
        auto* this_ptr = const_cast<AutogradTensor<T>*>(this);
        auto* other_auto = extract_autograd(other);
        auto* other_ptr = const_cast<TensorBase<T>*>(other_data);
        auto this_data_shared = std::shared_ptr<TensorBase<T>>(clone_tensor(_data.get()));
        
        setup_binary_op_grad(
            result.get(),
            this_ptr,
            other_auto,
            [other_ptr, this_data_shared](const TensorBase<T>* grad_output, const TensorBase<T>* input, const TensorBase<T>* /*other_input*/) {
                // d/dx (x * y) = y, so grad = grad_output * y
                if (input->total_size() == this_data_shared->total_size()) {
                    return grad_output->multiply(other_ptr);
                } else {
                    return grad_output->multiply(this_data_shared.get());
                }
            }
        );
        
        return result;
    }
    
    std::unique_ptr<TensorBase<T>> divide(const TensorBase<T>* other) const override {
        auto* other_data = extract_data(other);
        auto result_data = _data->divide(other_data);
        auto result = std::make_unique<AutogradTensor<T>>(std::move(result_data), false);
        
        auto* this_ptr = const_cast<AutogradTensor<T>*>(this);
        auto* other_auto = extract_autograd(other);
        auto* other_ptr = const_cast<TensorBase<T>*>(other_data);
        auto this_data_shared = std::shared_ptr<TensorBase<T>>(clone_tensor(_data.get()));
        
        setup_binary_op_grad(
            result.get(),
            this_ptr,
            other_auto,
            [other_ptr, this_data_shared](const TensorBase<T>* grad_output, const TensorBase<T>* input, const TensorBase<T>* /*other_input*/) {
                // d/dx (x / y) = 1/y, so grad = grad_output / y
                if (input->total_size() == this_data_shared->total_size()) {
                    return grad_output->divide(other_ptr);
                } else {
                    // d/dy (x / y) = -x / y^2
                    auto y_sq = other_ptr->multiply(other_ptr);
                    auto neg_x = this_data_shared->negate();
                    auto grad = neg_x->divide(y_sq.get());
                    return grad_output->multiply(grad.get());
                }
            }
        );
        
        return result;
    }
    
    std::unique_ptr<TensorBase<T>> add_scalar(T scalar) const override {
        auto result_data = _data->add_scalar(scalar);
        auto result = std::make_unique<AutogradTensor<T>>(std::move(result_data), false);
        
        setup_unary_op_grad(
            result.get(),
            const_cast<AutogradTensor<T>*>(this),
            [](const TensorBase<T>* grad_output, const TensorBase<T>* /*input*/) {
                // d/dx (x + c) = 1
                return clone_tensor_ptr(grad_output);
            }
        );
        
        return result;
    }
    
    std::unique_ptr<TensorBase<T>> subtract_scalar(T scalar) const override {
        auto result_data = _data->subtract_scalar(scalar);
        auto result = std::make_unique<AutogradTensor<T>>(std::move(result_data), false);
        
        setup_unary_op_grad(
            result.get(),
            const_cast<AutogradTensor<T>*>(this),
            [](const TensorBase<T>* grad_output, const TensorBase<T>* /*input*/) {
                return clone_tensor_ptr(grad_output);
            }
        );
        
        return result;
    }
    
    std::unique_ptr<TensorBase<T>> multiply_scalar(T scalar) const override {
        auto result_data = _data->multiply_scalar(scalar);
        auto result = std::make_unique<AutogradTensor<T>>(std::move(result_data), false);
        
        auto scalar_val = scalar;
        setup_unary_op_grad(
            result.get(),
            const_cast<AutogradTensor<T>*>(this),
            [scalar_val](const TensorBase<T>* grad_output, const TensorBase<T>* /*input*/) {
                return grad_output->multiply_scalar(scalar_val);
            }
        );
        
        return result;
    }
    
    std::unique_ptr<TensorBase<T>> divide_scalar(T scalar) const override {
        auto result_data = _data->divide_scalar(scalar);
        auto result = std::make_unique<AutogradTensor<T>>(std::move(result_data), false);
        
        auto scalar_val = scalar;
        setup_unary_op_grad(
            result.get(),
            const_cast<AutogradTensor<T>*>(this),
            [scalar_val](const TensorBase<T>* grad_output, const TensorBase<T>* /*input*/) {
                return grad_output->divide_scalar(scalar_val);
            }
        );
        
        return result;
    }
    
    // ========================================================================
    // Unary operations
    // ========================================================================
    
    std::unique_ptr<TensorBase<T>> negate() const override {
        auto result_data = _data->negate();
        auto result = std::make_unique<AutogradTensor<T>>(std::move(result_data), false);
        
        setup_unary_op_grad(
            result.get(),
            const_cast<AutogradTensor<T>*>(this),
            [](const TensorBase<T>* grad_output, const TensorBase<T>* /*input*/) {
                return grad_output->negate();
            }
        );
        
        return result;
    }
    
    std::unique_ptr<TensorBase<T>> abs() const override {
        auto result_data = _data->abs();
        auto result = std::make_unique<AutogradTensor<T>>(std::move(result_data), false);
        
        // d/dx |x| = sign(x)
        setup_unary_op_grad(
            result.get(),
            const_cast<AutogradTensor<T>*>(this),
            [this](const TensorBase<T>* grad_output, const TensorBase<T>* /*input*/) {
                // sign(x) = x / |x|
                auto abs_x = this->_data->abs();
                auto sign = this->_data->divide(abs_x.get());
                return grad_output->multiply(sign.get());
            }
        );
        
        return result;
    }
    
    // ========================================================================
    // Reductions (simplified - just compute, no grad for now)
    // ========================================================================
    
    T sum() const override { return _data->sum(); }
    T mean() const override { return _data->mean(); }
    T max() const override { return _data->max(); }
    T min() const override { return _data->min(); }
    T dot(const TensorBase<T>* other) const override { return _data->dot(other); }
    
    // ========================================================================
    // Transforms
    // ========================================================================
    
    std::unique_ptr<TensorBase<T>> reshape(const std::size_t* new_shape, std::size_t new_ndim) const override {
        auto result_data = _data->reshape(new_shape, new_ndim);
        auto result = std::make_unique<AutogradTensor<T>>(std::move(result_data), false);
        
        auto old_shape = get_shape_vector();
        setup_unary_op_grad(
            result.get(),
            const_cast<AutogradTensor<T>*>(this),
            [old_shape](const TensorBase<T>* grad_output, const TensorBase<T>* /*input*/) {
                // Gradient of reshape is just reshape back
                std::vector<std::size_t> shape_vec(old_shape);
                return grad_output->reshape(shape_vec.data(), shape_vec.size());
            }
        );
        
        return result;
    }
    
    std::unique_ptr<TensorBase<T>> transpose() const override {
        auto result_data = _data->transpose();
        auto result = std::make_unique<AutogradTensor<T>>(std::move(result_data), false);
        
        setup_unary_op_grad(
            result.get(),
            const_cast<AutogradTensor<T>*>(this),
            [](const TensorBase<T>* grad_output, const TensorBase<T>* /*input*/) {
                // Gradient of transpose is just transpose again
                return grad_output->transpose();
            }
        );
        
        return result;
    }
    
    std::unique_ptr<TensorBase<T>> clamp(T min_val, T max_val) const override {
        auto result_data = _data->clamp(min_val, max_val);
        auto result = std::make_unique<AutogradTensor<T>>(std::move(result_data), false);
        
        auto this_data_shared = std::shared_ptr<TensorBase<T>>(clone_tensor(_data.get()));
        
        // d/dx clamp(x) = 1 if min < x < max, else 0
        setup_unary_op_grad(
            result.get(),
            const_cast<AutogradTensor<T>*>(this),
            [this_data_shared, min_val, max_val](const TensorBase<T>* grad_output, const TensorBase<T>* /*input*/) {
                // Create mask: 1 where min < x < max, 0 otherwise
                auto gt_min = this_data_shared->greater_than(min_val);
                auto lt_max = this_data_shared->less_than(max_val);
                
                // Simplified: just pass through gradient where clamp was active
                // Proper implementation needs bool-to-float conversion
                return clone_tensor_ptr(grad_output);
            }
        );
        
        return result;
    }
    
    // ========================================================================
    // Comparisons (no gradient tracking)
    // ========================================================================
    
    std::unique_ptr<TensorBase<bool>> greater_than(T threshold) const override {
        return _data->greater_than(threshold);
    }
    
    std::unique_ptr<TensorBase<bool>> less_than(T threshold) const override {
        return _data->less_than(threshold);
    }
    
    // ========================================================================
    // Export and Print
    // ========================================================================
    
    std::unique_ptr<T[]> to_array() const override { return _data->to_array(); }
    
    void print(std::ostream& os) const override { _data->print(os); }
    
    void batch_print(std::ostream& os, std::size_t batch_size = 100) const override {
        _data->batch_print(os, batch_size);
    }
    
    // ========================================================================
    // Backward pass
    // ========================================================================
    
    void backward(const TensorBase<T>* gradient = nullptr) {
        if (!_requires_grad && !_grad) {
            throw std::runtime_error("backward() called on tensor that doesn't require grad");
        }
        
        // Build topological order
        std::vector<AutogradTensor<T>*> topo_order;
        std::unordered_set<AutogradTensor<T>*> visited;
        build_topological_order(const_cast<AutogradTensor<T>*>(this), topo_order, visited);
        
        // Initialize gradient for this tensor
        if (gradient) {
            _grad = clone_tensor(gradient);
        } else {
            // Default: gradient of scalar is 1.0
            auto shape_vec = get_shape_vector();
            if (shape_vec.empty() || (shape_vec.size() == 1 && shape_vec[0] == 1)) {
                _grad = create_ones(shape_vec.data(), shape_vec.size());
            } else {
                throw std::runtime_error("backward() on non-scalar requires gradient argument");
            }
        }
        
        // Traverse in reverse topological order
        for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
            auto* node = *it;
            
            // Skip if no gradient
            if (!node->_grad) continue;
            
            // Skip leaf nodes
            if (node->is_leaf()) continue;
            
            // Apply backward function
            if (node->_backward_fn) {
                node->_backward_fn(node->_grad.get());
            }
        }
    }
    
    // ========================================================================
    // Helper methods
    // ========================================================================
    
private:
    std::vector<std::size_t> get_shape_vector() const {
        std::vector<std::size_t> shape_vec;
        auto* s = this->shape();
        for (std::size_t i = 0; i < ndim(); ++i) {
            shape_vec.push_back(s[i]);
        }
        return shape_vec;
    }
    
    std::unique_ptr<TensorBase<T>> clone_tensor(const TensorBase<T>* other) const {
        std::vector<std::size_t> shape_vec;
        auto* s = other->shape();
        for (std::size_t i = 0; i < other->ndim(); ++i) {
            shape_vec.push_back(s[i]);
        }
        auto result = std::make_unique<DenseTensor<T>>(shape_vec.data(), shape_vec.size());
        auto arr = other->to_array();
        for (std::size_t i = 0; i < other->total_size(); ++i) {
            result->set_element(i, arr[i]);
        }
        return result;
    }
    
    static std::unique_ptr<TensorBase<T>> clone_tensor_ptr(const TensorBase<T>* other) {
        std::vector<std::size_t> shape_vec;
        auto* s = other->shape();
        for (std::size_t i = 0; i < other->ndim(); ++i) {
            shape_vec.push_back(s[i]);
        }
        auto result = std::make_unique<DenseTensor<T>>(shape_vec.data(), shape_vec.size());
        auto arr = other->to_array();
        for (std::size_t i = 0; i < other->total_size(); ++i) {
            result->set_element(i, arr[i]);
        }
        return result;
    }
    
    std::unique_ptr<TensorBase<T>> create_zeros(const std::size_t* shape, std::size_t ndim) const {
        return std::make_unique<DenseTensor<T>>(shape, ndim, T{0});
    }
    
    std::unique_ptr<TensorBase<T>> create_ones(const std::size_t* shape, std::size_t ndim) const {
        return std::make_unique<DenseTensor<T>>(shape, ndim, T{1});
    }
    
    std::vector<std::size_t> get_shape_vector(const TensorBase<T>* other) const {
        std::vector<std::size_t> shape;
        auto* s = other->shape();
        for (std::size_t i = 0; i < other->ndim(); ++i) {
            shape.push_back(s[i]);
        }
        return shape;
    }
    
    static AutogradTensor<T>* extract_autograd(const TensorBase<T>* tensor) {
        return dynamic_cast<AutogradTensor<T>*>(const_cast<TensorBase<T>*>(tensor));
    }
    
    void setup_binary_op_grad(
        AutogradTensor<T>* result,
        AutogradTensor<T>* left,
        AutogradTensor<T>* right,
        std::function<std::unique_ptr<TensorBase<T>>(const TensorBase<T>*, const TensorBase<T>*, const TensorBase<T>*)> grad_fn) const {
        
        bool track_grad = (left && left->should_track_grad()) || (right && right->should_track_grad());
        
        if (track_grad) {
            result->_parents.clear();
            if (left) result->_parents.push_back(left);
            if (right) result->_parents.push_back(right);
            
            result->_is_leaf = false;
            result->_backward_fn = [grad_fn, left, right](const TensorBase<T>* grad_output) {
                if (left && left->_requires_grad) {
                    auto grad_left = grad_fn(grad_output, left->_data.get(), right ? right->_data.get() : nullptr);
                    left->accumulate_grad(grad_left.get());
                }
                if (right && right->_requires_grad) {
                    auto grad_right = grad_fn(grad_output, right->_data.get(), left ? left->_data.get() : nullptr);
                    right->accumulate_grad(grad_right.get());
                }
            };
        }
    }
    
    void setup_unary_op_grad(
        AutogradTensor<T>* result,
        AutogradTensor<T>* input,
        std::function<std::unique_ptr<TensorBase<T>>(const TensorBase<T>*, const TensorBase<T>*)> grad_fn) const {
        
        if (input && input->should_track_grad()) {
            result->_parents.clear();
            result->_parents.push_back(input);
            result->_is_leaf = false;
            result->_backward_fn = [grad_fn, input](const TensorBase<T>* grad_output) {
                if (input->_requires_grad) {
                    auto grad_input = grad_fn(grad_output, input->_data.get());
                    input->accumulate_grad(grad_input.get());
                }
            };
        }
    }
    
    void build_topological_order(
        AutogradTensor<T>* node,
        std::vector<AutogradTensor<T>*>& order,
        std::unordered_set<AutogradTensor<T>*>& visited) {
        
        if (visited.count(node)) return;
        visited.insert(node);
        
        for (auto* parent : node->_parents) {
            build_topological_order(parent, order, visited);
        }
        
        order.push_back(node);
    }
    
    static std::unique_ptr<TensorBase<T>> broadcast_gradient_back(
        const TensorBase<T>* grad_output, const TensorBase<T>* target) {
        // Simplified: just clone the gradient
        // Proper implementation needs broadcasting logic
        return clone_tensor_ptr(grad_output);
    }
};

// Static member initialization
template<typename T>
std::vector<AutogradTensor<T>*> AutogradTensor<T>::_all_tensors;

// ============================================================================
// Factory functions
// ============================================================================

template<typename T>
std::unique_ptr<AutogradTensor<T>> make_tensor(
    const T* data, std::size_t size, bool requires_grad = false) {
    auto dense = std::make_unique<DenseTensor<T>>(data, size);
    return std::make_unique<AutogradTensor<T>>(std::move(dense), requires_grad);
}

template<typename T>
std::unique_ptr<AutogradTensor<T>> make_tensor(
    const std::size_t* shape, std::size_t ndim, T fill_value = T{}, bool requires_grad = false) {
    return std::make_unique<AutogradTensor<T>>(shape, ndim, fill_value, requires_grad);
}

template<typename T>
std::unique_ptr<AutogradTensor<T>> make_tensor(
    std::initializer_list<T> data, bool requires_grad = false) {
    return std::make_unique<AutogradTensor<T>>(data, requires_grad);
}

template<typename T>
std::unique_ptr<AutogradTensor<T>> make_tensor(
    std::unique_ptr<TensorBase<T>> data, bool requires_grad = false) {
    return std::make_unique<AutogradTensor<T>>(std::move(data), requires_grad);
}

// ============================================================================
// Autograd Engine (global state management)
// ============================================================================

template<typename T>
class AutogradEngine {
private:
    static bool _enabled;
    
public:
    static void enable_grad() { _enabled = true; }
    static void disable_grad() { _enabled = false; }
    static bool is_grad_enabled() { return _enabled; }
};

template<typename T>
bool AutogradEngine<T>::_enabled = true;
