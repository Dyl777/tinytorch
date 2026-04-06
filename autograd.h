#pragma once
#include <memory>
#include <vector>
#include <functional>
#include <unordered_map>
#include <algorithm>
#include <cassert>
#include <iostream>

// ============================================================================
// Autograd Engine for TinyTorch
// ============================================================================
// This implements automatic differentiation using a computational graph approach.
// Each tensor operation creates a node in the graph, and backward() traverses
// the graph in reverse topological order to compute gradients.
//
// Key Concepts:
// -------------
// 1. requires_grad: If true, operations on this tensor are tracked
// 2. grad: The gradient tensor (same shape as data tensor)
// 3. grad_fn: Function to compute gradient during backward pass
// 4. is_leaf: True if tensor was created by user (not by an operation)
// 5. Computational graph: DAG of operations
//
// How it works:
// -------------
// 1. Forward pass: Build computational graph, store grad_fn for each op
// 2. Backward pass: Start from output, apply chain rule via grad_fn
// 3. Gradient accumulation: Leaf tensors accumulate gradients from all paths
// ============================================================================

// Forward declaration
template<typename T>
class TensorBase;

// ============================================================================
// Gradient Function Base Class
// ============================================================================
// Each operation defines its own gradient function. During backward pass,
// grad_fn receives the gradient w.r.t. output and computes gradient w.r.t. inputs.

template<typename T>
struct GradContext {
    // Shape information needed for gradient computation
    std::vector<std::size_t> shape;
    std::vector<std::size_t> stride;
    std::size_t ndim;
    std::size_t total_size;
    
    // Additional context for specific operations
    void* extra_data;
    
    GradContext() : extra_data(nullptr), ndim(0), total_size(0) {}
    
    GradContext(const std::vector<std::size_t>& s, const std::vector<std::size_t>& st)
        : shape(s), stride(st), extra_data(nullptr) {
        ndim = s.size();
        total_size = 1;
        for (auto dim : s) total_size *= dim;
    }
};

// Gradient function: takes grad_output, returns grad_input
using GradFn = std::function<std::unique_ptr<TensorBase<T>>(const TensorBase<T>*)>;

// ============================================================================
// AutogradTensorBase: Extends TensorBase with gradient tracking
// ============================================================================

template<typename T>
class AutogradTensorBase : public TensorBase<T> {
protected:
    // Gradient tracking
    bool _requires_grad;
    bool _is_leaf;
    std::unique_ptr<TensorBase<T>> _grad;  // Accumulated gradient
    std::weak_ptr<AutogradTensorBase<T>> _grad_owner;  // Shared ownership of grad
    
    // Computational graph
    std::vector<std::shared_ptr<AutogradTensorBase<T>>> _parents;
    GradFn _backward_fn;
    
public:
    AutogradTensorBase() 
        : _requires_grad(false), _is_leaf(true) {}
    
    virtual ~AutogradTensorBase() = default;
    
    // ========================================================================
    // Gradient tracking interface
    // ========================================================================
    
    bool requires_grad() const { return _requires_grad; }
    bool is_leaf() const { return _is_leaf; }
    
    // Enable gradient tracking
    void set_requires_grad(bool requires_grad) {
        if (!is_leaf()) {
            throw std::runtime_error("Can only set requires_grad on leaf tensors");
        }
        _requires_grad = requires_grad;
    }
    
    // Get gradient (may be null if not computed yet)
    const TensorBase<T>* grad() const { return _grad.get(); }
    TensorBase<T>* grad() { return _grad.get(); }
    
    // Set gradient (for manual gradient injection)
    void set_grad(std::unique_ptr<TensorBase<T>> grad) {
        _grad = std::move(grad);
    }
    
    // Initialize gradient to zeros (if not already initialized)
    void zero_grad() {
        if (_requires_grad) {
            // Create zero tensor of same shape
            auto shape_vec = get_shape_vector();
            _grad = create_zeros(shape_vec.data(), shape_vec.size());
        }
    }
    
    // Accumulate gradient
    void accumulate_grad(const TensorBase<T>* new_grad) {
        if (!_grad) {
            // First gradient - just copy it
            auto shape_vec = get_shape_vector();
            _grad = clone_tensor(new_grad);
        } else {
            // Add to existing gradient
            auto old_grad = _grad.get();
            auto sum_result = old_grad->add(new_grad);
            _grad = std::move(sum_result);
        }
    }
    
    // ========================================================================
    // Computational graph interface
    // ========================================================================
    
    const std::vector<std::shared_ptr<AutogradTensorBase<T>>>& parents() const {
        return _parents;
    }
    
    const GradFn& backward_fn() const { return _backward_fn; }
    
    // Set up the computational graph for this tensor
    void set_graph(
        std::vector<std::shared_ptr<AutogradTensorBase<T>>> parents,
        GradFn backward_fn,
        bool is_leaf = false) {
        _parents = std::move(parents);
        _backward_fn = std::move(backward_fn);
        _is_leaf = is_leaf;
    }
    
    // ========================================================================
    // Helper methods for creating tensors with gradient tracking
    // ========================================================================
    
    // Create a new tensor that shares gradient tracking with this one
    virtual std::shared_ptr<AutogradTensorBase<T>> create_like(
        const std::vector<std::size_t>& shape,
        bool requires_grad = false) const = 0;
    
    virtual std::shared_ptr<AutogradTensorBase<T>> create_like(
        const TensorBase<T>* other,
        bool requires_grad = false) const = 0;
    
    // ========================================================================
    // Pure virtual methods to be implemented by derived classes
    // ========================================================================
    
    virtual std::vector<std::size_t> get_shape_vector() const = 0;
    virtual std::unique_ptr<TensorBase<T>> clone_tensor(const TensorBase<T>* other) const = 0;
    virtual std::unique_ptr<TensorBase<T>> create_zeros(
        const std::size_t* shape, std::size_t ndim) const = 0;
    virtual std::unique_ptr<TensorBase<T>> create_ones(
        const std::size_t* shape, std::size_t ndim) const = 0;
    virtual std::unique_ptr<TensorBase<T>> broadcast_to(
        const TensorBase<T>* target_shape_tensor) const = 0;
    
    // ========================================================================
    // Backward pass
    // ========================================================================
    
    void backward(const TensorBase<T>* gradient = nullptr) {
        if (!this->_requires_grad && this->_grad == nullptr) {
            throw std::runtime_error("backward() called on tensor that doesn't require grad");
        }
        
        // Build computational graph
        std::vector<std::shared_ptr<AutogradTensorBase<T>>> topo_order;
        build_topological_order(this->shared_from_this(), topo_order);
        
        // Initialize gradient for this tensor
        if (gradient) {
            this->_grad = clone_tensor(gradient);
        } else {
            // Default: gradient of scalar output is 1.0
            auto shape_vec = get_shape_vector();
            if (shape_vec.empty() || (shape_vec.size() == 1 && shape_vec[0] == 1)) {
                // Scalar tensor - create tensor with value 1.0
                this->_grad = create_ones(shape_vec.data(), shape_vec.size());
            } else {
                throw std::runtime_error("backward() on non-scalar tensor requires gradient argument");
            }
        }
        
        // Traverse graph in reverse topological order
        for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
            auto& node = *it;
            
            // Skip if no gradient has flowed to this node
            if (!node->_grad) continue;
            
            // Skip leaf nodes (they don't have backward functions)
            if (node->is_leaf()) continue;
            
            // Apply backward function
            if (node->_backward_fn) {
                auto input_grads = node->_backward_fn(node->_grad.get());
                
                // Accumulate gradients to parents
                for (std::size_t i = 0; i < node->_parents.size(); ++i) {
                    auto& parent = node->_parents[i];
                    if (parent && parent->_requires_grad) {
                        // For now, assume single input gradient
                        // In reality, backward_fn should return gradients for all inputs
                        parent->accumulate_grad(input_grads.get());
                    }
                }
            }
        }
    }
    
private:
    // Build topological ordering of the computational graph
    void build_topological_order(
        std::shared_ptr<AutogradTensorBase<T>> node,
        std::vector<std::shared_ptr<AutogradTensorBase<T>>>& order,
        std::unordered_map<AutogradTensorBase<T>*, bool>& visited) {
        
        if (visited[node.get()]) return;
        visited[node.get()] = true;
        
        for (auto& parent : node->_parents) {
            build_topological_order(parent, order, visited);
        }
        
        order.push_back(node);
    }
    
    void build_topological_order(
        std::shared_ptr<AutogradTensorBase<T>> node,
        std::vector<std::shared_ptr<AutogradTensorBase<T>>>& order) {
        std::unordered_map<AutogradTensorBase<T>*, bool> visited;
        build_topological_order(node, order, visited);
    }
};

// ============================================================================
// Autograd Helper Functions
// ============================================================================

// Helper to extract AutogradTensorBase from TensorBase
template<typename T>
std::shared_ptr<AutogradTensorBase<T>> get_autograd_tensor(TensorBase<T>* tensor) {
    auto* auto_tensor = dynamic_cast<AutogradTensorBase<T>*>(tensor);
    if (!auto_tensor) {
        throw std::runtime_error("Tensor is not an AutogradTensor");
    }
    // This is a simplification - in reality we need shared_from_this
    return nullptr; // Placeholder
}

// Check if any input requires grad
template<typename T>
bool any_requires_grad(const TensorBase<T>* a, const TensorBase<T>* b) {
    auto* auto_a = dynamic_cast<const AutogradTensorBase<T>*>(a);
    auto* auto_b = dynamic_cast<const AutogradTensorBase<T>*>(b);
    
    bool req_a = auto_a && auto_a->requires_grad();
    bool req_b = auto_b && auto_b->requires_grad();
    
    return req_a || req_b;
}

// Check if tensor requires grad
template<typename T>
bool tensor_requires_grad(const TensorBase<T>* tensor) {
    auto* auto_tensor = dynamic_cast<const AutogradTensorBase<T>*>(tensor);
    return auto_tensor && auto_tensor->requires_grad();
}
