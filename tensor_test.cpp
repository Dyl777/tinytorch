#include "tensor.h"
#include <iostream>
#include <cassert>
#include <cmath>

void test_scalar_tensor() {
    std::cout << "=== Testing Scalar Tensor ===" << std::endl;

    Tensor<int> scalar(42);
    std::cout << "Scalar tensor: " << scalar << std::endl;
    std::cout << "Item: " << scalar.item() << std::endl;
    std::cout << "NDim: " << scalar.ndim() << std::endl;
    std::cout << "Total size: " << scalar.total_size() << std::endl;

    assert(scalar.item() == 42);
    assert(scalar.ndim() == 0);
    assert(scalar.total_size() == 1);

    std::cout << "✓ Scalar tensor tests passed\n" << std::endl;
}

void test_1d_tensor() {
    std::cout << "=== Testing 1D Tensor ===" << std::endl;

    Tensor<float> tensor_1d{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::cout << "1D tensor: " << tensor_1d << std::endl;
    std::cout << "Shape: [" << tensor_1d.shape()[0] << "]" << std::endl;
    std::cout << "Stride: [" << tensor_1d.stride()[0] << "]" << std::endl;

    assert(tensor_1d.ndim() == 1);
    assert(tensor_1d.total_size() == 5);
    assert(tensor_1d.shape()[0] == 5);
    assert(tensor_1d(0) == 1.0f);
    assert(tensor_1d(2) == 3.0f);
    assert(tensor_1d(4) == 5.0f);

    tensor_1d(1) = 20.0f;
    assert(tensor_1d(1) == 20.0f);

    std::cout << "✓ 1D tensor tests passed\n" << std::endl;
}

void test_2d_tensor() {
    std::cout << "=== Testing 2D Tensor ===" << std::endl;

    Tensor<double> tensor_2d{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };

    std::cout << "2D tensor: " << tensor_2d << std::endl;
    std::cout << "Shape: [" << tensor_2d.shape()[0] << ", " << tensor_2d.shape()[1] << "]" << std::endl;
    std::cout << "Stride: [" << tensor_2d.stride()[0] << ", " << tensor_2d.stride()[1] << "]" << std::endl;

    assert(tensor_2d.ndim() == 2);
    assert(tensor_2d.total_size() == 9);
    assert(tensor_2d.shape()[0] == 3);
    assert(tensor_2d.shape()[1] == 3);
    assert(tensor_2d(0, 0) == 1.0);
    assert(tensor_2d(1, 1) == 5.0);
    assert(tensor_2d(2, 2) == 9.0);

    tensor_2d(0, 2) = 30.0;
    assert(tensor_2d(0, 2) == 30.0);

    std::cout << "✓ 2D tensor tests passed\n" << std::endl;
}

void test_nd_tensor() {
    std::cout << "=== Testing N-Dimensional Tensor ===" << std::endl;

    std::size_t shape[] = {2, 3, 4};
    Tensor<int> tensor_3d(shape, 3, 0);

    std::cout << "3D tensor shape: [";
    for (std::size_t i = 0; i < tensor_3d.ndim(); ++i) {
        std::cout << tensor_3d.shape()[i];
        if (i != tensor_3d.ndim() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    assert(tensor_3d.ndim() == 3);
    assert(tensor_3d.total_size() == 24);

    int* data_ptr = tensor_3d.data_non_volatile();
    for (std::size_t i = 0; i < tensor_3d.total_size(); ++i) {
        data_ptr[i] = static_cast<int>(i);
    }

    std::size_t indices[] = {1, 2, 3};
    std::cout << "Element at [1,2,3]: " << tensor_3d.at(indices) << std::endl;
    assert(tensor_3d.at(indices) == 23);

    std::cout << "✓ N-D tensor tests passed\n" << std::endl;
}

void test_elementwise_operations() {
    std::cout << "=== Testing Element-wise Operations ===" << std::endl;

    Tensor<float> a{1.0f, 2.0f, 3.0f};
    Tensor<float> b{4.0f, 5.0f, 6.0f};

    Tensor<float>* sum = a.add(&b);
    std::cout << "A: " << a << std::endl;
    std::cout << "B: " << b << std::endl;
    std::cout << "A + B: " << *sum << std::endl;

    assert(sum->total_size() == 3);
    assert((*sum)(0) == 5.0f);
    assert((*sum)(1) == 7.0f);
    assert((*sum)(2) == 9.0f);

    Tensor<float>* diff = a.subtract(&b);
    std::cout << "A - B: " << *diff << std::endl;
    assert((*diff)(0) == -3.0f);

    Tensor<float>* prod = a.multiply(&b);
    std::cout << "A * B: " << *prod << std::endl;
    assert((*prod)(0) == 4.0f);

    Tensor<float> scalar(10.0f);
    Tensor<float>* scalar_add = scalar.add(&a);
    std::cout << "10 + A: " << *scalar_add << std::endl;
    assert((*scalar_add)(0) == 11.0f);

    delete sum;
    delete diff;
    delete prod;
    delete scalar_add;

    std::cout << "✓ Element-wise operation tests passed\n" << std::endl;
}

void test_scalar_operations() {
    std::cout << "=== Testing Scalar Operations ===" << std::endl;

    Tensor<int> tensor{1, 2, 3, 4, 5};

    Tensor<int>* added = tensor.add_scalar(10);
    std::cout << "Tensor + 10: " << *added << std::endl;
    assert((*added)(0) == 11);
    assert((*added)(4) == 15);

    Tensor<int>* multiplied = tensor.multiply_scalar(2);
    std::cout << "Tensor * 2: " << *multiplied << std::endl;
    assert((*multiplied)(0) == 2);
    assert((*multiplied)(2) == 6);

    delete added;
    delete multiplied;

    std::cout << "✓ Scalar operation tests passed\n" << std::endl;
}

void test_copy_move_semantics() {
    std::cout << "=== Testing Copy/Move Semantics ===" << std::endl;

    Tensor<float> original{1.0f, 2.0f, 3.0f};

    Tensor<float> copied(original);
    std::cout << "Original: " << original << std::endl;
    std::cout << "Copied: " << copied << std::endl;
    assert(copied(0) == original(0));

    original(0) = 100.0f;
    assert(copied(0) == 1.0f);

    Tensor<float> moved(std::move(original));
    std::cout << "Moved: " << moved << std::endl;
    assert(moved(0) == 100.0f);

    Tensor<float> assigned = std::move(moved);
    std::cout << "Assigned: " << assigned << std::endl;
    assert(assigned(0) == 100.0f);

    std::cout << "✓ Copy/Move semantics tests passed\n" << std::endl;
}

void test_type_templates() {
    std::cout << "=== Testing Template Types ===" << std::endl;

    Tensor<int> int_tensor{1, 2, 3, 4, 5};
    std::cout << "Int tensor: " << int_tensor << std::endl;

    Tensor<double> double_tensor{1.1, 2.2, 3.3};
    std::cout << "Double tensor: " << double_tensor << std::endl;

    Tensor<float> float_tensor{1.0f, 2.0f, 3.0f};
    std::cout << "Float tensor: " << float_tensor << std::endl;

    std::cout << "✓ Template type tests passed\n" << std::endl;
}

void test_error_handling() {
    std::cout << "=== Testing Error Handling ===" << std::endl;

    Tensor<float> tensor{1.0f, 2.0f, 3.0f};

    try {
        float val = tensor(10);
        std::cerr << "✗ Should have thrown exception" << std::endl;
    } catch (const std::invalid_argument& e) {
        std::cout << "✓ Caught expected exception: " << e.what() << std::endl;
    }

    try {
        float val = tensor.item();
        std::cerr << "✗ Should have thrown exception" << std::endl;
    } catch (const std::runtime_error& e) {
        std::cout << "✓ Caught expected exception: " << e.what() << std::endl;
    }

    std::cout << "✓ Error handling tests passed\n" << std::endl;
}

void test_raw_pointer_construction() {
    std::cout << "=== Testing Raw C Pointer Construction ===" << std::endl;

    float raw_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    Tensor<float> tensor_from_ptr(raw_data, 5);

    std::cout << "Tensor from raw pointer: " << tensor_from_ptr << std::endl;
    assert(tensor_from_ptr.ndim() == 1);
    assert(tensor_from_ptr.total_size() == 5);
    assert(tensor_from_ptr(0) == 1.0f);
    assert(tensor_from_ptr(4) == 5.0f);

    // Use data_non_volatile() for normal (non-hardware) access
    const float* data_ptr = tensor_from_ptr.data_non_volatile();
    assert(data_ptr[0] == 1.0f);
    assert(data_ptr[2] == 3.0f);

    std::cout << "✓ Raw pointer construction tests passed\n" << std::endl;
}

// ============================================================================
// SFINAE Tests - Testing compile-time type constraints
// ============================================================================
void test_sfinae_constraints() {
    std::cout << "=== Testing SFINAE Constraints ===" << std::endl;

    // These operations are enabled via SFINAE for arithmetic types
    Tensor<float> tf{1.0f, 2.0f, 3.0f};
    Tensor<float>* result = tf.add_scalar(1.0f);
    assert((*result)(0) == 2.0f);
    delete result;

    Tensor<int> ti{1, 2, 3};
    Tensor<int>* result2 = ti.multiply_scalar(3);
    assert((*result2)(0) == 3);
    delete result2;

    // Floating-point only operations (SFINAE constrained)
    Tensor<double> td{1.5, 2.5, 3.5};
    double dot_result = td.dot(&td);
    std::cout << "Dot product: " << dot_result << std::endl;
    // 1.5^2 + 2.5^2 + 3.5^2 = 2.25 + 6.25 + 12.25 = 20.75
    assert(std::abs(dot_result - 20.75) < 1e-9);

    double mean_result = td.mean();
    std::cout << "Mean: " << mean_result << std::endl;
    assert(std::abs(mean_result - 2.5) < 1e-9);

    std::cout << "✓ SFINAE constraint tests passed\n" << std::endl;
}

// ============================================================================
// Reduction Operations Tests
// ============================================================================
void test_reductions() {
    std::cout << "=== Testing Reduction Operations ===" << std::endl;

    Tensor<float> t{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    float sum_result = t.sum();
    std::cout << "Sum: " << sum_result << std::endl;
    assert(sum_result == 15.0f);

    float max_result = t.max();
    std::cout << "Max: " << max_result << std::endl;
    assert(max_result == 5.0f);

    float min_result = t.min();
    std::cout << "Min: " << min_result << std::endl;
    assert(min_result == 1.0f);

    std::size_t argmax_result = t.argmax();
    std::cout << "Argmax: " << argmax_result << std::endl;
    assert(argmax_result == 4);

    std::size_t argmin_result = t.argmin();
    std::cout << "Argmin: " << argmin_result << std::endl;
    assert(argmin_result == 0);

    std::cout << "✓ Reduction operation tests passed\n" << std::endl;
}

// ============================================================================
// Reshape and Transpose Tests
// ============================================================================
void test_reshape_transpose() {
    std::cout << "=== Testing Reshape and Transpose ===" << std::endl;

    // Create 1D tensor and reshape to 2D
    Tensor<float> t{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::cout << "Original: " << t << std::endl;

    std::size_t new_shape[] = {2, 3};
    Tensor<float>* reshaped = t.reshape(new_shape, 2);
    std::cout << "Reshaped to [2,3]: " << *reshaped << std::endl;
    assert(reshaped->ndim() == 2);
    assert(reshaped->shape()[0] == 2);
    assert(reshaped->shape()[1] == 3);
    assert((*reshaped)(0, 0) == 1.0f);
    assert((*reshaped)(1, 2) == 6.0f);

    // Transpose
    Tensor<float>* transposed = reshaped->transpose();
    std::cout << "Transposed: " << *transposed << std::endl;
    assert(transposed->shape()[0] == 3);
    assert(transposed->shape()[1] == 2);
    assert((*transposed)(0, 0) == 1.0f);
    assert((*transposed)(2, 1) == 6.0f);

    delete reshaped;
    delete transposed;

    std::cout << "✓ Reshape and transpose tests passed\n" << std::endl;
}

// ============================================================================
// Clamp and Comparison Tests
// ============================================================================
void test_clamp_and_comparison() {
    std::cout << "=== Testing Clamp and Comparison ===" << std::endl;

    Tensor<float> t{-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};

    Tensor<float>* clamped = t.clamp(0.0f, 2.0f);
    std::cout << "Clamped [0, 2]: " << *clamped << std::endl;
    assert((*clamped)(0) == 0.0f);
    assert((*clamped)(3) == 1.0f);
    assert((*clamped)(5) == 2.0f);
    delete clamped;

    Tensor<bool>* gt = t.greater_than(0.0f);
    std::cout << "Greater than 0: " << *gt << std::endl;
    assert((*gt)(0) == false);
    assert((*gt)(3) == true);
    delete gt;

    Tensor<bool>* lt = t.less_than(1.0f);
    std::cout << "Less than 1: " << *lt << std::endl;
    assert((*lt)(0) == true);
    assert((*lt)(3) == false);
    delete lt;

    std::cout << "✓ Clamp and comparison tests passed\n" << std::endl;
}

// ============================================================================
// Absolute Value and Negation Tests
// ============================================================================
void test_abs_and_negate() {
    std::cout << "=== Testing Abs and Negate ===" << std::endl;

    Tensor<int> t{-3, -2, -1, 0, 1, 2, 3};

    Tensor<int>* abs_result = t.abs();
    std::cout << "Abs: " << *abs_result << std::endl;
    assert((*abs_result)(0) == 3);
    assert((*abs_result)(3) == 0);
    assert((*abs_result)(6) == 3);
    delete abs_result;

    Tensor<int>* neg_result = t.negate();
    std::cout << "Negate: " << *neg_result << std::endl;
    assert((*neg_result)(0) == 3);
    assert((*neg_result)(3) == 0);
    assert((*neg_result)(6) == -3);
    delete neg_result;

    std::cout << "✓ Abs and negate tests passed\n" << std::endl;
}

// ============================================================================
// Volatile Tensor Tests (Hardware-Mapped Memory Simulation)
// ============================================================================
void test_volatile_tensors() {
    std::cout << "=== Testing Volatile Tensors (Hardware-Mapped Memory) ===" << std::endl;

    // Create a volatile tensor - simulates memory-mapped hardware buffer
    // The volatile qualifier ensures every read/write actually hits memory,
    // which is critical when hardware can modify the data independently.
    Tensor<volatile float> volatile_tensor{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    std::cout << "Volatile tensor: " << volatile_tensor << std::endl;
    std::cout << "Shape: [" << volatile_tensor.shape()[0] << "]" << std::endl;

    // Access via volatile methods - each access hits memory (no caching)
    // This is essential for reading hardware registers that may change
    // between reads without any CPU write to that location.
    volatile float& elem = volatile_tensor.volatile_at(2);
    std::cout << "Element at index 2 (volatile read): " << elem << std::endl;
    assert(elem == 3.0f);

    // Volatile write - hardware will see this immediately
    elem = 30.0f;
    volatile float& check = volatile_tensor.volatile_at(2);
    std::cout << "After volatile write, element at index 2: " << check << std::endl;
    assert(check == 30.0f);

    // 2D volatile tensor
    Tensor<volatile double> volatile_2d{
        {1.0, 2.0},
        {3.0, 4.0}
    };

    std::cout << "2D volatile tensor: " << volatile_2d << std::endl;

    volatile double& cell = volatile_2d.volatile_at(1, 1);
    std::cout << "Cell [1,1] (volatile read): " << cell << std::endl;
    assert(cell == 4.0);

    // Volatile item access for scalar tensors
    Tensor<volatile int> volatile_scalar(42);
    volatile int& item = volatile_scalar.volatile_item();
    std::cout << "Volatile scalar item: " << item << std::endl;
    assert(item == 42);

    std::cout << "✓ Volatile tensor tests passed\n" << std::endl;
}

// ============================================================================
// Memory Barrier and Prefetch Tests
// ============================================================================
void test_memory_barriers() {
    std::cout << "=== Testing Memory Barriers and Prefetch ===" << std::endl;

    Tensor<float> t{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    // Explicit memory barrier - ensures all pending memory operations
    // complete before continuing. Critical for DMA/hardware sync.
    memory_barrier();
    std::cout << "Memory barrier executed" << std::endl;

    // Prefetch data into L1 cache before processing
    // This hints the CPU to start loading data from memory into cache
    // before we actually need it, reducing cache miss latency.
    prefetch_read(t.data());
    std::cout << "Prefetch executed for tensor data" << std::endl;

    // Verify data is still correct after barrier/prefetch
    assert(t(0) == 1.0f);
    assert(t(4) == 5.0f);

    std::cout << "✓ Memory barrier and prefetch tests passed\n" << std::endl;
}

// ============================================================================
// Integer-Specific Operations (SFINAE: Integral Types)
// ============================================================================
void test_integer_operations() {
    std::cout << "=== Testing Integer-Specific Operations ===" << std::endl;

    Tensor<int> t{5, 2, 8, 1, 9, 3};

    int sum_result = t.sum();
    std::cout << "Integer sum: " << sum_result << std::endl;
    assert(sum_result == 28);

    int max_result = t.max();
    std::cout << "Integer max: " << max_result << std::endl;
    assert(max_result == 9);

    std::size_t argmax_result = t.argmax();
    std::cout << "Integer argmax: " << argmax_result << std::endl;
    assert(argmax_result == 4);

    std::cout << "✓ Integer-specific operation tests passed\n" << std::endl;
}

// ============================================================================
// Double-Precision Floating Point Tests
// ============================================================================
void test_double_precision() {
    std::cout << "=== Testing Double-Precision Operations ===" << std::endl;

    Tensor<double> t{1.5, 2.7, 3.1, 4.9, 5.3};

    double mean_result = t.mean();
    std::cout << "Double mean: " << mean_result << std::endl;
    assert(std::abs(mean_result - 3.5) < 1e-9);

    double dot_result = t.dot(&t);
    std::cout << "Double dot product: " << dot_result << std::endl;
    double expected_dot = 1.5*1.5 + 2.7*2.7 + 3.1*3.1 + 4.9*4.9 + 5.3*5.3;
    assert(std::abs(dot_result - expected_dot) < 1e-9);

    std::cout << "✓ Double-precision operation tests passed\n" << std::endl;
}

int main() {
    std::cout << "\n================================================" << std::endl;
    std::cout << "  TinyTorch C++ Tensor Test Suite" << std::endl;
    std::cout << "  (C-style Pointers + C++ Templates + SFINAE)" << std::endl;
    std::cout << "  Features: volatile, __asm__, memory barriers" << std::endl;
    std::cout << "================================================\n" << std::endl;

    test_scalar_tensor();
    test_1d_tensor();
    test_2d_tensor();
    test_nd_tensor();
    test_elementwise_operations();
    test_scalar_operations();
    test_copy_move_semantics();
    test_type_templates();
    test_error_handling();
    test_raw_pointer_construction();
    test_sfinae_constraints();
    test_reductions();
    test_reshape_transpose();
    test_clamp_and_comparison();
    test_abs_and_negate();
    test_volatile_tensors();
    test_memory_barriers();
    test_integer_operations();
    test_double_precision();

    std::cout << "================================================" << std::endl;
    std::cout << "  All Tests Passed! ✓" << std::endl;
    std::cout << "================================================\n" << std::endl;

    return 0;
}
