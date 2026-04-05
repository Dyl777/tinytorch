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
    
    // Initializer list constructor
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
    
    // Test mutable access
    tensor_1d(1) = 20.0f;
    assert(tensor_1d(1) == 20.0f);
    
    std::cout << "✓ 1D tensor tests passed\n" << std::endl;
}

void test_2d_tensor() {
    std::cout << "=== Testing 2D Tensor ===" << std::endl;
    
    // Nested initializer list constructor
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
    
    // Test mutable access
    tensor_2d(0, 2) = 30.0;
    assert(tensor_2d(0, 2) == 30.0);
    
    std::cout << "✓ 2D tensor tests passed\n" << std::endl;
}

void test_nd_tensor() {
    std::cout << "=== Testing N-Dimensional Tensor ===" << std::endl;
    
    // Create a 3D tensor with shape [2, 3, 4] using raw C pointer
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
    
    // Fill with values using raw pointer access
    int* data_ptr = tensor_3d.data();
    for (std::size_t i = 0; i < tensor_3d.total_size(); ++i) {
        data_ptr[i] = static_cast<int>(i);
    }
    
    // Test N-dimensional indexing using raw C pointer array
    std::size_t indices[] = {1, 2, 3};
    std::cout << "Element at [1,2,3]: " << tensor_3d.at(indices) << std::endl;
    assert(tensor_3d.at(indices) == 23);
    
    std::cout << "✓ N-D tensor tests passed\n" << std::endl;
}

void test_elementwise_operations() {
    std::cout << "=== Testing Element-wise Operations ===" << std::endl;
    
    // 1D + 1D using raw C pointers
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
    
    // 1D - 1D
    Tensor<float>* diff = a.subtract(&b);
    std::cout << "A - B: " << *diff << std::endl;
    assert((*diff)(0) == -3.0f);
    assert((*diff)(1) == -3.0f);
    assert((*diff)(2) == -3.0f);
    
    // 1D * 1D
    Tensor<float>* prod = a.multiply(&b);
    std::cout << "A * B: " << *prod << std::endl;
    assert((*prod)(0) == 4.0f);
    assert((*prod)(1) == 10.0f);
    assert((*prod)(2) == 18.0f);
    
    // Scalar + 1D
    Tensor<float> scalar(10.0f);
    Tensor<float>* scalar_add = scalar.add(&a);
    std::cout << "10 + A: " << *scalar_add << std::endl;
    assert((*scalar_add)(0) == 11.0f);
    assert((*scalar_add)(1) == 12.0f);
    assert((*scalar_add)(2) == 13.0f);
    
    // Clean up raw pointers
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
    
    // Clean up raw pointers
    delete added;
    delete multiplied;
    
    std::cout << "✓ Scalar operation tests passed\n" << std::endl;
}

void test_copy_move_semantics() {
    std::cout << "=== Testing Copy/Move Semantics ===" << std::endl;
    
    Tensor<float> original{1.0f, 2.0f, 3.0f};
    
    // Copy constructor
    Tensor<float> copied(original);
    std::cout << "Original: " << original << std::endl;
    std::cout << "Copied: " << copied << std::endl;
    assert(copied(0) == original(0));
    assert(copied(1) == original(1));
    
    // Modify original - copy should be independent
    original(0) = 100.0f;
    assert(copied(0) == 1.0f);
    
    // Move constructor
    Tensor<float> moved(std::move(original));
    std::cout << "Moved: " << moved << std::endl;
    assert(moved(0) == 100.0f);
    
    // Move assignment
    Tensor<float> assigned = std::move(moved);
    std::cout << "Assigned: " << assigned << std::endl;
    assert(assigned(0) == 100.0f);
    
    std::cout << "✓ Copy/Move semantics tests passed\n" << std::endl;
}

void test_type_templates() {
    std::cout << "=== Testing Template Types ===" << std::endl;
    
    // Integer tensor
    Tensor<int> int_tensor{1, 2, 3, 4, 5};
    std::cout << "Int tensor: " << int_tensor << std::endl;
    
    // Double tensor
    Tensor<double> double_tensor{1.1, 2.2, 3.3};
    std::cout << "Double tensor: " << double_tensor << std::endl;
    
    // Float tensor
    Tensor<float> float_tensor{1.0f, 2.0f, 3.0f};
    std::cout << "Float tensor: " << float_tensor << std::endl;
    
    std::cout << "✓ Template type tests passed\n" << std::endl;
}

void test_error_handling() {
    std::cout << "=== Testing Error Handling ===" << std::endl;
    
    Tensor<float> tensor{1.0f, 2.0f, 3.0f};
    
    // Test out of bounds
    try {
        float val = tensor(10);
        std::cerr << "✗ Should have thrown exception" << std::endl;
    } catch (const std::invalid_argument& e) {
        std::cout << "✓ Caught expected exception: " << e.what() << std::endl;
    }
    
    // Test item() on multi-element tensor
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
    
    // Create tensor from raw C pointer
    float raw_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    Tensor<float> tensor_from_ptr(raw_data, 5);
    
    std::cout << "Tensor from raw pointer: " << tensor_from_ptr << std::endl;
    assert(tensor_from_ptr.ndim() == 1);
    assert(tensor_from_ptr.total_size() == 5);
    assert(tensor_from_ptr(0) == 1.0f);
    assert(tensor_from_ptr(4) == 5.0f);
    
    // Get raw data pointer back
    const float* data_ptr = tensor_from_ptr.data();
    assert(data_ptr[0] == 1.0f);
    assert(data_ptr[2] == 3.0f);
    
    std::cout << "✓ Raw pointer construction tests passed\n" << std::endl;
}

int main() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  TinyTorch C++ Tensor Test Suite" << std::endl;
    std::cout << "  (C-style Pointers + C++ Templates)" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
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
    
    std::cout << "========================================" << std::endl;
    std::cout << "  All Tests Passed! ✓" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    return 0;
}
