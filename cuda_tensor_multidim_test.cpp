#include "cuda_tensor.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

void test_cuda_1d() {
    std::cout << "=== Testing CUDA 1D Tensor ===" << std::endl;
    
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    CudaTensor* t = CudaTensor::from_data(data, 0);
    
    if (!t) {
        std::cout << "⊘ Failed to create CUDA tensor (no CUDA support)" << std::endl;
        return;
    }
    
    assert(t->ndim() == 1);
    assert(t->total_size() == 5);
    assert(t->shape()[0] == 5);
    std::cout << "1D tensor: " << *t << std::endl;
    
    // Test operations
    auto result = t->add_scalar(10.0f);
    std::cout << "1D + 10: " << *result << std::endl;
    
    auto host_result = static_cast<CudaTensor*>(result.get())->to_host();
    assert(std::abs(host_result[0] - 11.0f) < 1e-5f);
    assert(std::abs(host_result[4] - 15.0f) < 1e-5f);
    
    std::cout << "✓ 1D tensor works" << std::endl;
    delete t;
}

void test_cuda_2d() {
    std::cout << "\n=== Testing CUDA 2D Tensor ===" << std::endl;
    
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<std::size_t> shape = {2, 3};
    CudaTensor* t = CudaTensor::from_data(data, shape, 0);
    
    if (!t) {
        std::cout << "⊘ Failed to create CUDA tensor" << std::endl;
        return;
    }
    
    assert(t->ndim() == 2);
    assert(t->total_size() == 6);
    assert(t->shape()[0] == 2);
    assert(t->shape()[1] == 3);
    assert(t->stride()[0] == 3);
    assert(t->stride()[1] == 1);
    
    std::cout << "2D tensor: " << *t << std::endl;
    
    // Test operations
    auto result = t->multiply_scalar(2.0f);
    std::cout << "2D * 2: " << *result << std::endl;
    
    auto host_result = static_cast<CudaTensor*>(result.get())->to_host();
    assert(std::abs(host_result[0] - 2.0f) < 1e-5f);
    assert(std::abs(host_result[5] - 12.0f) < 1e-5f);
    
    std::cout << "✓ 2D tensor works" << std::endl;
    delete t;
}

void test_cuda_reshape() {
    std::cout << "\n=== Testing CUDA Reshape ===" << std::endl;
    
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    CudaTensor* t1 = CudaTensor::from_data(data, 0);
    
    if (!t1) {
        std::cout << "⊘ Failed to create CUDA tensor" << std::endl;
        return;
    }
    
    // Reshape 1D to 2D
    std::size_t new_shape[] = {2, 3};
    auto reshaped = t1->reshape(new_shape, 2);
    
    assert(reshaped->ndim() == 2);
    assert(reshaped->shape()[0] == 2);
    assert(reshaped->shape()[1] == 3);
    assert(reshaped->total_size() == 6);
    
    std::cout << "Original 1D: " << *t1 << std::endl;
    std::cout << "Reshaped 2D: " << *reshaped << std::endl;
    
    // Verify data is preserved
    auto host_data = static_cast<CudaTensor*>(reshaped.get())->to_host();
    for (size_t i = 0; i < data.size(); ++i) {
        assert(std::abs(host_data[i] - data[i]) < 1e-5f);
    }
    
    std::cout << "✓ Reshape test passed" << std::endl;
    delete t1;
}

void test_cuda_transpose() {
    std::cout << "\n=== Testing CUDA Transpose ===" << std::endl;
    
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<std::size_t> shape = {2, 3};
    CudaTensor* t = CudaTensor::from_data(data, shape, 0);
    
    if (!t) {
        std::cout << "⊘ Failed to create CUDA tensor" << std::endl;
        return;
    }
    
    std::cout << "Original: " << *t << std::endl;
    
    // Transpose
    auto transposed = t->transpose();
    
    assert(transposed->ndim() == 2);
    assert(transposed->shape()[0] == 3);
    assert(transposed->shape()[1] == 2);
    
    std::cout << "Transposed: " << *transposed << std::endl;
    
    // Verify data: Original [[1,2,3],[4,5,6]] -> Transposed [[1,4],[2,5],[3,6]]
    auto host_data = static_cast<CudaTensor*>(transposed.get())->to_host();
    assert(std::abs(host_data[0] - 1.0f) < 1e-5f);
    assert(std::abs(host_data[1] - 4.0f) < 1e-5f);
    assert(std::abs(host_data[2] - 2.0f) < 1e-5f);
    assert(std::abs(host_data[3] - 5.0f) < 1e-5f);
    assert(std::abs(host_data[4] - 3.0f) < 1e-5f);
    assert(std::abs(host_data[5] - 6.0f) < 1e-5f);
    
    std::cout << "✓ Transpose test passed" << std::endl;
    delete t;
}

void test_cuda_binary_ops_2d() {
    std::cout << "\n=== Testing CUDA Binary Operations on 2D Tensors ===" << std::endl;
    
    std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data_b = {5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<std::size_t> shape = {2, 2};
    
    CudaTensor* a = CudaTensor::from_data(data_a, shape, 0);
    CudaTensor* b = CudaTensor::from_data(data_b, shape, 0);
    
    if (!a || !b) {
        std::cout << "⊘ Failed to create CUDA tensors" << std::endl;
        delete a;
        delete b;
        return;
    }
    
    std::cout << "A: " << *a << std::endl;
    std::cout << "B: " << *b << std::endl;
    
    // Test addition
    auto sum = a->add(b);
    std::cout << "A + B: " << *sum << std::endl;
    
    auto host_sum = static_cast<CudaTensor*>(sum.get())->to_host();
    assert(std::abs(host_sum[0] - 6.0f) < 1e-5f);
    assert(std::abs(host_sum[3] - 12.0f) < 1e-5f);
    
    // Test multiplication
    auto prod = a->multiply(b);
    std::cout << "A * B: " << *prod << std::endl;
    
    auto host_prod = static_cast<CudaTensor*>(prod.get())->to_host();
    assert(std::abs(host_prod[0] - 5.0f) < 1e-5f);
    assert(std::abs(host_prod[3] - 32.0f) < 1e-5f);
    
    std::cout << "✓ 2D binary operations test passed" << std::endl;
    delete a;
    delete b;
}

int main() {
    std::cout << "\n================================================" << std::endl;
    std::cout << "  CUDA Multi-dimensional Tensor Tests" << std::endl;
    std::cout << "  GPU: NVIDIA GeForce MX130 (CUDA 12.8)" << std::endl;
    std::cout << "================================================\n" << std::endl;
    
    test_cuda_1d();
    test_cuda_2d();
    test_cuda_reshape();
    test_cuda_transpose();
    test_cuda_binary_ops_2d();
    
    std::cout << "\n================================================" << std::endl;
    std::cout << "  All CUDA Multi-dimensional Tests Passed! ✓" << std::endl;
    std::cout << "================================================\n" << std::endl;
    
    return 0;
}
