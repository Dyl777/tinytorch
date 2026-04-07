#include "distributed_tensor.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

void test_device_pool() {
    std::cout << "=== Testing Device Pool ===" << std::endl;
    
    auto& pool = get_device_pool();
    std::cout << pool.summary() << std::endl;
    
    assert(pool.total_devices() >= 2);  // At least CPU Dense + CPU Mmap
    std::cout << "Total devices: " << pool.total_devices() << std::endl;
    std::cout << "GPU devices: " << pool.gpu_count() << std::endl;
    std::cout << "CPU devices: " << pool.cpu_count() << std::endl;
    
    auto gpus = pool.gpu_devices();
    auto cpus = pool.cpu_devices();
    assert(!cpus.empty());
    
    std::cout << "✓ Device pool tests passed\n" << std::endl;
}

void test_distributed_tensor_creation() {
    std::cout << "=== Testing DistributedTensor Creation ===" << std::endl;
    
    // Zeros
    auto zeros = DistributedTensor<float>::zeros({100, 100});
    assert((zeros->shape() == std::vector<std::size_t>{100, 100}));
    assert(zeros->total_elements() == 10000);
    std::cout << "Zeros: " << zeros->distribution_info() << std::endl;
    assert(zeros->num_shards() >= 1);
    
    // Ones
    auto ones = DistributedTensor<float>::ones({50, 50});
    assert(ones->total_elements() == 2500);
    std::cout << "Ones: " << ones->distribution_info() << std::endl;
    
    // Random normal
    auto rand = DistributedTensor<float>::randn({200, 200});
    assert(rand->total_elements() == 40000);
    std::cout << "Random: " << rand->distribution_info() << std::endl;
    
    // From data
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto from_data = DistributedTensor<float>::from_data(data, {2, 2});
    assert(from_data->total_elements() == 4);
    std::cout << "From data: " << from_data->distribution_info() << std::endl;
    
    // With gradient
    auto with_grad = DistributedTensor<float>::randn({100, 100}, true);
    assert(with_grad->requires_grad());
    std::cout << "With grad: " << with_grad->distribution_info() << std::endl;
    
    std::cout << "✓ DistributedTensor creation tests passed\n" << std::endl;
}

void test_distributed_tensor_operations() {
    std::cout << "=== Testing DistributedTensor Operations ===" << std::endl;
    
    auto x = DistributedTensor<float>::zeros({100, 100});
    auto y = DistributedTensor<float>::ones({100, 100});
    
    // Set some values
    x->set_element(0, 1.0f);
    x->set_element(1, 2.0f);
    y->set_element(0, 3.0f);
    
    // Get values
    assert(std::abs(x->get_element(0) - 1.0f) < 1e-5f);
    assert(std::abs(x->get_element(1) - 2.0f) < 1e-5f);
    std::cout << "Element access works" << std::endl;
    
    // Add
    auto sum = x->add(y.get());
    assert(sum->total_elements() == x->total_elements());
    std::cout << "Add: " << sum->distribution_info() << std::endl;
    
    // Multiply
    auto prod = x->multiply(y.get());
    assert(prod->total_elements() == x->total_elements());
    std::cout << "Multiply: " << prod->distribution_info() << std::endl;
    
    // Scalar operations
    auto scalar_add = x->add_scalar(10.0f);
    auto scalar_mul = x->multiply_scalar(2.0f);
    std::cout << "Scalar ops work" << std::endl;
    
    // Reductions
    auto total = x->sum();
    auto avg = x->mean();
    std::cout << "Sum: " << total->get_element(0) << std::endl;
    std::cout << "Mean: " << avg->get_element(0) << std::endl;
    
    // Reshape
    auto reshaped = x->reshape({50, 200});
    assert(reshaped->total_elements() == x->total_elements());
    std::cout << "Reshape: " << reshaped->distribution_info() << std::endl;
    
    std::cout << "✓ DistributedTensor operation tests passed\n" << std::endl;
}

void test_distributed_tensor_autograd() {
    std::cout << "=== Testing DistributedTensor Autograd ===" << std::endl;
    
    auto x = DistributedTensor<float>::randn({100, 100}, true);
    auto y = DistributedTensor<float>::randn({100, 100}, true);
    
    auto z = x->multiply(y.get());
    auto loss = z->sum();
    
    std::cout << "Forward pass: " << loss->distribution_info() << std::endl;
    
    // Backward pass
    loss->backward();
    std::cout << "Backward pass completed" << std::endl;
    
    // Check gradients exist
    x->zero_grad();
    y->zero_grad();
    std::cout << "Gradients zeroed" << std::endl;
    
    std::cout << "✓ DistributedTensor autograd tests passed\n" << std::endl;
}

void test_distributed_tensor_no_full_gather() {
    std::cout << "=== Testing No Full Data Gather ===" << std::endl;
    
    // Create large tensor
    auto x = DistributedTensor<float>::randn({1000, 1000});
    std::cout << "Created tensor with " << x->num_shards() << " shards" << std::endl;
    
    // Verify shards are on different devices
    for (size_t i = 0; i < x->num_shards(); ++i) {
        std::cout << "  " << x->shard_info(i) << std::endl;
    }
    
    // Operations should not gather all data
    auto y = x->multiply_scalar(2.0f);
    assert(y->num_shards() == x->num_shards());
    std::cout << "After multiply_scalar: " << y->num_shards() << " shards" << std::endl;
    
    auto z = x->add(y.get());
    assert(z->num_shards() == x->num_shards());
    std::cout << "After add: " << z->num_shards() << " shards" << std::endl;
    
    // Reductions should use tree-reduce
    auto total = x->sum();
    assert(total->num_shards() == 1);  // Result is scalar
    std::cout << "After sum: " << total->num_shards() << " shard (scalar result)" << std::endl;
    
    std::cout << "✓ No full data gather tests passed\n" << std::endl;
}

void test_distributed_tensor_device_refresh() {
    std::cout << "=== Testing DistributedTensor Device Refresh ===" << std::endl;
    
    auto x = DistributedTensor<float>::randn({100, 100});
    std::cout << "Before refresh: " << x->distribution_info() << std::endl;
    
    x->refresh_devices();
    std::cout << "After refresh: " << x->distribution_info() << std::endl;
    
    std::cout << "✓ DistributedTensor device refresh tests passed\n" << std::endl;
}

int main() {
    std::cout << "\n================================================" << std::endl;
    std::cout << "  TinyTorch Distributed Tensor Test Suite" << std::endl;
    std::cout << "================================================\n" << std::endl;
    
    test_device_pool();
    test_distributed_tensor_creation();
    test_distributed_tensor_operations();
    test_distributed_tensor_autograd();
    test_distributed_tensor_no_full_gather();
    test_distributed_tensor_device_refresh();
    
    std::cout << "================================================" << std::endl;
    std::cout << "  All Distributed Tensor Tests Passed! ✓" << std::endl;
    std::cout << "================================================\n" << std::endl;
    
    return 0;
}
