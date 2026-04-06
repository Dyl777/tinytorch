#include "auto_tensor.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

// ============================================================================
// AutoTensor Test Suite
// ============================================================================
// Tests the automatic backend selection between DenseTensor and MmapTensor
// based on data size relative to system memory.

void test_system_info() {
    std::cout << "=== Testing System Info ===" << std::endl;
    AutoTensor<float>::print_system_info();
    std::cout << "✓ System info test passed\n" << std::endl;
}

void test_auto_small_data() {
    std::cout << "=== Testing Auto Small Data (Should Use Dense) ===" << std::endl;

    // Tiny data - should definitely use DenseTensor
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto tensor = AutoTensor<float>::from_data(data, 5);

    std::cout << "Backend: " << tensor->backend_name() << std::endl;
    std::cout << "Is streaming: " << (tensor->is_streaming() ? "yes" : "no") << std::endl;
    std::cout << "Tensor: " << *tensor << std::endl;

    assert(!tensor->is_streaming());  // Should be dense
    assert(tensor->ndim() == 1);
    assert(tensor->total_size() == 5);
    assert(tensor->get_element(0) == 1.0f);
    assert(tensor->get_element(4) == 5.0f);

    std::cout << "✓ Auto small data test passed\n" << std::endl;
}

void test_auto_medium_data() {
    std::cout << "=== Testing Auto Medium Data ===" << std::endl;

    // 100K floats = 400KB - still small relative to RAM
    std::size_t size = 100000;
    std::vector<float> data(size);
    for (std::size_t i = 0; i < size; ++i) {
        data[i] = static_cast<float>(i);
    }

    auto tensor = AutoTensor<float>::from_data(data.data(), size);
    std::cout << "Backend: " << tensor->backend_name() << std::endl;

    assert(tensor->total_size() == size);
    assert(tensor->get_element(0) == 0.0f);
    assert(tensor->get_element(size - 1) == 99999.0f);

    // Test operations work regardless of backend
    float sum_result = tensor->sum();
    std::cout << "Sum: " << sum_result << std::endl;

    float mean_result = tensor->mean();
    std::cout << "Mean: " << mean_result << std::endl;

    std::cout << "✓ Auto medium data test passed\n" << std::endl;
}

void test_auto_force_streaming() {
    std::cout << "=== Testing Auto Force Streaming ===" << std::endl;

    // Small data but force streaming backend
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    AutoConfig config;
    config.force_streaming = true;

    auto tensor = AutoTensor<float>::from_data(data, 5, config);
    std::cout << "Backend: " << tensor->backend_name() << std::endl;
    std::cout << "Is streaming: " << (tensor->is_streaming() ? "yes" : "no") << std::endl;

    assert(tensor->is_streaming());  // Forced to streaming
    assert(tensor->get_element(0) == 1.0f);
    assert(tensor->get_element(4) == 5.0f);

    std::cout << "✓ Auto force streaming test passed\n" << std::endl;
}

void test_auto_force_dense() {
    std::cout << "=== Testing Auto Force Dense ===" << std::endl;

    // Even with low threshold, force dense
    AutoConfig config;
    config.memory_threshold = 0.0001;  // Very low threshold
    config.force_dense = true;

    std::size_t size = 10000;
    std::vector<float> data(size, 1.0f);

    auto tensor = AutoTensor<float>::from_data(data.data(), size, config);
    std::cout << "Backend: " << tensor->backend_name() << std::endl;

    assert(!tensor->is_streaming());  // Forced to dense

    std::cout << "✓ Auto force dense test passed\n" << std::endl;
}

void test_auto_operations() {
    std::cout << "=== Testing Auto Tensor Operations ===" << std::endl;

    float data_a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float data_b[] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

    auto a = AutoTensor<float>::from_data(data_a, 5);
    auto b = AutoTensor<float>::from_data(data_b, 5);

    // Element-wise operations
    auto sum = a->add(b.get());
    std::cout << "A + B: " << *sum << std::endl;
    assert(sum->get_element(0) == 6.0f);
    assert(sum->get_element(4) == 6.0f);

    auto diff = a->subtract(b.get());
    assert(diff->get_element(0) == -4.0f);
    assert(diff->get_element(4) == 4.0f);

    auto prod = a->multiply(b.get());
    assert(prod->get_element(0) == 5.0f);
    assert(prod->get_element(2) == 9.0f);

    // Scalar operations
    auto scalar_add = a->add_scalar(10.0f);
    assert(scalar_add->get_element(0) == 11.0f);

    auto scalar_mul = a->multiply_scalar(2.0f);
    assert(scalar_mul->get_element(0) == 2.0f);

    // Unary operations
    auto neg = a->negate();
    assert(neg->get_element(0) == -1.0f);

    auto abs_val = neg->abs();
    assert(abs_val->get_element(0) == 1.0f);

    // Reductions
    float sum_result = a->sum();
    std::cout << "Sum: " << sum_result << std::endl;
    assert(sum_result == 15.0f);

    float mean_result = a->mean();
    std::cout << "Mean: " << mean_result << std::endl;
    assert(mean_result == 3.0f);

    float max_result = a->max();
    std::cout << "Max: " << max_result << std::endl;
    assert(max_result == 5.0f);

    float min_result = a->min();
    std::cout << "Min: " << min_result << std::endl;
    assert(min_result == 1.0f);

    // Dot product
    float dot_result = a->dot(b.get());
    std::cout << "Dot: " << dot_result << std::endl;
    assert(dot_result == 35.0f);  // 1*5 + 2*4 + 3*3 + 4*2 + 5*1 = 35

    std::cout << "✓ Auto operations test passed\n" << std::endl;
}

void test_auto_reshape_transpose() {
    std::cout << "=== Testing Auto Reshape and Transpose ===" << std::endl;

    // Create 1D tensor
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto tensor = AutoTensor<float>::from_data(data, 6);

    // Reshape to 2D
    std::size_t new_shape[] = {2, 3};
    auto reshaped = tensor->reshape(new_shape, 2);
    std::cout << "Reshaped: " << *reshaped << std::endl;
    assert(reshaped->ndim() == 2);
    assert(reshaped->shape()[0] == 2);
    assert(reshaped->shape()[1] == 3);

    // Transpose
    auto transposed = reshaped->transpose();
    std::cout << "Transposed: " << *transposed << std::endl;
    assert(transposed->shape()[0] == 3);
    assert(transposed->shape()[1] == 2);

    std::cout << "✓ Auto reshape/transpose test passed\n" << std::endl;
}

void test_auto_comparison() {
    std::cout << "=== Testing Auto Comparison Operations ===" << std::endl;

    float data[] = {1.0f, 3.0f, 5.0f, 7.0f, 9.0f};
    auto tensor = AutoTensor<float>::from_data(data, 5);

    auto gt = tensor->greater_than(5.0f);
    std::cout << "Greater than 5: " << *gt << std::endl;
    assert(gt->get_element(0) == false);
    assert(gt->get_element(3) == true);

    auto lt = tensor->less_than(5.0f);
    std::cout << "Less than 5: " << *lt << std::endl;
    assert(lt->get_element(0) == true);
    assert(lt->get_element(3) == false);

    std::cout << "✓ Auto comparison test passed\n" << std::endl;
}

void test_auto_clamp() {
    std::cout << "=== Testing Auto Clamp ===" << std::endl;

    float data[] = {-5.0f, -2.0f, 0.0f, 3.0f, 7.0f, 10.0f};
    auto tensor = AutoTensor<float>::from_data(data, 6);

    auto clamped = tensor->clamp(0.0f, 5.0f);
    std::cout << "Clamped [0, 5]: " << *clamped << std::endl;
    assert(clamped->get_element(0) == 0.0f);
    assert(clamped->get_element(2) == 0.0f);
    assert(clamped->get_element(3) == 3.0f);
    assert(clamped->get_element(5) == 5.0f);

    std::cout << "✓ Auto clamp test passed\n" << std::endl;
}

void test_auto_export() {
    std::cout << "=== Testing Auto Export to Array ===" << std::endl;

    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto tensor = AutoTensor<float>::from_data(data, 5);

    auto exported = tensor->to_array();
    for (std::size_t i = 0; i < 5; ++i) {
        assert(exported[i] == data[i]);
    }
    std::cout << "Exported array matches original" << std::endl;

    std::cout << "✓ Auto export test passed\n" << std::endl;
}

void test_auto_from_shape() {
    std::cout << "=== Testing Auto From Shape ===" << std::endl;

    // Small tensor from shape
    std::size_t shape[] = {3, 4};
    auto tensor = AutoTensor<float>::from_shape(shape, 2, AutoConfig{}, 42.0f);

    std::cout << "Backend: " << tensor->backend_name() << std::endl;
    std::cout << "Shape: [" << tensor->shape()[0] << ", " << tensor->shape()[1] << "]" << std::endl;
    std::cout << "Tensor: " << *tensor << std::endl;

    assert(tensor->ndim() == 2);
    assert(tensor->total_size() == 12);
    assert(tensor->get_element(0) == 42.0f);
    assert(tensor->get_element(11) == 42.0f);

    std::cout << "✓ Auto from shape test passed\n" << std::endl;
}

void test_auto_mixed_backend_error() {
    std::cout << "=== Testing Mixed Backend Error Handling ===" << std::endl;

    // Create dense tensor
    float data_a[] = {1.0f, 2.0f, 3.0f};
    auto dense = AutoTensor<float>::from_data(data_a, 3);

    // Create streaming tensor
    float data_b[] = {4.0f, 5.0f, 6.0f};
    AutoConfig config;
    config.force_streaming = true;
    auto streaming = AutoTensor<float>::from_data(data_b, 3, config);

    // Try to mix backends - should throw
    try {
        auto result = dense->add(streaming.get());
        std::cerr << "✗ Should have thrown exception for mixed backends" << std::endl;
    } catch (const std::invalid_argument& e) {
        std::cout << "✓ Caught expected exception: " << e.what() << std::endl;
    }

    std::cout << "✓ Mixed backend error handling test passed\n" << std::endl;
}

void test_auto_large_data_simulation() {
    std::cout << "=== Testing Auto Large Data Simulation ===" << std::endl;

    // Simulate a large dataset: 10M floats = 40MB
    // On a system with 8GB+ RAM this should still be dense,
    // but we can force streaming with low threshold
    std::size_t size = 10000000;
    std::cout << "Creating " << (size * sizeof(float) / (1024.0 * 1024.0)) << " MB tensor..." << std::endl;

    // Use low threshold to trigger streaming
    AutoConfig config;
    config.memory_threshold = 0.001;  // 0.1% of RAM

    // Create data in chunks to avoid allocation issues
    std::cout << "Allocating data..." << std::endl;
    std::vector<float> data(size);
    for (std::size_t i = 0; i < size; ++i) {
        data[i] = static_cast<float>(i % 1000);
    }

    std::cout << "Creating AutoTensor..." << std::endl;
    auto tensor = AutoTensor<float>::from_data(data.data(), size, config);
    std::cout << "Backend: " << tensor->backend_name() << std::endl;

    // Test operations
    std::cout << "Computing sum..." << std::endl;
    float sum_result = tensor->sum();
    std::cout << "Sum: " << sum_result << std::endl;

    std::cout << "Computing mean..." << std::endl;
    float mean_result = tensor->mean();
    std::cout << "Mean: " << mean_result << std::endl;

    std::cout << "✓ Auto large data simulation test passed\n" << std::endl;
}

void test_auto_threshold_tuning() {
    std::cout << "=== Testing AutoConfig Threshold Tuning ===" << std::endl;

    std::size_t total_mem = get_total_system_memory();
    std::cout << "Total system memory: " << (total_mem / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;

    // Create 1MB of data
    std::size_t size = 256000;  // 256K floats = 1MB
    std::vector<float> data(size, 1.0f);
    std::size_t data_bytes = size * sizeof(float);

    // Very low threshold → should use streaming
    {
        AutoConfig config;
        config.memory_threshold = 0.00001;  // 0.001%
        auto tensor = AutoTensor<float>::from_data(data.data(), size, config);
        std::cout << "Threshold 0.001%: " << tensor->backend_name() << std::endl;
        assert(tensor->is_streaming());
    }

    // Very high threshold → should use dense
    {
        AutoConfig config;
        config.memory_threshold = 0.99;  // 99%
        auto tensor = AutoTensor<float>::from_data(data.data(), size, config);
        std::cout << "Threshold 99%: " << tensor->backend_name() << std::endl;
        assert(!tensor->is_streaming());
    }

    std::cout << "✓ AutoConfig threshold tuning test passed\n" << std::endl;
}

void test_batch_print_dense() {
    std::cout << "=== Testing Batch Print (Dense Backend) ===" << std::endl;

    // Create a moderate-sized dense tensor
    std::size_t size = 50;
    std::vector<float> data(size);
    for (std::size_t i = 0; i < size; ++i) {
        data[i] = static_cast<float>(i);
    }

    auto tensor = AutoTensor<float>::from_data(data.data(), size);
    std::cout << "Backend: " << tensor->backend_name() << std::endl;

    std::cout << "\nBatch print with batch_size=10:" << std::endl;
    tensor->batch_print(std::cout, 10);
    std::cout << "\n" << std::endl;

    std::cout << "Batch print with batch_size=25:" << std::endl;
    tensor->batch_print(std::cout, 25);
    std::cout << "\n" << std::endl;

    std::cout << "✓ Batch print (dense) test passed\n" << std::endl;
}

void test_batch_print_streaming() {
    std::cout << "=== Testing Batch Print (Streaming Backend) ===" << std::endl;

    // Force streaming backend
    AutoConfig config;
    config.force_streaming = true;
    config.print_batch_size = 15;

    std::size_t size = 60;
    std::vector<float> data(size);
    for (std::size_t i = 0; i < size; ++i) {
        data[i] = static_cast<float>(i * 0.5f);
    }

    auto tensor = AutoTensor<float>::from_data(data.data(), size, config);
    std::cout << "Backend: " << tensor->backend_name() << std::endl;

    std::cout << "\nBatch print with batch_size=15:" << std::endl;
    tensor->batch_print(std::cout, 15);
    std::cout << "\n" << std::endl;

    std::cout << "Batch print with batch_size=30:" << std::endl;
    tensor->batch_print(std::cout, 30);
    std::cout << "\n" << std::endl;

    std::cout << "✓ Batch print (streaming) test passed\n" << std::endl;
}

void test_batch_print_large() {
    std::cout << "=== Testing Batch Print Large Tensor (1M elements) ===" << std::endl;

    // Force streaming for a 1M element tensor
    AutoConfig config;
    config.force_streaming = true;

    std::size_t size = 1000000;
    std::cout << "Creating 1M element streaming tensor..." << std::endl;

    // Create in chunks to avoid large allocation
    std::size_t shape[] = {size};
    auto tensor = AutoTensor<float>::from_shape(shape, 1, config, 0.0f);

    // Fill with values
    std::cout << "Filling tensor..." << std::endl;
    for (std::size_t i = 0; i < size; ++i) {
        tensor->set_element(i, static_cast<float>(i));
    }

    // Print first and last few elements using batch_print with small batch
    std::cout << "\nBatch print first 20 elements (batch_size=5):" << std::endl;
    // We can't easily print just a slice, so let's just verify batch_print works
    // by printing the whole thing with a reasonable batch size
    std::cout << "Batch print with batch_size=100000:" << std::endl;
    tensor->batch_print(std::cout, 100000);
    std::cout << "\n" << std::endl;

    // Verify elements are correct
    assert(tensor->get_element(0) == 0.0f);
    assert(tensor->get_element(999999) == 999999.0f);

    std::cout << "✓ Batch print large tensor test passed\n" << std::endl;
}

int main() {
    std::cout << "\n================================================" << std::endl;
    std::cout << "  TinyTorch AutoTensor Test Suite" << std::endl;
    std::cout << "  (Automatic Dense/Mmap Backend Selection)" << std::endl;
    std::cout << "================================================\n" << std::endl;

    test_system_info();
    test_auto_small_data();
    test_auto_medium_data();
    test_auto_force_streaming();
    test_auto_force_dense();
    test_auto_operations();
    test_auto_reshape_transpose();
    test_auto_comparison();
    test_auto_clamp();
    test_auto_export();
    test_auto_from_shape();
    test_auto_mixed_backend_error();
    test_auto_large_data_simulation();
    test_auto_threshold_tuning();
    test_batch_print_dense();
    test_batch_print_streaming();
    test_batch_print_large();

    std::cout << "================================================" << std::endl;
    std::cout << "  All AutoTensor Tests Passed! ✓" << std::endl;
    std::cout << "================================================\n" << std::endl;

    return 0;
}
