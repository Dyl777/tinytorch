#include "stream_tensor.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

// ============================================================================
// StreamTensor Test Suite
// ============================================================================
// These tests verify the memory-mapped streaming tensor implementation.
// StreamTensor stores data in mmap-backed files and processes in batches,
// allowing tensors larger than physical RAM to be processed.

void test_stream_tensor_creation() {
    std::cout << "=== Testing StreamTensor Creation ===" << std::endl;

    // Create a small StreamTensor with shape [5]
    std::size_t shape[] = {5};
    StreamConfig config;
    config.batch_size = 2;  // Small batch for testing
    config.auto_cleanup = true;

    StreamTensor<float> st(shape, 1, config, 0.0f);

    std::cout << "StreamTensor: " << st << std::endl;
    std::cout << "Shape: [" << st.shape()[0] << "]" << std::endl;
    std::cout << "Total size: " << st.total_size() << std::endl;
    std::cout << "NDim: " << st.ndim() << std::endl;
    std::cout << "Batch size: " << st.effective_batch_size() << std::endl;

    assert(st.ndim() == 1);
    assert(st.total_size() == 5);
    assert(st.shape()[0] == 5);

    // Verify initial fill
    for (std::size_t i = 0; i < 5; ++i) {
        assert(st.get_element(i) == 0.0f);
    }

    std::cout << "✓ StreamTensor creation tests passed\n" << std::endl;
}

void test_stream_element_access() {
    std::cout << "=== Testing StreamTensor Element Access ===" << std::endl;

    std::size_t shape[] = {5};
    StreamConfig config;
    config.batch_size = 2;

    StreamTensor<float> st(shape, 1, config, 0.0f);

    // Write elements
    for (std::size_t i = 0; i < 5; ++i) {
        st.set_element(i, static_cast<float>(i * 10));
    }

    // Read elements back
    assert(st.get_element(0) == 0.0f);
    assert(st.get_element(1) == 10.0f);
    assert(st.get_element(2) == 20.0f);
    assert(st.get_element(3) == 30.0f);
    assert(st.get_element(4) == 40.0f);

    // Test operator()
    assert(st(0) == 0.0f);
    assert(st(2) == 20.0f);
    assert(st(4) == 40.0f);

    // Mutable access
    st(1) = 100.0f;
    assert(st(1) == 100.0f);

    std::cout << "✓ StreamTensor element access tests passed\n" << std::endl;
}

void test_stream_2d_access() {
    std::cout << "=== Testing StreamTensor 2D Access ===" << std::endl;

    std::size_t shape[] = {3, 4};
    StreamConfig config;
    config.batch_size = 4;

    StreamTensor<float> st(shape, 2, config, 0.0f);

    // Fill with values
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            st.set_element(i * 4 + j, static_cast<float>(i * 10 + j));
        }
    }

    // Verify via operator()
    assert(st(0, 0) == 0.0f);
    assert(st(0, 3) == 3.0f);
    assert(st(1, 0) == 10.0f);
    assert(st(2, 3) == 23.0f);

    // Mutable access
    st(1, 2) = 99.0f;
    assert(st(1, 2) == 99.0f);

    std::cout << "StreamTensor 2D: " << st << std::endl;
    std::cout << "✓ StreamTensor 2D access tests passed\n" << std::endl;
}

void test_stream_batched_elementwise() {
    std::cout << "=== Testing StreamTensor Batched Element-wise Ops ===" << std::endl;

    std::size_t shape[] = {10};
    StreamConfig config;
    config.batch_size = 3;  // Small batch to test multiple iterations

    // Create tensors from flat arrays
    float data_a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    float data_b[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f, 90.0f, 100.0f};

    StreamTensor<float> a(data_a, 10, config);
    StreamTensor<float> b(data_b, 10, config);

    // Batched add
    StreamTensor<float>* sum = a.add(&b, config);
    std::cout << "A + B: " << *sum << std::endl;
    for (std::size_t i = 0; i < 10; ++i) {
        assert(sum->get_element(i) == data_a[i] + data_b[i]);
    }

    // Batched subtract
    StreamTensor<float>* diff = a.subtract(&b, config);
    for (std::size_t i = 0; i < 10; ++i) {
        assert(diff->get_element(i) == data_a[i] - data_b[i]);
    }

    // Batched multiply
    StreamTensor<float>* prod = a.multiply(&b, config);
    for (std::size_t i = 0; i < 10; ++i) {
        assert(prod->get_element(i) == data_a[i] * data_b[i]);
    }

    // Batched scalar add
    StreamTensor<float>* scalar_add = a.add_scalar(100.0f, config);
    for (std::size_t i = 0; i < 10; ++i) {
        assert(scalar_add->get_element(i) == data_a[i] + 100.0f);
    }

    // Batched scalar multiply
    StreamTensor<float>* scalar_mul = a.multiply_scalar(2.0f, config);
    for (std::size_t i = 0; i < 10; ++i) {
        assert(scalar_mul->get_element(i) == data_a[i] * 2.0f);
    }

    delete sum;
    delete diff;
    delete prod;
    delete scalar_add;
    delete scalar_mul;

    std::cout << "✓ StreamTensor batched element-wise tests passed\n" << std::endl;
}

void test_stream_batched_reductions() {
    std::cout << "=== Testing StreamTensor Batched Reductions ===" << std::endl;

    std::size_t shape[] = {100};
    StreamConfig config;
    config.batch_size = 7;  // Small batch to test accumulation

    StreamTensor<float> st(shape, 1, config, 0.0f);

    // Fill with 1.0, 2.0, ..., 100.0
    for (std::size_t i = 0; i < 100; ++i) {
        st.set_element(i, static_cast<float>(i + 1));
    }

    // Batched sum: 1 + 2 + ... + 100 = 5050
    float sum_result = st.batched_sum(config);
    std::cout << "Sum: " << sum_result << std::endl;
    assert(std::abs(sum_result - 5050.0f) < 1.0f);

    // Batched mean: 5050 / 100 = 50.5
    float mean_result = st.batched_mean(config);
    std::cout << "Mean: " << mean_result << std::endl;
    assert(std::abs(mean_result - 50.5f) < 0.1f);

    // Batched max
    float max_result = st.batched_max(config);
    std::cout << "Max: " << max_result << std::endl;
    assert(std::abs(max_result - 100.0f) < 0.1f);

    // Batched min
    float min_result = st.batched_min(config);
    std::cout << "Min: " << min_result << std::endl;
    assert(std::abs(min_result - 1.0f) < 0.1f);

    std::cout << "✓ StreamTensor batched reduction tests passed\n" << std::endl;
}

void test_stream_batched_dot() {
    std::cout << "=== Testing StreamTensor Batched Dot Product ===" << std::endl;

    std::size_t shape[] = {5};
    StreamConfig config;
    config.batch_size = 2;

    float data_a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float data_b[] = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    StreamTensor<float> a(data_a, 5, config);
    StreamTensor<float> b(data_b, 5, config);

    // Dot product: 1*2 + 2*3 + 3*4 + 4*5 + 5*6 = 2 + 6 + 12 + 20 + 30 = 70
    float dot_result = a.batched_dot(&b, config);
    std::cout << "Dot product: " << dot_result << std::endl;
    assert(std::abs(dot_result - 70.0f) < 0.1f);

    std::cout << "✓ StreamTensor batched dot product tests passed\n" << std::endl;
}

void test_stream_unary_ops() {
    std::cout << "=== Testing StreamTensor Unary Operations ===" << std::endl;

    std::size_t shape[] = {6};
    StreamConfig config;
    config.batch_size = 2;

    float data[] = {-3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    StreamTensor<float> st(data, 6, config);

    // Abs
    StreamTensor<float>* abs_result = st.abs(config);
    std::cout << "Abs: " << *abs_result << std::endl;
    assert(std::abs(abs_result->get_element(0) - 3.0f) < 0.1f);
    assert(std::abs(abs_result->get_element(3) - 0.0f) < 0.1f);
    assert(std::abs(abs_result->get_element(5) - 2.0f) < 0.1f);

    // Negate
    StreamTensor<float>* neg_result = st.negate(config);
    std::cout << "Negate: " << *neg_result << std::endl;
    assert(std::abs(neg_result->get_element(0) - 3.0f) < 0.1f);
    assert(std::abs(neg_result->get_element(5) - (-2.0f)) < 0.1f);

    delete abs_result;
    delete neg_result;

    std::cout << "✓ StreamTensor unary operation tests passed\n" << std::endl;
}

void test_stream_reshape() {
    std::cout << "=== Testing StreamTensor Reshape ===" << std::endl;

    std::size_t shape_1d[] = {12};
    StreamConfig config;
    config.batch_size = 3;

    StreamTensor<float> st(shape_1d, 1, config, 0.0f);
    for (std::size_t i = 0; i < 12; ++i) {
        st.set_element(i, static_cast<float>(i));
    }

    // Reshape to [3, 4]
    std::size_t new_shape[] = {3, 4};
    StreamTensor<float>* reshaped = st.reshape(new_shape, 2, config);

    std::cout << "Reshaped: " << *reshaped << std::endl;
    assert(reshaped->ndim() == 2);
    assert(reshaped->shape()[0] == 3);
    assert(reshaped->shape()[1] == 4);

    // Data should be identical, just different shape
    for (std::size_t i = 0; i < 12; ++i) {
        assert(reshaped->get_element(i) == static_cast<float>(i));
    }

    delete reshaped;

    std::cout << "✓ StreamTensor reshape tests passed\n" << std::endl;
}

void test_stream_transpose() {
    std::cout << "=== Testing StreamTensor Transpose ===" << std::endl;

    std::size_t shape[] = {2, 3};
    StreamConfig config;
    config.batch_size = 2;

    StreamTensor<float> st(shape, 2, config, 0.0f);

    // Fill: [[0, 1, 2], [3, 4, 5]]
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            st.set_element(i * 3 + j, static_cast<float>(i * 3 + j));
        }
    }

    StreamTensor<float>* transposed = st.transpose(config);
    std::cout << "Original: " << st << std::endl;
    std::cout << "Transposed: " << *transposed << std::endl;

    assert(transposed->shape()[0] == 3);
    assert(transposed->shape()[1] == 2);

    // Verify transpose: [i,j] -> [j,i]
    assert(std::abs(transposed->operator()(0, 0) - 0.0f) < 0.1f);
    assert(std::abs(transposed->operator()(0, 1) - 3.0f) < 0.1f);
    assert(std::abs(transposed->operator()(2, 1) - 5.0f) < 0.1f);

    delete transposed;

    std::cout << "✓ StreamTensor transpose tests passed\n" << std::endl;
}

void test_stream_clamp() {
    std::cout << "=== Testing StreamTensor Clamp ===" << std::endl;

    std::size_t shape[] = {7};
    StreamConfig config;
    config.batch_size = 2;

    float data[] = {-5.0f, -2.0f, 0.0f, 1.0f, 3.0f, 7.0f, 10.0f};
    StreamTensor<float> st(data, 7, config);

    StreamTensor<float>* clamped = st.clamp(0.0f, 5.0f, config);
    std::cout << "Clamped [0, 5]: " << *clamped << std::endl;

    assert(std::abs(clamped->get_element(0) - 0.0f) < 0.1f);  // -5 -> 0
    assert(std::abs(clamped->get_element(1) - 0.0f) < 0.1f);  // -2 -> 0
    assert(std::abs(clamped->get_element(2) - 0.0f) < 0.1f);  // 0 -> 0
    assert(std::abs(clamped->get_element(3) - 1.0f) < 0.1f);  // 1 -> 1
    assert(std::abs(clamped->get_element(4) - 3.0f) < 0.1f);  // 3 -> 3
    assert(std::abs(clamped->get_element(5) - 5.0f) < 0.1f);  // 7 -> 5
    assert(std::abs(clamped->get_element(6) - 5.0f) < 0.1f);  // 10 -> 5

    delete clamped;

    std::cout << "✓ StreamTensor clamp tests passed\n" << std::endl;
}

void test_stream_comparison() {
    std::cout << "=== Testing StreamTensor Comparison ===" << std::endl;

    std::size_t shape[] = {5};
    StreamConfig config;
    config.batch_size = 2;

    float data[] = {1.0f, 3.0f, 5.0f, 7.0f, 9.0f};
    StreamTensor<float> st(data, 5, config);

    StreamTensor<bool>* gt = st.greater_than(5.0f, config);
    std::cout << "Greater than 5: " << *gt << std::endl;
    assert(gt->get_element(0) == false);
    assert(gt->get_element(2) == false);
    assert(gt->get_element(3) == true);
    assert(gt->get_element(4) == true);

    StreamTensor<bool>* lt = st.less_than(5.0f, config);
    std::cout << "Less than 5: " << *lt << std::endl;
    assert(lt->get_element(0) == true);
    assert(lt->get_element(2) == false);
    assert(lt->get_element(4) == false);

    delete gt;
    delete lt;

    std::cout << "✓ StreamTensor comparison tests passed\n" << std::endl;
}

void test_stream_copy_move() {
    std::cout << "=== Testing StreamTensor Copy/Move ===" << std::endl;

    std::size_t shape[] = {5};
    StreamConfig config;
    config.batch_size = 2;

    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    StreamTensor<float> original(data, 5, config);

    // Copy constructor
    StreamTensor<float> copied(original);
    for (std::size_t i = 0; i < 5; ++i) {
        assert(copied.get_element(i) == original.get_element(i));
    }

    // Modify original - copy should be independent
    original.set_element(0, 100.0f);
    assert(copied.get_element(0) == 1.0f);

    // Move constructor
    StreamTensor<float> moved(std::move(original));
    assert(moved.get_element(0) == 100.0f);

    // Move assignment
    StreamTensor<float> assigned = std::move(moved);
    assert(assigned.get_element(0) == 100.0f);

    std::cout << "✓ StreamTensor copy/move tests passed\n" << std::endl;
}

void test_stream_large_tensor() {
    std::cout << "=== Testing StreamTensor Large Tensor (Simulated Out-of-Core) ===" << std::endl;

    // Create a tensor with 1 million elements using small batch size
    // This simulates processing data larger than RAM
    std::size_t shape[] = {1000000};
    StreamConfig config;
    config.batch_size = 1000;  // Very small batch relative to tensor size
    config.max_memory_bytes = 1024 * 1024;  // 1MB limit

    std::cout << "Creating 1M element StreamTensor with batch_size=1000..." << std::endl;
    StreamTensor<float> st(shape, 1, config, 0.0f);

    // Fill with values in batches
    std::cout << "Filling with values..." << std::endl;
    for (std::size_t i = 0; i < 1000000; ++i) {
        st.set_element(i, static_cast<float>(i));
    }

    // Verify some elements
    assert(std::abs(st.get_element(0) - 0.0f) < 0.1f);
    assert(std::abs(st.get_element(500000) - 500000.0f) < 0.1f);
    assert(std::abs(st.get_element(999999) - 999999.0f) < 0.1f);

    // Batched sum: 0 + 1 + ... + 999999 = 999999 * 1000000 / 2 = 499999500000
    std::cout << "Computing batched sum..." << std::endl;
    float sum_result = st.batched_sum(config);
    std::cout << "Sum: " << sum_result << std::endl;
    // float only has ~7 significant digits, so large sums lose precision.
    // 499999500000 has 12 digits, so we allow ~1% relative error.
    assert(std::abs(sum_result - 499999500000.0f) / 499999500000.0f < 0.01f);

    // Batched mean
    std::cout << "Computing batched mean..." << std::endl;
    float mean_result = st.batched_mean(config);
    std::cout << "Mean: " << mean_result << std::endl;
    assert(std::abs(mean_result - 499999.5f) < 1.0f);

    std::cout << "✓ StreamTensor large tensor tests passed\n" << std::endl;
}

void test_stream_free_functions() {
    std::cout << "=== Testing StreamTensor Free Function Operators ===" << std::endl;

    std::size_t shape[] = {5};
    StreamConfig config;
    config.batch_size = 2;

    float data_a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float data_b[] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

    StreamTensor<float> a(data_a, 5, config);
    StreamTensor<float> b(data_b, 5, config);

    // Free function operators
    StreamTensor<float>* sum = stream_add(&a, &b, config);
    assert(sum->get_element(0) == 6.0f);
    assert(sum->get_element(4) == 6.0f);

    StreamTensor<float>* diff = stream_subtract(&a, &b, config);
    assert(diff->get_element(0) == -4.0f);
    assert(diff->get_element(4) == 4.0f);

    StreamTensor<float>* prod = stream_multiply(&a, &b, config);
    assert(prod->get_element(0) == 5.0f);
    assert(prod->get_element(2) == 9.0f);

    delete sum;
    delete diff;
    delete prod;

    std::cout << "✓ StreamTensor free function tests passed\n" << std::endl;
}

void test_stream_to_array() {
    std::cout << "=== Testing StreamTensor to Flat Array ===" << std::endl;

    std::size_t shape[] = {5};
    StreamConfig config;
    config.batch_size = 2;

    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    StreamTensor<float> st(data, 5, config);

    // Export to flat array
    float* array = st.to_flat_array();
    for (std::size_t i = 0; i < 5; ++i) {
        assert(array[i] == data[i]);
    }
    delete[] array;

    std::cout << "✓ StreamTensor to array tests passed\n" << std::endl;
}

void test_stream_config_tuning() {
    std::cout << "=== Testing StreamConfig Tuning ===" << std::endl;

    StreamConfig config;
    config.batch_size = 1000000;
    config.max_memory_bytes = 256 * 1024 * 1024;  // 256MB

    // For float (4 bytes): effective batch = min(1M, 256MB/4) = 1M
    std::size_t eff_batch = config.effective_batch_size(sizeof(float));
    std::cout << "Effective batch for float: " << eff_batch << std::endl;
    assert(eff_batch == 1000000);

    // For double (8 bytes): effective batch = min(1M, 256MB/8) = 1M
    eff_batch = config.effective_batch_size(sizeof(double));
    std::cout << "Effective batch for double: " << eff_batch << std::endl;
    assert(eff_batch == 1000000);

    // Tight memory budget
    StreamConfig tight_config;
    tight_config.batch_size = 1000000;
    tight_config.max_memory_bytes = 4096;  // Only 4KB

    eff_batch = tight_config.effective_batch_size(sizeof(float));
    std::cout << "Effective batch with 4KB budget (float): " << eff_batch << std::endl;
    assert(eff_batch == 1024);  // 4096 / 4 = 1024

    std::cout << "✓ StreamConfig tuning tests passed\n" << std::endl;
}

void test_stream_sync() {
    std::cout << "=== Testing StreamTensor Sync ===" << std::endl;

    std::size_t shape[] = {10};
    StreamConfig config;
    config.batch_size = 3;

    StreamTensor<float> st(shape, 1, config, 42.0f);

    // Sync to ensure data is flushed to disk
    st.sync();
    std::cout << "Sync completed" << std::endl;

    // Verify data is still correct after sync
    for (std::size_t i = 0; i < 10; ++i) {
        assert(st.get_element(i) == 42.0f);
    }

    std::cout << "✓ StreamTensor sync tests passed\n" << std::endl;
}

int main() {
    std::cout << "\n================================================" << std::endl;
    std::cout << "  TinyTorch StreamTensor Test Suite" << std::endl;
    std::cout << "  (Memory-Mapped + Batched Streaming)" << std::endl;
    std::cout << "  Features: mmap, batch swap in/out, out-of-core" << std::endl;
    std::cout << "================================================\n" << std::endl;

    test_stream_tensor_creation();
    test_stream_element_access();
    test_stream_2d_access();
    test_stream_batched_elementwise();
    test_stream_batched_reductions();
    test_stream_batched_dot();
    test_stream_unary_ops();
    test_stream_reshape();
    test_stream_transpose();
    test_stream_clamp();
    test_stream_comparison();
    test_stream_copy_move();
    test_stream_large_tensor();
    test_stream_free_functions();
    test_stream_to_array();
    test_stream_config_tuning();
    test_stream_sync();

    std::cout << "================================================" << std::endl;
    std::cout << "  All StreamTensor Tests Passed! ✓" << std::endl;
    std::cout << "================================================\n" << std::endl;

    return 0;
}
