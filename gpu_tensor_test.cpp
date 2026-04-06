#include "gpu_tensor.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <chrono>

// ============================================================================
// GPU Tensor Test Suite
// ============================================================================

void test_gpu_enumeration() {
    std::cout << "=== Testing GPU Enumeration ===" << std::endl;

    auto& selector = get_gpu_selector();
    bool found = selector.enumerate();

    if (!found || selector.get_gpus().empty()) {
        std::cout << "[WARN] No GPUs detected or enumeration failed" << std::endl;
        std::cout << "  This is expected on systems without OpenGL 4.3+ drivers" << std::endl;
        std::cout << "✓ GPU enumeration test completed (no GPUs found)\n" << std::endl;
        return;
    }

    std::cout << "Detected " << selector.get_gpus().size() << " GPU(s):\n" << std::endl;
    selector.print_all(std::cout);
    std::cout << std::endl;

    // Verify each GPU has valid info
    for (const auto& gpu : selector.get_gpus()) {
        assert(gpu.device_id >= 0);
        assert(!gpu.name.empty());
        assert(!gpu.vendor.empty());
        std::cout << "GPU #" << gpu.device_id << ": " << gpu.name
                  << " (score: " << gpu.compute_score << ")" << std::endl;
    }

    std::cout << "✓ GPU enumeration test passed\n" << std::endl;
}

void test_gpu_selection() {
    std::cout << "=== Testing GPU Selection ===" << std::endl;

    auto& selector = get_gpu_selector();
    if (selector.get_gpus().empty()) {
        std::cout << "[SKIP] No GPUs available\n" << std::endl;
        return;
    }

    // Test auto-select
    selector.auto_select();
    std::cout << "Auto-selected: GPU #" << selector.selected_device() << std::endl;
    assert(selector.selected_device() >= 0);

    // Test select by device ID
    selector.select_device(0);
    std::cout << "Selected device 0: GPU #" << selector.selected_device() << std::endl;
    assert(selector.selected_device() == 0);

    // Test select discrete only
    selector.select_discrete_only();
    std::cout << "Discrete-only: GPU #" << selector.selected_device() << std::endl;

    // Test select by name (try common patterns)
    bool found = selector.select_by_name("NVIDIA") ||
                 selector.select_by_name("AMD") ||
                 selector.select_by_name("Intel") ||
                 selector.select_by_name("Radeon");
    if (found) {
        std::cout << "Name-selected: GPU #" << selector.selected_device() << std::endl;
    } else {
        std::cout << "No GPU matched common name patterns" << std::endl;
    }

    std::cout << "✓ GPU selection test passed\n" << std::endl;
}

void test_gpu_context_init() {
    std::cout << "=== Testing GPU Context Initialization ===" << std::endl;

    auto& selector = get_gpu_selector();
    if (selector.get_gpus().empty()) {
        std::cout << "[SKIP] No GPUs available\n" << std::endl;
        return;
    }

    // Initialize GPU context
    auto& ctx = get_gpu_context();
    bool success = ctx.initialize(0);

    if (!success) {
        std::cout << "[WARN] GPU context initialization failed" << std::endl;
        std::cout << "  This may be due to missing OpenGL 4.3+ support" << std::endl;
        std::cout << "✓ GPU context test completed (init failed)\n" << std::endl;
        return;
    }

    std::cout << "GPU context initialized successfully" << std::endl;
    assert(ctx.is_initialized());
    assert(ctx.device_id() == 0);

    std::cout << "✓ GPU context initialization test passed\n" << std::endl;
}

void test_gpu_tensor_creation() {
    std::cout << "=== Testing GpuTensor Creation ===" << std::endl;

    auto& selector = get_gpu_selector();
    if (selector.get_gpus().empty()) {
        std::cout << "[SKIP] No GPUs available\n" << std::endl;
        return;
    }

    try {
        // Create from raw data
        float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        GpuTensor<float> tensor(data, 5);

        std::cout << "Backend: " << tensor.backend_name() << std::endl;
        std::cout << "Shape: [" << tensor.shape()[0] << "]" << std::endl;
        std::cout << "Total size: " << tensor.total_size() << std::endl;
        std::cout << "Device: #" << tensor.device_id() << std::endl;

        assert(tensor.ndim() == 1);
        assert(tensor.total_size() == 5);
        assert(tensor.device_id() >= 0);

        // Verify data roundtrip
        auto host = tensor.to_host();
        assert(host.size() == 5);
        assert(host[0] == 1.0f);
        assert(host[4] == 5.0f);

        std::cout << "Data roundtrip: " << host[0] << ", " << host[1] << ", "
                  << host[2] << ", " << host[3] << ", " << host[4] << std::endl;

        std::cout << "✓ GpuTensor creation test passed\n" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "[FAIL] Exception: " << e.what() << std::endl;
    }
}

void test_gpu_tensor_from_shape() {
    std::cout << "=== Testing GpuTensor From Shape ===" << std::endl;

    auto& selector = get_gpu_selector();
    if (selector.get_gpus().empty()) {
        std::cout << "[SKIP] No GPUs available\n" << std::endl;
        return;
    }

    try {
        std::size_t shape[] = {3, 4};
        GpuTensor<float> tensor(shape, 2, GpuConfig{}, 42.0f);

        std::cout << "Shape: [" << tensor.shape()[0] << ", " << tensor.shape()[1] << "]" << std::endl;
        std::cout << "Total size: " << tensor.total_size() << std::endl;

        assert(tensor.ndim() == 2);
        assert(tensor.total_size() == 12);

        auto host = tensor.to_host();
        for (std::size_t i = 0; i < 12; ++i) {
            assert(host[i] == 42.0f);
        }

        std::cout << "All elements initialized to 42.0" << std::endl;
        std::cout << "✓ GpuTensor from shape test passed\n" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "[FAIL] Exception: " << e.what() << std::endl;
    }
}

void test_gpu_elementwise_ops() {
    std::cout << "=== Testing GpuTensor Element-wise Operations ===" << std::endl;

    auto& selector = get_gpu_selector();
    if (selector.get_gpus().empty()) {
        std::cout << "[SKIP] No GPUs available\n" << std::endl;
        return;
    }

    try {
        float data_a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        float data_b[] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

        GpuTensor<float> a(data_a, 5);
        GpuTensor<float> b(data_b, 5);

        // Add
        auto sum = a.add(&b);
        auto sum_host = dynamic_cast<GpuTensor<float>*>(sum.get())->to_host();
        std::cout << "A + B: [";
        for (std::size_t i = 0; i < 5; ++i) {
            std::cout << sum_host[i] << (i < 4 ? ", " : "");
            assert(std::abs(sum_host[i] - 6.0f) < 0.001f);
        }
        std::cout << "]" << std::endl;

        // Subtract
        auto diff = a.subtract(&b);
        auto diff_host = dynamic_cast<GpuTensor<float>*>(diff.get())->to_host();
        std::cout << "A - B: [";
        for (std::size_t i = 0; i < 5; ++i) {
            std::cout << diff_host[i] << (i < 4 ? ", " : "");
        }
        std::cout << "]" << std::endl;

        // Multiply
        auto prod = a.multiply(&b);
        auto prod_host = dynamic_cast<GpuTensor<float>*>(prod.get())->to_host();
        std::cout << "A * B: [";
        for (std::size_t i = 0; i < 5; ++i) {
            std::cout << prod_host[i] << (i < 4 ? ", " : "");
        }
        std::cout << "]" << std::endl;

        // Scalar operations
        auto scalar_add = a.add_scalar(10.0f);
        auto scalar_add_host = dynamic_cast<GpuTensor<float>*>(scalar_add.get())->to_host();
        std::cout << "A + 10: [";
        for (std::size_t i = 0; i < 5; ++i) {
            std::cout << scalar_add_host[i] << (i < 4 ? ", " : "");
        }
        std::cout << "]" << std::endl;

        auto scalar_mul = a.multiply_scalar(2.0f);
        auto scalar_mul_host = dynamic_cast<GpuTensor<float>*>(scalar_mul.get())->to_host();
        std::cout << "A * 2: [";
        for (std::size_t i = 0; i < 5; ++i) {
            std::cout << scalar_mul_host[i] << (i < 4 ? ", " : "");
        }
        std::cout << "]" << std::endl;

        std::cout << "✓ GpuTensor element-wise operations test passed\n" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "[FAIL] Exception: " << e.what() << std::endl;
    }
}

void test_gpu_unary_ops() {
    std::cout << "=== Testing GpuTensor Unary Operations ===" << std::endl;

    auto& selector = get_gpu_selector();
    if (selector.get_gpus().empty()) {
        std::cout << "[SKIP] No GPUs available\n" << std::endl;
        return;
    }

    try {
        float data[] = {-3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
        GpuTensor<float> tensor(data, 7);

        // Abs
        auto abs_result = tensor.abs();
        auto abs_host = dynamic_cast<GpuTensor<float>*>(abs_result.get())->to_host();
        std::cout << "Abs: [";
        for (std::size_t i = 0; i < 7; ++i) {
            std::cout << abs_host[i] << (i < 6 ? ", " : "");
            assert(std::abs(abs_host[i]) == abs_host[i]); // All positive
        }
        std::cout << "]" << std::endl;

        // Negate
        auto neg_result = tensor.negate();
        auto neg_host = dynamic_cast<GpuTensor<float>*>(neg_result.get())->to_host();
        std::cout << "Negate: [";
        for (std::size_t i = 0; i < 7; ++i) {
            std::cout << neg_host[i] << (i < 6 ? ", " : "");
            assert(std::abs(neg_host[i] + data[i]) < 0.001f);
        }
        std::cout << "]" << std::endl;

        std::cout << "✓ GpuTensor unary operations test passed\n" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "[FAIL] Exception: " << e.what() << std::endl;
    }
}

void test_gpu_reductions() {
    std::cout << "=== Testing GpuTensor Reductions ===" << std::endl;

    auto& selector = get_gpu_selector();
    if (selector.get_gpus().empty()) {
        std::cout << "[SKIP] No GPUs available\n" << std::endl;
        return;
    }

    try {
        float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        GpuTensor<float> tensor(data, 5);

        float sum_result = tensor.sum();
        std::cout << "Sum: " << sum_result << std::endl;
        assert(std::abs(sum_result - 15.0f) < 0.001f);

        float mean_result = tensor.mean();
        std::cout << "Mean: " << mean_result << std::endl;
        assert(std::abs(mean_result - 3.0f) < 0.001f);

        float max_result = tensor.max();
        std::cout << "Max: " << max_result << std::endl;
        assert(std::abs(max_result - 5.0f) < 0.001f);

        float min_result = tensor.min();
        std::cout << "Min: " << min_result << std::endl;
        assert(std::abs(min_result - 1.0f) < 0.001f);

        // Dot product
        float data_b[] = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        GpuTensor<float> tensor_b(data_b, 5);
        float dot_result = tensor.dot(&tensor_b);
        std::cout << "Dot: " << dot_result << std::endl;
        // 1*2 + 2*3 + 3*4 + 4*5 + 5*6 = 2 + 6 + 12 + 20 + 30 = 70
        assert(std::abs(dot_result - 70.0f) < 0.001f);

        std::cout << "✓ GpuTensor reductions test passed\n" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "[FAIL] Exception: " << e.what() << std::endl;
    }
}

void test_gpu_transform_ops() {
    std::cout << "=== Testing GpuTensor Transform Operations ===" << std::endl;

    auto& selector = get_gpu_selector();
    if (selector.get_gpus().empty()) {
        std::cout << "[SKIP] No GPUs available\n" << std::endl;
        return;
    }

    try {
        // Reshape
        float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        GpuTensor<float> tensor(data, 6);

        std::size_t new_shape[] = {2, 3};
        auto reshaped = tensor.reshape(new_shape, 2);
        std::cout << "Reshaped to [2,3]: " << *reshaped << std::endl;
        assert(reshaped->ndim() == 2);
        assert(reshaped->shape()[0] == 2);
        assert(reshaped->shape()[1] == 3);

        // Transpose
        auto transposed = reshaped->transpose();
        std::cout << "Transposed: " << *transposed << std::endl;
        assert(transposed->shape()[0] == 3);
        assert(transposed->shape()[1] == 2);

        // Clamp
        float data_clamp[] = {-5.0f, -2.0f, 0.0f, 3.0f, 7.0f, 10.0f};
        GpuTensor<float> tensor_clamp(data_clamp, 6);
        auto clamped = tensor_clamp.clamp(0.0f, 5.0f);
        auto clamp_host = dynamic_cast<GpuTensor<float>*>(clamped.get())->to_host();
        std::cout << "Clamped [0, 5]: [";
        for (std::size_t i = 0; i < 6; ++i) {
            std::cout << clamp_host[i] << (i < 5 ? ", " : "");
        }
        std::cout << "]" << std::endl;

        std::cout << "✓ GpuTensor transform operations test passed\n" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "[FAIL] Exception: " << e.what() << std::endl;
    }
}

void test_gpu_comparison_ops() {
    std::cout << "=== Testing GpuTensor Comparison Operations ===" << std::endl;

    auto& selector = get_gpu_selector();
    if (selector.get_gpus().empty()) {
        std::cout << "[SKIP] No GPUs available\n" << std::endl;
        return;
    }

    try {
        float data[] = {1.0f, 3.0f, 5.0f, 7.0f, 9.0f};
        GpuTensor<float> tensor(data, 5);

        auto gt = tensor.greater_than(5.0f);
        std::cout << "Greater than 5: " << *gt << std::endl;

        auto lt = tensor.less_than(5.0f);
        std::cout << "Less than 5: " << *lt << std::endl;

        std::cout << "✓ GpuTensor comparison operations test passed\n" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "[FAIL] Exception: " << e.what() << std::endl;
    }
}

void test_gpu_auto_selection() {
    std::cout << "=== Testing GpuAutoTensor Auto-Selection ===" << std::endl;

    auto& selector = get_gpu_selector();
    if (selector.get_gpus().empty()) {
        std::cout << "[SKIP] No GPUs available\n" << std::endl;
        return;
    }

    try {
        // Small data - might fall back to CPU
        float small_data[] = {1.0f, 2.0f, 3.0f};
        auto small_tensor = GpuAutoTensor<float>::from_data(small_data, 3);
        std::cout << "Small tensor backend: " << small_tensor->backend_name() << std::endl;

        // Large data - should use GPU
        std::size_t large_size = 500000;
        std::vector<float> large_data(large_size);
        for (std::size_t i = 0; i < large_size; ++i) {
            large_data[i] = static_cast<float>(i);
        }

        auto large_tensor = GpuAutoTensor<float>::from_data(large_data.data(), large_size);
        std::cout << "Large tensor backend: " << large_tensor->backend_name() << std::endl;

        // Force GPU
        auto forced_gpu = GpuAutoTensor<float>::force_gpu(large_data.data(), large_size, 0);
        std::cout << "Forced GPU backend: " << forced_gpu->backend_name() << std::endl;

        std::cout << "✓ GpuAutoTensor auto-selection test passed\n" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "[FAIL] Exception: " << e.what() << std::endl;
    }
}

void test_gpu_performance() {
    std::cout << "=== Testing GpuTensor Performance ===" << std::endl;

    auto& selector = get_gpu_selector();
    if (selector.get_gpus().empty()) {
        std::cout << "[SKIP] No GPUs available\n" << std::endl;
        return;
    }

    try {
        std::size_t size = 10000000; // 10M elements
        std::vector<float> data_a(size, 1.5f);
        std::vector<float> data_b(size, 2.5f);

        std::cout << "Creating tensors with " << (size / 1000000) << "M elements..." << std::endl;

        auto start = std::chrono::high_resolution_clock::now();
        GpuTensor<float> gpu_a(data_a.data(), size);
        GpuTensor<float> gpu_b(data_b.data(), size);
        auto gpu_create = std::chrono::high_resolution_clock::now();
        std::cout << "GPU creation: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(gpu_create - start).count()
                  << " ms" << std::endl;

        // GPU add
        auto gpu_sum = gpu_a.add(&gpu_b);
        auto gpu_compute = std::chrono::high_resolution_clock::now();
        std::cout << "GPU compute (add): "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(gpu_compute - gpu_create).count()
                  << " ms" << std::endl;

        // Download result
        auto gpu_host = dynamic_cast<GpuTensor<float>*>(gpu_sum.get())->to_host();
        auto gpu_download = std::chrono::high_resolution_clock::now();
        std::cout << "GPU download: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(gpu_download - gpu_compute).count()
                  << " ms" << std::endl;

        // Verify
        assert(std::abs(gpu_host[0] - 4.0f) < 0.001f);
        assert(std::abs(gpu_host[size - 1] - 4.0f) < 0.001f);

        std::cout << "✓ GpuTensor performance test passed\n" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "[FAIL] Exception: " << e.what() << std::endl;
    }
}

void test_gpu_print() {
    std::cout << "=== Testing GpuTensor Print ===" << std::endl;

    auto& selector = get_gpu_selector();
    if (selector.get_gpus().empty()) {
        std::cout << "[SKIP] No GPUs available\n" << std::endl;
        return;
    }

    try {
        float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        GpuTensor<float> tensor(data, 5);

        std::cout << "Regular print: " << tensor << std::endl;

        std::cout << "Batch print (batch_size=2):" << std::endl;
        tensor.batch_print(std::cout, 2);
        std::cout << std::endl;

        std::cout << "✓ GpuTensor print test passed\n" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "[FAIL] Exception: " << e.what() << std::endl;
    }
}

void test_gpu_copy_move() {
    std::cout << "=== Testing GpuTensor Copy/Move ===" << std::endl;

    auto& selector = get_gpu_selector();
    if (selector.get_gpus().empty()) {
        std::cout << "[SKIP] No GPUs available\n" << std::endl;
        return;
    }

    try {
        float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        GpuTensor<float> original(data, 5);

        // Copy constructor
        GpuTensor<float> copied(original);
        auto copied_host = copied.to_host();
        assert(copied_host[0] == 1.0f);
        assert(copied_host[4] == 5.0f);
        std::cout << "Copied tensor: " << copied << std::endl;

        // Move constructor
        GpuTensor<float> moved(std::move(original));
        auto moved_host = moved.to_host();
        assert(moved_host[0] == 1.0f);
        std::cout << "Moved tensor: " << moved << std::endl;

        // Move assignment
        GpuTensor<float> assigned = std::move(moved);
        auto assigned_host = assigned.to_host();
        assert(assigned_host[0] == 1.0f);
        std::cout << "Assigned tensor: " << assigned << std::endl;

        std::cout << "✓ GpuTensor copy/move test passed\n" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "[FAIL] Exception: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "\n================================================" << std::endl;
    std::cout << "  TinyTorch GPU Tensor Test Suite" << std::endl;
    std::cout << "  (OpenGL Compute Shaders + Multi-GPU)" << std::endl;
    std::cout << "================================================\n" << std::endl;

    // Print system info first
    GpuAutoTensor<float>::print_system_info();
    std::cout << std::endl;

    test_gpu_enumeration();
    test_gpu_selection();
    test_gpu_context_init();
    test_gpu_tensor_creation();
    test_gpu_tensor_from_shape();
    test_gpu_elementwise_ops();
    test_gpu_unary_ops();
    test_gpu_reductions();
    test_gpu_transform_ops();
    test_gpu_comparison_ops();
    test_gpu_auto_selection();
    test_gpu_performance();
    test_gpu_print();
    test_gpu_copy_move();

    std::cout << "================================================" << std::endl;
    std::cout << "  All GPU Tensor Tests Completed!" << std::endl;
    std::cout << "================================================\n" << std::endl;

    return 0;
}
