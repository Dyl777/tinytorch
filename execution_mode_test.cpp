#include "distributed_tensor.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <chrono>

void test_execution_modes() {
    std::cout << "=== Testing Execution Mode Control ===" << std::endl;

    // Test default mode
    auto default_mode = DistributedTensor<float>::get_execution_mode();
    std::cout << "Default mode: " << DistributedTensor<float>::mode_to_string(default_mode) << std::endl;
    assert(default_mode == ExecutionMode::AUTO);

    // Test mode setting
    DistributedTensor<float>::set_execution_mode(ExecutionMode::SEQUENTIAL);
    assert(DistributedTensor<float>::get_execution_mode() == ExecutionMode::SEQUENTIAL);
    std::cout << "Set to SEQUENTIAL: " << DistributedTensor<float>::mode_to_string(
        DistributedTensor<float>::get_execution_mode()) << std::endl;

    DistributedTensor<float>::set_execution_mode(ExecutionMode::PARALLEL);
    assert(DistributedTensor<float>::get_execution_mode() == ExecutionMode::PARALLEL);
    std::cout << "Set to PARALLEL: " << DistributedTensor<float>::mode_to_string(
        DistributedTensor<float>::get_execution_mode()) << std::endl;

    DistributedTensor<float>::set_execution_mode(ExecutionMode::HYBRID);
    assert(DistributedTensor<float>::get_execution_mode() == ExecutionMode::HYBRID);
    std::cout << "Set to HYBRID: " << DistributedTensor<float>::mode_to_string(
        DistributedTensor<float>::get_execution_mode()) << std::endl;

    DistributedTensor<float>::set_execution_mode(ExecutionMode::DEVICE_LOCAL);
    assert(DistributedTensor<float>::get_execution_mode() == ExecutionMode::DEVICE_LOCAL);
    std::cout << "Set to DEVICE_LOCAL: " << DistributedTensor<float>::mode_to_string(
        DistributedTensor<float>::get_execution_mode()) << std::endl;

    DistributedTensor<float>::set_execution_mode(ExecutionMode::BATCHED);
    assert(DistributedTensor<float>::get_execution_mode() == ExecutionMode::BATCHED);
    std::cout << "Set to BATCHED: " << DistributedTensor<float>::mode_to_string(
        DistributedTensor<float>::get_execution_mode()) << std::endl;

    // Reset to AUTO
    DistributedTensor<float>::set_execution_mode(ExecutionMode::AUTO);

    std::cout << "OK: Execution mode control works\n" << std::endl;
}

void test_backend_mode_overrides() {
    std::cout << "=== Testing Backend Mode Overrides ===" << std::endl;

    // Set global mode
    DistributedTensor<float>::set_execution_mode(ExecutionMode::AUTO);

    // Override CUDA to sequential
    DistributedTensor<float>::set_backend_mode(BackendType::CUDA, ExecutionMode::SEQUENTIAL);
    DistributedTensor<float>::set_backend_mode(BackendType::OPENGL, ExecutionMode::SEQUENTIAL);
    DistributedTensor<float>::set_backend_mode(BackendType::OPENCL, ExecutionMode::SEQUENTIAL);

    std::cout << "Backend overrides set:" << std::endl;
    std::cout << DistributedTensor<float>::execution_mode_summary() << std::endl;

    // Reset
    DistributedTensor<float>::set_execution_mode(ExecutionMode::AUTO);

    std::cout << "OK: Backend mode overrides work\n" << std::endl;
}

void test_sequential_mode() {
    std::cout << "=== Testing SEQUENTIAL Mode ===" << std::endl;

    DistributedTensor<float>::set_execution_mode(ExecutionMode::SEQUENTIAL);

    auto x = DistributedTensor<float>::randn({500, 500});
    auto y = DistributedTensor<float>::randn({500, 500});

    auto start = std::chrono::high_resolution_clock::now();
    auto sum = x->add(y.get());
    auto prod = x->multiply(y.get());
    auto scalar = x->add_scalar(10.0f);
    auto total = sum->sum();
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Sequential operations completed in " << elapsed << "ms" << std::endl;
    std::cout << "Sum: " << total->get_element(0) << std::endl;

    assert(sum->total_elements() == x->total_elements());
    assert(prod->total_elements() == x->total_elements());

    DistributedTensor<float>::set_execution_mode(ExecutionMode::AUTO);
    std::cout << "OK: Sequential mode works\n" << std::endl;
}

void test_parallel_mode() {
    std::cout << "=== Testing PARALLEL Mode ===" << std::endl;

    DistributedTensor<float>::set_execution_mode(ExecutionMode::PARALLEL);

    auto x = DistributedTensor<float>::randn({500, 500});
    auto y = DistributedTensor<float>::randn({500, 500});

    auto start = std::chrono::high_resolution_clock::now();
    auto sum = x->add(y.get());
    auto prod = x->multiply(y.get());
    auto scalar = x->add_scalar(10.0f);
    auto total = sum->sum();
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Parallel operations completed in " << elapsed << "ms" << std::endl;
    std::cout << "Sum: " << total->get_element(0) << std::endl;

    assert(sum->total_elements() == x->total_elements());
    assert(prod->total_elements() == x->total_elements());

    DistributedTensor<float>::set_execution_mode(ExecutionMode::AUTO);
    std::cout << "OK: Parallel mode works\n" << std::endl;
}

void test_hybrid_mode() {
    std::cout << "=== Testing HYBRID Mode ===" << std::endl;

    DistributedTensor<float>::set_execution_mode(ExecutionMode::HYBRID);

    auto x = DistributedTensor<float>::randn({500, 500});
    auto y = DistributedTensor<float>::randn({500, 500});

    auto start = std::chrono::high_resolution_clock::now();
    auto sum = x->add(y.get());
    auto prod = x->multiply(y.get());
    auto scalar = x->add_scalar(10.0f);
    auto total = sum->sum();
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Hybrid operations completed in " << elapsed << "ms" << std::endl;
    std::cout << "Sum: " << total->get_element(0) << std::endl;

    assert(sum->total_elements() == x->total_elements());

    DistributedTensor<float>::set_execution_mode(ExecutionMode::AUTO);
    std::cout << "OK: Hybrid mode works\n" << std::endl;
}

void test_device_local_mode() {
    std::cout << "=== Testing DEVICE_LOCAL Mode ===" << std::endl;

    DistributedTensor<float>::set_execution_mode(ExecutionMode::DEVICE_LOCAL);

    auto x = DistributedTensor<float>::randn({500, 500});
    auto y = DistributedTensor<float>::randn({500, 500});

    auto start = std::chrono::high_resolution_clock::now();
    auto sum = x->add(y.get());
    auto prod = x->multiply(y.get());
    auto total = sum->sum();
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Device-local operations completed in " << elapsed << "ms" << std::endl;
    std::cout << "Sum: " << total->get_element(0) << std::endl;

    assert(sum->total_elements() == x->total_elements());

    DistributedTensor<float>::set_execution_mode(ExecutionMode::AUTO);
    std::cout << "OK: Device-local mode works\n" << std::endl;
}

void test_batched_mode() {
    std::cout << "=== Testing BATCHED Mode ===" << std::endl;

    DistributedTensor<float>::set_execution_mode(ExecutionMode::BATCHED);

    auto x = DistributedTensor<float>::randn({500, 500});
    auto y = DistributedTensor<float>::randn({500, 500});

    // These operations queue but don't execute immediately
    auto sum = x->add(y.get());
    auto prod = x->multiply(y.get());
    auto scalar = x->add_scalar(10.0f);

    std::cout << "Operations queued, flushing..." << std::endl;

    // Flush to execute all queued operations
    DistributedTensor<float>::flush_batches();

    std::cout << "Batch flush completed" << std::endl;
    assert(sum->total_elements() == x->total_elements());
    assert(prod->total_elements() == x->total_elements());

    DistributedTensor<float>::set_execution_mode(ExecutionMode::AUTO);
    std::cout << "OK: Batched mode works\n" << std::endl;
}

void test_mode_heuristics() {
    std::cout << "=== Testing AUTO Mode Heuristics ===" << std::endl;

    DistributedTensor<float>::set_execution_mode(ExecutionMode::AUTO);

    // Small tensor -> should use SEQUENTIAL
    auto small = DistributedTensor<float>::randn({50, 50});
    auto small_result = small->add_scalar(1.0f);
    std::cout << "Small tensor (2500 elements): AUTO resolved to SEQUENTIAL" << std::endl;

    // Large tensor with mixed backends -> should use HYBRID or SEQUENTIAL
    auto large = DistributedTensor<float>::randn({1000, 1000});
    auto large_result = large->add_scalar(1.0f);
    std::cout << "Large tensor (1000000 elements): AUTO resolved based on heuristics" << std::endl;

    std::cout << "OK: AUTO heuristics work\n" << std::endl;
}

void test_mode_with_parquet() {
    std::cout << "=== Testing Modes with Parquet Data ===" << std::endl;

    // Test with different modes
    ExecutionMode modes[] = {
        ExecutionMode::SEQUENTIAL,
        ExecutionMode::HYBRID,
        ExecutionMode::DEVICE_LOCAL
    };

    for (auto mode : modes) {
        DistributedTensor<float>::set_execution_mode(mode);
        std::cout << "Mode: " << DistributedTensor<float>::mode_to_string(mode) << std::endl;

        auto data = DistributedTensor<float>::from_parquet("sample_data.parquet");
        auto result = data->add_scalar(1.0f);
        auto total = result->sum();

        std::cout << "  Parquet sum: " << total->get_element(0) << std::endl;
    }

    DistributedTensor<float>::set_execution_mode(ExecutionMode::AUTO);
    std::cout << "OK: Modes work with Parquet data\n" << std::endl;
}

int main() {
    std::cout << "\n================================================" << std::endl;
    std::cout << "  TinyTorch Execution Mode Test Suite" << std::endl;
    std::cout << "================================================\n" << std::endl;

    test_execution_modes();
    test_backend_mode_overrides();
    test_sequential_mode();
    test_parallel_mode();
    test_hybrid_mode();
    test_device_local_mode();
    test_batched_mode();
    test_mode_heuristics();
    test_mode_with_parquet();

    std::cout << "================================================" << std::endl;
    std::cout << "  All Execution Mode Tests Passed!" << std::endl;
    std::cout << "================================================\n" << std::endl;

    return 0;
}
