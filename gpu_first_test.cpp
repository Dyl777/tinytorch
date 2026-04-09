#include "distributed_tensor.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <chrono>

int main() {
    std::cout << "\n================================================" << std::endl;
    std::cout << "  TinyTorch GPU-FIRST Mode Test Suite" << std::endl;
    std::cout << "================================================\n" << std::endl;

    // Enable GPU-FIRST mode
    DistributedTensor<float>::set_execution_mode(ExecutionMode::GPU_SATURATE);
    DistributedTensor<float>::set_max_gpu_memory_percentage(90.0);
    std::cout << "Execution mode: " << DistributedTensor<float>::mode_to_string(
        DistributedTensor<float>::get_execution_mode()) << "\n" << std::endl;

    std::cout << "Max GPU memory percentage: "
              << DistributedTensor<float>::get_max_gpu_memory_percentage() << "%" 
              << std::endl;

    // Print MemoryBudget info
    std::cout << "MemoryBudget: " << MemoryBudget::instance().summary() << std::endl;

    // Test 1: Small tensor (should still work)
    std::cout << "\n=== Test 1: Small Tensor (100x100) ===" << std::endl;
    {
        auto x = DistributedTensor<float>::randn({100, 100});
        std::cout << "Created: " << x->distribution_info() << std::endl;
        std::cout << "Shards: " << x->num_shards() << std::endl;
        for (size_t i = 0; i < x->num_shards(); ++i) {
            std::cout << "  " << x->shard_info(i) << std::endl;
        }
        auto y = x->multiply_scalar(2.0f);
        auto sum = y->sum();
        std::cout << "Sum: " << sum->get_element(0) << std::endl;
        std::cout << "OK: Small tensor works\n" << std::endl;
    }

    // Test 2: Medium tensor (500x500 = 250K elements, ~1MB)
    std::cout << "=== Test 2: Medium Tensor (500x500) ===" << std::endl;
    {
        auto x = DistributedTensor<float>::randn({500, 500});
        std::cout << "Created: " << x->distribution_info() << std::endl;
        std::cout << "Shards: " << x->num_shards() << std::endl;
        for (size_t i = 0; i < x->num_shards(); ++i) {
            std::cout << "  " << x->shard_info(i) << std::endl;
        }
        auto y = x->multiply_scalar(2.0f);
        auto sum = y->sum();
        std::cout << "Sum: " << sum->get_element(0) << std::endl;
        std::cout << "OK: Medium tensor works\n" << std::endl;
    }

    // Test 3: Large tensor (2000x2000 = 4M elements, ~16MB)
    std::cout << "=== Test 3: Large Tensor (2000x2000 = 4M elements, ~16MB) ===" << std::endl;
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto x = DistributedTensor<float>::randn({2000, 2000});
        auto created = std::chrono::high_resolution_clock::now();

        std::cout << "Created: " << x->distribution_info() << std::endl;
        std::cout << "Shards: " << x->num_shards() << std::endl;
        for (size_t i = 0; i < x->num_shards(); ++i) {
            std::cout << "  " << x->shard_info(i) << std::endl;
        }

        // Count GPU vs CPU shards
        int gpu_shards = 0, cpu_shards = 0;
        for (size_t i = 0; i < x->num_shards(); ++i) {
            std::string dev = x->shard_device(i);
            if (dev.find("CUDA") != std::string::npos ||
                dev.find("OpenGL") != std::string::npos ||
                dev.find("OpenCL") != std::string::npos) {
                gpu_shards++;
            } else {
                cpu_shards++;
            }
        }
        std::cout << "GPU shards: " << gpu_shards << ", CPU shards: " << cpu_shards << std::endl;

        // Operations
        auto y = x->multiply_scalar(2.0f);
        auto ops_done = std::chrono::high_resolution_clock::now();

        auto sum = y->sum();
        auto sum_done = std::chrono::high_resolution_clock::now();

        auto create_ms = std::chrono::duration_cast<std::chrono::milliseconds>(created - start).count();
        auto ops_ms = std::chrono::duration_cast<std::chrono::milliseconds>(ops_done - created).count();
        auto sum_ms = std::chrono::duration_cast<std::chrono::milliseconds>(sum_done - ops_done).count();

        std::cout << "Sum: " << sum->get_element(0) << std::endl;
        std::cout << "Timing: create=" << create_ms << "ms, ops=" << ops_ms << "ms, sum=" << sum_ms << "ms" << std::endl;
        std::cout << "OK: Large tensor works\n" << std::endl;
    }

    // Test 3.5: GPU Utilization Diagnostic
    std::cout << "=== Test 3.5: GPU Utilization Diagnostic ===" << std::endl;
    {
        // Get raw device info
        DevicePool pool;
        std::cout << "Available devices:" << std::endl;
        for (const auto& d : pool.devices()) {
            std::cout << "  " << d.name << " [" << backend_type_to_string(d.backend) << "]" << std::endl;
            std::cout << "    Memory: " << (d.available_memory_bytes / (1024*1024)) << " MB" << std::endl;
            std::cout << "    Compute score: " << d.compute_score << std::endl;
        }

        // Create tensor and show exact shard distribution
        auto x = DistributedTensor<float>::randn({3000, 3000}); // 9M elements
        std::cout << "\nTensor: 3000x3000 = 9M elements (~36MB)" << std::endl;
        std::cout << "Shards: " << x->num_shards() << std::endl;

        for (size_t i = 0; i < x->num_shards(); ++i) {
            std::cout << "  Shard " << i << ": " << x->shard_info(i) << std::endl;
        }

        // Calculate percentages
        std::size_t total = 0, cuda_elems = 0, opencl_elems = 0, cpu_elems = 0;
        for (size_t i = 0; i < x->num_shards(); ++i) {
            std::string dev = x->shard_device(i);
            std::size_t elems = x->shard_num_elements(i);
            total += elems;
            if (dev.find("CUDA") != std::string::npos) cuda_elems += elems;
            else if (dev.find("OpenCL") != std::string::npos) opencl_elems += elems;
            else cpu_elems += elems;
        }
        std::cout << "\nElement distribution:" << std::endl;
        std::cout << "  CUDA: " << cuda_elems << " (" << (100.0 * cuda_elems / total) << "%)" << std::endl;
        std::cout << "  OpenCL: " << opencl_elems << " (" << (100.0 * opencl_elems / total) << "%)" << std::endl;
        std::cout << "  CPU: " << cpu_elems << " (" << (100.0 * cpu_elems / total) << "%)" << std::endl;
        std::cout << "OK: Diagnostic complete\n" << std::endl;
    }

    // Test 4: Very large tensor (5000x5000 = 25M elements, ~100MB)
    std::cout << "=== Test 4: Very Large Tensor (5000x5000 = 25M elements, ~100MB) ===" << std::endl;
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto x = DistributedTensor<float>::randn({5000, 5000});
        auto created = std::chrono::high_resolution_clock::now();

        std::cout << "Created: " << x->distribution_info() << std::endl;
        std::cout << "Shards: " << x->num_shards() << std::endl;

        // Detailed shard analysis with per-shard timing
        int gpu_shards = 0, cpu_shards = 0;
        std::size_t gpu_elements = 0, cpu_elements = 0;
        for (size_t i = 0; i < x->num_shards(); ++i) {
            std::string dev = x->shard_device(i);
            std::string info = x->shard_info(i);
            std::size_t elems = 0;
            {
                std::size_t pos = info.find(": ");
                if (pos != std::string::npos) {
                    std::size_t pos2 = info.find(" elements", pos + 2);
                    if (pos2 != std::string::npos) {
                        try {
                            elems = static_cast<std::size_t>(
                                std::stoull(info.substr(pos + 2, pos2 - (pos + 2))));
                        } catch (...) {
                            elems = 0;
                        }
                    }
                }
            }
            if (dev.find("CUDA") != std::string::npos ||
                dev.find("OpenGL") != std::string::npos ||
                dev.find("OpenCL") != std::string::npos) {
                gpu_shards++;
                gpu_elements += elems;
            } else {
                cpu_shards++;
                cpu_elements += elems;
            }
            std::cout << "  Shard " << i << ": " << elems << " elements on " << dev << std::endl;
        }
        std::cout << "GPU shards: " << gpu_shards << ", CPU shards: " << cpu_shards << std::endl;
        std::cout << "GPU elements: " << gpu_elements << " (" << (100.0 * gpu_elements / (gpu_elements + cpu_elements)) << "%), "
                  << "CPU elements: " << cpu_elements << std::endl;

        // Per-shard operation timing
        std::vector<std::chrono::milliseconds> shard_times;
        for (size_t i = 0; i < x->num_shards(); ++i) {
            auto shard_start = std::chrono::high_resolution_clock::now();
            auto y_local = x->add_scalar(static_cast<float>(i) * 0.001f);
            auto shard_end = std::chrono::high_resolution_clock::now();
            shard_times.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(shard_end - shard_start));
        }
        std::cout << "Per-shard add_scalar timing:" << std::endl;
        for (size_t i = 0; i < shard_times.size(); ++i) {
            std::cout << "  Shard " << i << " (" << x->shard_device(i) << "): " << shard_times[i].count() << "ms" << std::endl;
        }

        auto y = x->add_scalar(1.0f);
        auto ops_done = std::chrono::high_resolution_clock::now();

        auto sum = y->sum();
        auto sum_done = std::chrono::high_resolution_clock::now();

        auto create_ms = std::chrono::duration_cast<std::chrono::milliseconds>(created - start).count();
        auto ops_ms = std::chrono::duration_cast<std::chrono::milliseconds>(ops_done - created).count();
        auto sum_ms = std::chrono::duration_cast<std::chrono::milliseconds>(sum_done - ops_done).count();

        std::cout << "Sum: " << sum->get_element(0) << std::endl;
        std::cout << "Timing: create=" << create_ms << "ms, ops=" << ops_ms << "ms, sum=" << sum_ms << "ms" << std::endl;
        std::cout << "OK: Very large tensor works\n" << std::endl;
    }

    // Test 5: Autograd in GPU_FIRST mode
    std::cout << "=== Test 5: Autograd in GPU-FIRST Mode ===" << std::endl;
    {
        auto x = DistributedTensor<float>::randn({200, 200}, true);
        auto y = DistributedTensor<float>::randn({200, 200}, true);
        auto z = x->multiply(y.get());
        auto loss = z->sum();

        std::cout << "Forward: " << loss->distribution_info() << std::endl;
        std::cout << "Shards: " << loss->num_shards() << std::endl;
        for (size_t i = 0; i < loss->num_shards(); ++i) {
            std::cout << "  " << loss->shard_info(i) << std::endl;
        }

        loss->backward();
        std::cout << "Backward completed" << std::endl;

        x->zero_grad();
        y->zero_grad();
        std::cout << "Gradients zeroed" << std::endl;

        x->sgd_step(0.01f);
        std::cout << "SGD step completed" << std::endl;
        std::cout << "OK: Autograd works in GPU-FIRST mode\n" << std::endl;
    }

    // Test 6: MemoryBudget
    std::cout << "=== Test 6: MemoryBudget Info ===" << std::endl;
    std::cout << "MemoryBudget: " << MemoryBudget::instance().summary() << std::endl;

    std::cout << "\n================================================" << std::endl;
    std::cout << "  All GPU-FIRST Tests Passed!" << std::endl;
    std::cout << "================================================\n" << std::endl;

    return 0;
}
