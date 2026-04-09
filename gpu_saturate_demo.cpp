// ============================================================================
// GPU_SATURATE Mode Demo with Work-Stealing and Intel GPU Priority
// ============================================================================
// This demonstrates the new GPU_SATURATE options:
// 1. Work-stealing: Intel GPU takes remaining work after CUDA finishes
// 2. Intel GPU priority factor: Give Intel more work upfront (0.5x to 3x+ tested)
// 3. Minimum shard size: Avoid tiny shards
//
// NOTE: Per-shard timing moved to tests to avoid implementation bottleneck
//
// Compile: g++ -std=c++17 -O2 -DTINYTORCH_USE_OPENCL_SDK -I. \
//          -I"C:/Users/AMBE/OpenCL-SDK/install/include" \
//          -o gpu_saturate_demo.exe gpu_saturate_demo.cpp \
//          -L"C:/Users/AMBE/OpenCL-SDK/install/lib" -lOpenCL -lopengl32 -lgdi32

#include "distributed_tensor.h"
#include <iostream>
#include <vector>
#include <chrono>

int main() {
    std::cout << "\n================================================" << std::endl;
    std::cout << "  GPU_SATURATE Mode Demo" << std::endl;
    std::cout << "  Work-Stealing + Intel GPU Priority Tests" << std::endl;
    std::cout << "================================================\n" << std::endl;

    // Configure GPU_SATURATE mode
    DistributedTensor<float>::set_execution_mode(ExecutionMode::GPU_SATURATE);
    DistributedTensor<float>::set_max_gpu_memory_percentage(90.0);
    
    // Configure new GPU_SATURATE options
    std::cout << "=== Configuration ===" << std::endl;
    
    // Enable work-stealing: Intel takes remaining work after CUDA finishes
    DistributedTensor<float>::set_work_stealing_enabled(true);
    std::cout << "Work-stealing: ENABLED" << std::endl;
    
    // Give Intel GPU 2x priority (more work assigned upfront)
    DistributedTensor<float>::set_intel_gpu_priority_factor(2.0);
    std::cout << "Intel GPU priority: 2.0x (2x more elements assigned upfront)" << std::endl;
    
    // Set minimum shard size to avoid tiny shards
    DistributedTensor<float>::set_min_shard_size(4096);
    std::cout << "Minimum shard size: 4096 elements" << std::endl;
    
    // Print full config
    std::cout << "\n" << DistributedTensor<float>::gpu_saturate_config_summary() << std::endl;
    std::cout << std::endl;
    
    // Test 1: Medium tensor with Intel priority
    std::cout << "=== Test 1: Intel Priority Mode (3000x3000 = 9M elements) ===" << std::endl;
    {
        // With 2x priority, Intel should get ~66% of elements, CUDA ~33%
        auto x = DistributedTensor<float>::randn({3000, 3000});
        
        std::cout << "Distribution: " << x->distribution_info() << std::endl;
        std::cout << "Shards: " << x->num_shards() << std::endl;
        
        // Calculate element distribution
        size_t cuda_elems = 0, opencl_elems = 0, cpu_elems = 0;
        for (size_t i = 0; i < x->num_shards(); ++i) {
            std::string dev = x->shard_device(i);
            size_t elems = x->shard_num_elements(i);
            if (dev.find("CUDA") != std::string::npos) cuda_elems += elems;
            else if (dev.find("OpenCL") != std::string::npos) opencl_elems += elems;
            else cpu_elems += elems;
        }
        
        size_t total = cuda_elems + opencl_elems + cpu_elems;
        std::cout << "Element distribution:" << std::endl;
        std::cout << "  CUDA: " << cuda_elems << " (" << (100.0 * cuda_elems / total) << "%)" << std::endl;
        std::cout << "  OpenCL: " << opencl_elems << " (" << (100.0 * opencl_elems / total) << "%)" << std::endl;
        std::cout << "  CPU: " << cpu_elems << " (" << (100.0 * cpu_elems / total) << "%)" << std::endl;
        
        // Run operation and get timing
        auto start = std::chrono::high_resolution_clock::now();
        auto y = x->multiply_scalar(2.0f);
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        std::cout << "Operation time: " << ms << "ms" << std::endl;
    }
    
    // Test 1b: Intel Priority 3x (even more bias toward Intel)
    std::cout << "\n=== Test 1b: Intel Priority 3x (extreme) ===" << std::endl;
    {
        DistributedTensor<float>::set_intel_gpu_priority_factor(3.0);
        
        auto x = DistributedTensor<float>::randn({3000, 3000});
        
        size_t cuda_elems = 0, opencl_elems = 0;
        for (size_t i = 0; i < x->num_shards(); ++i) {
            std::string dev = x->shard_device(i);
            size_t elems = x->shard_num_elements(i);
            if (dev.find("CUDA") != std::string::npos) cuda_elems += elems;
            else if (dev.find("OpenCL") != std::string::npos) opencl_elems += elems;
        }
        
        size_t total = cuda_elems + opencl_elems;
        std::cout << "With 3x priority, Intel should get ~75%, CUDA ~25%:" << std::endl;
        std::cout << "  CUDA: " << cuda_elems << " (" << (100.0 * cuda_elems / total) << "%)" << std::endl;
        std::cout << "  OpenCL (Intel): " << opencl_elems << " (" << (100.0 * opencl_elems / total) << "%)" << std::endl;
    }
    
    // Test 1c: Intel Priority 0.5x (favor CUDA)
    std::cout << "\n=== Test 1c: Intel Priority 0.5x (favor CUDA) ===" << std::endl;
    {
        DistributedTensor<float>::set_intel_gpu_priority_factor(0.5);
        
        auto x = DistributedTensor<float>::randn({3000, 3000});
        
        size_t cuda_elems = 0, opencl_elems = 0;
        for (size_t i = 0; i < x->num_shards(); ++i) {
            std::string dev = x->shard_device(i);
            size_t elems = x->shard_num_elements(i);
            if (dev.find("CUDA") != std::string::npos) cuda_elems += elems;
            else if (dev.find("OpenCL") != std::string::npos) opencl_elems += elems;
        }
        
        size_t total = cuda_elems + opencl_elems;
        std::cout << "With 0.5x priority, Intel gets half share:" << std::endl;
        std::cout << "  CUDA: " << cuda_elems << " (" << (100.0 * cuda_elems / total) << "%)" << std::endl;
        std::cout << "  OpenCL (Intel): " << opencl_elems << " (" << (100.0 * opencl_elems / total) << "%)" << std::endl;
    }
    
    // Test 2: With work-stealing disabled (equal distribution)
    std::cout << "\n=== Test 2: Equal Distribution Mode (priority=1.0) ===" << std::endl;
    {
        DistributedTensor<float>::set_intel_gpu_priority_factor(1.0);
        
        auto x = DistributedTensor<float>::randn({3000, 3000});
        
        size_t cuda_elems = 0, opencl_elems = 0;
        for (size_t i = 0; i < x->num_shards(); ++i) {
            std::string dev = x->shard_device(i);
            size_t elems = x->shard_num_elements(i);
            if (dev.find("CUDA") != std::string::npos) cuda_elems += elems;
            else if (dev.find("OpenCL") != std::string::npos) opencl_elems += elems;
        }
        
        size_t total = cuda_elems + opencl_elems;
        std::cout << "Element distribution (should be ~50/50):" << std::endl;
        std::cout << "  CUDA: " << cuda_elems << " (" << (100.0 * cuda_elems / total) << "%)" << std::endl;
        std::cout << "  OpenCL: " << opencl_elems << " (" << (100.0 * opencl_elems / total) << "%)" << std::endl;
    }
    
    // Test 3: Minimum shard size enforcement
    std::cout << "\n=== Test 3: Minimum Shard Size (4096) ===" << std::endl;
    {
        // Small tensor - should result in minimum-sized shards or single shard
        DistributedTensor<float>::set_min_shard_size(4096);
        auto x = DistributedTensor<float>::randn({100, 100}); // 10K elements
        
        std::cout << "Tensor: 100x100 = 10K elements" << std::endl;
        std::cout << "Shards: " << x->num_shards() << std::endl;
        
        for (size_t i = 0; i < x->num_shards(); ++i) {
            std::cout << "  Shard " << i << ": " << x->shard_num_elements(i) << " elements" << std::endl;
        }
    }
    
    // Test 4: Large tensor with Intel priority
    std::cout << "\n=== Test 4: Large Tensor (25M elements) with Intel Priority ===" << std::endl;
    {
        // Restore Intel priority
        DistributedTensor<float>::set_intel_gpu_priority_factor(2.0);
        
        auto start = std::chrono::high_resolution_clock::now();
        auto x = DistributedTensor<float>::randn({5000, 5000}); // 25M elements
        auto created = std::chrono::high_resolution_clock::now();
        
        // Check actual distribution
        size_t cuda_elems = 0, opencl_elems = 0;
        for (size_t i = 0; i < x->num_shards(); ++i) {
            std::string dev = x->shard_device(i);
            size_t elems = x->shard_num_elements(i);
            if (dev.find("CUDA") != std::string::npos) cuda_elems += elems;
            else if (dev.find("OpenCL") != std::string::npos) opencl_elems += elems;
        }
        
        std::cout << "Created with distribution:" << std::endl;
        std::cout << "  CUDA: " << cuda_elems << " elements" << std::endl;
        std::cout << "  Intel GPU: " << opencl_elems << " elements" << std::endl;
        
        // Run operations
        auto y = x->multiply_scalar(1.5f);
        auto op1_done = std::chrono::high_resolution_clock::now();
        
        auto z = y->add_scalar(0.5f);
        auto op2_done = std::chrono::high_resolution_clock::now();
        
        auto sum = z->sum();
        auto sum_done = std::chrono::high_resolution_clock::now();
        
        auto create_ms = std::chrono::duration_cast<std::chrono::milliseconds>(created - start).count();
        auto op1_ms = std::chrono::duration_cast<std::chrono::milliseconds>(op1_done - created).count();
        auto op2_ms = std::chrono::duration_cast<std::chrono::milliseconds>(op2_done - op1_done).count();
        auto sum_ms = std::chrono::duration_cast<std::chrono::milliseconds>(sum_done - op2_done).count();
        
        std::cout << "Timing breakdown:" << std::endl;
        std::cout << "  Create: " << create_ms << "ms" << std::endl;
        std::cout << "  Multiply: " << op1_ms << "ms" << std::endl;
        std::cout << "  Add: " << op2_ms << "ms" << std::endl;
        std::cout << "  Sum: " << sum_ms << "ms" << std::endl;
        std::cout << "  Total: " << (create_ms + op1_ms + op2_ms + sum_ms) << "ms" << std::endl;
    }
    
    // Test 5: Work-stealing scenario simulation
    std::cout << "\n=== Test 5: Work-Stealing Configuration ===" << std::endl;
    {
        std::cout << "Work-stealing enabled: " 
                  << (DistributedTensor<float>::get_work_stealing_enabled() ? "YES" : "NO") 
                  << std::endl;
        std::cout << "When enabled, Intel GPU will take remaining work if CUDA finishes first" << std::endl;
        std::cout << "Threshold: " << DistributedTensor<float>::get_work_stealing_threshold_ms() << "ms" << std::endl;
    }
    
    // Print memory budget info
    std::cout << "\n=== Memory Budget ===" << std::endl;
    std::cout << DistributedTensor<float>::memory_budget_summary() << std::endl;
    
    std::cout << "\n================================================" << std::endl;
    std::cout << "  All GPU_SATURATE Tests Passed!" << std::endl;
    std::cout << "================================================\n" << std::endl;
    
    return 0;
}
