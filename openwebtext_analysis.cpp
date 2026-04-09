// ============================================================================
// OpenWebtext Dataset Analysis with Distributed Tensor
// ============================================================================
// This file demonstrates how to:
// 1. Load large-scale text datasets (OpenWebtext) from Parquet format
// 2. Use distributed tensors to shard data across GPU + CPU
// 3. Run analysis operations (tokenization, embeddings, statistics)
// 4. Respect memory budgets using streaming/batching
//
// Usage:
//   Compile: g++ -std=c++17 -O2 -DTINYTORCH_USE_OPENCL_SDK -I. -Iparquet_parser/carquet/include \
//            -I"C:/Users/AMBE/OpenCL-SDK/install/include" -o openwebtext_analysis.exe \
//            openwebtext_analysis.cpp -L"C:/Users/AMBE/OpenCL-SDK/install/lib" -Lparquet_parser/carquet/build/Release \
//            -lOpenCL -lopengl32 -lgdi32 -lcarquet -lzlibstatic -llz4
//   Run: ./openwebtext_analysis.exe path/to/openwebtext.parquet

#include "distributed_tensor.h"
#include "distributed_parquet.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>

using namespace tinytorch;

// ============================================================================
// Analysis Operations
// ============================================================================

// Simple tokenization: split by spaces and count tokens
struct TokenStats {
    size_t total_tokens = 0;
    size_t total_chars = 0;
    size_t max_length = 0;
    size_t min_length = SIZE_MAX;
    double avg_length = 0.0;
    std::vector<size_t> length_histogram; // Binned by 100s
};

// Compute basic statistics on text lengths
template<typename T>
TokenStats analyze_text_lengths(DistributedTensor<T>* tensor) {
    TokenStats stats;
    
    // Get shard info
    size_t num_shards = tensor->num_shards();
    std::cout << "Analyzing " << num_shards << " shards..." << std::endl;
    
    // Per-shard analysis
    std::vector<std::future<TokenStats>> futures;
    
    for (size_t shard_idx = 0; shard_idx < num_shards; ++shard_idx) {
        // Note: In a real implementation, we'd need to access shard data
        // For now, this is a placeholder showing the API
    }
    
    return stats;
}

// ============================================================================
// Main Analysis Pipeline
// ============================================================================

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_openwebtext.parquet> [column_name]" << std::endl;
        std::cerr << "Example: " << argv[0] << " openwebtext.parquet text" << std::endl;
        return 1;
    }
    
    std::string parquet_file = argv[1];
    std::string column_name = (argc >= 3) ? argv[2] : "text";
    
    std::cout << "\n================================================" << std::endl;
    std::cout << "  OpenWebtext Dataset Analysis" << std::endl;
    std::cout << "================================================\n" << std::endl;
    
    // Print GPU_SATURATE configuration
    std::cout << DistributedTensor<float>::gpu_saturate_config_summary() << std::endl;
    std::cout << std::endl;
    
    // Get file info first
    std::cout << "=== Parquet File Info ===" << std::endl;
    std::string info = DistributedParquetLoader::get_file_info(parquet_file);
    std::cout << info << std::endl;
    
    // Configure GPU_SATURATE mode for optimal performance
    std::cout << "=== Configuring GPU_SATURATE Mode ===" << std::endl;
    
    // Enable work-stealing: Intel GPU takes remaining work after CUDA finishes
    DistributedTensor<float>::set_work_stealing_enabled(true);
    std::cout << "Work-stealing: ENABLED (Intel takes remaining work)" << std::endl;
    
    // Give Intel GPU higher priority (2x more work upfront)
    // This is because Intel UHD 620 has more memory than MX130 in many laptops
    DistributedTensor<float>::set_intel_gpu_priority_factor(2.0);
    std::cout << "Intel GPU priority factor: 2.0x (more work assigned upfront)" << std::endl;
    
    // Set minimum shard size to avoid tiny shards
    DistributedTensor<float>::set_min_shard_size(4096);
    std::cout << "Minimum shard size: 4096 elements" << std::endl;
    
    // Enable per-shard timing instrumentation
    DistributedTensor<float>::set_per_shard_timing_enabled(true);
    std::cout << "Per-shard timing: ENABLED" << std::endl;
    
    // Set work-stealing threshold (Intel takes over if CUDA takes >100ms)
    DistributedTensor<float>::set_work_stealing_threshold_ms(100);
    std::cout << "Work-stealing threshold: 100ms" << std::endl;
    
    std::cout << std::endl;
    
    // Load configuration
    ParquetLoadOptions opts;
    opts.column_name = column_name;
    opts.batch_size = 65536;      // 64K rows per batch
    opts.use_mmap = true;         // Use memory-mapped I/O
    opts.memory_budget_pct = 30.0; // 30% RAM threshold
    opts.min_shard_size = 4096;   // Same as above
    
    std::cout << "=== Loading Dataset ===" << std::endl;
    std::cout << "File: " << parquet_file << std::endl;
    std::cout << "Column: " << column_name << std::endl;
    std::cout << "Batch size: " << opts.batch_size << std::endl;
    std::cout << "Memory budget: " << opts.memory_budget_pct << "%" << std::endl;
    std::cout << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Load the dataset
    // For text data, we'd typically load as int32 (token IDs) or float (embeddings)
    // Here we demonstrate with float for embeddings
    auto tensor = DistributedParquetLoader::load<float>(parquet_file, opts);
    
    if (!tensor) {
        std::cerr << "Failed to load dataset" << std::endl;
        return 1;
    }
    
    auto load_done = std::chrono::high_resolution_clock::now();
    auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(load_done - start).count();
    
    std::cout << "\n=== Dataset Loaded ===" << std::endl;
    std::cout << "Load time: " << load_ms << "ms" << std::endl;
    std::cout << "Distribution: " << tensor->distribution_info() << std::endl;
    std::cout << "Number of shards: " << tensor->num_shards() << std::endl;
    
    // Print per-shard details
    std::cout << "\nShard details:" << std::endl;
    for (size_t i = 0; i < tensor->num_shards(); ++i) {
        std::cout << "  " << tensor->shard_info(i) << std::endl;
    }
    
    // Print shard timing report if available
    std::cout << "\n" << ExecutionContextManager::format_shard_timings() << std::endl;
    
    // Run distributed analysis operations
    std::cout << "=== Running Analysis Operations ===" << std::endl;
    
    // Operation 1: Sum (total of all elements)
    auto sum_start = std::chrono::high_resolution_clock::now();
    auto sum_tensor = tensor->sum();
    auto sum_done = std::chrono::high_resolution_clock::now();
    auto sum_ms = std::chrono::duration_cast<std::chrono::milliseconds>(sum_done - sum_start).count();
    
    std::cout << "Sum operation: " << sum_ms << "ms" << std::endl;
    std::cout << "Sum result: " << sum_tensor->get_element(0) << std::endl;
    
    // Operation 2: Element-wise multiply (simulate embedding scaling)
    auto mul_start = std::chrono::high_resolution_clock::now();
    auto scaled = tensor->multiply_scalar(0.5f);
    auto mul_done = std::chrono::high_resolution_clock::now();
    auto mul_ms = std::chrono::duration_cast<std::chrono::milliseconds>(mul_done - mul_start).count();
    
    std::cout << "Scale operation (x0.5): " << mul_ms << "ms" << std::endl;
    std::cout << "Scaled tensor: " << scaled->distribution_info() << std::endl;
    
    // Operation 3: Add scalar (shift embeddings)
    auto add_start = std::chrono::high_resolution_clock::now();
    auto shifted = scaled->add_scalar(1.0f);
    auto add_done = std::chrono::high_resolution_clock::now();
    auto add_ms = std::chrono::duration_cast<std::chrono::milliseconds>(add_done - add_start).count();
    
    std::cout << "Shift operation (+1.0): " << add_ms << "ms" << std::endl;
    
    // Verify operations
    auto verify_sum = shifted->sum();
    std::cout << "Final sum after operations: " << verify_sum->get_element(0) << std::endl;
    
    // Print final shard timing report
    std::cout << "\n" << ExecutionContextManager::format_shard_timings() << std::endl;
    
    // Print memory usage
    std::cout << "\n=== Memory Usage ===" << std::endl;
    std::cout << DistributedTensor<float>::memory_budget_summary() << std::endl;
    
    // Total time
    auto total_done = std::chrono::high_resolution_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_done - start).count();
    
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Total time: " << total_ms << "ms" << std::endl;
    std::cout << "  - Load: " << load_ms << "ms (" << (100.0 * load_ms / total_ms) << "%)" << std::endl;
    std::cout << "  - Sum: " << sum_ms << "ms (" << (100.0 * sum_ms / total_ms) << "%)" << std::endl;
    std::cout << "  - Scale: " << mul_ms << "ms (" << (100.0 * mul_ms / total_ms) << "%)" << std::endl;
    std::cout << "  - Shift: " << add_ms << "ms (" << (100.0 * add_ms / total_ms) << "%)" << std::endl;
    
    std::cout << "\n================================================" << std::endl;
    std::cout << "  Analysis Complete" << std::endl;
    std::cout << "================================================\n" << std::endl;
    
    return 0;
}
