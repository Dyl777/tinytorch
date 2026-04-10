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

// Compute basic statistics on text lengths - runs on each shard's device
template<typename T>
TokenStats analyze_text_lengths(DistributedTensor<T>* tensor) {
    TokenStats stats;
    
    // Get shard info
    size_t num_shards = tensor->num_shards();
    std::cout << "Analyzing " << num_shards << " shards..." << std::endl;
    
    // Per-shard analysis using parallel execution
    std::vector<std::future<TokenStats>> futures;
    
    for (size_t shard_idx = 0; shard_idx < num_shards; ++shard_idx) {
        futures.push_back(std::async(std::launch::async, [tensor, shard_idx]() {
            TokenStats shard_stats;
            size_t num_elems = tensor->shard_num_elements(shard_idx);
            
            // Get offset for this shard
            size_t global_offset = 0;
            for (size_t i = 0; i < shard_idx; ++i) {
                global_offset += tensor->shard_num_elements(i);
            }
            
            // Process each element in this shard
            for (size_t i = 0; i < num_elems; ++i) {
                float val = tensor->get_element(global_offset + i);
                size_t len = static_cast<size_t>(val);
                
                shard_stats.total_chars += len;
                shard_stats.max_length = std::max(shard_stats.max_length, len);
                shard_stats.min_length = std::min(shard_stats.min_length, len);
            }
            
            return shard_stats;
        }));
    }
    
    // Aggregate results from all shards
    for (auto& f : futures) {
        TokenStats shard_result = f.get();
        stats.total_chars += shard_result.total_chars;
        stats.max_length = std::max(stats.max_length, shard_result.max_length);
        if (shard_result.min_length < stats.min_length) {
            stats.min_length = shard_result.min_length;
        }
    }
    
    if (tensor->total_elements() > 0) {
        stats.avg_length = static_cast<double>(stats.total_chars) / tensor->total_elements();
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
    
    // CRITICAL: Set execution mode to GPU_SATURATE (was missing!)
    DistributedTensor<float>::set_execution_mode(ExecutionMode::GPU_SATURATE);
    std::cout << "Mode: GPU_SATURATE" << std::endl;
    
    // Work-stealing and Intel priority DISABLED
    std::cout << "Work-stealing: DISABLED" << std::endl;
    std::cout << "Intel GPU priority: DISABLED (no scaling)" << std::endl;
    
    // Set minimum shard size to avoid tiny shards
    DistributedTensor<float>::set_min_shard_size(4096);
    std::cout << "Minimum shard size: 4096 elements" << std::endl;
    
    // Timing instrumentation removed (not available in core)
    std::cout << "Per-shard timing: DISABLED (moved to tests)" << std::endl;
    
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
    
    // Load text lengths as float tensor for GPU operations
    auto tensor = DistributedParquetLoader::load_text_lengths(parquet_file, opts);
    // For text data, we'd typically load as int32 (token IDs) or float (embeddings)
    // Here we demonstrate with float for embeddings
    //auto tensor = DistributedParquetLoader::load<float>(parquet_file, opts);
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
    
    
    // Run distributed analysis operations
    std::cout << "=== Running Analysis Operations ===" << std::endl;
    
    // Print first few text samples
    std::cout << "\n=== Text Samples ===" << std::endl;
    auto samples = DistributedParquetLoader::load_text_samples(parquet_file, opts, 3);
    for (size_t i = 0; i < samples.size(); ++i) {
        std::cout << "Row " << i << " (length=" << samples[i].length() << "): " 
                  << samples[i].substr(0, 100) << (samples[i].length() > 100 ? "..." : "") << std::endl;
    }
    
    // Run text length analysis
    std::cout << "\n=== Analyzing Text Lengths ===" << std::endl;
    auto stats = analyze_text_lengths(tensor.get());
    std::cout << "Total characters: " << stats.total_chars << std::endl;
    std::cout << "Max length: " << stats.max_length << std::endl;
    std::cout << "Min length: " << stats.min_length << std::endl;
    std::cout << "Avg length: " << stats.avg_length << std::endl;
    
    // Operation 1: Sum (total of all elements)
    auto sum_start = std::chrono::high_resolution_clock::now();
    auto sum_tensor = tensor->sum();
    auto sum_done = std::chrono::high_resolution_clock::now();
    auto sum_ms = std::chrono::duration_cast<std::chrono::milliseconds>(sum_done - sum_start).count();
    
    std::cout << "\nSum operation: " << sum_ms << "ms" << std::endl;
    std::cout << "Sum result (total lengths): " << sum_tensor->get_element(0) << std::endl;
    
    // Print sample values before operations
    std::cout << "\n=== Sample Values (First 10 lengths) ===" << std::endl;
    std::cout << "Original: ";
    for (size_t i = 0; i < std::min(size_t(10), tensor->total_elements()); ++i) {
        std::cout << tensor->get_element(i) << " ";
    }
    std::cout << std::endl;
    
    // Operation 2: Element-wise multiply (scale lengths)
    auto mul_start = std::chrono::high_resolution_clock::now();
    auto scaled = tensor->multiply_scalar(0.5f);
    auto mul_done = std::chrono::high_resolution_clock::now();
    auto mul_ms = std::chrono::duration_cast<std::chrono::milliseconds>(mul_done - mul_start).count();
    // std::cout << "Scale operation (x0.5): " << mul_ms << "ms" << std::endl;
    // std::cout << "Scaled tensor: " << scaled->distribution_info() << std::endl;
    std::cout << "\nScale operation (x0.5): " << mul_ms << "ms" << std::endl;
    std::cout << "After scale: ";
    for (size_t i = 0; i < std::min(size_t(10), scaled->total_elements()); ++i) {
        std::cout << scaled->get_element(i) << " ";
    }
    std::cout << std::endl;
    
    // Operation 3: Add scalar (shift lengths)
    auto add_start = std::chrono::high_resolution_clock::now();
    auto shifted = scaled->add_scalar(1.0f);
    auto add_done = std::chrono::high_resolution_clock::now();
    auto add_ms = std::chrono::duration_cast<std::chrono::milliseconds>(add_done - add_start).count();
    
    std::cout << "\nAdd operation (+1.0): " << add_ms << "ms" << std::endl;
    std::cout << "After add: ";
    for (size_t i = 0; i < std::min(size_t(10), shifted->total_elements()); ++i) {
        std::cout << shifted->get_element(i) << " ";
    }
    std::cout << std::endl;
    
    // Verify operations
    auto verify_sum = shifted->sum();
    std::cout << "Final sum after operations: " << verify_sum->get_element(0) << std::endl;
    
    
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
