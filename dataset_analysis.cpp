// ============================================================================
// OpenWebtext Dataset Analysis - Distributed Tensor Loading
// ============================================================================
// Loads the OpenWebtext parquet dataset and performs distributed operations
// Uses GPU_SATURATE mode without Intel priority (equal distribution)
//
// Compile: g++ -std=c++17 -O2 -DTINYTORCH_USE_OPENCL_SDK -I. \
//          -I"C:/Users/AMBE/OpenCL-SDK/install/include" \
//          -I"../parquet_parser/carquet/include" \
//          -o dataset_analysis.exe dataset_analysis.cpp \
//          -L"C:/Users/AMBE/OpenCL-SDK/install/lib" -lOpenCL -lopengl32 -lgdi32 \
//          -L"../parquet_parser/carquet/build/Release" -lcarquet -lzlibstatic -llz4

#include "distributed_tensor.h"
#include "distributed_parquet.h"
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    std::string dataset_path = "../dataset/train-openweb.parquet";
    if (argc > 1) {
        dataset_path = argv[1];
    }

    std::cout << "\n================================================" << std::endl;
    std::cout << "  OpenWebtext Dataset Analysis" << std::endl;
    std::cout << "  Distributed Tensor Loading" << std::endl;
    std::cout << "================================================\n" << std::endl;

    // Configure GPU_SATURATE mode - NO Intel priority
    DistributedTensor<float>::set_execution_mode(ExecutionMode::GPU_SATURATE);
    DistributedTensor<float>::set_max_gpu_memory_percentage(90.0);
    // Intel priority factor NOT set - disabled completely
    DistributedTensor<float>::set_min_shard_size(4096);

    std::cout << "=== Configuration ===" << std::endl;
    std::cout << "Mode: GPU_SATURATE" << std::endl;
    std::cout << "Intel priority: DISABLED (no scaling)" << std::endl;
    std::cout << "Dataset: " << dataset_path << std::endl;
    std::cout << std::endl;

    // Get file info
    std::cout << "=== Dataset Info ===" << std::endl;
    std::string info = tinytorch::DistributedParquetLoader::get_file_info(dataset_path);
    std::cout << info << std::endl;

    // Load the dataset
    std::cout << "=== Loading Dataset ===" << std::endl;
    tinytorch::ParquetLoadOptions opts;
    opts.column_name = "text";  // Common column name for text datasets
    opts.batch_size = 65536;
    opts.use_mmap = true;
    opts.min_shard_size = 4096;

    std::cout << "=== Text Samples (from Parquet) ===" << std::endl;
    {
        auto samples = tinytorch::DistributedParquetLoader::load_text_samples(dataset_path, opts, 5);
        for (size_t i = 0; i < samples.size(); ++i) {
            std::cout << "Row " << i << ": " << samples[i] << std::endl;
        }
        std::cout << std::endl;
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto tensor = tinytorch::DistributedParquetLoader::load_text_lengths(dataset_path, opts);
    auto load_done = std::chrono::high_resolution_clock::now();

    if (!tensor) {
        std::cerr << "Failed to load dataset text lengths." << std::endl;
        return 1;
    }

    auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(load_done - start).count();
    std::cout << "\nDataset loaded in " << load_ms << "ms" << std::endl;
    std::cout << "Tensor info: " << tensor->distribution_info() << std::endl;
    std::cout << "Total elements: " << tensor->total_elements() << std::endl;

    // Print sample values from original tensor (text lengths)
    std::cout << "\n=== Sample Values (Original) ===" << std::endl;
    std::cout << "First 10 elements: ";
    for (size_t i = 0; i < std::min(size_t(10), tensor->total_elements()); ++i) {
        std::cout << tensor->get_element(i) << " ";
    }
    std::cout << std::endl;

    // Show shard distribution
    std::cout << "\n=== Shard Distribution ===" << std::endl;
    size_t total_elements = 0;
    for (size_t i = 0; i < tensor->num_shards(); ++i) {
        std::cout << "  Shard " << i << ": " << tensor->shard_info(i) << std::endl;
        total_elements += tensor->shard_num_elements(i);
    }
    std::cout << "Total: " << total_elements << " elements" << std::endl;

    // Perform operations
    std::cout << "\n=== Performing Operations ===" << std::endl;

    // Operation 1: Sum (get total)
    auto sum_start = std::chrono::high_resolution_clock::now();
    auto sum_tensor = tensor->sum();
    auto sum_done = std::chrono::high_resolution_clock::now();
    auto sum_ms = std::chrono::duration_cast<std::chrono::milliseconds>(sum_done - sum_start).count();
    float total_sum = sum_tensor->get_element(0);
    std::cout << "Sum operation: " << sum_ms << "ms" << std::endl;
    std::cout << "Total sum: " << total_sum << std::endl;
    std::cout << "Average value: " << (total_sum / tensor->total_elements()) << std::endl;

    // Print sample values after sum (original data)
    std::cout << "Sample values [0-9]: ";
    for (size_t i = 0; i < std::min(size_t(10), tensor->total_elements()); ++i) {
        std::cout << tensor->get_element(i) << " ";
    }
    std::cout << std::endl;

    // Operation 2: Scale values
    auto scale_start = std::chrono::high_resolution_clock::now();
    auto scaled = tensor->multiply_scalar(0.5f);
    auto scale_done = std::chrono::high_resolution_clock::now();
    auto scale_ms = std::chrono::duration_cast<std::chrono::milliseconds>(scale_done - scale_start).count();
    std::cout << "\nScale (x0.5) operation: " << scale_ms << "ms" << std::endl;

    // Print sample values after scale
    std::cout << "Sample values after scale [0-9]: ";
    for (size_t i = 0; i < std::min(size_t(10), scaled->total_elements()); ++i) {
        std::cout << scaled->get_element(i) << " ";
    }
    std::cout << std::endl;

    // Verify scale by summing again
    auto scaled_sum = scaled->sum();
    std::cout << "Scaled sum: " << scaled_sum->get_element(0) << " (should be ~" << (total_sum * 0.5f) << ")" << std::endl;

    // Operation 3: Add scalar
    auto add_start = std::chrono::high_resolution_clock::now();
    auto shifted = scaled->add_scalar(1.0f);
    auto add_done = std::chrono::high_resolution_clock::now();
    auto add_ms = std::chrono::duration_cast<std::chrono::milliseconds>(add_done - add_start).count();
    std::cout << "\nAdd (+1.0) operation: " << add_ms << "ms" << std::endl;

    // Print sample values after add
    std::cout << "Sample values after add [0-9]: ";
    for (size_t i = 0; i < std::min(size_t(10), shifted->total_elements()); ++i) {
        std::cout << shifted->get_element(i) << " ";
    }
    std::cout << std::endl;

    // Final verification
    auto final_sum = shifted->sum();
    float final_total = final_sum->get_element(0);
    std::cout << "Final sum after scale+add: " << final_total << std::endl;

    // Calculate expected: (original_avg * 0.5 + 1.0)
    float expected = (total_sum / tensor->total_elements()) * 0.5f + 1.0f;
    std::cout << "Expected average after ops: ~" << expected << std::endl;
    std::cout << "Actual average: " << (final_total / tensor->total_elements()) << std::endl;

    // Summary
    auto total_done = std::chrono::high_resolution_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_done - start).count();

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Total time: " << total_ms << "ms" << std::endl;
    std::cout << "  Load: " << load_ms << "ms" << std::endl;
    std::cout << "  Sum: " << sum_ms << "ms" << std::endl;
    std::cout << "  Scale: " << scale_ms << "ms" << std::endl;
    std::cout << "  Add: " << add_ms << "ms" << std::endl;

    // Check if distribution was roughly equal (50/50)
    std::cout << "\n=== Distribution Check (should be ~50/50) ===" << std::endl;
    size_t cuda_elems = 0, opencl_elems = 0, cpu_elems = 0;
    for (size_t i = 0; i < tensor->num_shards(); ++i) {
        std::string dev = tensor->shard_device(i);
        size_t elems = tensor->shard_num_elements(i);
        if (dev.find("CUDA") != std::string::npos) cuda_elems += elems;
        else if (dev.find("OpenCL") != std::string::npos) opencl_elems += elems;
        else cpu_elems += elems;
    }

    size_t gpu_total = cuda_elems + opencl_elems;
    if (gpu_total > 0) {
        std::cout << "  CUDA: " << cuda_elems << " (" << (100.0 * cuda_elems / gpu_total) << "%)" << std::endl;
        std::cout << "  OpenCL (Intel): " << opencl_elems << " (" << (100.0 * opencl_elems / gpu_total) << "%)" << std::endl;
    }
    if (cpu_elems > 0) {
        std::cout << "  CPU: " << cpu_elems << " (overflow)" << std::endl;
    }

    std::cout << "\n================================================" << std::endl;
    std::cout << "  Dataset Analysis Complete" << std::endl;
    std::cout << "================================================\n" << std::endl;

    return 0;
}
