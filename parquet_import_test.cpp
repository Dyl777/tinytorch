#include "distributed_tensor.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

void test_parquet_metadata() {
    std::cout << "=== Testing Parquet Metadata Parsing ===" << std::endl;

    auto meta = DistributedTensor<float>::parse_parquet_metadata("sample_data.parquet");

    std::cout << "Rows: " << meta.num_rows << std::endl;
    std::cout << "Cols: " << meta.num_cols << std::endl;
    std::cout << "Columns:" << std::endl;
    for (const auto& col : meta.column_names) {
        std::cout << "  - " << col << std::endl;
    }

    assert(meta.num_rows == 1000);
    assert(meta.num_cols == 5);
    assert(meta.column_names.size() == 5);
    assert(meta.column_names[0] == "feature_0");
    assert(meta.column_names[4] == "label");

    std::cout << "OK: Metadata parsed correctly\n" << std::endl;
}

void test_parquet_import() {
    std::cout << "=== Testing Parquet Import ===" << std::endl;

    auto tensor = DistributedTensor<float>::from_parquet("sample_data.parquet");

    std::cout << "Shape: [";
    auto shape = tensor->shape();
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << shape[i];
    }
    std::cout << "]" << std::endl;

    std::cout << "Total elements: " << tensor->total_elements() << std::endl;
    std::cout << "Number of shards: " << tensor->num_shards() << std::endl;

    // Print each shard's device assignment
    std::cout << "Shard distribution:" << std::endl;
    for (size_t i = 0; i < tensor->num_shards(); ++i) {
        std::cout << "  Shard " << i << ": " << tensor->shard_info(i) << std::endl;
    }

    assert(tensor->shape().size() == 2);
    assert(tensor->shape()[0] == 1000);
    assert(tensor->shape()[1] == 5);
    assert(tensor->total_elements() == 5000);
    assert(tensor->num_shards() >= 1);

    std::cout << "OK: Parquet imported successfully\n" << std::endl;
}

void test_parquet_data_integrity() {
    std::cout << "=== Testing Parquet Data Integrity ===" << std::endl;

    auto tensor = DistributedTensor<float>::from_parquet("sample_data.parquet");

    // Check first few values (known from the sample data)
    // Row 0: feature_0=0.496714, feature_1=7.798711, feature_2=-3.337589, feature_3=1.907808, label=0.0
    float val_0_0 = tensor->get_element(0);
    float val_0_1 = tensor->get_element(1);
    float val_0_2 = tensor->get_element(2);
    float val_0_3 = tensor->get_element(3);
    float val_0_4 = tensor->get_element(4);

    std::cout << "Row 0 values:" << std::endl;
    std::cout << "  feature_0: " << val_0_0 << " (expected ~0.496714)" << std::endl;
    std::cout << "  feature_1: " << val_0_1 << " (expected ~7.798711)" << std::endl;
    std::cout << "  feature_2: " << val_0_2 << " (expected ~-3.337589)" << std::endl;
    std::cout << "  feature_3: " << val_0_3 << " (expected ~1.907808)" << std::endl;
    std::cout << "  label:     " << val_0_4 << " (expected 0.0)" << std::endl;

    assert(std::abs(val_0_0 - 0.496714f) < 0.001f);
    assert(std::abs(val_0_1 - 7.798711f) < 0.01f);
    assert(std::abs(val_0_2 - (-3.337589f)) < 0.01f);
    assert(std::abs(val_0_3 - 1.907808f) < 0.01f);
    assert(std::abs(val_0_4 - 0.0f) < 0.001f);

    // Check row-major layout: row 1 starts at index 5
    float val_1_0 = tensor->get_element(5);
    std::cout << "\nRow 1, feature_0: " << val_1_0 << " (expected ~-0.138264)" << std::endl;
    assert(std::abs(val_1_0 - (-0.138264f)) < 0.001f);

    std::cout << "\nOK: Data integrity verified\n" << std::endl;
}

void test_parquet_operations() {
    std::cout << "=== Testing Parquet Tensor Operations ===" << std::endl;

    auto x = DistributedTensor<float>::from_parquet("sample_data.parquet");

    // Scalar operations
    auto x_plus_one = x->add_scalar(1.0f);
    std::cout << "After add_scalar(1.0): " << x_plus_one->distribution_info() << std::endl;

    auto x_doubled = x->multiply_scalar(2.0f);
    std::cout << "After multiply_scalar(2.0): " << x_doubled->distribution_info() << std::endl;

    // Verify operation results
    float original = x->get_element(0);
    float plus_one = x_plus_one->get_element(0);
    float doubled = x_doubled->get_element(0);

    std::cout << "\nVerification:" << std::endl;
    std::cout << "  Original[0]: " << original << std::endl;
    std::cout << "  Original[0] + 1: " << plus_one << " (expected " << (original + 1.0f) << ")" << std::endl;
    std::cout << "  Original[0] * 2: " << doubled << " (expected " << (original * 2.0f) << ")" << std::endl;

    assert(std::abs(plus_one - (original + 1.0f)) < 0.001f);
    assert(std::abs(doubled - (original * 2.0f)) < 0.001f);

    // Reductions
    auto total = x->sum();
    auto mean = x->mean();

    std::cout << "\nReductions:" << std::endl;
    std::cout << "  Sum: " << total->get_element(0) << std::endl;
    std::cout << "  Mean: " << mean->get_element(0) << std::endl;

    std::cout << "\nOK: Operations work on Parquet data\n" << std::endl;
}

void test_parquet_with_grad() {
    std::cout << "=== Testing Parquet Import with Gradient ===" << std::endl;

    auto x = DistributedTensor<float>::from_parquet("sample_data.parquet", true);

    std::cout << "Requires grad: " << (x->requires_grad() ? "yes" : "no") << std::endl;
    assert(x->requires_grad());

    // Simple forward pass
    auto y = x->multiply_scalar(2.0f);
    auto loss = y->sum();

    std::cout << "Forward pass: " << loss->distribution_info() << std::endl;

    // Backward pass
    loss->backward();
    std::cout << "Backward pass completed" << std::endl;

    x->zero_grad();
    std::cout << "Gradients zeroed" << std::endl;

    std::cout << "\nOK: Parquet with gradient works\n" << std::endl;
}

int main() {
    std::cout << "\n================================================" << std::endl;
    std::cout << "  TinyTorch Parquet Import Test Suite" << std::endl;
    std::cout << "================================================\n" << std::endl;

    test_parquet_metadata();
    test_parquet_import();
    test_parquet_data_integrity();
    test_parquet_operations();
    test_parquet_with_grad();

    std::cout << "================================================" << std::endl;
    std::cout << "  All Parquet Import Tests Passed!" << std::endl;
    std::cout << "================================================\n" << std::endl;

    return 0;
}
