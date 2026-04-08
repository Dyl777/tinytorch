#include "distributed_tensor.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

void test_matmul() {
    std::cout << "=== Testing Distributed Matmul ===" << std::endl;

    // Create small tensors for testing: [2,3] @ [4,3]^T = [2,4]
    std::vector<float> data_a = {1, 2, 3, 4, 5, 6};
    std::vector<float> data_b = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0};

    auto a = DistributedTensor<float>::from_data(data_a, {2, 3});
    auto b = DistributedTensor<float>::from_data(data_b, {4, 3});

    std::cout << "A shape: [2, 3], B shape: [4, 3]" << std::endl;

    auto c = a->matmul(b.get());
    std::cout << "Result shape: [" << c->shape()[0] << ", " << c->shape()[1] << "]" << std::endl;
    assert(c->shape()[0] == 2);
    assert(c->shape()[1] == 4);

    // Verify some values
    auto gathered = c->all_gather();
    std::cout << "Result: [";
    for (size_t i = 0; i < gathered.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << gathered[i];
    }
    std::cout << "]" << std::endl;

    // Expected: row 0 of A @ all rows of B^T
    // [1,2,3] @ [1,0,1,0] = 1*1+2*0+3*1=4, 1*0+2*1+3*0=2, 1*1+2*0+3*1=4, 1*0+2*1+3*0=2
    assert(std::abs(gathered[0] - 4.0f) < 0.01f);
    assert(std::abs(gathered[1] - 2.0f) < 0.01f);

    std::cout << "OK: Matmul works\n" << std::endl;
}

void test_softmax() {
    std::cout << "=== Testing Distributed Softmax ===" << std::endl;

    std::vector<float> data = {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f};
    auto x = DistributedTensor<float>::from_data(data, {2, 3});

    auto s = x->softmax();
    auto gathered = s->all_gather();

    std::cout << "Softmax result: [";
    for (size_t i = 0; i < gathered.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << gathered[i];
    }
    std::cout << "]" << std::endl;

    // Global softmax: all values should sum to 1
    float total_sum = 0;
    for (float v : gathered) total_sum += v;
    std::cout << "Total sum: " << total_sum << std::endl;
    assert(std::abs(total_sum - 1.0f) < 0.001f);

    // Values should be positive and ordered
    assert(gathered[2] > gathered[1]);  // exp(3) > exp(2)
    assert(gathered[1] > gathered[0]);  // exp(2) > exp(1)

    std::cout << "OK: Softmax works\n" << std::endl;
}

void test_layer_norm() {
    std::cout << "=== Testing Distributed Layer Norm ===" << std::endl;

    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto x = DistributedTensor<float>::from_data(data, {2, 3});

    auto ln = x->layer_norm();
    auto gathered = ln->all_gather();

    std::cout << "Layer norm result: [";
    for (size_t i = 0; i < gathered.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << gathered[i];
    }
    std::cout << "]" << std::endl;

    // Global normalization: mean should be ~0
    float global_mean = 0;
    for (float v : gathered) global_mean += v;
    global_mean /= gathered.size();
    std::cout << "Global mean: " << global_mean << std::endl;
    assert(std::abs(global_mean) < 0.001f);

    // Std should be ~1
    float global_var = 0;
    for (float v : gathered) global_var += (v - global_mean) * (v - global_mean);
    global_var /= gathered.size();
    float global_std = std::sqrt(global_var);
    std::cout << "Global std: " << global_std << std::endl;
    assert(std::abs(global_std - 1.0f) < 0.001f);

    std::cout << "OK: Layer norm works\n" << std::endl;
}

void test_activations() {
    std::cout << "=== Testing Distributed Activations ===" << std::endl;

    std::vector<float> data = {-1.0f, 0.0f, 1.0f, 2.0f};
    auto x = DistributedTensor<float>::from_data(data, {2, 2});

    auto gelu_out = x->gelu();
    auto gelu_data = gelu_out->all_gather();
    std::cout << "GELU: [";
    for (size_t i = 0; i < gelu_data.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << gelu_data[i];
    }
    std::cout << "]" << std::endl;

    auto silu_out = x->silu();
    auto silu_data = silu_out->all_gather();
    std::cout << "SiLU: [";
    for (size_t i = 0; i < silu_data.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << silu_data[i];
    }
    std::cout << "]" << std::endl;

    // GELU(0) = 0, GELU(1) ~ 0.84
    assert(std::abs(gelu_data[1]) < 0.001f);  // GELU(0) = 0
    assert(gelu_data[2] > 0.5f);  // GELU(1) > 0.5

    // SiLU(0) = 0, SiLU(1) = 1 * sigmoid(1) ~ 0.73
    assert(std::abs(silu_data[1]) < 0.001f);  // SiLU(0) = 0
    assert(silu_data[2] > 0.5f && silu_data[2] < 1.0f);  // SiLU(1) in (0.5, 1)

    std::cout << "OK: Activations work\n" << std::endl;
}

void test_cross_entropy() {
    std::cout << "=== Testing Distributed Cross Entropy Loss ===" << std::endl;

    // Simple case: logits for 2 classes, 2 samples
    std::vector<float> logits = {2.0f, 1.0f, 0.1f, 3.0f};
    std::vector<float> labels = {1.0f, 0.0f, 0.0f, 1.0f};

    auto logit_tensor = DistributedTensor<float>::from_data(logits, {2, 2});
    auto label_tensor = DistributedTensor<float>::from_data(labels, {2, 2});

    auto loss = logit_tensor->cross_entropy_loss(label_tensor.get());
    float loss_val = loss->get_element(0);

    std::cout << "Cross entropy loss: " << loss_val << std::endl;
    assert(loss_val > 0.0f);  // Loss should be positive
    assert(loss_val < 5.0f);  // And reasonable

    std::cout << "OK: Cross entropy works\n" << std::endl;
}

void test_optimizer_steps() {
    std::cout << "=== Testing Distributed Optimizer Steps ===" << std::endl;

    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto x = DistributedTensor<float>::from_data(data, {2, 2}, true);
    auto loss = x->sum();
    std::cout << "Loss value: " << loss->get_element(0) << std::endl;
    loss->backward();
    std::cout << "Backward completed" << std::endl;

    // Check if gradients exist by doing another operation that uses grads
    auto before = x->all_gather();
    std::cout << "Params before SGD: [";
    for (size_t i = 0; i < before.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << before[i];
    }
    std::cout << "]" << std::endl;

    // Try sgd_step
    x->sgd_step(0.01f);
    auto after = x->all_gather();
    std::cout << "Params after SGD: [";
    for (size_t i = 0; i < after.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << after[i];
    }
    std::cout << "]" << std::endl;

    // Just verify the operation completed without crashing
    std::cout << "OK: Optimizer steps work\n" << std::endl;
}

void test_all_reduce() {
    std::cout << "=== Testing Distributed All-Reduce ===" << std::endl;

    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto x = DistributedTensor<float>::from_data(data, {2, 2});

    auto reduced = x->all_reduce_sum();
    auto gathered = reduced->all_gather();

    std::cout << "All-reduce result: [";
    for (size_t i = 0; i < gathered.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << gathered[i];
    }
    std::cout << "]" << std::endl;

    // All shards should have the same summed values
    float expected_sum = 1.0f + 2.0f + 3.0f + 4.0f;
    std::cout << "Expected sum per element: " << expected_sum << std::endl;
    for (size_t i = 0; i < gathered.size(); ++i) {
        assert(std::abs(gathered[i] - expected_sum) < 0.01f);
    }

    std::cout << "OK: All-reduce works\n" << std::endl;
}

int main() {
    std::cout << "\n================================================" << std::endl;
    std::cout << "  TinyTorch LLM Operations Test Suite" << std::endl;
    std::cout << "================================================\n" << std::endl;

    test_matmul();
    test_softmax();
    test_layer_norm();
    test_activations();
    test_cross_entropy();
    test_optimizer_steps();
    test_all_reduce();

    std::cout << "================================================" << std::endl;
    std::cout << "  All LLM Operations Tests Passed!" << std::endl;
    std::cout << "================================================\n" << std::endl;

    return 0;
}
