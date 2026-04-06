#include "gpu_tensor.h"
#include "cuda_tensor.h"
#include "gpu_kernels.h"
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cmath>

static const size_t NUM_ELEMENTS = 100'000'000;  // 100M floats = ~381 MB

void print_separator(const std::string& title) {
    std::cout << "\n================================================" << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << "================================================\n" << std::endl;
}

void wait_for_user(const std::string& msg) {
    std::cout << msg << std::endl;
    std::cout << "Press Enter to continue..." << std::endl;
    std::cin.get();
}

void test_opengl_gpu() {
    print_separator("OpenGL GPU Test (GpuTensor)");
    auto& selector = get_gpu_selector();
    selector.enumerate();
    selector.print_all(std::cout);
    const auto& gpus = selector.get_gpus();
    if (gpus.empty()) { std::cout << "No OpenGL GPUs found.\n"; return; }

    for (size_t gpu_idx = 0; gpu_idx < gpus.size(); ++gpu_idx) {
        const auto& gpu = gpus[gpu_idx];
        std::cout << "\n--- GPU #" << gpu_idx << ": " << gpu.name << " ---" << std::endl;
        size_t num = NUM_ELEMENTS;
        std::cout << "Allocating " << (num * sizeof(float) / (1024*1024)) << " MB..." << std::endl;
        std::vector<float> data(num);
        for (size_t i = 0; i < num; ++i) data[i] = static_cast<float>(i % 1000) * 0.001f;
        GpuConfig cfg; cfg.device_id = gpu.device_id;
        auto* t = new GpuTensor<float>(data.data(), data.size(), cfg);
        std::vector<float>().swap(data);
        std::cout << "Created: " << t->total_size() << " elements, " << (t->total_size()*sizeof(float)/(1024*1024)) << " MB" << std::endl;
        std::cout << "Backend: " << t->backend_name() << std::endl;
        std::cout << "Verifying..." << std::endl;
        bool ok = true;
        size_t indices[] = {0, 1000, num/2, num-1};
        for (size_t idx : indices) {
            float val = t->get_element(idx);
            float expected = static_cast<float>(idx % 1000) * 0.001f;
            if (std::abs(val - expected) > 1e-5f) { std::cout << "  MISMATCH at " << idx << std::endl; ok = false; }
        }
        if (ok) std::cout << "  All verified!" << std::endl;
        std::cout << "Running compute (multiply 2.0)..." << std::endl;
        auto result = t->multiply_scalar(2.0f);
        std::cout << "  Result[0] = " << result->get_element(0) << std::endl;
        wait_for_user("GPU memory allocated. Check Task Manager / GPU-Z.");
        delete t;
        std::cout << "Tensor deleted.\n" << std::endl;
    }
}

void test_cuda_gpu() {
    print_separator("CUDA GPU Test (CudaTensor)");
    auto& F = get_cuda_driver_functions();
    if (!F.is_loaded()) { std::cout << "CUDA not found.\n"; return; }
    F.cuInit(0);
    int count = 0; F.cuDeviceGetCount(&count);
    std::cout << "Found " << count << " CUDA device(s)" << std::endl;
    if (count == 0) return;

    for (int dev = 0; dev < count; ++dev) {
        char name[256] = {0}; F.cuDeviceGetName(name, sizeof(name), dev);
        std::cout << "\n--- CUDA Device #" << dev << ": " << name << " ---" << std::endl;
        size_t num = NUM_ELEMENTS;
        std::cout << "Allocating " << (num * sizeof(float) / (1024*1024)) << " MB..." << std::endl;
        std::vector<float> data(num);
        for (size_t i = 0; i < num; ++i) data[i] = static_cast<float>(i % 1000) * 0.001f;
        auto* t = CudaTensor::from_data(data, dev);
        std::vector<float>().swap(data);
        if (!t) { std::cout << "Failed to create CudaTensor.\n"; continue; }
        std::cout << "Created: " << t->size() << " elements, " << (t->size()*sizeof(float)/(1024*1024)) << " MB" << std::endl;
        std::cout << "Backend: " << t->backend_name() << std::endl;
        std::cout << "Verifying..." << std::endl;
        bool ok = true;
        size_t indices[] = {0, 1000, num/2, num-1};
        for (size_t idx : indices) {
            float val = t->get_element(idx);
            float expected = static_cast<float>(idx % 1000) * 0.001f;
            if (std::abs(val - expected) > 1e-3f) { std::cout << "  MISMATCH at " << idx << std::endl; ok = false; }
        }
        if (ok) std::cout << "  All verified!" << std::endl;
        std::cout << "Running compute (multiply 2.0)..." << std::endl;
        auto result = t->multiply_scalar(2.0f);
        std::cout << "  Result[0] = " << result->get_element(0) << std::endl;
        wait_for_user("CUDA memory allocated. Check nvidia-smi.");
        delete t;
        std::cout << "Tensor deleted.\n" << std::endl;
    }
}

void test_opencl_gpu() {
    print_separator("OpenCL GPU Test (OpenClTensor)");
    auto& cl_mgr = get_opencl_manager();
    if (!cl_mgr.initialize()) { std::cout << "OpenCL not found.\n"; return; }
    const auto& devices = cl_mgr.devices();
    std::cout << "Found " << devices.size() << " OpenCL device(s)" << std::endl;
    for (size_t i = 0; i < devices.size(); ++i)
        std::cout << "  #" << i << ": " << devices[i].name << " (" << devices[i].vendor << ")" << std::endl;
    if (devices.empty()) return;

    for (size_t dev_idx = 0; dev_idx < devices.size(); ++dev_idx) {
        const auto& dev_info = devices[dev_idx];
        std::cout << "\n--- OpenCL Device #" << dev_idx << ": " << dev_info.name << " ---" << std::endl;
        size_t num = NUM_ELEMENTS;
        std::cout << "Allocating " << (num * sizeof(float) / (1024*1024)) << " MB..." << std::endl;
        std::vector<float> data(num);
        for (size_t i = 0; i < num; ++i) data[i] = static_cast<float>(i % 1000) * 0.001f;
        cl_int err;
        cl_context context = clCreateContext(nullptr, 1, &dev_info.device, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) { std::cout << "Failed to create context.\n"; continue; }
        cl_command_queue queue = clCreateCommandQueue(context, dev_info.device, 0, &err);
        if (err != CL_SUCCESS) { clReleaseContext(context); continue; }
        auto* t = OpenClTensor::from_data(data, dev_info.device, context, queue, true);
        std::vector<float>().swap(data);
        if (!t) { std::cout << "Failed to create OpenClTensor.\n"; continue; }
        std::cout << "Created: " << t->size() << " elements, " << (t->size()*sizeof(float)/(1024*1024)) << " MB" << std::endl;
        std::cout << "Backend: " << t->backend_name() << std::endl;
        std::cout << "Verifying..." << std::endl;
        bool ok = true;
        size_t indices[] = {0, 1000, num/2, num-1};
        for (size_t idx : indices) {
            float val = t->get_element(idx);
            float expected = static_cast<float>(idx % 1000) * 0.001f;
            if (std::abs(val - expected) > 1e-3f) { std::cout << "  MISMATCH at " << idx << std::endl; ok = false; }
        }
        if (ok) std::cout << "  All verified!" << std::endl;
        std::cout << "Running compute (multiply 2.0)..." << std::endl;
        auto result = t->multiply_scalar(2.0f);
        std::cout << "  Result[0] = " << result->get_element(0) << std::endl;
        wait_for_user("OpenCL memory allocated. Check Task Manager.");
        delete t;
        std::cout << "Tensor deleted.\n" << std::endl;
    }
}

void test_all_gpus_simultaneously() {
    print_separator("All GPUs Simultaneous Test");
    std::cout << "Allocating ~381 MB on ALL available GPUs at once." << std::endl;
    std::vector<TensorBase<float>*> gpu_tensors;
    std::vector<size_t> gpu_sizes;

    // OpenGL
    auto& selector = get_gpu_selector();
    selector.enumerate();
    const auto& gpus = selector.get_gpus();
    for (size_t i = 0; i < gpus.size(); ++i) {
        std::cout << "OpenGL GPU #" << i << ": " << gpus[i].name << std::endl;
        std::vector<float> data(NUM_ELEMENTS, 1.0f);
        GpuConfig cfg; cfg.device_id = gpus[i].device_id;
        auto* t = new GpuTensor<float>(data.data(), data.size(), cfg);
        std::vector<float>().swap(data);
        if (t) { gpu_tensors.push_back(t); gpu_sizes.push_back(NUM_ELEMENTS); }
    }

    // CUDA
    auto& F = get_cuda_driver_functions();
    if (F.is_loaded()) {
        F.cuInit(0);
        int count = 0; F.cuDeviceGetCount(&count);
        for (int dev = 0; dev < count; ++dev) {
            char name[256] = {0}; F.cuDeviceGetName(name, sizeof(name), dev);
            std::cout << "CUDA Device #" << dev << ": " << name << std::endl;
            std::vector<float> data(NUM_ELEMENTS, 2.0f);
            auto* t = CudaTensor::from_data(data, dev);
            std::vector<float>().swap(data);
            if (t) { gpu_tensors.push_back(t); gpu_sizes.push_back(NUM_ELEMENTS); }
        }
    }

    // OpenCL
    auto& cl_mgr = get_opencl_manager();
    if (cl_mgr.initialize()) {
        const auto& devices = cl_mgr.devices();
        for (size_t dev_idx = 0; dev_idx < devices.size(); ++dev_idx) {
            const auto& dev_info = devices[dev_idx];
            std::cout << "OpenCL Device #" << dev_idx << ": " << dev_info.name << std::endl;
            std::vector<float> data(NUM_ELEMENTS, 3.0f);
            cl_int err;
            cl_context context = clCreateContext(nullptr, 1, &dev_info.device, nullptr, nullptr, &err);
            if (err != CL_SUCCESS) continue;
            cl_command_queue queue = clCreateCommandQueue(context, dev_info.device, 0, &err);
            if (err != CL_SUCCESS) { clReleaseContext(context); continue; }
            auto* t = OpenClTensor::from_data(data, dev_info.device, context, queue, true);
            std::vector<float>().swap(data);
            if (t) { gpu_tensors.push_back(t); gpu_sizes.push_back(NUM_ELEMENTS); }
        }
    }

    std::cout << "\nTotal GPU tensors: " << gpu_tensors.size() << std::endl;
    size_t total_mb = 0;
    for (size_t s : gpu_sizes) total_mb += s * sizeof(float) / (1024*1024);
    std::cout << "Total GPU memory: ~" << total_mb << " MB" << std::endl;
    wait_for_user("ALL GPU memory allocated. Check Task Manager / nvidia-smi.");
    std::cout << "Cleaning up..." << std::endl;
    for (auto* t : gpu_tensors) delete t;
    std::cout << "All GPU memory freed.\n" << std::endl;
}

int main() {
    std::cout << "\n================================================" << std::endl;
    std::cout << "  TinyTorch GPU Memory Verification" << std::endl;
    std::cout << "  Large Tensor Allocation Test (~381 MB each)" << std::endl;
    std::cout << "================================================\n" << std::endl;
    std::cout << "Open your GPU monitoring tool (Task Manager, nvidia-smi, GPU-Z) now." << std::endl;
    std::cout << "Press Enter to begin..." << std::endl;
    std::cin.get();

    test_opengl_gpu();
    test_cuda_gpu();
    test_opencl_gpu();
    test_all_gpus_simultaneously();

    std::cout << "\n================================================" << std::endl;
    std::cout << "  GPU Memory Verification Complete!" << std::endl;
    std::cout << "================================================\n" << std::endl;
    return 0;
}
