#include "gpu_backends.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

int main() {
    std::cout << "\n================================================" << std::endl;
    std::cout << "  TinyTorch Multi-Backend GPU Test Suite" << std::endl;
    std::cout << "  (OpenGL + CUDA + OpenCL)" << std::endl;
    std::cout << "================================================\n" << std::endl;

    // ========================================================================
    // 1. Unified GPU Enumeration
    // ========================================================================
    std::cout << "=== Unified GPU Enumeration ===" << std::endl;

    auto& selector = get_unified_gpu_selector();
    bool found = selector.enumerate();

    if (!found) {
        std::cout << "[WARN] No GPUs detected across any backend" << std::endl;
        std::cout << "  This is expected if CUDA/OpenCL runtimes are not installed" << std::endl;
    } else {
        selector.print_all(std::cout);
        std::cout << std::endl;

        // Auto-select best GPU
        selector.auto_select();
        std::cout << "Auto-selected: GPU #" << selector.selected_device() << std::endl;

        const auto* selected = selector.selected_gpu();
        if (selected) {
            std::cout << "Selected GPU: " << selected->name << " (" << selected->vendor << ")" << std::endl;
            std::cout << "Backend: " << (selected->backend == GpuBackend::CUDA ? "CUDA" :
                                         selected->backend == GpuBackend::OpenCL ? "OpenCL" : "OpenGL") << std::endl;
        }
    }

    std::cout << std::endl;

    // ========================================================================
    // 2. OpenCL Backend Test
    // ========================================================================
    std::cout << "=== OpenCL Backend ===" << std::endl;
    auto& cl_mgr = get_opencl_manager();
    if (cl_mgr.initialize()) {
        cl_mgr.print_devices(std::cout);
        cl_mgr.auto_select();
        std::cout << "Selected OpenCL device: #" << cl_mgr.selected_device() << std::endl;
    } else {
        std::cout << "[SKIP] OpenCL not available (install OpenCL runtime)" << std::endl;
    }
    std::cout << std::endl;

    // ========================================================================
    // 3. CUDA Backend Test
    // ========================================================================
    std::cout << "=== CUDA Backend ===" << std::endl;
    auto& cuda_mgr = get_cuda_manager();
    if (cuda_mgr.initialize()) {
        cuda_mgr.print_devices(std::cout);
        cuda_mgr.auto_select();
        std::cout << "Selected CUDA device: #" << cuda_mgr.selected_device() << std::endl;
    } else {
        std::cout << "[SKIP] CUDA not available (install CUDA toolkit)" << std::endl;
    }
    std::cout << std::endl;

    // ========================================================================
    // 4. OpenGL Backend Test
    // ========================================================================
    std::cout << "=== OpenGL Backend ===" << std::endl;
    auto& gl_selector = get_gpu_selector();
    if (gl_selector.get_gpus().empty()) gl_selector.enumerate();
    if (!gl_selector.get_gpus().empty()) {
        gl_selector.print_all(std::cout);
        gl_selector.auto_select();
        std::cout << "Selected OpenGL device: #" << gl_selector.selected_device() << std::endl;
    } else {
        std::cout << "[SKIP] No OpenGL 4.3+ devices found" << std::endl;
    }
    std::cout << std::endl;

    // ========================================================================
    // 5. GPU Selection Tests
    // ========================================================================
    std::cout << "=== GPU Selection Tests ===" << std::endl;

    if (!selector.gpus().empty()) {
        // Select by name
        bool found_nvidia = selector.select_by_name("MX130") ||
                            selector.select_by_name("GeForce") ||
                            selector.select_by_name("NVIDIA");
        if (found_nvidia) {
            std::cout << "Selected NVIDIA GPU by name: #" << selector.selected_device() << std::endl;
        } else {
            std::cout << "No NVIDIA GPU found by name search" << std::endl;
        }

        // Select discrete only
        selector.select_discrete_only();
        std::cout << "Discrete-only selection: #" << selector.selected_device() << std::endl;
    }

    std::cout << std::endl;

    // ========================================================================
    // Summary
    // ========================================================================
    std::cout << "================================================" << std::endl;
    std::cout << "  Summary:" << std::endl;
    std::cout << "  - OpenCL: " << (cl_mgr.is_available() ? "Available" : "Not available") << std::endl;
    std::cout << "  - CUDA:   " << (cuda_mgr.is_available() ? "Available" : "Not available") << std::endl;
    std::cout << "  - OpenGL: " << (gl_selector.get_gpus().empty() ? "Not available" : "Available") << std::endl;
    std::cout << "  - Total GPUs detected: " << selector.gpus().size() << std::endl;
    std::cout << "================================================\n" << std::endl;

    return 0;
}
