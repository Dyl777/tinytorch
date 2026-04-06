#include "gpu_kernels.h"
#include <iostream>

int main() {
    GpuKernelTestSuite suite;
    suite.run_all_tests(100000);  // Test with 100K elements
    return 0;
}
