#include <hip/hip_runtime.h>

#include <cassert>
#include <iostream>
#include <vector>

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t error = (call);                                              \
        if (error != hipSuccess) {                                              \
            std::cerr << "HIP error: " << hipGetErrorString(error) << "\n";    \
            std::exit(1);                                                       \
        }                                                                      \
    } while (0)

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    constexpr int n = 1024;
    constexpr size_t bytes = n * sizeof(float);

    std::vector<float> host_a(n);
    std::vector<float> host_b(n);
    std::vector<float> host_c(n);

    for (int i = 0; i < n; ++i) {
        host_a[i] = static_cast<float>(i);
        host_b[i] = static_cast<float>(i * 2);
    }

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;

    HIP_CHECK(hipMalloc(&device_a, bytes));
    HIP_CHECK(hipMalloc(&device_b, bytes));
    HIP_CHECK(hipMalloc(&device_c, bytes));

    HIP_CHECK(hipMemcpy(device_a, host_a.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_b, host_b.data(), bytes, hipMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(device_a, device_b, device_c, n);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(host_c.data(), device_c, bytes, hipMemcpyDeviceToHost));

    for (int i = 0; i < n; ++i) {
        assert(host_c[i] == host_a[i] + host_b[i]);
    }

    HIP_CHECK(hipFree(device_a));
    HIP_CHECK(hipFree(device_b));
    HIP_CHECK(hipFree(device_c));

    std::cout << "ok: " << n << " HIP additions verified\n";
    return 0;
}
