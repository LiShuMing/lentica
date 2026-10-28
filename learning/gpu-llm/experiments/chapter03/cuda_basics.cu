#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << " - " << cudaGetErrorString(err) << std::endl;      \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                    \
    } while (0)

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

template <int TILE>
__global__ void matmul_tiled_kernel(const float* a, const float* b, float* c, int n) {
    __shared__ float tile_a[TILE][TILE];
    __shared__ float tile_b[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int tile = 0; tile < (n + TILE - 1) / TILE; ++tile) {
        int a_col = tile * TILE + threadIdx.x;
        int b_row = tile * TILE + threadIdx.y;

        tile_a[threadIdx.y][threadIdx.x] = (row < n && a_col < n) ? a[row * n + a_col] : 0.0f;
        tile_b[threadIdx.y][threadIdx.x] = (b_row < n && col < n) ? b[b_row * n + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

void run_vector_add() {
    const int n = 1 << 20;
    const int bytes = n * static_cast<int>(sizeof(float));
    const int threads_per_block = 256;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;

    std::vector<float> h_a(n);
    std::vector<float> h_b(n);
    std::vector<float> h_c(n, 0.0f);

    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i) * 0.5f;
        h_b[i] = static_cast<float>(i) * 2.0f;
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    vector_add_kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    float max_abs_error = 0.0f;
    for (int i = 0; i < n; ++i) {
        float expected = h_a[i] + h_b[i];
        max_abs_error = std::max(max_abs_error, std::fabs(h_c[i] - expected));
    }

    std::cout << "Vector add" << std::endl;
    std::cout << "  n=" << n << ", blockDim.x=" << threads_per_block
              << ", gridDim.x=" << blocks << std::endl;
    std::cout << "  idx = blockIdx.x * blockDim.x + threadIdx.x" << std::endl;
    std::cout << "  example: blockIdx.x=1, threadIdx.x=17 -> idx="
              << 1 * threads_per_block + 17 << std::endl;
    std::cout << "  c[123]=" << h_c[123] << ", expected=" << h_a[123] + h_b[123]
              << std::endl;
    std::cout << "  max_abs_error=" << max_abs_error << std::endl;

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

void run_matrix_multiply() {
    constexpr int n = 64;
    constexpr int tile = 16;
    const int elements = n * n;
    const int bytes = elements * static_cast<int>(sizeof(float));

    std::vector<float> h_a(elements);
    std::vector<float> h_b(elements);
    std::vector<float> h_c(elements, 0.0f);

    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            h_a[row * n + col] = static_cast<float>((row + col) % 7);
            h_b[row * n + col] = static_cast<float>((row == col) ? 1.0f : 0.0f);
        }
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(tile, tile);
    dim3 grid((n + tile - 1) / tile, (n + tile - 1) / tile);
    matmul_tiled_kernel<tile><<<grid, block>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    float max_abs_error = 0.0f;
    for (int i = 0; i < elements; ++i) {
        max_abs_error = std::max(max_abs_error, std::fabs(h_c[i] - h_a[i]));
    }

    std::cout << "Tiled matrix multiplication" << std::endl;
    std::cout << "  n=" << n << "x" << n << ", tile=" << tile << "x" << tile
              << ", grid=(" << grid.x << "," << grid.y << ")" << std::endl;
    std::cout << "  B is identity, so C should equal A" << std::endl;
    std::cout << "  c[3,5]=" << h_c[3 * n + 5] << ", expected=" << h_a[3 * n + 5]
              << std::endl;
    std::cout << "  max_abs_error=" << max_abs_error << std::endl;

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

int main() {
    int device = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    CUDA_CHECK(cudaSetDevice(device));

    std::cout << "CUDA device: " << prop.name << std::endl;
    std::cout << std::endl;

    run_vector_add();
    std::cout << std::endl;
    run_matrix_multiply();

    return 0;
}
