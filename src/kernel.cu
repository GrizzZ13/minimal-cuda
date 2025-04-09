#include <cuda_runtime.h>

#include <cstdio>

#include "kernel.h"

__global__ void vector_addition_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void launch_vector_addition(const float* d_a, const float* d_b, float* d_c, int n) {
    // 启动核函数
    dim3 blockSize = 256;
    dim3 numBlocks = (n + blockSize.x - 1) / blockSize.x;
    vector_addition_kernel<<<numBlocks, blockSize, 0>>>(d_a, d_b, d_c, n);
    // 检查错误
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
}

__global__ void stencil_1d_kernel(const int* in, int* out, int pixels, int radius) {
    extern __shared__ int temp[];
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x + radius;

    temp[lindex] = in[gindex];
    if (threadIdx.x < radius) {
        temp[lindex - radius] = in[gindex - radius];
        temp[lindex + pixels] = in[gindex + pixels];
    }

    __syncthreads();

    int result = 0;
    for (int i = -radius; i <= radius; i++) {
        result += temp[lindex + i];
    }
    out[gindex] = result;
}

void launch_stencil_1d(const int* in, int* out, int pixels, int radius) {
    int shared_mem = sizeof(int) * (pixels + 2 * radius);
    stencil_1d_kernel<<<1, pixels, shared_mem>>>(in, out, pixels, radius);
    // 检查错误
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
}