#include <cuda_runtime.h>

#include "kernel.h"

__global__ void vectorAddKernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void vectorAdd(const float* d_a, const float* d_b, float* d_c, int n) {
    // 启动核函数
    dim3 blockSize = 256;
    dim3 numBlocks = (n + blockSize.x - 1) / blockSize.x;
    vectorAddKernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
}
