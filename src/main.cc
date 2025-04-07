#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

#include "kernel.h"

int main() {
    const int N = 1024;
    float a[N], b[N], c[N];
    size_t size = N * sizeof(float);

    // 初始化数据
    for (int i = 0; i < N; ++i) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    float *d_a, *d_b, *d_c;

    // 分配设备内存
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // 拷贝数据到设备
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // 调用CUDA函数
    vectorAdd(d_a, d_b, d_c, N);

    // 拷贝结果回主机
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // 清理设备内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // 验证结果
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(c[i] - 3.0f) > 1e-6) {
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Vector add succeeded!" << std::endl;
    } else {
        std::cerr << "Vector add failed!" << std::endl;
    }

    return 0;
}