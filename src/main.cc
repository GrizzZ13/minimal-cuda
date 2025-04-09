#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

#include "kernel.h"

void test_vector_addition() {
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
    launch_vector_addition(d_a, d_b, d_c, N);

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
}

void test_stencil_1d() {
    const int N = 1024;
    int in[N], out[N];
    size_t size = N * sizeof(int);

    // 初始化数据
    for (int i = 0; i < N; ++i) {
        in[i] = i;
    }

    int *d_in, *d_out;

    // 分配设备内存
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    // 拷贝数据到设备
    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);

    // 调用CUDA函数
    launch_stencil_1d(d_in, d_out, N, 4);

    // 拷贝结果回主机
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

    // 清理设备内存
    cudaFree(d_in);
    cudaFree(d_out);
}

int main() {
    test_vector_addition();
    test_stencil_1d();
    return 0;
}