#ifndef _KERNEL_H_
#define _KERNEL_H_

#ifdef __cplusplus
extern "C" {
#endif

void launch_vector_addition(const float* a, const float* b, float* c, int n);
void launch_stencil_1d(const int* in, int* out, int block_size, int radius);

#ifdef __cplusplus
}
#endif

#endif  // _KERNEL_H_