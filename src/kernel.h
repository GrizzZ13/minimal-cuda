#ifndef KERNEL_H
#define KERNEL_H

#ifdef __cplusplus
extern "C" {
#endif

void vectorAdd(const float* a, const float* b, float* c, int n);

#ifdef __cplusplus
}
#endif

#endif  // KERNEL_H