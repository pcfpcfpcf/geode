#ifndef GEODE_KERNELS_H
#define GEODE_KERNELS_H

#include <stdint.h>

#define QK_K 256

float gemv_row(const void *data, unsigned type, int n_in, const float *x);
void dequant_row(const void *data, unsigned type, int n, float *dst);

#endif
