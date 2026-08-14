#ifndef GEODE_FLOPS_H
#define GEODE_FLOPS_H

/* Returns achieved aggregate FP32 FLOPS/s across all online CPUs, 0 on
   failure. *q4k_dequant_flops receives the effective useful FLOPS/s of a
   Q4_K dequant+dot kernel (0 when the cpu lacks AVX2/FMA/F16C), and
   *threads the worker count used. */
double flops_probe(int *threads, double *q4k_dequant_flops);

#endif
