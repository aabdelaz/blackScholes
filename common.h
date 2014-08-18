#ifndef COMMONH
#define COMMONH

#define MAX(a,b) ({ typeof (a) _a = (a); \
                   typeof (b) _b = (b); \
                   _a > _b ? _a : _b; })

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math_functions.h>
#include <cula_blas.hpp>
#include <cula.hpp>

__global__ void initabc(float *a, float *b, float *c, int n, float t, float h);
__global__ void initA(float *a, float *b, float *c, float *A, int n);
__global__ void findScale(float *p, int idx);
__global__ void addDiag(float *A, int n, int c);
__global__ void addRows(float *A, float *sums, int n);

void pm(float *A, int n);
void checkCublasStatus(cublasStatus_t status);
void checkError(cudaError_t error);
void checkCulaStatus(culaStatus status);
void padeExp(cublasHandle_t handle, float *A, float *E, int n);
void phi(cublasHandle_t handle, float *A, float *E, int n);

#endif

