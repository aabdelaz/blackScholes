#include "common.h"

// volatility; given t, returns a vector containing the volatility at that time
// and x
__device__ float sigma(float t, float x) {
  return 0.2 + 0.2*(1 - t)*((x/25 - 1.2)*(x/25 - 1.2)/((x/25)*(x/25) + 1.44));
}

__global__ void
initabc(float *a, float *b, float *c, int n, float t, float h)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n)
    {
        float r = .06;
        float x = i*h;
        float sigSq = sigma(t, x);
        sigSq *= sigSq;
        
        a[i] = sigSq*i*i/2 - r*i/2;
        b[i] = -sigSq*i*i - r;
        c[i] = sigSq*i*i/2 + r*i/2;
    }
}

__global__ void
initA(float *a, float *b, float *c, float *A, int n)
{
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n && j < n) {
        if (i == j) {
            A[j*n + i] = b[i];
            return;
        }
        if (i == j + 1) {
            A[j*n + i] = a[i];
            return;
        }
        if (i == j - 1) {
            A[j*n + i] = c[i];
            return;
        }
    }

}


__global__ void
addDiag(float *A, int n, int c)
{
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n && j < n) {
        if (i == j) A[j*n + i] += c;
   }
}

__global__ void
addRows(float *A, float *sums, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    int sum = 0;
    if (i < n) {
        for (int j = 0; j < n; j++) { 
            float A_ij = A[j*n + i];
            if (A_ij < 0) A_ij = -A_ij;
            sum += A_ij;
        }
        sums[i] = sum;  
   }
}


__global__ void
findScale(float *p, int idx)
{

    // put s_factor in p[0], s_factor^-1 in p[1]
    if (p[idx] < 0.5) {
        p[0] = 1;
        p[1] = 1;
        return;
    }

    int f = (int) (log(p[idx])/log((float)2));  
    int s = MAX(0, f + 2);
    float s_f = 1.;
    for (int i = 0; i < s; i++) {
        s_f /= 2;
    }
    p[0] = s_f;
    p[1] = 1./s_f;
}



