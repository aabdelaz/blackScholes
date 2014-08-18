//#define DOPRINT 

// A is the matrix, i is the column index, and m is the # of rows
#define COL(A, i, m) (A + (i)*m)

// A is the matrix, i is the row index, and n is the # of columns 
#define ROW(A, i) (A + i)

// A is the matrix, i is the row number, j is the column number, and m is the # of rows
#define ENTRY(A, i, j, m) (A + (j)*m + i)

#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>

#include "common.h"

static float *V;
float *H, *p;

// defined in eq 2.4, p. 3
// never gets past the if statement in our examples 
float pi_epsilon(const float y, const float ep) {
    if (y >= ep) return y;
    if (y <= -ep) return 0;

    float c0, c1, c2, c4, c6, c8;
    float e2 = ep*ep;
    float e3 = e2*ep;
    c0 = 35./256*ep;
    c1 = 0.5;
    c2 = 35./(64*ep);
    c4 = -35./(128*e3);
    c6 = 7./(64*e3*e2);
    c8 = -5./(256*e3*e3*ep);

    float y2 = y*y;
    float y4 = y2*y2;
    float y6 = y4*y2;
    float y8 = y4*y4;

    return c0 + c1*y + c2*y2 + c4*y4 + c6*y6 + c8*y8; 

}

// volatility; given t, returns a vector containing the volatility at that time
// and xi

inline float sigma(const float t, const float x) {
  return 0.2 + 0.2*(1 - t)*((x/25 - 1.2)*(x/25 - 1.2)/((x/25)*(x/25) + 1.44));
} 

// returns beta; fills in H and V
// I'll assume that these are all device pointers and that cuda has been initialized properly,
// etc: except for H, which is a host pointer, for convenience (to me, but probably not to anyone
// else reading this)
// Also, I'm assuming that V is zeroed out; it will mess me up if it isn't  
float arnoldi(cublasHandle_t handle, const float *A, const float *v, const int n, const int m, float *H_h) {

    cublasStatus_t status;
    cudaError_t error;
    float beta, beta_inv, h_ij, one = 1, zero = 0;
    int i, j, k;
    status = cublasSnrm2(handle, n, v, 1, &beta);
    checkCublasStatus(status);

    // so beta is correct

    beta_inv = 1./beta;

    // V.col(0) = v/beta;
    status = cublasSaxpy(handle, n, &beta_inv, v, 1, V, 1);
    checkCublasStatus(status);
    
    for (j = 0; j < m - 1; j++) {
        // p = A*V.col(j);
        status = cublasSgemv(handle, CUBLAS_OP_N, n, n, &one, A, n, COL(V, j, n), 1, &zero, p, 1);
        checkCublasStatus(status);
        for (i = 0; i <= j; i++) {
            // H(i, j) = cdot(V.col(i), p);
            status = cublasSdot(handle, n, p, 1, COL(V, i, n), 1, ENTRY(H_h, i, j, m));
            checkCublasStatus(status);
            // p -= H(i, j)*V.col(i);
            h_ij = -(*ENTRY(H_h, i, j, m));
            status = cublasSaxpy(handle, n, &h_ij, COL(V, i, n), 1, p, 1); 
            checkCublasStatus(status);
        }   
        // so p is correct when j == 0, as is norm(p)
 
        // H(j + 1, j) = norm(p);
        status = cublasSnrm2(handle, n, p, 1, ENTRY(H_h, j + 1, j, m));
        checkCublasStatus(status);

        h_ij = 1./(*(ENTRY(H_h, j + 1, j, m))); 
        // V.col(j + 1) = p/H(j + 1, j);
        status = cublasSaxpy(handle, n, &h_ij, p, 1, COL(V, j + 1, n), 1);
        checkCublasStatus(status);
    }

    // p = A*V.col(m - 1);
    status = cublasSgemv(handle, CUBLAS_OP_N, n, n, &one, A, n, COL(V, m - 1, n), 1, &zero, p, 1);
    checkCublasStatus(status);

    for (i = 0; i <= m - 1; i++) {
        // H(i, m - 1) = cdot(V.col(i), p);
        status = cublasSdot(handle, n, p, 1, COL(V, i, n), 1, ENTRY(H_h, i, m - 1, m));
        checkCublasStatus(status);

        // p -= H(i, m - 1)*V.col(i);
        h_ij = -(*ENTRY(H_h, i, m - 1, m));
        status = cublasSaxpy(handle, n, &h_ij, COL(V, i, n), 1, p, 1); 
        checkCublasStatus(status);
    }

    return beta;
}

// TODO: Can overwrite H with E in expcuda.c, since we only need it as input to that function
// actually not anymore, with phi as it is
// MAKE SURE THAT I KEEP TRACK OF DEVICE/HOST PTRS
// all the pointers passed here are device pointers 
void krylov(cublasHandle_t handle, float *w, int m, float l, const float *A, int n, const float *v, int expo) {
    // remember to zero out V 
    cudaError_t error;
    cublasStatus_t status;
    float *H_h = 0, *E = 0, zero = 0, beta;
    int i, j;

    error = cudaMemset((void*)V, 0, n*m*sizeof(V[0]));
    checkError(error);

    H_h = (float *)malloc(m*m*sizeof(H_h[0]));
    if (H_h == 0) {
        fprintf(stderr, "Malloc of H failed\n");
        exit(1);
    }

    memset((void*)H_h, 0, m*m*sizeof(H_h[0]));
    beta = arnoldi(handle, A, v, n, m, H_h);

    error = cudaMalloc((void**)&E, m*m*sizeof(E[0]));
    checkError(error);

    error = cudaMemcpy(H, H_h, m*m*sizeof(H_h[0]), cudaMemcpyHostToDevice);
    checkError(error); 

    // scale H by l
    status = cublasSscal(handle, m*m, &l, H, 1); 
    checkCublasStatus(status);

    if (expo == 1) padeExp(handle, H, E, m);
    else phi(handle, H, E, m);

    // w = beta*V*matrix_exp(l*H)*e_0;
    // so instead of having e_0, I can calculate the product w/o it, and 
    // copy the first row of it into w  

    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, m, &beta, V, n, E, m, &zero, V, n);
    checkCublasStatus(status);

    // get first row of bVe and copy it into w
    status = cublasScopy(handle, n, V, 1, w, 1);  
    checkCublasStatus(status);

    // free everything
    free(H_h);

    error = cudaFree(E);
    checkError(error);
}

int main(int argc, char *argv[]) {

    // we are going to need a vector of x_i's
    // which depend on S_max, K, and N
    // we'll hardcode those here
    cudaError_t error;
    cublasStatus_t status;
    cublasHandle_t handle;
    culaStatus culaS; 

    //clock_t begin, end;
    //double time_spent;

    // k, confusingly named, is the dimensions to be projected upon
    FILE *fp;
    int k = 50;
    float K = 25;
    float S_max = 4*K;
    int N = atoi(argv[1]);
    int M = 200;
    float epsilon = .0001;
    float r = .06;
    char filename[100];
    sprintf(filename,"outputs/cuda%d_%d.txt", M, N);

    // l is the time step
    float l = 1./(M - 1);

    // h is the x tep
    float h = S_max/(N - 1);
 
    float *U = 0, *U_h, *A;
    float zero = 0;
    int i, j;
    float t, *a, *b, *c, *f, *v, *w1, *w2;

    //begin = clock();

    status = cublasCreate(&handle);
    checkCublasStatus(status);

    culaS = culaInitialize();
    checkCulaStatus(culaS);

    // we will have a matrix U, where the ij^th entry is the value of the option at time t = t_i 
    // and x = x_j
    // the dimensions of this matrix will be M*N
    // M is the time axis
    // N is the x axis
    // We keep U as device storage, because we don't need it till the end
    error = cudaMalloc((void **)&U, M*N*sizeof(U[0]));
    checkError(error);

    U_h = (float*)malloc(M*N*sizeof(U_h[0]));
    if (U_h == 0) {
        fprintf(stderr, "Error with malloc of U_h\n");
        exit(1);
    }

    error = cudaMalloc((void **)&p, N*sizeof(p[0]));
    checkError(error);

    error = cudaMalloc((void **)&H, k*k*sizeof(H[0]));
    checkError(error);

    error = cudaMalloc((void **)&V, k*N*sizeof(V[0]));
    checkError(error);

    // fill with zeros
    error = cudaMemset(U, 0, M*N*sizeof(U[0]));
    checkError(error);

    // we can determine the values of the U_0j^th entries
    // This could be done in ||, but it's only done once so its 
    // probably not worth it
    for (j = 0; j < N; j++) {
        *ENTRY(U_h, 0, j, M) = pi_epsilon(h*j - K, epsilon);
    }
    // copy row of U_h to U 
    status = cublasSetVector(N, sizeof(U_h[0]), U_h, M, U, M);  
    checkCublasStatus(status);  

    // now we need to fill in the A matrix, which is a function of t 
    // so let's loop over t
    // we'll allocate A only once, same for a, b, and c, except we don't have 
    // to zero those out
    error = cudaMalloc((void **)&A, N*N*sizeof(A[0]));
    checkError(error);

    error = cudaMemset((void*)A, 0, N*N*sizeof(A[0]));
    checkError(error);

    error = cudaMalloc((void **)&f, N*sizeof(f[0]));
    checkError(error);

    error = cudaMemset((void*)f, 0, N*sizeof(f[0]));
    checkError(error);
    
    error = cudaMalloc((void **)&a, N*N*sizeof(a[0]));
    checkError(error);

    error = cudaMalloc((void **)&b, N*N*sizeof(b[0]));
    checkError(error);

    error = cudaMalloc((void **)&c, N*N*sizeof(c[0]));
    checkError(error);

    error = cudaMalloc((void **)&v, N*sizeof(v[0]));
    checkError(error);

    error = cudaMalloc((void **)&w1, N*sizeof(w1[0]));
    checkError(error);

    error = cudaMalloc((void **)&w2, N*sizeof(w2[0]));
    checkError(error);

    int blockSize = 16;
    int threadsPerBlock = blockSize*blockSize;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    dim3 threads(blockSize, blockSize);
    dim3 grid((N + blockSize - 1)/blockSize, (N + blockSize - 1)/blockSize);

    for (i = 0; i < M - 1; i++) {
        t = i*l;

        initabc<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, N, t, h);
        initA<<<grid, threads>>>(a, b, c, A, N);
  
        // now we need f(t) at x_0 and x_N
        // see 2.14
        float sigSq = sigma(t, (N-1)*h);
        sigSq *= sigSq; 
        float cN = sigSq*(N-1)*(N-1)/2. + r*(N-1)/2.;

        float f_N = (S_max - K*exp(-t*r))*cN;

        status = cublasSetVector(1, sizeof(f_N), &f_N, 1, f + N - 1, 1); 
        checkCublasStatus(status);

        status = cublasScopy(handle, N, ROW(U, i), M, v, 1);
        checkCublasStatus(status);

        // so A and v are correct; the issue is something inside krylov
        krylov(handle, w1, k, l, A, N, v, 1);
        krylov(handle, w2, k, l, A, N, f, 0);

        status = cublasSaxpy(handle, N, &l, w2, 1, w1, 1);
        checkCublasStatus(status);

        status = cublasScopy(handle, N, w1, 1, ROW(U, i + 1), M);
        checkCublasStatus(status);
    }


    #ifdef DOPRINT
    // copy to U_h
    status = cublasGetVector(M*N, sizeof(U[0]), U, 1, U_h, 1);
    fp = fopen(filename, "w");
    for (i = 0; i < M; i++) {
        float t = i*l;
        for (j = 0; j < N; j++) {
            float x = j*h;
           fprintf(fp, "%f %f %f\n", t, x, *ENTRY(U_h, i, j, M));
        }
    }

    #endif 

    // cleanup
    free(U_h);

    error = cudaFree(p);
    checkError(error);

    error = cudaFree(f);
    checkError(error);

    error = cudaFree(A);
    checkError(error);

    error = cudaFree(V);
    checkError(error);

    error = cudaFree(H);
    checkError(error);

    error = cudaFree(U);
    checkError(error);

    error = cudaFree(v);
    checkError(error);

    error = cudaFree(w1);
    checkError(error);

    error = cudaFree(w2);
    checkError(error);

    error = cudaFree(a);
    checkError(error);

    error = cudaFree(b);
    checkError(error);

    error = cudaFree(c);
    checkError(error);

    status = cublasDestroy(handle);
    checkCublasStatus(status);
    culaShutdown();

    //end = clock();
    //time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    //printf("%d %f\n", N, time_spent); 

    return 0;

}

