#define DOPRINT 

// A is the matrix, i is the column index, and m is the # of rows
#define COL(A, i, m) (A + (i)*m)

// A is the matrix, i is the row index, and n is the # of columns 
#define ROW(A, i) (A + i)

// A is the matrix, i is the row number, j is the column number, and m is the # of rows
#define ENTRY(A, i, j, m) (A + (j)*m + i)

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cula_blas.h>
#include <cula_lapack.h>

float *I;

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
// and x
inline float sigma(const float t, const float x) {
  return 0.2 + 0.2*(1 - t)*((x/25 - 1.2)*(x/25 - 1.2)/((x/25)*(x/25) + 1.44));
} 

// returns beta; fills in H and V
// I'll assume that these are all device pointers and that cuda has been initialized properly,
// etc: except for H, which is a host pointer, for convenience (to me, but probably not to anyone
// else reading this)
// Also, I'm assuming that V is zeroed out; it will mess me up if it isn't  
float arnoldi(cublasHandle_t handle, const float *A, const float *v, const int n, float *V, 
const int m, float *H) {

    cublasStatus_t status;
    cudaError_t error;
    float beta, beta_inv, h_ij, one = 1, zero = 0;
    float *p = 0;
    int i, j, k;
    status = cublasSnrm2(handle, n, v, 1, &beta);
    checkCublasStatus(status);

    // so beta is correct

    beta_inv = 1./beta;
    // V.col(0) = v/beta;
    status = cublasSaxpy(handle, n, &beta_inv, v, 1, V, 1);
    checkCublasStatus(status);
    
    // allocate p here? or do it outside and pass as a parameter? Since p 
    // is a local variable, I'll do it here, even if it's a little less efficient
    error = cudaMalloc((void **)&p, n*sizeof(p[0]));
    checkError(error);

    // these two loops seem to work fine, although they eventually accumulate numerical errors
    for (j = 0; j < m - 1; j++) {
        // p = A*V.col(j);
        status = cublasSgemv(handle, CUBLAS_OP_N, n, n, &one, A, n, COL(V, j, n), 1, &zero, p, 1);
        checkCublasStatus(status);
        for (i = 0; i <= j; i++) {
            // H(i, j) = cdot(V.col(i), p);
            status = cublasSdot(handle, n, p, 1, COL(V, i, n), 1, ENTRY(H, i, j, m));
            checkCublasStatus(status);
            // p -= H(i, j)*V.col(i);
            h_ij = -(*ENTRY(H, i, j, m));
            status = cublasSaxpy(handle, n, &h_ij, COL(V, i, n), 1, p, 1); 
            checkCublasStatus(status);
        }   
        // so p is correct when j == 0, as is norm(p)
 
        // H(j + 1, j) = norm(p);
        status = cublasSnrm2(handle, n, p, 1, ENTRY(H, j + 1, j, m));
        checkCublasStatus(status);

        h_ij = 1./(*(ENTRY(H, j + 1, j, m))); 
        // V.col(j + 1) = p/H(j + 1, j);
        status = cublasSaxpy(handle, n, &h_ij, p, 1, COL(V, j + 1, n), 1);
        checkCublasStatus(status);
    }

    // p = A*V.col(m - 1);
    status = cublasSgemv(handle, CUBLAS_OP_N, n, n, &one, A, n, COL(V, m - 1, n), 1, &zero, p, 1);
    checkCublasStatus(status);

    for (i = 0; i <= m - 1; i++) {
        // H(i, m - 1) = cdot(V.col(i), p);
        status = cublasSdot(handle, n, p, 1, COL(V, i, n), 1, ENTRY(H, i, m - 1, m));
        checkCublasStatus(status);

        // p -= H(i, m - 1)*V.col(i);
        h_ij = -(*ENTRY(H, i, m - 1, m));
        status = cublasSaxpy(handle, n, &h_ij, COL(V, i, n), 1, p, 1); 
        checkCublasStatus(status);
    }

    error = cudaFree(p);
    checkError(error);
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
    float *V = 0, *H = 0, *H_d, *E = 0, *bVe = 0, zero = 0, beta;
    int i, j;

    error = cudaMalloc((void **)&V, n*m*sizeof(V[0]));
    checkError(error);

    error = cudaMemset((void*)V, 0, n*m*sizeof(V[0]));
    checkError(error);

    H = (float *)malloc(m*m*sizeof(H[0]));
    if (H == 0) {
        fprintf(stderr, "Malloc of H or E failed\n");
        exit(1);
    }

    memset((void*)H, 0, m*m*sizeof(H[0]));
    beta = arnoldi(handle, A, v, n, V, m, H);

    error = cudaMalloc((void**)&E, m*m*sizeof(E[0]));
    checkError(error);
    
    error = cudaMalloc((void**)&H_d, m*m*sizeof(H_d[0]));  
    checkError(error);
 
    error = cudaMemcpy(H_d, H, m*m*sizeof(H[0]), cudaMemcpyHostToDevice);
    checkError(error); 

    // scale H by l
    status = cublasSscal(handle, m*m, &l, H_d, 1); 
    checkCublasStatus(status);

    status = cudaMalloc((void**)&bVe, m*n*sizeof(bVe[0]));
    checkCublasStatus(status);

    if (expo == 1) padeExp(handle, H_d, E, m);
    else phi(handle, H_d, E, m);

    // w = beta*V*matrix_exp(l*H)*e_0;
    // so instead of having e_0, I can calculate the product w/o it, and 
    // copy the first row of it into w  

    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, m, &beta, V, n, E, m, &zero, bVe, n); 
    checkCublasStatus(status);

    // get first row of bVe and copy it into w
    status = cublasScopy(handle, n, bVe, 1, w, 1);  
    checkCublasStatus(status);

    // free everything
    free(H);

    error = cudaFree(bVe);
    checkError(error);

    error = cudaFree(E);
    checkError(error);

    error = cudaFree(H_d);
    checkError(error);

    error = cudaFree(V);
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
 
    // k, confusingly named, is the dimensions to be projected upon
    FILE *fp;
    int k = 50;
    float K = 25;
    float S_max = 4*K;
    int N = 200;
    int M = 200;
    float epsilon = .0001;
    float r = .06;
    char filename[100];
    sprintf(filename,"outputs/cuda%d_%d.txt", M, N);

    // l is the time step
    float l = 1./(M - 1);

    // h is the x tep
    float h = S_max/(N - 1);
    
    float *U = 0, *U_h, *A, *A_h;
    float zero = 0;
    int i, j, j2;
    float t, x, sigSq, *a, *b, *c, *f, *f_h, *v, *w1, *w2;

    float *I_h = (float *)malloc(k*k*sizeof(I_h[0]));
    memset(I_h, 0, k*k*sizeof(I_h[0]));

    for (i = 0; i < k; i++) {
        I_h[i*k + i] = 1;
    }   

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

    error = cudaMalloc((void**)&I, k*k*sizeof(I[0]));
    checkError(error);
   
    error = cudaMemcpy(I, I_h, k*k*sizeof(I_h[0]), cudaMemcpyHostToDevice);
    checkError(error);
 
    // fill with zeros
    status = cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, &zero, NULL, N, &zero, NULL, N, U, N);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "Zeroing out U didn't work: status code is %d\n", status);
        exit(1);
    }

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

    error = cudaMalloc((void **)&f, N*sizeof(f[0]));
    checkError(error);
    
    a = (float*)malloc(N*sizeof(a[0]));
    b = (float*)malloc(N*sizeof(b[0]));
    c = (float*)malloc(N*sizeof(c[0]));

    A_h = (float*)malloc(N*N*sizeof(A_h[0]));
    memset(A_h, 0, N*N*sizeof(A_h[0]));    

    f_h =  (float*)malloc(N*sizeof(f_h[0]));
    memset(f_h, 0, N*sizeof(f_h[0]));    

    error = cudaMalloc((void **)&v, N*sizeof(v[0]));
    checkError(error);

    error = cudaMalloc((void **)&w1, N*sizeof(w1[0]));
    checkError(error);

    error = cudaMalloc((void **)&w2, N*sizeof(w2[0]));
    checkError(error);

    for (i = 0; i < M - 1; i++) {
        t = i*l;

        // see 3.7 for a definition of these a_i, b_i, and c_i
        for (j = 0; j < N; j++) {
            x = j*h;
            sigSq = sigma(t, x);
            sigSq *= sigSq;
            a[j] = sigSq*j*j/2 - r*j/2;
            b[j] = -sigSq*j*j - r;
            c[j] = sigSq*j*j/2 + r*j/2;
        }

        // now we can fill in A
        // rows 1 through N - 1 get an a
        for (j = 1; j < N; j++) {
            *ENTRY(A_h, j, j - 1, N) = a[j];
        }

        // rows 0 through N - 1 get a b
        for (j = 0; j < N; j++) {
            *ENTRY(A_h, j, j, N) = b[j];
        }

        // rows 0 to N - 2 get a c
        for (j = 0; j < N - 1; j++) {
            *ENTRY(A_h, j, j + 1, N) = c[j];
        }

        // now copy A_h to A
        status = cublasSetVector(N*N, sizeof(A_h[0]), A_h, 1, A, 1);  
        checkCublasStatus(status);  

        // ok, so A works fine 

        // now we need f(t) at x_0 and x_N
        // see 2.14
        f_h[N - 1] = (S_max - K*exp(-t*r))*c[N-1];

        status = cublasSetVector(N, sizeof(f_h[0]), f_h, 1, f, 1);  
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
    free(a);
    free(b);
    free(c);
    free(A_h);
    free(f_h);
    free(U_h);
    free(I_h);

    error = cudaFree(f);
    checkError(error);

    error = cudaFree(A);
    checkError(error);

    error = cudaFree(U);
    checkError(error);

    error = cudaFree(v);
    checkError(error);

    error = cudaFree(w1);
    checkError(error);

    error = cudaFree(w2);
    checkError(error);

    error = cudaFree(I);
    checkError(error);

    status = cublasDestroy(handle);
    checkCublasStatus(status);
    culaShutdown();

    return 0;

}

