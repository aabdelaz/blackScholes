// The caller of this code is responsible for calling culaInitialize and culaShutdown
#define POL 6
#define IDX(i, j, n) n*(j) + i 
#define MAX(a,b) ({ typeof (a) _a = (a); \
                   typeof (b) _b = (b); \
                   _a > _b ? _a : _b; })

#include <math.h>
#include <stdlib.h>
#include <string.h> 
#include <stdio.h>
#include <float.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cula_blas.h>
#include <cula_lapack.h>

void pm(float *A, int n) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) 
            printf("%f ", A[i + j*n]);
        printf("\n");
    }
    printf("\n");
}

inline void checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS error: Status code %d\n", status);
        exit(1);
    }
}

inline void checkError(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "!!!! CUDA error: Error code %d\n", error);
        exit(1);
    }
}

inline void checkCulaStatus(culaStatus status)
{
    char buf[256];
	
    if(!status)
        return;
		printf("nooooo %d\n", status);

    culaGetErrorInfoString(status, culaGetErrorInfo(), buf, sizeof(buf));
    printf("%s\n", buf);

    culaShutdown();
    exit(EXIT_FAILURE);
}

float infinity_norm(float *A, int n) {
    int i, j;
    float sum, max = FLT_MIN, ent;
    for (i = 0; i < n; i++) {
		sum = 0;
		for (j = 0; j < n; j++) {
            ent = A[IDX(i, j, n)];
            if (ent < 0) ent *= -1;
	        sum += ent;
		}

			if (sum > max) max = sum;
    }
			
    return max;
}

// a is the matrix you have, e is the one you'll fill in
// now I'm going to use cublas, assume A and E are device pointers
void padeExp(cublasHandle_t handle, float *A, float *E, int n) {
    culaStatus status;
    cublasStatus_t bS;
    cudaError_t error;
    float s, s_factor, one = 1, zero = 0, minus = -1, two = 2, m_two = -2;
    float *Q, *A2, *P;
    extern float *I;
    int i, j, f, *piv, scaled = 0;
    float c[POL + 1];
    float *A_h = (float*)malloc(sizeof(A_h[0])*n*n);

    c[0] = 1;
    for (i = 0; i < POL; i++) {
      	c[i + 1] = c[i]*((double)(POL - i)/((i + 1)*(2*POL - i)));
    } 
    
    bS = cublasGetVector(n*n, sizeof(A[0]), A, 1, A_h, 1);
    checkCublasStatus(bS);

    // scale here
    s = infinity_norm(A_h, n);
    if (s > 0.5) {
        scaled = 1;
        f = (int) (log(s)/log(2));  
        s = MAX(0,f + 2);
        s_factor = pow(2, -s);
        bS = cublasSscal(handle, n*n, &s_factor, A, 1); 
        checkCublasStatus(bS);
    }
  
    error = cudaMalloc((void**)&Q, n*n*sizeof(Q[0]));
    checkError(error);

    error = cudaMalloc((void**)&P, n*n*sizeof(P[0]));
    checkError(error);

    bS = cublasScopy(handle, n*n, I, 1, Q, 1);
    checkCublasStatus(bS);

    bS = cublasScopy(handle, n*n, I, 1, P, 1);
    checkCublasStatus(bS);   

    // allocate space for A2; no need to initialize memory

    error = cudaMalloc((void**)&A2, n*n*sizeof(A2[0]));
    checkError(error);

    bS = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, A, n, A, n, &zero, A2, n);
    checkCublasStatus(bS);

    bS = cublasSscal(handle, n*n, c + POL, Q, 1); 
    checkCublasStatus(bS);
 
    bS = cublasSscal(handle, n*n, c + POL - 1, P, 1); 
    checkCublasStatus(bS);

    int odd = 1;
    for (i = POL - 2; i >= 0; i--) {
        if (odd == 1) {
            // Q = Q*A2
            bS = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, Q, n, A2, n, &zero, Q, n);            
            checkCublasStatus(bS);

            // Q = Q + c[k]*I
            bS = cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &one, Q, n, c + i, I, n, Q, n);
            checkCublasStatus(bS);            
        }
        else {
            // P = P*A2 
            bS = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, P, n, A2, n, &zero, P, n);            
            checkCublasStatus(bS);

            // P = P + c[k]*I
            bS = cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &one, P, n, c + i, I, n, P, n);
            checkCublasStatus(bS);            
        }
        odd = 1-odd;
  	}  

  	if (odd == 1) {
    	// Q = Q*A
        bS = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, Q, n, A, n, &zero, Q, n);            
        checkCublasStatus(bS);
  	}
  	else {
    	// P = P*A
        bS = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, P, n, A, n, &zero, P, n);            
        checkCublasStatus(bS);
  	}
  
  	// Q = Q - P
    bS = cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &one, Q, n, &minus, P, n, Q, n);
    checkCublasStatus(bS);            

  	// Find X s.t. QX = P
  	error = cudaMalloc((void**)&piv, n*sizeof(int));
  	checkError(error);

    error = cudaMemset((void*)piv, 0, n*sizeof(int));
    checkError(error);

  	status = culaDeviceSgesv(n, n, Q, n, piv, P, n);
  	checkCulaStatus(status);

 	// now P = X
 
    bS = cublasScopy(handle, n*n, I, 1, E, 1);
    checkCublasStatus(bS);

  	if (odd == 0) bS = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &two, I, n, P, n, &one, E, n);            
  	else bS = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &m_two, I, n, P, n, &minus, E, n);
    checkCublasStatus(bS);

  	for(i = 0; i < s; i++) {
        bS = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, E, n, E, n, &zero, E, n);            
        checkCublasStatus(bS);
    }

    if (scaled == 1) {
        s_factor = 1./s_factor;
        bS = cublasSscal(handle, n*n, &s_factor, A, 1); 
        checkCublasStatus(bS);
    }

    free(A_h);

    error = cudaFree(piv);
    checkError(error);

    error = cudaFree(Q);
    checkError(error);

    error = cudaFree(P);
    checkError(error);

    error = cudaFree(A2);
    checkError(error);

    return;
}

void phi(cublasHandle_t handle, float *A, float *E, int n) {
    extern float *I;
    float one = 1, minus = -1;
    int i, *piv;
    cublasStatus_t bS;
    cudaError_t error;
    culaStatus status;
 
    // we want AX = e^A - I

    padeExp(handle, A, E, n);

    bS = cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &minus, I, n, &one, E, n, E, n);
    checkCublasStatus(bS);

    // now E = e^A - I

  	// Find X s.t. AX = E
  	error = cudaMalloc((void**)&piv, n*sizeof(int));
    checkError(error);

    error = cudaMemset((void*)piv, 0, n*sizeof(int));
    checkError(error);

    status = culaDeviceSgesv(n, n, A, n, piv, E, n);
  	checkCulaStatus(status);

  	// now E = X

    // cleanup
    error = cudaFree(piv);
    checkError(error);

}

/*
int main(void) {
    #define N 5
    cublasStatus_t status;
    cublasHandle_t handle;
    cudaError_t error;
    float A_h[N*N] = { -0.16580, 0.22570, 0.00000, 0.00000, 0.00000, 0.25460,
  -0.73720,
   0.29330,
   0.00000,
   0.00000,
  -0.00220,
   0.33500,
  -0.53390,
   0.20670,
   0.00000,
  -0.002100,
  -0.0019000,
   0.2369000,
  -0.3663000,
   0.1378000,
  -0.0019000,
  -0.0018000,
  -0.0025000,
   0.1566000,
  -0.2340000
};

    float E_h[N*N];
    int i, j;
    //float A[4] = {1, 0, 0, 1};
    float *E, *A;

  	error = cudaMalloc((void**)&E, N*N*sizeof(float));
    checkError(error);

  	error = cudaMalloc((void**)&A, N*N*sizeof(float));
    checkError(error);

    status = cublasCreate(&handle);
    checkCublasStatus(status);
    culaInitialize();

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) 
            printf("%f ", A_h[i + j*N]);
        printf("\n");
    }
    printf("\n");

    error = cudaMemcpy((void*)A, (void*)A_h, N*N*sizeof(float), cudaMemcpyHostToDevice);
    checkError(error); 

    //padeExp(handle, A, E, N);
    phi(handle, A, E, N);

    error = cudaMemcpy((void*)E_h, (void*)E, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    checkError(error); 

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) 
            printf("%f ", E_h[i + j*N]);
        printf("\n");
    }

    status = cublasDestroy(handle);
    checkCublasStatus(status);

    error = cudaFree(E);
    checkError(error);

    error = cudaFree(A);
    checkError(error);


    culaShutdown();
}*/ 
