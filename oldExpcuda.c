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

void checkCulaStatus(culaStatus status)
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
void padeExp(float *A, float *E, int n) {
    culaStatus status;
    float s;
    float *Q;
    float *I;	
    float *A2;
    float *P;
    int i, j, f, *piv, scaled = 0;
    float c[POL + 1];

    c[0] = 1;
    for (i = 0; i < POL; i++) {
      	c[i + 1] = c[i]*((double)(POL - i)/((i + 1)*(2*POL - i)));
    } 
    
    // scale here
    s = infinity_norm(A, n);
    if (s > 0.5) {
        scaled = 1;
        f = (int) (log(s)/log(2));  
        s = MAX(0,f + 2);
        status = culaSgemm('n', 'n', n, n, n, 0, A, n, A, n, pow(2, -s), A, n);
        checkCulaStatus(status);
    }

    // set up identity
    I = (float*)malloc(n*n*sizeof(float));
    memset(I, 0, n*n*sizeof(float));
    for (i = 0; i < n; i++) {
        I[i*n + i] = 1;
    }
    
    Q = (float*)malloc(n*n*sizeof(float));
    P = (float*)malloc(n*n*sizeof(float));
    memcpy(Q, I, n*n*sizeof(float));
    memcpy(P, I, n*n*sizeof(float));
    
    // allocate space for A2; no need to initialize memory
    A2 = (float*)malloc(n*n*sizeof(float));
 
    status = culaSgemm('n', 'n', n, n, n, 1, A, n, A, n, 0, A2, n);
    checkCulaStatus(status);

    status = culaSgemm('n', 'n', n, n, n, 0, Q, n, Q, n, c[POL], Q, n); 
    checkCulaStatus(status);
    status = culaSgemm('n', 'n', n, n, n, 0, P, n, P, n, c[POL - 1], P, n); 
    checkCulaStatus(status);

    int odd = 1;
    for (i = POL - 2; i >= 0; i--) {
        if (odd == 1) {
            // Q = Q*A2 + c[k]*I;
            status = culaSgemm('n', 'n', n, n, n, 1, Q, n, A2, n, 0, Q, n); 
            checkCulaStatus(status);
            
            status = culaSgemm('n', 'n', n, n, n, c[i], I, n, I, n, 1, Q, n); 
            checkCulaStatus(status); 
        }
        else {
            // P = P*A2 + c[k]*I 
            status = culaSgemm('n', 'n', n, n, n, 1, P, n, A2, n, 0, P, n); 
            checkCulaStatus(status);
            
            status = culaSgemm('n', 'n', n, n, n, c[i], I, n, I, n, 1, P, n); 
            checkCulaStatus(status);
        }
        
        odd = 1-odd;
  	}  

  	if (odd == 1) {
    	// Q = Q*A
    	status = culaSgemm('n', 'n', n, n, n, 1, Q, n, A, n, 0, Q, n); 
    	checkCulaStatus(status);
  	}
  	else {
    	// P = P*A
    	status = culaSgemm('n', 'n', n, n, n, 1, P, n, A, n, 0, P, n); 
    	checkCulaStatus(status);
  	}
  
  	// Q = Q - P
  	status = culaSgemm('n', 'n', n, n, n, -1, P, n, I, n, 1, Q, n); 
  	checkCulaStatus(status);

  	// Find X s.t. QX = P
  	piv = (int*)malloc(n*sizeof(int));
  	memset(piv, 0, n*sizeof(int));
  
  	status = culaSgesv(n, n, Q, n, piv, P, n);
  	checkCulaStatus(status);
  
  	// now P = X
 
  	memcpy(E, I, n*n*sizeof(float));
  	if (odd == 0) status = culaSgemm('n', 'n', n, n, n, 2, I, n, P, n, 1, E, n);
  	else status = culaSgemm('n', 'n', n, n, n, -2, I, n, P, n, -1, E, n);
  
  	checkCulaStatus(status);

  	for(i = 0; i < s; i++) {
    	status = culaSgemm('n', 'n', n, n, n, 1, E, n, E, n, 0, E, n);
        checkCulaStatus(status);
    }


    if (scaled == 1) {
        status = culaSgemm('n', 'n', n, n, n, 0, A, n, A, n, 1./pow(2, -s), A, n);
        checkCulaStatus(status);
    }

    free(I);
    free(A2);
    free(P);
    free(Q);
    free(piv);
    return;
}

void phi(float *A, float *E, int n) {

    float *I;
    int i, *piv;
    culaStatus status;
 
    // we want AX = e^A - I

    // set up identity
    I = (float*)malloc(n*n*sizeof(float));
    memset(I, 0, n*n*sizeof(float));
    for (i = 0; i < n; i++) {
        I[i*n + i] = 1;
    }

    padeExp(A, E, n);

    status = culaSgemm('n', 'n', n, n, n, -1, I, n, I, n, 1, E, n);
    checkCulaStatus(status);

    // now E = e^A - I

  	// Find X s.t. AX = E
  	piv = (int*)malloc(n*sizeof(int));
  	memset(piv, 0, n*sizeof(int));


  	status = culaSgesv(n, n, A, n, piv, E, n);
  	checkCulaStatus(status);
  	// now E = X

    // cleanup
    free(piv);
    free(I);

}

/*
int main(void) {
    #define N 5
    float A[N*N] = { -0.16580, 0.22570, 0.00000, 0.00000, 0.00000, 0.25460,
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

    int i, j;
    //float A[4] = {1, 0, 0, 1};
    float E[N*N];

    culaInitialize();
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) 
            printf("%f ", A[i + j*N]);
        printf("\n");
    }
    printf("\n");
    padeExp(A, E, N);

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) 
            printf("%f ", E[i + j*N]);
        printf("\n");
    }
    culaShutdown();
}*/ 
