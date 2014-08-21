#include <armadillo>
#include <cmath>

#define N 200
#define M_MAX 200
#define NUM_K_VALS 5
#define NUM_M_VALS 5

#define MAX(a,b) ({ typeof (a) _a = (a); \
                   typeof (b) _b = (b); \
                   _a > _b ? _a : _b; }) 

using namespace std;
using namespace arma;

// defined in eq 2.4, p. 3
// never gets past the if statement in our examples 
float pi_epsilon(const float &y, const float &ep) {
    if (y >= ep) return y;
    if (y <= -ep) return 0;

    float c0, c1, c2, c4, c6, c8;
    c0 = 35./256*ep;
    c1 = 0.5;
    c2 = 35./(64*ep);
    c4 = -35./(128*ep*ep*ep);
    c6 = 7./(64*ep*ep*ep*ep*ep);
    c8 = -5./(256*ep*ep*ep*ep*ep*ep*ep);

    float y2 = y*y;
    float y4 = y2*y2;
    float y6 = y4*y2;
    float y8 = y4*y4;

    return c0 + c1*y + c2*y2 + c4*y4 + c6*y6 + c8*y8; 

}

// volatility; given t, returns a vector containing the volatility at that time
// and x
inline float sigma(const float &t, const float &x) {
    return 0.2 + 0.2*(1 - t)*((x/25 - 1.2)*(x/25 - 1.2)/((x/25)*(x/25) + 1.44));
} 


// returns beta; fills in H and V
float arnoldi(const fmat &A, const fmat &v, fmat &V, fmat &H) {
    int m = V.n_cols;
    float beta = norm(v);
    V.col(0) = v/beta; 
    for (int j = 0; j < m - 1; j++) {
        fmat p = A*V.col(j);
        for (int i = 0; i <= j; i++) {
            H(i, j) = dot(V.col(i), p);
            p -= H(i, j)*V.col(i);
        }
        H(j + 1, j) = norm(p);
        V.col(j + 1) = p/H(j + 1, j);
    }

    fmat p = A*V.col(m - 1);
    for (int i = 0; i <= m - 1; i++) {
        H(i, m - 1) = dot(V.col(i), p);
        p -= H(i, m - 1)*V.col(i);
    }

    return beta;
}

fmat matrix_exp(fmat &A) {

    int p = 6, scaled = 0;
    int n = A.n_rows;
    
    fmat c(p + 1, 1);
    c(0) = 1;
    for (int i = 0; i < p; i++) c(i + 1) = c(i)*((float)(p - i)/((i + 1)*(2*p - i)));

    float s_factor, s = norm(A,"inf");
    if (s > 0.5) { 
        scaled = 1;
        s = MAX(0, ((int)(log(s)/log(2)))+2);
        s_factor = pow(2, -s);
        A = s_factor*A;
    }

    fmat I(A.n_rows, A.n_cols, fill::eye);
    fmat A2 = A*A;
    fmat Q = c(p)*I;
    fmat P = c(p- 1)*I;

    int odd = 1;
    for (int k = p - 2; k >= 0; k--) {
        if (odd == 1) 
            Q = Q*A2 + c(k)*I;
        else
            P = P*A2 + c(k)*I;
    
        odd = 1-odd;
    }

    fmat E;
    if (odd == 1) { 
        Q = Q*A;
        Q = Q - P;
        E = -(I + 2*solve(Q,P));
    }

    else {
        P = P*A;
        Q = Q - P;
        E = I + 2*solve(Q,P);
    }

    for (int k = 0; k < s; k++) E = E*E;

    if (scaled == 1) A = A/s_factor; 

    return E;
}

// computes phi(A) = A^-1*(e^A - I)
fmat phi(fmat &A) {
    return solve(A, matrix_exp(A) - fmat(A.n_rows, A.n_cols, fill::eye));
}

fmat krylov(int m, float l, const fmat &A, const fmat &v, int expo) {
    fmat V(A.n_rows, m, fill::zeros);
    fmat H(m, m, fill::zeros);

    float beta = arnoldi(A, v, V, H);
    fmat e_0(m, 1, fill::zeros);
    e_0(0) = 1;
    fmat w;
    H = l*H;
    if (expo == 1) { 
        w = beta*V*matrix_exp(H)*e_0;
    }
    else {
        w = beta*V*phi(H)*e_0;
    }
    return w;
}

fvec run_solver(int k, int M) {
    // so let's define some parameters
    float K = 25;
    float S_max = 4*K;
    float epsilon = .0001;
    float r = .06;

    // l is the time step
    float l = 1./(M - 1);

    // h is the x tep
    float h = S_max/(N - 1);

    // we will have a matrix U, where the ij^th entry is the value of the option at time t = t_i 
    // and x = x_j
    // the dimensions of this matrix will be M*N
    // M is the time axis
    // N is the x axis
    fmat U(M, N, fill::zeros);

    // we can determine the values of the U_0j^th entries
    for (int j = 0; j < N; j++) {
        U(0, j) = pi_epsilon(h*j - K, epsilon);
    }

    // now we need to fill in the A matrix, which is a function of t 
    // so let's loop over t
    //for (int i = 0; i < 1; i++) {
    for (int i = 0; i < M - 1; i++) {
        float t = i*l;
        fmat A(N, N, fill::zeros);
        fmat a(N, 1);
        fmat b(N, 1);
        fmat c(N, 1);
        // see 3.7 for a definition of these a_i, b_i, and c_i
        for (int j = 0; j < N; j++) {
            float x = j*h;
            float sigSq = sigma(t, x);
            sigSq *= sigSq;
            a(j) = sigSq*j*j/2 - r*j/2;
            b(j) = -sigSq*j*j - r;
            c(j) = sigSq*j*j/2 + r*j/2;
        }

        // now we can fill in A
        // rows 1 through N - 1 get an a
        for (int j = 1; j < N; j++) {
            A(j, j - 1) = a(j);
        }


        // rows 0 through N - 1 get a b
        for (int j = 0; j < N; j++) {
            A(j, j) = b(j);
        }

        // rows 0 to N - 2 get a c
        for (int j = 0; j < N - 1; j++) {
            A(j, j + 1) = c(j);
        }

        // now we need f(t) at x_0 and x_N
        fmat f(N, 1, fill::zeros);
        
        // see 2.14
        f(N - 1) = (S_max - K*exp(-t*r))*c(N-1);

        // alrighty, lets do dis
        U.row(i + 1) = (krylov(k, l, A, U.row(i).st(), 1) + l*krylov(k, l, A, f, 0)).st();
    }

    return U.row(M - 1).st();

}

int main(int argc, char **argv) {

    // this is the code for finding error as a function of k
    fvec ref = run_solver(N - 1, M_MAX);
    float refNorm = norm(ref);

    /*
    int k_vals[NUM_K_VALS] = { N/32, N/16, N/8, N/4, N/2 };
    for (int i = 0; i < NUM_K_VALS; i++) {
        fvec result = run_solver(k_vals[i], M_MAX);
        float rel_err = norm(ref - result)/refNorm; 
        cout << k_vals[i] << " " << rel_err << '\n';
    }*/

    int M_vals[NUM_M_VALS] = { M_MAX/32, M_MAX/16, M_MAX/8, M_MAX/4, M_MAX/2 };
    for (int i = 0; i < NUM_M_VALS; i++) {
        fvec result = run_solver(50, M_vals[i]);
        float rel_err = norm(ref - result)/refNorm; 
        cout << M_vals[i] << " " << rel_err << '\n';
    }
    
    
    

    return 0;
}

