#define MAX(a,b) ({ typeof (a) _a = (a); \
                   typeof (b) _b = (b); \
                   _a > _b ? _a : _b; })

#define STEP_WIDTH 8    

#define DOPRINT

#include <armadillo>
#include <cmath>
#include <fstream>
#include <complex> 
#include <ctime>

using namespace std;
using namespace arma;

ofstream imaginary;
ofstream absval;

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


// for some reason I'm getting nans in H 
float rational_arnoldi(const fmat &A, const fmat &v, fmat &V, fmat &H, const fmat &U, const fvec &xi) {
    int m = V.n_cols;
    float beta = norm(v);
    V.col(0) = v/beta;
 
    fmat I(A.n_rows, A.n_cols, fill::eye);

    for (int j = 0; j < m - 1; j++) {
        fmat y = V.cols(0, j)*U.col(j).rows(0, j);
        fmat x = solve(I - A/xi(j), A*y);
        for (int i = 0; i <= j; i++) {
            H(i, j) = dot(V.col(i), x);
            x -= H(i, j)*V.col(i);
        }

       H(j + 1, j) = norm(x);
       V.col(j + 1) = x/H(j + 1, j);
       //cout << "j is " << j << '\n';
       //cout << "x is \n" << x << '\n';
       //cout << "V is \n" << V << '\n';
       //cout << "H is \n" << H << '\n';
    }

    fmat y = V*U.col(m - 1);

    fmat x = solve(I - A/xi(m - 1), A*y);
    for (int i = 0; i <= m - 1; i++) {
        H(i, m - 1) = dot(V.col(i), x);
        x -= H(i, m - 1)*V.col(i);
    }
    
    return beta;
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
    fvec xi(m);
    //xi.fill((float)-1.0*m/sqrt(2));
    xi.fill(1e15);
    //fmat U(m, m, fill::zeros);
    fmat U(m, m, fill::eye);
    // one entry per column
    /*for (int j = 0; j < m; j++) {
        U(j/STEP_WIDTH*STEP_WIDTH, j) = 1.;
    }*/
    
    /*for (int j = 0; j < STEP_WIDTH; j++) {
        U(j, j) = 1.;
    }

    /*for (int j = STEP_WIDTH; j < m; j++) {
        U(j - STEP_WIDTH, j) = 1.;
    }*/

    float beta = /*arnoldi*/ rational_arnoldi(A, v, V, H, U, xi);
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

    template <typename T>
string itoa(T Number)
{
    stringstream ss;
    ss << Number;
    return ss.str();
}

int main(int argc, char *argv[]) {

    // so let's define some parameters

    // we are going to need a vector of x_i's
    // which depend on S_max, K, and N
    // we'll hardcode those here

    // k, confusingly named, is the dimensions to be projected upon
    int k = 50;
    float K = 25;
    float S_max = 4*K;
    float N = atoi(argv[1]);
    float M = 200;
    float epsilon = .0001;
    float r = .06;
    string filename = "outputs/bench" + itoa(M) + '_' + itoa(N) + ".txt";

    clock_t start, end;
    start = clock();

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

        fmat buf1 = krylov(k, l, A, U.row(i).st(), 1);

        fmat buf2 = l*krylov(k, l, A, f, 0);
        U.row(i + 1) = (buf1 + buf2).st();


        // alrighty, lets do dis
        //U.row(i + 1) = (krylov(k, l, A, U.row(i).st(), 1) + l*krylov(k, l, A, f, 0)).st();
    }

#ifdef DOPRINT
    ofstream out;
    out.open(filename.c_str(), ofstream::out);
    for (int i = 0; i < M; i++) {
        float t = i*l;
        for (int j = 0; j < N; j++) {
            float x = j*h;
            out << t << ' ' << x << ' ' << U(i, j) << '\n';   
        }
    }
    out.close();
#endif 

    end = clock();
    cout << "time: " << (double)(end - start)/CLOCKS_PER_SEC << '\n';

    return 0;

}

