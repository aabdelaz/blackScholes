#define MAX(a,b) ({ typeof (a) _a = (a); \
                   typeof (b) _b = (b); \
                   _a > _b ? _a : _b; })

fmat matrix_exp(fmat &A) {

    int p = 6, scaled = 0;
    int n = A.n_rows;
    
    fmat c(p + 1);
    c(0) = 1;
    for (int i = 0; i < p; i++) c(i + 1) = c(i)*((double)(p - i)/((i + 1)*(2*p - i)));

    float s_factor, s = norm(A,'inf');
    if (s > 0.5) { 
        scaled = 1;
        s = MAX(0, ((int)(log(s)/log(2)))+2);
        s_factor = pow(2, -s);
        A = s_factor*A;
    }

    fmat I = eye(A.n_rows, A.n_cols);
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

