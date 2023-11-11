#include <iostream>
#include <random>
#include <iomanip>
#include <algorithm>
#include <fstream>

// matrix order wi-1j wij-1 wij wij+1 wij

const int Mx = 9;
const int Ny = 9;
const int size = (Mx + 1) * (Ny + 1);

const double f_xy = 1.;
const double x_start = -1.;
const double x_end = 1.;
const double y_start = -1. / 2.;
const double y_end = 1. / 2.;

const double dx = (x_end - x_start) / Mx;
const double dy = (y_end - y_start) / Ny;
const double k_eps = std::max(dx, dy);

void print(double *a) {
    for (int i = 0; i <= Mx; ++i) {
        for (int j = 0; j <= Ny; ++j) {
            std::cout << std::setw(5) << std::setprecision(5) << a[i * Ny  + j] << " ";
        }
        std::cout << std::endl;
    }
}

inline bool inDomain(double x, double y) {
    return x * x + 4 * y * y < 1;
}

inline double k(double x, double y) {
    return inDomain(x, y) ? 1. : 1. / k_eps;
}

inline double Fxy(double x, double y) {
    return inDomain(x, y) ? 1. : 0.;
}

inline bool isInnerDomain(double x1, double x2, double y1, double y2) {
    return inDomain(x1, y1) and inDomain(x1, y2)
           and inDomain(x2, y1) and inDomain(x2, y2);
}

inline bool isOuterDomain(double x1, double x2, double y1, double y2) {
    return !inDomain(x1, y1) and !inDomain(x1, y2)
           and !inDomain(x2, y1) and !inDomain(x2, y2);
}

inline double findIntersectWithDomainFixedX(double x, double y1, double y2) {
    double result, res_y1 = sqrt(1 - x * x) / 2.;
    double res_y2 = -1. * sqrt(1 - x * x) / 2.;
    if ((res_y1 > y1) and (res_y1 < y2))
        result = res_y1;
    if ((res_y2 > y1) and (res_y2 < y2))
        result = res_y2;
    return result;
}

inline double findIntersectWithDomainFixedY(double y, double x1, double x2) {
    double result, res_x1 = sqrt(1 - 4 * y * y);
    double res_x2 = -1. * sqrt(1 - 4 * y * y);
    if ((res_x1 > x1) and (res_x1 < x2))
        result = res_x1;
    if ((res_x2 > x1) and (res_x2 < x2))
        result = res_x2;
    return result;
}

double monteCarlo(double x0, double x1, double y0, double y1, double (*func)(double, double)) {
    int numSamples = 1000000;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> xDistribution(x0, x1);
    std::uniform_real_distribution<double> yDistribution(y0, y1);
    double sum = 0.0;
    for (int i = 0; i < numSamples; ++i) {
        double x = xDistribution(gen);
        double y = yDistribution(gen);
        double fValue = func(x, y);
        sum += fValue;
    }
    double average = sum / numSamples;
    double integral = average;
    return integral;
}

double dot(int n, double *a, double *b) {
    double sum = 0.0;
    for (int i = 0; i <= Mx; ++i) {
        for (int j = 0; j <= Ny; ++j) {
            sum += dx * a[i * Ny + j] * dy * b[i * Ny + j];
        }
    }
    return sum;
}

double norm(int n, double *a) {
    return dot(n, a, a);
}

double add(double *a, double *b, double alpha, double beta, double *result) {
    for (int i = 0; i <= Mx; ++i) {
        for (int j = 0; j <= Ny; ++j) {
            result[i * Ny + j] = alpha * a[i * Ny + j] + beta * b[i * Ny + j];
        }
    }
}

// matrix order wi-1j wij-1 wij wij+1 wi+1j
double mul(double *A, double *b, double *result) {
    double xi, yj, top_point, bot_point, left_point, right_point;
    for (int i = 0; i <= Mx; ++i) {
        for (int j = 0; j <= Ny; ++j) {
            if ((i > 0) and (j > 0) and (i < Mx) and (j < Ny)) {
                top_point = b[(i - 1) * Ny + j];
                bot_point = b[(i + 1) * Ny + j];
                left_point = b[i * Ny + j - 1];
                right_point = b[i * Ny + j + 1];
                result[i * Ny + j] = A[i * Ny * 5 + j * 5 + 0] * top_point +
                                     A[i * Ny * 5 + j * 5 + 1] * left_point +
                                     A[i * Ny * 5 + j * 5 + 2] * b[i * Ny + j] +
                                     A[i * Ny * 5 + j * 5 + 3] * right_point +
                                     A[i * Ny * 5 + j * 5 + 4] * bot_point;
            } else result[i * Ny + j] = b[i * Ny + j];
        }
    }
}

void optimize(double *matrix, double *f_vector, double *w_vector,
              double *r_vector, double *Ar_vector) {

    double tau_ar_proc[2] = {0., 0.}, tau_ar[2] = {0., 0.}, tau_k;
    double eps_proc = 0., eps = 10.;
    int iter = 0;
    while (eps > 1e-06) {
        mul(matrix, w_vector, Ar_vector);

        add(Ar_vector, f_vector, 1., -1., r_vector);

        mul(matrix, r_vector, Ar_vector);

        tau_ar[0] = dot(size, Ar_vector, r_vector);
        tau_ar[1] = dot(size, Ar_vector, Ar_vector);

        tau_k = tau_ar[0] / tau_ar[1];
        add(w_vector, r_vector, 1.0, -tau_k, w_vector);

        //e2 norm
        eps = norm(size, r_vector);

        eps = sqrt(eps);

        if (iter % 100000 == 0) std::cout << iter<< ": " << std::setprecision(15) << "eps: " << eps << " (Ar,r): " << tau_ar[0] << " (r, r): " << tau_ar[1] << std::endl;

        ++iter;
    }

    std::cout << "iter: " << iter << std::endl;
}

int main() {
    int countInDomain = 0;
    int countOuterDomain = 0;
    double xi, yj;
    for (int i = 0; i <= Mx; ++i) {
        xi = x_start + i * dx;
        for (int j = 0; j <= Ny; ++j) {
            yj = y_start + j * dy;
            if (inDomain(xi, yj)) countInDomain += 1;
            else countOuterDomain += 1;
        }
    }
    std::cout << "in domain: " << countInDomain << std::endl;
    std::cout << "out domain: " << countOuterDomain << std::endl;

    double xi_2m, yj_2m, xi_2p, yj_2p, aij, ai_pj, bij, bij_p, lij;
    double F_vector[(Mx + 1) * (Ny + 1)];
    double w_vector[(Mx + 1) * (Ny + 1)];
    double matrix[(Mx + 1) * (Ny + 1) * 5];
    double r_vector[(Mx + 1) * (Ny + 1)];
    double Ar_vector[(Mx + 1) * (Ny + 1)];

    std::cout.setf(std::ios::fixed);
    // filling
    for (int i = 0; i <= Mx; ++i) {
        xi = x_start + i * dx;
        for (int j = 0; j <= Ny; ++j) {
            yj = y_start + j * dy;

            // initial solution
            w_vector[i * Ny + j] = 0.;


            xi_2m = xi - 0.5 * dx;
            xi_2p = xi + 0.5 * dx;

            yj_2m = yj - 0.5 * dy;
            yj_2p = yj + 0.5 * dy;

            // aij
            if (isInnerDomain(xi_2m, xi_2m, yj_2m, yj_2p)) {
                aij = 1;
            } else {
                if (isOuterDomain(xi_2m, xi_2m, yj_2m, yj_2p)) {
                    aij = 1./k_eps;
                } else {
                    lij = findIntersectWithDomainFixedX(xi_2m, yj_2m, yj_2p);
                    aij = lij / dy + (1 - lij / dy) / k_eps;
                }
            }
            // ai+1j
            if (isInnerDomain(xi_2p, xi_2p, yj_2m, yj_2p)) {
                ai_pj = 1;
            } else {
                if (isOuterDomain(xi_2p, xi_2p, yj_2m, yj_2p)) {
                    ai_pj = 1./k_eps;
                } else {
                    lij = findIntersectWithDomainFixedX(xi_2p, yj_2m, yj_2p);
                    ai_pj = lij / dy + (1 - lij / dy) / k_eps;
                }
            }

            // bij
            if (isInnerDomain(xi_2m, xi_2p, yj_2m, yj_2m)) {
                bij = 1;
            } else {
                if (isOuterDomain(xi_2m, xi_2p, yj_2m, yj_2m)) {
                    bij = 1./k_eps;
                } else {
                    lij = findIntersectWithDomainFixedY(yj_2m, xi_2m, xi_2p);
                    bij = lij / dx + (1. - lij / dx) / k_eps;
                }
            }

            // bij+1
            if (isInnerDomain(xi_2m, xi_2p, yj_2p, yj_2p)) {
                bij_p = 1;
            } else {
                if (isOuterDomain(xi_2m, xi_2p, yj_2p, yj_2p)) {
                    bij_p = 1./k_eps;
                } else {
                    lij = findIntersectWithDomainFixedY(yj_2p, xi_2m, xi_2p);
                    bij_p = lij / dx + (1. - lij / dx) / k_eps;
                }
            }

            if ((i > 0) and (j > 0) and (i < Mx) and (j < Ny)) {
                // filling matrix order wi-1j wij-1 wij wij+1 wij
                matrix[i * Ny * 5 + j * 5 + 0] = -1. * aij / (dx * dx);
                matrix[i * Ny * 5 + j * 5 + 1] = -1. * bij / (dy * dy);
                matrix[i * Ny * 5 + j * 5 + 2] = ((ai_pj + aij) / (dx * dx) + (bij_p + bij) / (dy * dy));
                matrix[i * Ny * 5 + j * 5 + 3] = -1. * bij_p / (dy * dy);
                matrix[i * Ny * 5 + j * 5 + 4] = -1. * ai_pj / (dx * dx);
            }
            else matrix[i * Ny * 5 + j * 5 + 2] = 1;
            // filling F
            if ((i > 0) and (j > 0) and (i < Mx) and (j < Ny)) {
                if (isInnerDomain(xi_2m, xi_2p, yj_2m, yj_2p)) {
                    F_vector[i * Ny + j] = 1;
                } else {
                    if (isOuterDomain(xi_2m, xi_2p, yj_2m, yj_2p)) {
                        F_vector[i * Ny + j] = 0;
                    } else {
                        F_vector[i * Ny + j] = monteCarlo(xi_2m, xi_2p, yj_2m, yj_2p, Fxy);
                    }
                }
            }
            else F_vector[i * Ny + j] = 0;
//            std::cout << std::setw(5) << std::setprecision(5) << matrix[i * Ny * 5 + j * 5 + 2] << " ";
        }
//        std::cout << std::endl;
    }

//    for (int i = 0; i <= Mx; ++i) {
//        for (int j = 0; j <= Ny; ++j) {
//            std::cout << "(" <<i<<", "<<j<<"): ";
//            for (int k = 0; k < 5; ++k) std::cout << matrix[i*Ny*5 + j * 5 + k] << " ";
//            std::cout<<std::endl;
//        }
//    }


    optimize(matrix, F_vector, w_vector, r_vector, Ar_vector);
    std::ofstream myfile;
    myfile.open ("result");
    for (int i = 0; i <= Mx; ++i) {
        for (int j = 0; j <= Ny; ++j) {
            myfile << w_vector[i * Ny  + j] << ",";
        }
    }
    myfile.close();

    return 0;
}
