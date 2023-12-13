#include <iostream>
#include <random>
#include <iomanip>
#include <algorithm>
#include <omp.h>
#include <fstream>
#include <mpi.h>

// matrix order wi-1j wij-1 wij wij+1 wij
const int Mx = 159;
const int Ny = 159;
const int size = (Mx + 1) * (Ny + 1);

struct InfoMPI
{
    int rank, size;
    int dims_size[2] = {0, 0};
    int periods[2] = {0, 0};
    int coords[2] = {0, 0};

    int x_start, y_start;
    int x_end, y_end;

    MPI_Comm comm;

    InfoMPI(){
        MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
        MPI_Comm_size(MPI_COMM_WORLD, &this->size);
        MPI_Dims_create(this->size, 2, this->dims_size);
        MPI_Cart_create(MPI_COMM_WORLD, 2, this->dims_size, this->periods,
                        1, &this->comm);
        MPI_Comm_rank(this->comm, &this->rank);
        MPI_Cart_coords(this->comm, this->rank, 2, this->coords);

        int block_size_x = Mx / this->dims_size[0];
        int block_size_y = Ny / this->dims_size[1];
        this->x_start = this->coords[0] * block_size_x;
        this->y_start = this->coords[1] * block_size_y;

        this->x_end = this->coords[0] == this->dims_size[0] - 1 ? Mx : (this->coords[0] + 1) * block_size_x;
        this->y_end = this->coords[1] == this->dims_size[1] - 1 ? Ny : (this->coords[1] + 1) * block_size_y;
    }
};

const double f_xy = 1.;
const double x_start = -1.;
const double x_end = 1.;
const double y_start = -1. / 2.;
const double y_end = 1. / 2.;

const double dx = (x_end - x_start) / Mx;
const double dy = (y_end - y_start) / Ny;
const double k_eps = dx * dx;

void print(double *a) {
    for (int i = 0; i <= Mx; ++i) {
        for (int j = 0; j <= Ny; ++j) {
            std::cout << "(" <<i<<", "<<j<<"): ";
            std::cout << std::setw(5) << std::setprecision(5) << a[i * Ny  + j] << " ";
        }
        std::cout << std::endl;
    }
}

inline bool inDomain(double x, double y) {
    return x * x + 4. * y * y < 1.;
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
    double result = 0., res_y1 = sqrt(1 - x * x) / 2.;
    double res_y2 = -1. * sqrt(1 - x * x) / 2.;
    if ((res_y1 > y1) and (res_y1 < y2))
        result = res_y1;
    if ((res_y2 > y1) and (res_y2 < y2))
        result = res_y2;
    if (inDomain(x, y1)) {
        result = std::abs(result - y1);
    }
    if (inDomain(x, y2)) {
        result = std::abs(result - y2);
    }
    return result;
}

inline double findIntersectWithDomainFixedY(double y, double x1, double x2) {
    double result = 0., res_x1 = sqrt(1 - 4 * y * y);
    double res_x2 = -1. * sqrt(1 - 4 * y * y);
    if ((res_x1 > x1) and (res_x1 < x2))
        result = res_x1;
    if ((res_x2 > x1) and (res_x2 < x2))
        result = res_x2;
    if (inDomain(x1, y)) {
        result = std::abs(result - x1);
    }
    if (inDomain(x2, y)) {
        result = std::abs(result - x2);
    }
    return result;
}

double monteCarlo(double x0, double x1, double y0, double y1, double (*func)(double, double)) {
    int numSamples = 10000;
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
    return sum / numSamples;
}

double dot(int n, double *a, double *b, const InfoMPI& info) {
    double sum = 0.0;
    #pragma omp parallel for collapse(2) reduction(+ : sum)
    for (int i = 1 + info.x_start; i < info.x_end; ++i) {
        for (int j = 1 + info.y_start; j < info.y_end; ++j) {
            sum += a[i * Ny + j] * b[i * Ny + j];
        }
    }
    return dx * dy * sum;
}

void add(double *a, double *b, double alpha, double beta, double *result, const InfoMPI& info) {
    #pragma omp parallel for collapse(2)
    for (int i = 1 + info.x_start; i < info.x_end; ++i) {
        for (int j = 1 + info.y_start; j < info.y_end; ++j) {
            result[i * Ny + j] = alpha * a[i * Ny + j] + beta * b[i * Ny + j];
        }
    }
}

// matrix order wi-1j wij-1 wij wij+1 wi+1j
void mul(double *A, double *b, double *result, const InfoMPI& info) {
    double top_point, bot_point, left_point, right_point;
    #pragma omp parallel for collapse(2) firstprivate(top_point, bot_point, left_point, right_point)
    for (int i = 1 + info.x_start; i < info.x_end; ++i) {
        for (int j = 1 + info.y_start; j < info.y_end; ++j) {
            top_point = b[(i - 1) * Ny + j];
            bot_point = b[(i + 1) * Ny + j];
            left_point = b[i * Ny + j - 1];
            right_point = b[i * Ny + j + 1];
            result[i * Ny + j] = A[i * Ny * 5 + j * 5 + 0] * top_point +
                                 A[i * Ny * 5 + j * 5 + 1] * left_point +
                                 A[i * Ny * 5 + j * 5 + 2] * b[i * Ny + j] +
                                 A[i * Ny * 5 + j * 5 + 3] * right_point +
                                 A[i * Ny * 5 + j * 5 + 4] * bot_point;
        }
    }
}

void copy(double *x, double *y, const InfoMPI& info) {
    #pragma omp parallel for collapse(2)
    for (int i = 1 + info.x_start; i < info.x_end; ++i)
        for (int j = 1 + info.y_start; j < info.y_end; ++j) {
            y[i * Ny + j] = x[i * Ny + j];
        }
}

void optimize(double *matrix, double *f_vector, double *w_vector, double *r_vector, double *Ar_vector, const InfoMPI& info)
{
    double p_vector[(Mx + 1) * (Ny + 1)];
    double tau_ar_proc[2] = {0., 0.}, tau_ar[2] = {0., 0.};
    double beta_ar[2] = {0., 0.}, alpha, beta, beta_proc;
    double eps_proc = 0., eps = 10., f_eps = 10.;
    double wk_vector[(Mx + 1) * (Ny + 1)];
    int iter = 0;

    mul(matrix, w_vector, Ar_vector, info);
    add(Ar_vector, f_vector, -1., 1., r_vector, info);
    copy(r_vector, p_vector, info);

    while (eps > 1e-12)
    {
        copy(w_vector, wk_vector, info);

        tau_ar_proc[0] = dot(size, r_vector, r_vector, info);

        mul(matrix, p_vector, Ar_vector, info);

        tau_ar_proc[1] = dot(size, p_vector, Ar_vector, info);

        MPI_Allreduce(&tau_ar_proc, &tau_ar, 2, MPI_DOUBLE, MPI_SUM, info.comm);

        alpha = tau_ar[0]/tau_ar[1];

        add(w_vector, p_vector, 1.0, alpha, w_vector, info);

        mul(matrix, w_vector, Ar_vector, info);

        add(Ar_vector, f_vector, -1., 1., r_vector, info);

        beta_proc = dot(size, r_vector, r_vector, info)/tau_ar[0];

        MPI_Allreduce(&beta_proc, &beta, 1, MPI_DOUBLE, MPI_SUM, info.comm);

        add(r_vector, p_vector, 1.0, beta, p_vector, info);

        add(w_vector, wk_vector, 1.0, -1.0, wk_vector, info);

        eps_proc = dot(size, wk_vector, wk_vector, info);

        MPI_Allreduce(&eps_proc, &eps, 1, MPI_DOUBLE, MPI_SUM, info.comm);

        eps = sqrt(eps);

        ++iter;
    }
    eps_proc = dot(size, r_vector, r_vector, info);
    MPI_Allreduce(&eps_proc, &eps, 1, MPI_DOUBLE, MPI_SUM, info.comm);
    if (info.rank == 0) {
        std::cout<<"||(Aw - f)|| = "<<sqrt(eps)<<std::endl;
        std::cout << "iter: " << iter << std::endl;
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);

    InfoMPI info;

    omp_set_num_threads(2);
    if (info.rank == 0) {
        std::cout << "size_x: " << info.dims_size[0] << std::endl;
        std::cout << "size_y: " << info.dims_size[1] << std::endl;
    }

//    std::cout<<"rank: "<<info.rank<<std::endl<<"x_start: "<<info.x_start<<std::endl<<"x_end: "<<info.x_end<<std::endl
//             <<"y_start: "<<info.y_start<<std::endl<<"y_end: "<<info.y_end<<std::endl;

    double xi, yj;
    double xi_2m, yj_2m, xi_2p, yj_2p, aij, ai_pj, bij, bij_p, lij;
    double F_vector[(Mx + 1) * (Ny + 1)];
    double w_vector[(Mx + 1) * (Ny + 1)];
    double matrix[(Mx + 1) * (Ny + 1) * 5];
    double r_vector[(Mx + 1) * (Ny + 1)];
    double Ar_vector[(Mx + 1) * (Ny + 1)];

    for (int i = 0; i <= Mx; ++i)
        for (int j = 0; j <= Ny; ++j) {
            F_vector[i * Ny + j] = 0.;
            w_vector[i * Ny + j] = 0.;
            r_vector[i * Ny + j] = 0.;
            Ar_vector[i * Ny + j] = 0.;
            matrix[i * Ny * 5 + j * 5 + 0] = 0.;
            matrix[i * Ny * 5 + j * 5 + 1] = 0.;
            matrix[i * Ny * 5 + j * 5 + 2] = 0.;
            matrix[i * Ny * 5 + j * 5 + 3] = 0.;
            matrix[i * Ny * 5 + j * 5 + 4] = 0.;
        }

    double start_time = MPI_Wtime();
    // filling
    #pragma omp parallel for collapse(2) firstprivate(xi, yj, xi_2m, xi_2p, yj_2m, yj_2p, aij, ai_pj, bij, bij_p, lij)
    for (int i = 1 + info.x_start; i < info.x_end; ++i) {
        for (int j = 1 + info.y_start; j < info.y_end; ++j) {
            xi = x_start + i * dx;
            yj = y_start + j * dy;
            // initial solution
            w_vector[i * Ny + j] = 0.;
            // xi +- h1/2, yj +- h2/2
            xi_2m = xi - 0.5 * dx;
            xi_2p = xi + 0.5 * dx;
            yj_2m = yj - 0.5 * dy;
            yj_2p = yj + 0.5 * dy;

            // aij
            if (inDomain(xi_2m, yj_2m) and inDomain(xi_2m, yj_2p)) {
                aij = 1;
            } else {
                if (!inDomain(xi_2m, yj_2m) and !inDomain(xi_2m, yj_2p)) {
                    aij = 1./k_eps;
                } else {
                    lij = findIntersectWithDomainFixedX(xi_2m, yj_2m, yj_2p);
                    aij = lij / dy + (1 - lij / dy) / k_eps;
                }
            }
            // ai+1j
            if (inDomain(xi_2p, yj_2m) and inDomain(xi_2p, yj_2p)) {
                ai_pj = 1;
            } else {
                if (!inDomain(xi_2p, yj_2m) and !inDomain(xi_2p, yj_2p)) {
                    ai_pj = 1./k_eps;
                } else {
                    lij = findIntersectWithDomainFixedX(xi_2p, yj_2m, yj_2p);
                    ai_pj = lij / dy + (1 - lij / dy) / k_eps;
                }
            }

            // bij
            if (inDomain(xi_2m, yj_2m) and inDomain(xi_2p, yj_2m)) {
                bij = 1;
            } else {
                if (!inDomain(xi_2m, yj_2m) and !inDomain(xi_2p, yj_2m)) {
                    bij = 1./k_eps;
                } else {
                    lij = findIntersectWithDomainFixedY(yj_2m, xi_2m, xi_2p);
                    bij = lij / dx + (1. - lij / dx) / k_eps;
                }
            }

            // bij+1
            if (inDomain(xi_2m, yj_2p) and inDomain(xi_2p, yj_2p)) {
                bij_p = 1;
            } else {
                if (!inDomain(xi_2m, yj_2p) and !inDomain(xi_2p, yj_2p)) {
                    bij_p = 1./k_eps;
                } else {
                    lij = findIntersectWithDomainFixedY(yj_2p, xi_2m, xi_2p);
                    bij_p = lij / dx + (1. - lij / dx) / k_eps;
                }
            }

            // filling matrix order wi-1j wij-1 wij wij+1 wi+1j
            matrix[i * Ny * 5 + j * 5 + 0] = -1. * aij / (dx * dx);
            matrix[i * Ny * 5 + j * 5 + 1] = -1. * bij / (dy * dy);
            matrix[i * Ny * 5 + j * 5 + 2] = (ai_pj + aij) / (dx * dx) + (bij_p + bij) / (dy * dy);
            matrix[i * Ny * 5 + j * 5 + 3] = -1. * bij_p / (dy * dy);
            matrix[i * Ny * 5 + j * 5 + 4] = -1. * ai_pj / (dx * dx);

            // filling F
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
    }

    optimize(matrix, F_vector, w_vector, r_vector, Ar_vector, info);

    double end_time = MPI_Wtime();
    double result_time, time = end_time - start_time;

    MPI_Reduce(&time, &result_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (info.rank == 0) {
        std::cout<<"result time: "<<result_time<<std::endl;
    }

    MPI_Finalize();
    return 0;
}

