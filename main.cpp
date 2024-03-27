#include <iostream>
#include <sstream>
#include <array>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>

#include <mpi.h>
#include <fftw3-mpi.h>

// init std::array<double*, kDimsNumber> vector_grid;
// out derivative 

constexpr size_t kDimsNumber = 3;
constexpr size_t kRootRank = 0;

// Example functions 1
double sin_3d(const std::array<double, 3>& cords, const std::array<double, 3>& coefs) {
    return sin(std::inner_product(cords.begin(), cords.end(), coefs.begin(), 0.0));
}

double sin_3d_derivative(const std::array<double, 3>& cords, const std::array<double, 3>& coefs, int axis) {
    return coefs[axis] * cos(std::inner_product(cords.begin(), cords.end(), coefs.begin(), 0.0));
}

// Example functions 2
double cos_3d(const std::array<double, 3>& cords, const std::array<double, 3>& coefs) {
    return cos(std::inner_product(cords.begin(), cords.end(), coefs.begin(), 0.0));
}

double cos_3d_derivative(const std::array<double, 3>& cords, const std::array<double, 3>& coefs, int axis) {
    return -coefs[axis] * sin(std::inner_product(cords.begin(), cords.end(), coefs.begin(), 0.0));
}

// Example functions 3
double expsin_3d(const std::array<double, 3>& cords, const std::array<double, 3>& coefs) {
    return exp(sin(std::inner_product(cords.begin(), cords.end(), coefs.begin(), 0.0)));
}

double expsin_3d_derivative(const std::array<double, 3>& cords, const std::array<double, 3>& coefs, int axis) {
    return exp(sin(std::inner_product(cords.begin(), cords.end(), coefs.begin(), 0.0))) * coefs[axis] * cos(std::inner_product(cords.begin(), cords.end(), coefs.begin(), 0.0));
}

// Example functions 4
double exptrig_3d(const std::array<double, 3>& cords, const std::array<double, 3>& coefs) {
    return exp(sin(coefs[0] * cords[0]) + cos(coefs[1] * cords[1]) + sin(coefs[2] * cords[2]));
}

double exptrig_3d_derivative(const std::array<double, 3>& cords, const std::array<double, 3>& coefs, int axis) {
    if (axis == 0) {
        return exptrig_3d(cords, coefs) * coefs[0] * cos(coefs[0] * cords[0]);
    } else if (axis == 1) {
        return -exptrig_3d(cords, coefs) * coefs[1] * sin(coefs[1] * cords[1]);
    } else if (axis == 2) {
        return exptrig_3d(cords, coefs) * coefs[2] * cos(coefs[2] * cords[2]);
    }

    return 0;
}

// Help functions
void fill_double_grid_mpi(double* grid, const std::array<ptrdiff_t, 3>& dims_sizes, ptrdiff_t local_n0, ptrdiff_t local_0_start, double (*filler)(const std::array<double, 3>&, const std::array<double, 3>&), const std::array<double, 3>& coefs) {
    std::array<double, 3> cords_multipliers;

    for(ptrdiff_t i = 0; i < 3; i++) {
        cords_multipliers[i] = (2 * M_PI) / dims_sizes[i];
    }

    for(ptrdiff_t local_i = 0; local_i < local_n0; local_i++) {
        for(ptrdiff_t j = 0; j < dims_sizes[1]; j++) {
            for(ptrdiff_t k = 0; k < dims_sizes[2]; k++) {
                ptrdiff_t i = local_0_start + local_i;

                ptrdiff_t idx = (local_i * dims_sizes[1] + j) * 2 * (dims_sizes[2] / 2  + 1) + k;
                std::array<double, 3> cords;
                
                cords[0] = i * cords_multipliers[0];
                cords[1] = j * cords_multipliers[1];
                cords[2] = k * cords_multipliers[2];

                grid[idx] = filler(cords, coefs);
            }
        }
    }
}

void fill_double_grid_derivative_mpi(double* grid, const std::array<ptrdiff_t, 3>& dims_sizes, ptrdiff_t local_n0, ptrdiff_t local_0_start, double (*filler)(const std::array<double, 3>&, const std::array<double, 3>&, int), const std::array<double, 3>& coefs, int axis) {
    std::array<double, 3> cords_multipliers;

    for(ptrdiff_t i = 0; i < 3; i++) {
        cords_multipliers[i] = (2 * M_PI) / dims_sizes[i];
    }

    for(ptrdiff_t local_i = 0; local_i < local_n0; local_i++) {
        for(ptrdiff_t j = 0; j < dims_sizes[1]; j++) {
            for(ptrdiff_t k = 0; k < dims_sizes[2]; k++) {
                ptrdiff_t i = local_0_start + local_i;

                ptrdiff_t idx = (local_i * dims_sizes[1] + j) * 2 * (dims_sizes[2] / 2  + 1) + k;
                std::array<double, 3> cords;
                
                cords[0] = i * cords_multipliers[0];
                cords[1] = j * cords_multipliers[1];
                cords[2] = k * cords_multipliers[2];

                grid[idx] = filler(cords, coefs, axis);
            }
        }
    }
}

void add_complex_grids(fftw_complex* grid_a, fftw_complex* grid_b, size_t length) {
    for(size_t i = 0; i < length; i++) {
        grid_a[i][0] += grid_b[i][0];
        grid_a[i][1] += grid_b[i][1];
    }
}

void sub_complex_grids(fftw_complex* grid_a, fftw_complex* grid_b, size_t length) {
    for(size_t i = 0; i < length; i++) {
        grid_a[i][0] -= grid_b[i][0];
        grid_a[i][1] -= grid_b[i][1];
    }
}

void add_double_grids(double* grid_a, double* grid_b, size_t length) {
    for(size_t i = 0; i < length; i++) {
        grid_a[i] += grid_b[i];
    }
}

void sub_double_grids(double* grid_a, double* grid_b, size_t length) {
    for(size_t i = 0; i < length; i++) {
        grid_a[i] -= grid_b[i];
    }
}

// FFT
inline double distance(fftw_complex a, fftw_complex b) {
    return sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2));
}

double max_abs_diff(fftw_complex* arr_1, fftw_complex* arr_2, size_t length) {
    double max_diff = distance(arr_1[0], arr_2[0]);

    for(size_t i = 0; i < length; i++) {
        double diff = distance(arr_1[i], arr_2[i]);

        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    return max_diff;
}

double max_abs_diff(double* arr_1, double* arr_2, size_t length) {
    double max_diff = abs(arr_1[0] - arr_2[0]);

    for(int i = 0; i < length; i++) {
        double diff = abs(arr_1[i] - arr_2[i]);

        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    return max_diff;
}

void fft_derivative_fourier_mpi(fftw_plan forward, double* in, fftw_complex* fourier_coeffs, int axis, const std::array<ptrdiff_t, 3>& dims_sizes, MPI_Comm comm) {
    ptrdiff_t local_n0, local_0_start;
    auto local_length = fftw_mpi_local_size_3d(dims_sizes[0], dims_sizes[1], dims_sizes[2] / 2 + 1, comm, &local_n0, &local_0_start);

    // To Fourier Space
    fftw_execute(forward);

    // Take derivative
    for(ptrdiff_t local_i = 0; local_i < local_n0; local_i++) {
        for(ptrdiff_t j = 0; j < dims_sizes[1]; j++) {
            for(ptrdiff_t k = 0; k < dims_sizes[2] / 2 + 1; k++) {
                ptrdiff_t i = local_i + local_0_start;

                std::array<ptrdiff_t, 3> global_cords({i, j, k});
                
                ptrdiff_t k_idx = global_cords[axis];
                double k_mult = (k_idx < dims_sizes[axis] / 2) ? k_idx : k_idx - dims_sizes[axis];

                ptrdiff_t idx = (local_i * dims_sizes[1] + j) * (dims_sizes[2] / 2 + 1) + k;

                // (real + i * imag) * i * k_mult = -imag * k_mult + i * real * k_mult
                double real = fourier_coeffs[idx][0];
                fourier_coeffs[idx][0] = -fourier_coeffs[idx][1] * k_mult;
                fourier_coeffs[idx][1] = real * k_mult;
            }
        }
    }
}

void fft_derivative_mpi(fftw_plan forward, fftw_plan backward, double* in, fftw_complex* fourrier_coeffs, double* out, int axis, const std::array<ptrdiff_t, 3>& dims_sizes, MPI_Comm comm) {
    ptrdiff_t local_n0, local_0_start;
    auto local_length = fftw_mpi_local_size_3d(dims_sizes[0], dims_sizes[1], dims_sizes[2] / 2 + 1, comm, &local_n0, &local_0_start);

    // out contains fourier coefs after derivative operation
    fft_derivative_fourier_mpi(forward, in, fourrier_coeffs, axis, dims_sizes, comm);

    // return to real space
    fftw_execute(backward);

    // normalize
    double divider = dims_sizes[0] * dims_sizes[1] * dims_sizes[2];
    for(int i = 0; i < local_length * 2; i++) {
        out[i] /= divider;
    }
}

void fft_divirgence_mpi(std::vector<fftw_plan> forwards, fftw_plan backward, std::array<double*, kDimsNumber> in, fftw_complex* fourrier_coeffs, fftw_complex* fourier_coeffs_result, double* out, const std::array<ptrdiff_t, 3>& dims_sizes, MPI_Comm comm) {
    ptrdiff_t local_n0, local_0_start;
    auto local_length = fftw_mpi_local_size_3d(dims_sizes[0], dims_sizes[1], dims_sizes[2] / 2 + 1, comm, &local_n0, &local_0_start);

    std::memset(fourier_coeffs_result, 0, sizeof(fftw_complex) * local_length);

    for(int i = 0; i < kDimsNumber; i++) {
        fft_derivative_fourier_mpi(forwards[i], in[i], fourrier_coeffs, i, dims_sizes, comm);
        add_complex_grids(fourier_coeffs_result, fourrier_coeffs, local_length);
    }

    // return to real space
    fftw_execute(backward);

    // normalize
    double divider = dims_sizes[0] * dims_sizes[1] * dims_sizes[2];
    for(int i = 0; i < local_length * 2; i++) {
        out[i] /= divider;
    }
}


void fft_rotor_mpi(std::vector<fftw_plan> forwards, fftw_plan backward, std::array<double*, kDimsNumber> in, fftw_complex* fourrier_coeffs, fftw_complex* fourier_coeffs_result, double* out, std::array<double*, kDimsNumber> out_3d, const std::array<ptrdiff_t, 3>& dims_sizes, MPI_Comm comm) {
    ptrdiff_t local_n0, local_0_start;
    auto local_length = fftw_mpi_local_size_3d(dims_sizes[0], dims_sizes[1], dims_sizes[2] / 2 + 1, comm, &local_n0, &local_0_start);

    // First component
    std::memset(fourier_coeffs_result, 0, sizeof(fftw_complex) * local_length);

    fft_derivative_fourier_mpi(forwards[2], in[2], fourrier_coeffs, 1, dims_sizes, comm); // dFz/dy
    add_complex_grids(fourier_coeffs_result, fourrier_coeffs, local_length);

    fft_derivative_fourier_mpi(forwards[1], in[1], fourrier_coeffs, 2, dims_sizes, comm); // dFy/dz
    sub_complex_grids(fourier_coeffs_result, fourrier_coeffs, local_length);

    fftw_execute(backward);

    std::memcpy(out_3d[0], out, local_length * 2 * sizeof(double));

    // Second component
    std::memset(fourier_coeffs_result, 0, sizeof(fftw_complex) * local_length);

    fft_derivative_fourier_mpi(forwards[0], in[0], fourrier_coeffs, 2, dims_sizes, comm); // dFz/dy
    add_complex_grids(fourier_coeffs_result, fourrier_coeffs, local_length);

    fft_derivative_fourier_mpi(forwards[2], in[2], fourrier_coeffs, 0, dims_sizes, comm); // dFz/dy
    sub_complex_grids(fourier_coeffs_result, fourrier_coeffs, local_length);

    fftw_execute(backward);

    std::memcpy(out_3d[1], out, local_length * 2 * sizeof(double));

    // Second component
    std::memset(fourier_coeffs_result, 0, sizeof(fftw_complex) * local_length);

    fft_derivative_fourier_mpi(forwards[1], in[1], fourrier_coeffs, 0, dims_sizes, comm); // dFz/dy
    add_complex_grids(fourier_coeffs_result, fourrier_coeffs, local_length);

    fft_derivative_fourier_mpi(forwards[0], in[0], fourrier_coeffs, 1, dims_sizes, comm); // dFz/dy
    sub_complex_grids(fourier_coeffs_result, fourrier_coeffs, local_length);

    fftw_execute(backward);

    std::memcpy(out_3d[2], out, local_length * 2 * sizeof(double));

    // normalize
    double divider = dims_sizes[0] * dims_sizes[1] * dims_sizes[2];
    for(int j = 0; j < out_3d.size(); j++) {
        for(int i = 0; i < local_length * 2; i++) {
            out_3d[j][i] /= divider;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    fftw_mpi_init();

    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Parse arguments
    if(argc < 2) {
        if (rank == kRootRank) {
            std::cerr << "Invalid parameters number, usage: " << argv[0] << " N\n";
        }

        MPI_Finalize();
        exit(1);
    }

    std::istringstream ss(argv[1]);
    ptrdiff_t N;

    if (!(ss >> N)) {
        if (rank == kRootRank) {
            std::cerr << "Invalid number: " << argv[1] << '\n';
        }
        
        MPI_Finalize();
        exit(1);
    } else if (!ss.eof()) {
        if (rank == kRootRank) {
            std::cerr << "Trailing characters after number: " << argv[1] << '\n';
        }
        
        MPI_Finalize();
        exit(1);
    }

    if (N <= 0) {
        if (rank == kRootRank) {
            std::cerr << "N mus be positive integer\n";
        }

        exit(1);
    }

    // 
    ptrdiff_t local_n0, local_0_start;
    std::array<ptrdiff_t, kDimsNumber> dims_sizes;
    std::fill(dims_sizes.begin(), dims_sizes.end(), N);

    auto local_length = fftw_mpi_local_size_3d(dims_sizes[0], dims_sizes[1], dims_sizes[2] / 2 + 1, comm, &local_n0, &local_0_start);

    // Grids
    std::array<double*, kDimsNumber> in; // 3*3d grids

    fftw_complex* fourier_coeffs;
    fftw_complex* fourier_coeffs_result;
    double* out;

    double* result;
    std::array<double*, kDimsNumber> result_3d; // 3*3d grids for rotor;

    // Alloc grids
    for(auto& grid: in) {
        grid = fftw_alloc_real(local_length * 2);
    }
    
    fourier_coeffs_result = fftw_alloc_complex(local_length);
    fourier_coeffs = fftw_alloc_complex(local_length);
    out = fftw_alloc_real(local_length * 2);
    result = fftw_alloc_real(local_length * 2);


    for(auto& grid: result_3d) {
        grid = fftw_alloc_real(local_length * 2);
    }

    // Analitic grids
    std::array<double*, kDimsNumber * kDimsNumber> analitic_derivatives;
    double* analitic_divergence;
    std::array<double*, kDimsNumber> analitic_rotor;

    // Alloc analic grids
    for(auto& grid: analitic_derivatives) {
        grid = fftw_alloc_real(2 * local_length);
    }

    analitic_divergence = fftw_alloc_real(2 * local_length);

    for(auto& grid: analitic_rotor) {
        grid = fftw_alloc_real(2 * local_length);
    }

    // Prepare plans
    std::vector<fftw_plan> r2c_plans = {
        fftw_mpi_plan_dft_r2c_3d(dims_sizes[0], dims_sizes[1], dims_sizes[2], in[0], fourier_coeffs, comm, FFTW_ESTIMATE),
        fftw_mpi_plan_dft_r2c_3d(dims_sizes[0], dims_sizes[1], dims_sizes[2], in[1], fourier_coeffs, comm, FFTW_ESTIMATE),
        fftw_mpi_plan_dft_r2c_3d(dims_sizes[0], dims_sizes[1], dims_sizes[2], in[2], fourier_coeffs, comm, FFTW_ESTIMATE),
    };

    fftw_plan c2r_plan = fftw_mpi_plan_dft_c2r_3d(dims_sizes[0], dims_sizes[1], dims_sizes[2], fourier_coeffs, out, comm, FFTW_BACKWARD | FFTW_ESTIMATE);
    fftw_plan c2r_plan_result = fftw_mpi_plan_dft_c2r_3d(dims_sizes[0], dims_sizes[1], dims_sizes[2], fourier_coeffs_result, out, comm, FFTW_BACKWARD | FFTW_ESTIMATE);

    // Fill grids
    std::vector<std::array<double, 3>> coefs = {
        {1, 2, 2},
        {1, 2, 1},
        {2, 1, 2}
    };

    // Fill in grid
    for(int i = 0; i < in.size(); i++) {
        fill_double_grid_mpi(in[i], dims_sizes, local_n0, local_0_start, exptrig_3d, coefs[i]);
    }
    
    // Fill analitic derivatives
    for(int f_axis = 0; f_axis < kDimsNumber; f_axis++) {
        for(int d_axis = 0; d_axis < kDimsNumber; d_axis++) {
            fill_double_grid_derivative_mpi(analitic_derivatives[f_axis * kDimsNumber + d_axis], dims_sizes, local_n0, local_0_start, exptrig_3d_derivative, coefs[f_axis], d_axis);
        }
    }

    // Fill(calc) analitic divergence
    add_double_grids(analitic_divergence, analitic_derivatives[0], local_length * 2);
    add_double_grids(analitic_divergence, analitic_derivatives[4], local_length * 2); 
    add_double_grids(analitic_divergence, analitic_derivatives[8], local_length * 2);

    // Fill(calc) analitic rotor
    add_double_grids(analitic_rotor[0], analitic_derivatives[7], local_length * 2);
    sub_double_grids(analitic_rotor[0], analitic_derivatives[5], local_length * 2);

    add_double_grids(analitic_rotor[1], analitic_derivatives[2], local_length * 2);
    sub_double_grids(analitic_rotor[1], analitic_derivatives[6], local_length * 2);

    add_double_grids(analitic_rotor[2], analitic_derivatives[3], local_length * 2);
    sub_double_grids(analitic_rotor[2], analitic_derivatives[1], local_length * 2);

    // Calculations in Fourier space
    // Fourier derivatives
    for(int f_axis = 0; f_axis < kDimsNumber; f_axis++) {
        for(int d_axis = 0; d_axis < kDimsNumber; d_axis++) {
            MPI_Barrier(comm);
            auto time_start = MPI_Wtime();

            fft_derivative_mpi(r2c_plans[f_axis], c2r_plan, in[f_axis], fourier_coeffs, out, d_axis, dims_sizes, comm);

            MPI_Barrier(comm);
            auto time_fin = MPI_Wtime();

            double local_diff = max_abs_diff(out, analitic_derivatives[f_axis * kDimsNumber + d_axis], 2 * local_length);
            std::vector<double> all_diffs(size);
            MPI_Gather(&local_diff, 1, MPI_DOUBLE, all_diffs.data(), 1, MPI_DOUBLE, 0, comm);

            if (rank == kRootRank) {
                double max_diff = *std::max_element(all_diffs.begin(), all_diffs.end());
                std::cout << "dF_" << f_axis << "/d_" << d_axis << ": " << time_fin - time_start << " sec\n";
                std::cout << "Analitic max diff: " <<  max_diff << "\n\n";
            }
        }
    }

    // Fourier divergence
    MPI_Barrier(comm);
    auto time_start = MPI_Wtime();

    fft_divirgence_mpi(r2c_plans, c2r_plan_result, in, fourier_coeffs, fourier_coeffs_result, out, dims_sizes, comm);
    
    MPI_Barrier(comm);
    auto time_fin = MPI_Wtime();

    double local_diff = max_abs_diff(out, analitic_divergence, 2 * local_length);
    std::vector<double> all_diffs(size);
    MPI_Gather(&local_diff, 1, MPI_DOUBLE, all_diffs.data(), 1, MPI_DOUBLE, 0, comm);

    if (rank == kRootRank) {
        double max_diff = *std::max_element(all_diffs.begin(), all_diffs.end());
        std::cout << "divergence: " << time_fin - time_start << " sec\n";
        std::cout << "Analitic max diff: " <<  max_diff << "\n\n";
    }

    // Fourier rotor
    MPI_Barrier(comm);
    time_start = MPI_Wtime();

    fft_rotor_mpi(r2c_plans, c2r_plan_result, in, fourier_coeffs, fourier_coeffs_result, out, result_3d, dims_sizes, comm);
    
    MPI_Barrier(comm);
    time_fin = MPI_Wtime();

    std::array<double, kDimsNumber> local_diffs;
    for(int i = 0; i < kDimsNumber; i++) {
        local_diffs[i] = max_abs_diff(result_3d[i], analitic_rotor[i], local_length * 2);
    }
    local_diff = *std::max_element(local_diffs.begin(), local_diffs.end());

    MPI_Gather(&local_diff, 1, MPI_DOUBLE, all_diffs.data(), 1, MPI_DOUBLE, 0, comm);

    if (rank == kRootRank) {
        double max_diff = *std::max_element(all_diffs.begin(), all_diffs.end());
        std::cout << "rotor: " << time_fin - time_start << " sec\n";
        std::cout << "Analitic max diff: " <<  max_diff << "\n\n";
    }

    // Destroy plans
    for(auto& r2c_plan: r2c_plans) {
        fftw_destroy_plan(r2c_plan);
    }
    fftw_destroy_plan(c2r_plan);

    // Free grids
    for(auto& grid: result_3d) {
        fftw_free(grid);
    }

    fftw_free(result);
    fftw_free(out);
    fftw_free(fourier_coeffs);
    
    for(auto& grid: in) {
        fftw_free(grid);
    }

    MPI_Finalize();

    return 0;
}
