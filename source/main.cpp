#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include "mpi.h"

#include "ndarr.hpp"



float Lx = 1;
float Ly = 1;
float Lz = 1;
float T = 1e-2;

int N = 8;
int K = 10;

float hx = Lx / (N-1);
float hy = Ly / (N-1);
float hz = Lz / (N-1);
float tau = T / (K-1);


int rank = 0, comm_size;
std::vector<int> loc_shape(3);
int Nx, Ny, Nz;
int Sx, Sy, Sz;
int Px, Py, Pz;
int NPx, NPy, NPz;




float cmp_arr(NdArr& arr1, NdArr& arr2){
    float err;
    float max_err = 0;
    float max_max_err;

    for (int i = 0; i < Nx; ++i){
        for (int j = 0; j < Ny; ++j){
            for (int k = 0; k < Nz; ++k){
                err = abs(arr1(i, j, k) - arr2(i, j, k));
                if (err > max_err)
                    max_err = err;
            }
        }
    }

    MPI_Reduce(&max_err, &max_max_err, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    return (rank==0)?max_max_err:100500;
}


void anal(NdArr& res, int t){
    std::vector<float> xx(Nx, 0);
    std::vector<float> yy(Ny, 0);
    std::vector<float> zz(Nz, 0);

    for (int ii=0; ii < Nx; ++ii){
        float i = ii + Sx;
        xx[ii] = sin(M_PI/Lx * i/N * Lx);
    }
    for (int ii=0; ii < Ny; ++ii){
        float i = ii + Sy;
        yy[ii] = sin(M_PI/Lx * i/N * Ly);
    }
    for (int ii=0; ii < Nz; ++ii){
        float i = ii + Sz;
        zz[ii] = sin(M_PI/Lz * i/N * Lz);
    }

    float t_val = cos(((float)t)/K * T + 2*M_PI);
    for (int i = 0; i < Nx; ++i){
        for (int j = 0; j < Ny; ++j){
            for (int k = 0; k < Nz; ++k){
                res(i, j, k) = xx[i] * yy[j] * zz[k] * t_val;
            }
        }
    }
}

void laplacian(NdArr& a, NdArr& out){

    float hx2 = hx*hx;
    float hy2 = hy*hy;
    float hz2 = hz*hz;


    for (int i = 1; i < Nx-1; ++i){
        for (int j = 1; j < Ny-1; ++j){
            for (int k = 1; k < Nz-1; ++k){
                out(i, j, k) =
                    (a(i-1, j, k) - 2*a(i, j, k) + a(i+1, j, k)) / hx2 +
                    (a(i, j-1, k) - 2*a(i, j, k) + a(i, j+1, k)) / hy2 +
                    (a(i, j, k-1) - 2*a(i, j, k) + a(i, j, k+1)) / hz2;

                // std::cout << i << "  " << j << "  " << k << "  " << out(i, j, k) << std::endl;
            }
        }
    }

    // send, recv ...


    // periodic z
    for (int i = 1; i < Nx-1; ++i){
        for (int j = 1; j < Ny-1; ++j){
            out(i, j, Nz-1) =
                (a(i-1, j, Nz-2) - 2*a(i, j, Nz-1) + a(i+1, j, 1)) / hx2 +
                (a(i, j-1, Nz-2) - 2*a(i, j, Nz-1) + a(i, j+1, 1)) / hy2 +
                (a(i, j,   Nz-2) - 2*a(i, j, Nz-1) + a(i, j,   1)) / hz2;
        }
    }

    for (int i = 0; i < Nx; ++i){
        for (int j = 0; j < Ny; ++j){
            out(0   , i,    j) = 0;
            out(Nx-1, i,    j) = 0;
            out(i   , 0   , j) = 0;
            out(i   , Ny-1, j) = 0;

            // type 1
            // out(i, j, 0  ) = 0;
            // out(i, j, N-1) = 0;
        }
    }

}



int main(int argc, char** argv){


    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);


    try {
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        int n_side = round(cbrt(comm_size));
        if (n_side * n_side * n_side != comm_size)
            throw std::string("wrong num of processes");

    // std::cout << "KEK" << std::endl;    return 3;
        NPx = n_side;
        NPy = n_side;
        NPz = n_side;

        Nx = N/NPx;
        Ny = N/NPy;
        Nz = N/NPz;
        loc_shape[0] = Nx;
        loc_shape[1] = Ny;
        loc_shape[2] = Nz;


        Px = rank/NPy/NPz;
        Py = (rank - Px*NPy*NPz) / NPy;
        Pz = (rank - Px*NPy*NPz - Py*NPy);

        Sx = Px*Nx;
        Sy = Py*Ny;
        Sz = Pz*Nz;

        NdArr* a_p = new NdArr(loc_shape);
        NdArr* u_p = new NdArr(loc_shape);
        NdArr* u_prev_p = new NdArr(loc_shape);
        NdArr* u_prev_prev_p = new NdArr(loc_shape);
        NdArr* lap_p = new NdArr(loc_shape);


        NdArr& a = *a_p;
        NdArr& u = *u_p;
        NdArr& u_prev = *u_prev_p;
        NdArr& u_prev_prev = *u_prev_prev_p;
        NdArr& lap = *lap_p;


        std::ofstream out("out/out_" + std::to_string(rank) + ".txt", std::ios::out);

        out << rank
            << "    NPx: " << NPx << " Npy " << NPy << " Npz " << NPz
            << "    Px: " << Px << " Py " << Py << " Pz " << Pz
            << "    Sx: " << Sx << " Sy " << Sy << " Sz " << Sz
            << "    Nx: " << Nx << " Ny " << Ny << " Nz " << Nz
            << std::endl;


        // step 0
        {
            anal(a, 0);

            for (int i = 0; i < Nx; ++i){
                for (int j = 0; j < Ny; ++j){
                    for (int k = 0; k < Nz; ++k){
                        u_prev_prev(i, j, k) = a(i, j, k);
                    }
                }
            }
            float max_max_err = cmp_arr(u_prev_prev, a);
            if (rank == 0)
                std::cout << 0 << " Err: " << max_max_err << std::endl;
        }

        // step 1
        {
            anal(a, 1);
            laplacian(u_prev_prev, lap);

            for (int i = 0; i < Nx; ++i){
                for (int j = 0; j < Ny; ++j){
                    for (int k = 0; k < Nz; ++k){
                        u_prev(i, j, k) = u_prev_prev(i, j, k) + tau*tau * lap(i, j, k) / 2;
                    }
                }
            }
            float max_max_err = cmp_arr(u_prev, a);
            if (rank == 0)
                std::cout << 1 << " Err: " << max_max_err << std::endl;
        }

        // step 2 -> K
        for (int t = 2; t < K; t++){

            NdArr& u = *u_p;
            NdArr& u_prev = *u_prev_p;
            NdArr& u_prev_prev = *u_prev_prev_p;


            anal(a, t);
            laplacian(u_prev, lap);

            for (int i = 0; i < Nx; ++i){
                for (int j = 0; j < Ny; ++j){
                    for (int k = 0; k < Nz; ++k){
                        u(i, j, k) = 2*u_prev(i, j, k) - u_prev_prev(i, j, k) + tau*tau * lap(i, j, k);
                        // if (t == 4)
                        //     std::cout << u_prev(i,j,k) << std::endl;
                    }
                }
            }

            float max_max_err = cmp_arr(u, a);
            if (rank == 0)
                std::cout << t << " Err: " <<  max_max_err << std::endl;

            NdArr* tmp = u_prev_prev_p;
            u_prev_prev_p = u_prev_p;
            u_prev_p = u_p;
            u_p = tmp;

        }
    }
    catch (const std::string& e) {
        std::cerr << "rank " << rank << ": " << e << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    MPI_Finalize();

    return 0;
}
