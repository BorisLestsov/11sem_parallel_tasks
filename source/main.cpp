#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <math.h>
#include "mpi.h"

#include "ndarr.hpp"



float Lx = -1;
float Ly = -1;
float Lz = -1;
float T = -1;

int N = -1;
int K = -1;

float hx = -1;
float hy = -1;
float hz = -1;
float tau =-1;
float mult=-1;


int rank = 0, comm_size;
MPI_Comm MPI_CART_COMM;
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
                // if (i == 0 && j == 0 && k == 0)
                //     std::cout << "ERR " << i<<j<<k<< "    " << arr1(i,j,k) << "  " << arr2(i,j,k) << std::endl;
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
        xx[ii] = sin(M_PI/Lx * i/(N-1) * Lx);
    }
    for (int ii=0; ii < Ny; ++ii){
        float i = ii + Sy;
        yy[ii] = sin(M_PI/Lx * i/(N-1) * Ly);
    }
    for (int ii=0; ii < Nz; ++ii){
        float i = ii + Sz;
        zz[ii] = sin(M_PI/Lz * i/(N-1) * Lz);
    }

    float t_val = cos(((float)t)/K * T + 2*M_PI);
    for (int i = 0; i < Nx; ++i){
        for (int j = 0; j < Ny; ++j){
            for (int k = 0; k < Nz; ++k){
                res(i, j, k) = (xx[i] * yy[j] * zz[k] * t_val)/mult;
            }
        }
    }
}

void send_recv(int src, int dst, NdArr& send, NdArr& recv){
    MPI_Status status;

    // if (src >= 0){
    //     if (dst >= 0){
            MPI_Sendrecv(send.arr, send.size(), MPI_FLOAT,
                        dst, 0, recv.arr, recv.size(),
                        MPI_FLOAT, src, 0,
                        MPI_CART_COMM, &status);
    //     } else {
    //         MPI_Send(send.arr, send.size(), MPI_FLOAT, dst, 0, MPI_CART_COMM);
    //     }
    //
    // } else {
    //     if (dst >= 0){
    //         MPI_Recv(recv.arr, recv.size(), MPI_FLOAT, src, 0, MPI_CART_COMM, &status);
    //     } else {
    //     }
    // }
}


void laplacian(NdArr& a, NdArr& out){

    float hx2 = hx*hx;
    float hy2 = hy*hy;
    float hz2 = hz*hz;

    for (int i = 0; i < Nx-0; ++i)
        for (int j = 0; j < Ny-0; ++j)
            for (int k = 1; k < Nz-1; ++k)
                out(i, j, k) = 100000;
                // out(i, j, k) = 0;

    /*
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
    */


    // send, recv ...

    std::vector<int> shape_x(2);
    shape_x[0] = Ny;
    shape_x[1] = Nz;
    NdArr max_x_send0(shape_x), max_x_sendN(shape_x);
    for (int j = 0; j < Ny; ++j){
        for (int k = 0; k < Nz; ++k){
            out(0   , j, k) = 0;
            out(Nx-1, j, k) = 0;
            max_x_send0(j, k) = a(0, j, k);
            max_x_sendN(j, k) = a(Nx-1, j, k);
        }
    }
    NdArr max_x_recv0(shape_x), max_x_recvN(shape_x);

    std::vector<int> shape_y(2);
    shape_y[0] = Nx;
    shape_y[1] = Nz;
    NdArr max_y_send0(shape_y), max_y_sendN(shape_y);
    for (int i = 0; i < Nx; ++i){
        for (int k = 0; k < Nz; ++k){
            out(i, 0,    k) = 0;
            out(i, Ny-1, k) = 0;
            max_y_send0(i, k) = a(i, 0   , k);
            max_y_sendN(i, k) = a(i, Ny-1, k);
        }
    }
    NdArr max_y_recv0(shape_y), max_y_recvN(shape_y);

    std::vector<int> shape_z(2);
    shape_z[0] = Nx;
    shape_z[1] = Ny;
    NdArr max_z_send0(shape_z), max_z_sendN(shape_z);
    for (int i = 0; i < Nx; ++i){
        for (int j = 0; j < Ny; ++j){
            out(i, j, 0   ) = 0;
            out(i, j, Nz-1) = 0;
            max_z_send0(i, j) = a(i, j,    0);
            max_z_sendN(i, j) = a(i, j, Nz-1);
        }
    }
    NdArr max_z_recv0(shape_z), max_z_recvN(shape_z);


    int src = -1, dst = -1;

    MPI_Cart_shift(MPI_CART_COMM, 0, -1, &src, &dst);
    send_recv(src, dst, max_x_send0, max_x_recvN);
    MPI_Cart_shift(MPI_CART_COMM, 0, 1, &src, &dst);
    send_recv(src, dst, max_x_sendN, max_x_recv0);

    MPI_Cart_shift(MPI_CART_COMM, 1, -1, &src, &dst);
    send_recv(src, dst, max_y_send0, max_y_recvN);
    MPI_Cart_shift(MPI_CART_COMM, 1, 1, &src, &dst);
    send_recv(src, dst, max_y_sendN, max_y_recv0);

    MPI_Cart_shift(MPI_CART_COMM, 2, -1, &src, &dst);
    send_recv(src, dst, max_z_send0, max_z_recvN);
    MPI_Cart_shift(MPI_CART_COMM, 2, 1, &src, &dst);
    send_recv(src, dst, max_z_sendN, max_z_recv0);

    std::vector<int> p_shape(3);
    p_shape[0] = a.shape[0]+2;
    p_shape[1] = a.shape[1]+2;
    p_shape[2] = a.shape[2]+2;
    NdArr p(p_shape, 100000);

    for (int i = 0; i < Nx; ++i){
        for (int j = 0; j < Ny; ++j){
            for (int k = 0; k < Nz; ++k){
                p(i+1, j+1, k+1) = a(i, j, k);
            }
        }
    }
    for (int j = 0; j < Ny; ++j){
        for (int k = 0; k < Nz; ++k){
            int i;

            i = 0;
            p(i, j+1, k+1) = max_x_recv0(j, k);

            i = Nx+1;
            p(i, j+1, k+1) = max_x_recvN(j, k);
        }
    }
    for (int i = 0; i < Nx; ++i){
        for (int k = 0; k < Nz; ++k){
            int j;

            j = 0;
            p(i+1, j, k+1) = max_y_recv0(i, k);

            j = Ny+1;
            p(i+1, j, k+1) = max_y_recvN(i, k);
        }
    }
    for (int i = 0; i < Nx; ++i){
        for (int j = 0; j < Ny; ++j){
            int k;

            k = 0;
            p(i+1, j+1, k) = max_z_recv0(i, j);

            k = Nz+1;
            p(i+1, j+1, k) = max_z_recvN(i, j);
        }
    }
    /*
    */

    // for (int i = 0; i < Nx; ++i){
    //     for (int j = 0; j < Ny; ++j){
    //         for (int k = 0; k < Nz; ++k){
    //             std::cout << i<<j<<k<< "  " <<p(i,j,k) << std::endl;
    //         }
    //     }
    // }


    for (int i = 0; i < Nx; ++i){
        for (int j = 0; j < Ny; ++j){
            for (int k = 0; k < Nz; ++k){
                int p_i = i+1;
                int p_j = j+1;
                int p_k = k+1;

                out(i, j, k) =
                    (p(p_i-1, p_j, p_k) - 2*p(p_i, p_j, p_k) + p(p_i+1, p_j, p_k)) / hx2 +
                    (p(p_i, p_j-1, p_k) - 2*p(p_i, p_j, p_k) + p(p_i, p_j+1, p_k)) / hy2 +
                    (p(p_i, p_j, p_k-1) - 2*p(p_i, p_j, p_k) + p(p_i, p_j, p_k+1)) / hz2;
                out(i, j, k) /= mult;

                // std::cout << i << "  " << j << "  " << k << "  " << out(i, j, k) << std::endl;
            }
        }
    }

    //TODO: check ranges
    /*
    for (int j = 0; j < Ny; ++j){
        for (int k = 0; k < Nz; ++k){

            // if (rank == 0){
            //     std::cout << j << "  " << k << "    " << max_x_recvN(j, k) << std::endl;;
            // }
            a_j_m_1 = (j==0)?max_y_recv0(i, k):a(i, j-1, k);

            int i;

            i = 0;
            out(i, j, k) += (max_x_recv0(j, k)  -  2*a(i, j, k)  +  a(i+1, j, k)) / hx2;
            out(i, j, k) += (a(i, j-1, k)       -  2*a(i, j, k)  +  a(i, j+1, k)) / hy2;
            out(i, j, k) += (a(i, j, k-1)       -  2*a(i, j, k)  +  a(i, j, k+1)) / hz2;

            i = Nx-1;
            out(i, j, k) += (a(i-1, j, k)       -  2*a(i, j, k)  +  max_x_recvN(j, k)) / hx2;
            out(i, j, k) += (a(i, j-1, k)       -  2*a(i, j, k)  +  a(i, j+1, k)     ) / hy2;
            out(i, j, k) += (a(i, j, k-1)       -  2*a(i, j, k)  +  a(i, j, k+1)     ) / hz2;
        }
    }
    for (int i = 0; i < Nx; ++i){
        for (int k = 0; k < Nz-1; ++k){
            int j;

            j = 0;
            out(i, j, k) += (a(i-1, j, k)       -  2*a(i, j, k)  +  a(i+1, j, k)) / hx2;
            out(i, j, k) += (max_y_recv0(i, k)  -  2*a(i, j, k)  +  a(i, j+1, k)) / hy2;
            out(i, j, k) += (a(i, j, k-1)       -  2*a(i, j, k)  +  a(i, j, k+1)) / hz2;

            j = Ny-1;
            out(i, j, k) += (a(i-1, j, k)       -  2*a(i, j, k)  +  a(i+1, j, k)     ) / hx2;
            out(i, j, k) += (a(i, j-1, k)       -  2*a(i, j, k)  +  max_y_recvN(i, k)) / hy2;
            out(i, j, k) += (a(i, j, k-1)       -  2*a(i, j, k)  +  a(i, j, k+1)     ) / hz2;
        }
    }
    for (int i = 0; i < Nx-1; ++i){
        for (int j = 0; j < Ny-1; ++j){
            int k;

            k = 0;
            out(i, j, k) += (a(i-1, j, k)       -  2*a(i, j, k)  +  a(i+1, j, k)) / hx2;
            out(i, j, k) += (a(i, j-1, k)       -  2*a(i, j, k)  +  a(i, j+1, k)) / hy2;
            out(i, j, k) += (max_z_recv0(i, j)  -  2*a(i, j, k)  +  a(i, j, k+1)) / hz2;

            k = Nz-1;
            out(i, j, k) += (a(i-1, j, k)       -  2*a(i, j, k)  +  a(i+1, j, k)   ) / hx2;
            out(i, j, k) += (a(i, j-1, k)       -  2*a(i, j, k)  +  a(i, j+1, k)   ) / hy2;
            out(i, j, k) += (a(i, j, k-1)       -  2*a(i, j, k)  +  max_z_recvN(i, j)) / hz2;
        }
    }
    */


    // periodic z
    // for (int i = 1; i < Nx-1; ++i){
    //     for (int j = 1; j < Ny-1; ++j){
    //         out(i, j, Nz-1) =
    //             (a(i-1, j, Nz-2) - 2*a(i, j, Nz-1) + a(i+1, j, 1)) / hx2 +
    //             (a(i, j-1, Nz-2) - 2*a(i, j, Nz-1) + a(i, j+1, 1)) / hy2 +
    //             (a(i, j,   Nz-2) - 2*a(i, j, Nz-1) + a(i, j,   1)) / hz2;
    //     }
    // }

    if (Px == 0) {
        for (int j = 0; j < Ny; ++j){
            for (int k = 0; k < Nz; ++k){
                out(0   , j,    k) = 0;
            }
        }
    }
    if (Px == NPx-1) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k){
                out(Nx-1, j,    k) = 0;
            }
        }
    }



    if (Py == 0) {
        for (int i = 0; i < Nx; ++i){
            for (int k = 0; k < Nz; ++k){
                out(i   , 0   , k) = 0;
            }
        }
    }
    if (Py == NPy-1) {
        for (int i = 0; i < Nx; ++i){
            for (int k = 0; k < Nz; ++k){
                out(i   , Ny-1, k) = 0;
            }
        }
    }


            // type 1
    if (Pz == 0) {
        for (int i = 0; i < Nx; ++i){
            for (int j = 0; j < Ny; ++j){
                out(i, j, 0  ) = 0;
            }
        }
    }
    if (Pz == NPz-1) {
        for (int i = 0; i < Nx; ++i){
            for (int j = 0; j < Ny; ++j){
                out(i, j, Nz-1) = 0;
            }
        }
    }
    /*
    */

}



int main(int argc, char** argv){

    Lx = atof(argv[1]);
    Ly = atof(argv[2]);
    Lz = atof(argv[3]);
    T = atof(argv[4]);
    N = atoi(argv[5]);
    K = atoi(argv[6]);

    hx = Lx / (N-1);
    hy = Ly / (N-1);
    hz = Lz / (N-1);
    tau = T / (K-1);
    mult = M_PI*M_PI * (1.0/(Lx*Lx) + 1.0/(Ly*Ly) + 1./(Lz*Lz));


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
        int dims[] = {NPx, NPy, NPz};
        int coords[] = {-1, -1, -1};


        // TODO: FIX 0, 0, 1
        int periods[] = {0, 0, 0};

        MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, true, &MPI_CART_COMM);
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Cart_coords(MPI_CART_COMM, rank, 3, coords);


        Nx = N/NPx;
        Ny = N/NPy;
        Nz = N/NPz;
        loc_shape[0] = Nx;
        loc_shape[1] = Ny;
        loc_shape[2] = Nz;


        // Px = rank/NPy/NPz;
        // Py = (rank - Px*NPy*NPz) / NPy;
        // Pz = (rank - Px*NPy*NPz - Py*NPy);
        Px = coords[0];
        Py = coords[1];
        Pz = coords[2];

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
