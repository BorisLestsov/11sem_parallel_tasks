#include <iostream>
#include <vector>
#include <math.h>


class NdArr{
public:

    NdArr(const std::vector<int>& shape_):
        shape(shape_),
        Nd(shape_.size()),
        strides(shape.size(), 1)
    {
        arr_size = 1;
        for (int i = 0; i < shape_.size(); ++i){
            arr_size *= shape_[i];
        }
        strides[shape_.size()-1] = 1;
        for (int i = shape_.size()-2; i >= 0; --i){
            strides[i] = strides[i+1]*shape_[i+1];
        }
        // for (int i = 0; i < shape_.size(); ++i){
        //     std::cout << strides[i] << std::endl;
        // }
        arr = new double[arr_size];
    }

    ~NdArr(){
        delete []arr;
    }

    double& operator()(int i){
        return arr[i];
    }

    double& operator()(int i, int j){
        return arr[i*strides[0] + j];
    }

    double& operator()(int i, int j, int k){
        return arr[i*strides[0] + j*strides[1] + k];
    }

    double& operator()(int i, int j, int k, int l){
        return arr[i*strides[0] + j*strides[1] + k*strides[2] + l];
    }

    int size(){
        return arr_size;
    }


    double* arr;
    int Nd;
    std::vector<int> shape;
    std::vector<int> strides;
    int arr_size;

};

void anal(NdArr& res, double Lx, double Ly, double Lz, double T, int N, int K, int t){
    std::vector<double> xx(N, 0);
    std::vector<double> yy(N, 0);
    std::vector<double> zz(N, 0);

    for (int ii=0; ii < N; ++ii){
        double i = ii;
        xx[i] = sin(M_PI/Lx * i/N * Lx);
        yy[i] = sin(M_PI/Lx * i/N * Ly);
        zz[i] = sin(M_PI/Lz * i/N * Lz);
    }

    double t_val = cos(((double)t)/K * T + 2*M_PI);
    for (int i = 0; i < N; ++i){
        for (int j = 0; j < N; ++j){
            for (int k = 0; k < N; ++k){
                res(i, j, k) = xx[i] * yy[j] * zz[k] * t_val;
            }
        }
    }

}

void laplacian(NdArr& a, NdArr& out, double hx, double hy, double hz){
    int N = a.shape[0];
    int K = a.shape[3];

    double hx2 = hx*hx;
    double hy2 = hy*hy;
    double hz2 = hz*hz;


    for (int i = 1; i < N-1; ++i){
        for (int j = 1; j < N-1; ++j){
            for (int k = 1; k < N-1; ++k){
                out(i, j, k) =
                    (a(i-1, j, k) - 2*a(i, j, k) + a(i+1, j, k)) / hx2 +
                    (a(i, j-1, k) - 2*a(i, j, k) + a(i, j+1, k)) / hy2 +
                    (a(i, j, k-1) - 2*a(i, j, k) + a(i, j, k+1)) / hz2;

                // std::cout << i << "  " << j << "  " << k << "  " << out(i, j, k) << std::endl;
            }
        }
    }

    for (int i = 0; i < N; ++i){
        for (int j = 0; j < N; ++j){
            out(0  , i, j) = 0;
            out(N-1, i, j) = 0;
            out(i, 0  , j) = 0;
            out(i, N-1, j) = 0;
            out(i, j, 0  ) = 0;
            out(i, j, N-1) = 0;
        }
    }

    // out[1:-1, 1:-1, 1:-1] = (a[:-2,1:-1,1:-1] - 2 * a[1:-1,1:-1,1:-1] + a[2:,1:-1,1:-1]) / self.hx ** 2 + \
    //                         (a[1:-1,:-2,1:-1] - 2 * a[1:-1,1:-1,1:-1] + a[1:-1,2:,1:-1]) / self.hy ** 2 + \
    //                         (a[1:-1,1:-1,:-2] - 2 * a[1:-1,1:-1,1:-1] + a[1:-1,1:-1,2:]) / self.hz ** 2
    //
    // out[1:-1, 1:-1,   -1] = (a[:-2,1:-1,-2] - 2 * a[1:-1,1:-1,-1] + a[2:,1:-1,1]) / self.hx ** 2 + \
    //                         (a[1:-1,:-2,-2] - 2 * a[1:-1,1:-1,-1] + a[1:-1,2:,1]) / self.hy ** 2 + \
    //                         (a[1:-1,1:-1,-2] - 2 * a[1:-1,1:-1,-1] + a[1:-1,1:-1,1]) / self.hz ** 2
    // out[1:-1, 1:-1,    0] = out[1:-1, 1:-1,   -1]
    //
    // out[0,:,:] = 0
    // out[-1,:,:] = 0
    // out[:,0,:] = 0
    // out[:,-1,:] = 0

}


int main(){
    double Lx = 1;
    double Ly = 1;
    double Lz = 1;
    double T = 1e-2;

    int N = 6;
    int K = 20;

    double hx = Lx / (N-1);
    double hy = Ly / (N-1);
    double hz = Lz / (N-1);
    double tau = T / (K-1);

    std::vector<int> shape(3);
    shape[0] = N;
    shape[1] = N;
    shape[2] = N;

    NdArr* a_p = new NdArr(shape);
    NdArr* u_p = new NdArr(shape);
    NdArr* u_prev_p = new NdArr(shape);
    NdArr* u_prev_prev_p = new NdArr(shape);
    NdArr* lap_p = new NdArr(shape);

    NdArr& a = *a_p;
    NdArr& u = *u_p;
    NdArr& u_prev = *u_prev_p;
    NdArr& u_prev_prev = *u_prev_prev_p;
    NdArr& lap = *lap_p;

    anal(a, Lx, Ly, Lz, T, N, K, 0);

    for (int i = 0; i < N; ++i){
        for (int j = 0; j < N; ++j){
            for (int k = 0; k < N; ++k){
                u_prev_prev(i, j, k) = a(i, j, k);
            }
        }
    }

    laplacian(u_prev_prev, lap, hx, hy, hz);

    for (int i = 0; i < N; ++i){
        for (int j = 0; j < N; ++j){
            for (int k = 0; k < N; ++k){
                u_prev(i, j, k) = u_prev_prev(i, j, k) + tau*tau * lap(i, j, k) / 2;
            }
        }
    }

    for (int t = 2; t < K; t++){
        double err;
        double max_err = 0;


        anal(a, Lx, Ly, Lz, T, N, K, t);
        laplacian(u_prev, lap, hx, hy, hz);

        for (int i = 0; i < N; ++i){
            for (int j = 0; j < N; ++j){
                for (int k = 0; k < N; ++k){
                    u(i, j, k) = 2*u_prev(i, j, k) - u_prev_prev(i, j, k) + tau*tau * lap(i, j, k);
                    // if (t == 4)
                    //     std::cout << u_prev(i,j,k) << std::endl;

                    err = abs(u(i, j, k) - a(i, j, k));
                    if (err > max_err)
                        max_err = err;
                }
            }
        }

        NdArr* tmp = u_prev_prev_p;
        u_prev_prev_p = u_prev_p;
        u_prev_p = u_p;
        u_p = tmp;

        NdArr& u = *u_p;
        NdArr& u_prev = *u_prev_p;
        NdArr& u_prev_prev = *u_prev_prev_p;

        std::cout << "Err: " <<  max_err << std::endl;
    }

    return 0;
}
