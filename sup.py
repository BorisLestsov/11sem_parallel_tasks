
import numpy as np



class Solver:
    def __init__(self, Lx, Ly, Lz, T, N, K, anal):
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.T = T
        self.N = N
        self.K = K
        self.anal = anal

        self.hx = self.Lx / (N-1)
        self.hy = self.Ly / (N-1)
        self.hz = self.Lz / (N-1)
        self.tau = self.T / (K-1)


    def solve(self):
        self.a = self.anal(self.Lx, self.Ly, self.Lz, self.T, self.N, self.K)
        self.u = np.zeros_like(self.a)

        self.phi = self.a[:, :, :, 0]

        self.u[:, :, :, 0] = self.a[:, :, :, 0]
        self.u[:, :, :, 1] = self.u[:,:,:,0] + self.tau ** 2 * self.laplasian(self.u[:,:,:,0]) / 2


        for i in range(2, self.K):
            self.u[:,:,:,i] = 2*self.u[:,:,:,i-1] - self.u[:,:,:,i-2] + self.tau ** 2 * self.laplasian(self.u[:,:,:,i-1])

        # tmp = self.u[:, :, :, 3]
        # for i in tmp.flatten():
        #     print(i)

        for i in range(0, self.K):
            print(i, np.max(np.abs(self.a[:,:,:,i] - self.u[:,:,:,i])))


    def laplasian(self, a):
        # padded = np.pad(a, 1, 'constant')
        # out = (padded[:-2,1:-1,1:-1] - 2 * padded[1:-1,1:-1,1:-1] + padded[2:,1:-1,1:-1]) / self.hx ** 2 + \
        #     (padded[1:-1,:-2,1:-1] - 2 * padded[1:-1,1:-1,1:-1] + padded[1:-1,2:,1:-1]) / self.hy ** 2 + \
        #     (padded[1:-1,1:-1,:-2] - 2 * padded[1:-1,1:-1,1:-1] + padded[1:-1,1:-1,2:]) / self.hz ** 2

        out = np.zeros_like(a)
        out[1:-1, 1:-1, 1:-1] = (a[:-2,1:-1,1:-1] - 2 * a[1:-1,1:-1,1:-1] + a[2:,1:-1,1:-1]) / self.hx ** 2 + \
                                (a[1:-1,:-2,1:-1] - 2 * a[1:-1,1:-1,1:-1] + a[1:-1,2:,1:-1]) / self.hy ** 2 + \
                                (a[1:-1,1:-1,:-2] - 2 * a[1:-1,1:-1,1:-1] + a[1:-1,1:-1,2:]) / self.hz ** 2

        # out[1:-1, 1:-1,   -1] = (a[:-2,1:-1,-2] - 2 * a[1:-1,1:-1,-1] + a[2:,1:-1,1]) / self.hx ** 2 + \
        #                         (a[1:-1,:-2,-2] - 2 * a[1:-1,1:-1,-1] + a[1:-1,2:,1]) / self.hy ** 2 + \
        #                         (a[1:-1,1:-1,-2] - 2 * a[1:-1,1:-1,-1] + a[1:-1,1:-1,1]) / self.hz ** 2
        # out[1:-1, 1:-1,    0] = out[1:-1, 1:-1,   -1]

        out[0,:,:] = 0
        out[-1,:,:] = 0
        out[:,0,:] = 0
        out[:,-1,:] = 0
        out[:,:,0] = 0
        out[:,:,-1] = 0

        return out



def main():
    Lx = 1
    Ly = 1
    Lz = 1
    N = 8

    K = 10
    T = 1e-2

    def anal(Lx, Ly, Lz, T, N, K):
        xx = np.arange(N, dtype=np.float32) / N * Lx
        yy = np.arange(N, dtype=np.float32) / N * Ly
        zz = np.arange(N, dtype=np.float32) / N * Lz
        tt = np.arange(K, dtype=np.float32) / K * T

        xx = np.sin(np.pi/Lx*xx)
        yy = np.sin(np.pi/Ly*yy)
        zz = np.sin(np.pi/Lz*zz)
        tt = np.cos(tt+2*np.pi)

        ret = xx[:, None, None, None] * \
              yy[None, :, None, None] * \
              zz[None, None, :, None] * \
              tt[None, None, None, :]
        return ret



    solver = Solver(Lx, Ly, Lz, T, N, K, anal)

    solver.solve()






if __name__=="__main__":
    main()
