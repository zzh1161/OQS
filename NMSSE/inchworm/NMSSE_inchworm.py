import numpy as np
import math
import time

try:
    from ..utils.noise_generator import ColoredNoiseGenerator_FourierFiltering
    from ..utils.noise_generator import ColoredNoiseGenerator_Cholesky
except ImportError:
    from utils.noise_generator import ColoredNoiseGenerator_FourierFiltering
    from utils.noise_generator import ColoredNoiseGenerator_Cholesky

try:
    from . import pairing as pr
except ImportError:
    import pairing as pr


class NMSSE_Inchworm:
    def __init__(self,
                 Hs, L, tmax, bath_corr, N_steps,
                 N_trunc = 3, N_MC_samples = 300,
                 noise_generator = ColoredNoiseGenerator_Cholesky,
                ):
        self.dim = Hs.shape[0] # dimension of the physical hilbert space
        self.Hs = Hs
        self.L = L
        E, V = np.linalg.eigh(Hs)
        self.propagator = lambda t: V @ np.diag(np.exp(-1j * E * t)) @ V.conj().T
        self.Lint = lambda t: self.propagator(-t) @ L @ self.propagator(t)

        self.tmax = tmax
        self.bath_corr = bath_corr
        self.N_trunc = N_trunc
        self.N_MC_samples = N_MC_samples
        self.N_steps = N_steps
        self.t_grid = np.linspace(0, tmax, N_steps+1)
        self.dt = self.t_grid[1] - self.t_grid[0]

        self.noise_generator = noise_generator(
            bath_corr, tmax, N_steps
        )

        self.G = [[None for _ in range(self.N_steps+1)] for _ in range(self.N_steps+1)]


    def initialize(self):
        N = self.N_steps + 1
        for i in range(N):
            self.G[i][:] = [None] * N
        for n in range(N):
            self.G[n][n] = np.eye(self.dim)


    def compute_realization(
            self,
            N_traj,
            psi0,
        ):
        N = self.N_steps + 1
        psis = np.zeros((N_traj, N, self.dim), dtype=complex)
        psis[:, 0, :] = psi0

        for traj in range(N_traj):
            z = self.noise_generator.sample_process()
            self.initialize()

            for offset in range(1, N): 
                for j in range(offset, N):
                    k = j - offset

                    # Heun's step 1
                    # t0 = time.perf_counter()
                    int_temp = sum(self.MC_int(m, self.t_grid[j-1], self.t_grid[k]) for m in range(1, self.N_trunc + 1, 2))
                    self.G[j][k] = self.G[j-1][k] \
                                  + self.dt * z[j-1] * self.Lint(self.t_grid[j-1]) @ self.G[j-1][k] \
                                  + self.dt * int_temp
                    # t1 = time.perf_counter()
                    # print(f"Traj {traj+1}, G({j},{k}) Heun's Step 1 took {t1 - t0:.4f} seconds.")
                    
                    # Heun's step 2
                    # t0 = time.perf_counter()
                    int_temp = sum(self.MC_int(m, self.t_grid[j], self.t_grid[k]) for m in range(1, self.N_trunc + 1, 2))
                    self.G[j][k] = 0.5 * (self.G[j-1][k] + self.G[j][k]) \
                                  + 0.5 * self.dt * z[j] * self.Lint(self.t_grid[j]) @ self.G[j][k] \
                                  + 0.5 * self.dt * int_temp
                    # t1 = time.perf_counter()
                    # print(f"Traj {traj+1}, G({j},{k}) Heun's Step 2 took {t1 - t0:.4f} seconds.")
                    
            for step in range(1, N):
                psis[traj, step, :] = self.G[step][0] @ psi0

        return psis


    def L_hat(self, s, q):
        """
        s:   time snapshot
        q:   pairing rule
        """
        on_left = any(s == l for l, _ in q)
        return self.Lint(s) if on_left else self.Lint(-s)
    
    
    def G_interp(self, t_f, t_i):
        """
        Interpolate the propagator G(t_f, t_i) from stored values
        """
        if t_f < t_i:
            raise ValueError("t_f must be greater than or equal to t_i.")
        if t_f == t_i:
            return np.eye(self.dim)
        
        j = np.where(t_f >= self.t_grid)[0][-1]
        k = np.where(t_i >= self.t_grid)[0][-1]
        xi =  (t_f - self.t_grid[j]) / self.dt
        eta = (t_i - self.t_grid[k]) / self.dt

        if xi > eta:
            w1 = 1 - xi
            w2 = xi - eta
            w3 = eta
            return w1*self.G[j][k] + w2*self.G[j+1][k] + w3*self.G[j+1][k+1]
        else:
            w1 = 1 - eta
            w2 = eta - xi
            w3 = xi
            if w3 < 1e-14:
                return w1*self.G[j][k] + w2*self.G[j][k+1]
            else:
                return w1*self.G[j][k] + w2*self.G[j][k+1] + w3*self.G[j+1][k+1]
            

    def bath_corr_sum(self, q):
        return math.prod(-self.bath_corr(b - a) for a, b in q)
    

    def MC_int(self, m, t_f, t_i):
        """
        Perform the Monte Carlo integration for the inchworm step from t_i to t_f

        m:   truncation order
        t_i: initial time 
        t_f: final time
        """
        result = 0
        if (m % 2 != 1) or (t_i >= t_f):
            return result
        
        for _ in range(self.N_MC_samples):
            s = t_i + (t_f - t_i) * np.random.rand(m)
            s = np.sort(s)
            s = np.append(s, t_f)
            allpairs = pr.generate_linked_Pairs(list(s))
            for q in allpairs:
                temp = self.L_hat(s[0], q) @ self.G_interp(s[0], t_i)
                for i in range(1, m+1):
                    temp = self.L_hat(s[i], q) @ self.G_interp(s[i], s[i-1]) @ temp
                temp = self.bath_corr_sum(q) * temp
                result += temp

        result *= (t_f - t_i)**m / (math.factorial(m) * self.N_MC_samples)
        return result