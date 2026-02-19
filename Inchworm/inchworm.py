import numpy as np
import math
from typing import List

try:
    from numba import njit, prange  # type: ignore
except Exception:  # pragma: no cover
    njit = None
    prange = range

try:
    from . import pairing as pr
except ImportError:
    import pairing as pr

class Inchworm:
    def __init__(self, Hs, Ws, tmax, N_steps, bath_corr, N_trunc=3, N_MC_samples=300):
        self.dim = Hs.shape[0]
        self.Hs = Hs
        self.Ws = Ws
        self.tmax = tmax
        self.N_steps = N_steps
        self.dt = tmax / N_steps

        self.t_grid = np.linspace(0, tmax, self.N_steps + 1)
        self.t_contour = np.concatenate((self.t_grid, self.t_grid + self.tmax))

        self.bath_corr = bath_corr
        self.N_MC_samples = N_MC_samples
        self.N_trunc = N_trunc

        self.N_contour = 2 * (self.N_steps + 1)
        self.G = np.zeros(
            (self.N_contour, self.N_contour, self.dim, self.dim),
            dtype=np.complex128,
        )

        self._use_numba = (njit is not None)
        self._linked_pairings = {}
        self._linked_pairings_arr = {}
        self._factorials = {}
        self._prepare_pairing_cache()


    def initialize(self):
        self.G.fill(0)
        eye = np.eye(self.dim, dtype=np.complex128)
        for n in range(self.N_contour):
            self.G[n, n, :, :] = eye


    def _prepare_pairing_cache(self) -> None:
        """Cache linked pairings and factorials for all odd m up to N_trunc."""
        for m in range(1, self.N_trunc + 1, 2):
            linked = pr.generate_linked_Pairs(list(range(m + 1)))
            self._linked_pairings[m] = linked

            # Convert to a fixed-shape int64 array: (n_pairings, n_pairs, 2)
            n_pairs = (m + 1) // 2
            arr = np.empty((len(linked), n_pairs, 2), dtype=np.int64)
            for p_idx, pairing in enumerate(linked):
                for q_idx, (i, j) in enumerate(pairing):
                    arr[p_idx, q_idx, 0] = i
                    arr[p_idx, q_idx, 1] = j
            self._linked_pairings_arr[m] = arr
            self._factorials[m] = math.factorial(m)

    
    def bath_corr_sum(self, times, linked_idx):
        if len(times) % 2 != 0:
            return 0
        phys = lambda t: t if t <= self.tmax else 2*self.tmax - t
        return np.sum([np.prod([self.bath_corr(phys(times[j]) - phys(times[i])) for i, j in p]) for p in linked_idx])
    

    def G_interp(self, tf, ti):
        if tf < ti:
            raise ValueError("t_f must be greater than or equal to t_i.")
        if tf == ti:
            return np.eye(self.dim)
        
        j = np.where(tf >= self.t_contour)[0][-1]
        k = np.where(ti >= self.t_contour)[0][-1]
        if tf == self.tmax:
            j = self.N_steps
        if ti == self.tmax:
            k = self.N_steps + 1
        xi  = (tf - self.t_contour[j]) / self.dt
        eta = (ti - self.t_contour[k]) / self.dt

        if xi > eta:
            w1 = 1 - xi
            w2 = xi - eta
            w3 = eta
            return w1*self.G[j, k] + w2*self.G[j+1, k] + w3*self.G[j+1, k+1]
        else:
            w1 = 1 - eta
            w2 = eta - xi
            w3 = xi
            if w3 < 1e-14:
                return w1*self.G[j, k] + w2*self.G[j, k+1]
            else:
                return w1*self.G[j, k] + w2*self.G[j, k+1] + w3*self.G[j+1, k+1]
            

    def MC_int(self, m, tf, ti):
        if m % 2 != 1 or ti >= tf:
            return 0

        if self._use_numba:
            return self._MC_int_numba(m, tf, ti)

        linked_pairings = self._linked_pairings.get(m)
        if linked_pairings is None:
            linked_pairings = pr.generate_linked_Pairs(list(range(m + 1)))

        result = 0
        for _ in range(self.N_MC_samples):
            s = ti + (tf - ti) * np.random.rand(m)
            s = np.sort(s)
            count = np.sum(s < self.tmax)
            Lq_sum = self.bath_corr_sum(np.append(s, tf), linked_pairings)

            temp = self.Ws @ self.G_interp(s[0], ti)
            for i in range(1, m):
                temp = self.Ws @ self.G_interp(s[i], s[i - 1]) @ temp
            temp = self.Ws @ self.G_interp(tf, s[m - 1]) @ temp

            result += (-1) ** count * Lq_sum * temp

        result *= (1j) ** (m + 1) * (tf - ti) ** m / (math.factorial(m) * self.N_MC_samples)
        return result


    def _MC_int_numba(self, m: int, tf: float, ti: float):
        linked = self._linked_pairings.get(m)
        if linked is None:
            linked = pr.generate_linked_Pairs(list(range(m + 1)))
            self._linked_pairings[m] = linked
        linked_arr = self._linked_pairings_arr.get(m)
        if linked_arr is None:
            n_pairs = (m + 1) // 2
            linked_arr = np.empty((len(linked), n_pairs, 2), dtype=np.int64)
            for p_idx, pairing in enumerate(linked):
                for q_idx, (i, j) in enumerate(pairing):
                    linked_arr[p_idx, q_idx, 0] = i
                    linked_arr[p_idx, q_idx, 1] = j
            self._linked_pairings_arr[m] = linked_arr

        # Pre-generate all random samples (deterministic under np.random seed)
        s_samples = ti + (tf - ti) * np.random.rand(self.N_MC_samples, m)
        s_samples.sort(axis=1)

        # Compute the scalar bath-correlation weight per sample in Python (exact, uses user-supplied bath_corr)
        weights = np.empty(self.N_MC_samples, dtype=np.complex128)
        for n in range(self.N_MC_samples):
            s = s_samples[n]
            count = int(np.sum(s < self.tmax))
            Lq_sum = self.bath_corr_sum(np.append(s, tf), linked)
            weights[n] = ((-1) ** count) * Lq_sum

        out = _mc_chain_sum_numba(
            self.Ws,
            self.G,
            self.t_contour,
            float(self.dt),
            float(self.tmax),
            int(self.N_steps),
            float(tf),
            float(ti),
            s_samples,
            weights,
        )

        fac = self._factorials.get(m)
        if fac is None:
            fac = math.factorial(m)
            self._factorials[m] = fac
        out *= (1j) ** (m + 1) * (tf - ti) ** m / (fac * self.N_MC_samples)
        return out
    

    def sgn(self, j):
        return -1 if j <= self.N_steps else 1

    
    def inchworm_step(self, Os, rhos):
        self.initialize()
        N = self.N_steps
        # N is N-, N+1 is N+
        for offset in range(1, 2*N+2):
            for j in range(offset, 2*N+2):
                k = j - offset

                if j == N+1:
                    self.G[j, k] = Os @ self.G[N, k]
                    # print(f"Computed G[{j}][{k}]")
                    continue
                if k == N:
                    self.G[j, k] = self.G[j, N+1] @ Os
                    # print(f"Computed G[{j}][{k}]")
                    continue

                # Heun's Step 1
                int_temp = sum([self.MC_int(m, self.t_contour[j-1], self.t_contour[k]) for m in range(1, self.N_trunc+1, 2)]) 
                self.G[j, k] = self.G[j-1, k] + self.sgn(j-1) * self.dt * (1j * self.Hs @ self.G[j-1, k] + int_temp)

                # Heun's Step 2
                int_temp = sum([self.MC_int(m, self.t_contour[j], self.t_contour[k]) for m in range(1, self.N_trunc+1, 2)])
                self.G[j, k] = 0.5 * (self.G[j-1, k] + self.G[j, k]) + \
                               0.5 * self.sgn(j) * self.dt * (1j * self.Hs @ self.G[j, k] + int_temp)
                
                # print(f"Computed G[{j}][{k}]")
                
        result = [None for _ in range(N+1)]
        for n in range(N+1):
            result[n] = np.trace(rhos @ self.G[N+1+n, N-n])

        return result


def _numba_available() -> bool:
    return njit is not None


if _numba_available():
    @njit(cache=True)
    def _contour_index(t: float, tmax: float, dt: float, N_steps: int, for_initial: bool) -> int:
        # Special handling to disambiguate the duplicated contour point at t == tmax.
        if t == tmax:
            return N_steps + 1 if for_initial else N_steps
        if t < 0.0:
            return 0
        # t_contour runs from 0 to 2*tmax with step dt.
        idx = int(t / dt)
        if idx < 0:
            idx = 0
        max_idx = 2 * (N_steps + 1) - 1
        if idx > max_idx:
            idx = max_idx
        return idx


    @njit(cache=True)
    def _G_interp_numba(G: np.ndarray, t_contour: np.ndarray, dt: float, tmax: float, N_steps: int, tf: float, ti: float) -> np.ndarray:
        if tf < ti:
            raise ValueError("t_f must be greater than or equal to t_i.")
        dim = G.shape[2]
        if tf == ti:
            out = np.zeros((dim, dim), dtype=np.complex128)
            for d in range(dim):
                out[d, d] = 1.0
            return out

        j = _contour_index(tf, tmax, dt, N_steps, False)
        k = _contour_index(ti, tmax, dt, N_steps, True)

        xi = (tf - t_contour[j]) / dt
        eta = (ti - t_contour[k]) / dt

        if xi > eta:
            w1 = 1.0 - xi
            w2 = xi - eta
            w3 = eta
            return w1 * G[j, k] + w2 * G[j + 1, k] + w3 * G[j + 1, k + 1]

        w1 = 1.0 - eta
        w2 = eta - xi
        w3 = xi
        if w3 < 1e-14:
            return w1 * G[j, k] + w2 * G[j, k + 1]
        return w1 * G[j, k] + w2 * G[j, k + 1] + w3 * G[j + 1, k + 1]


    @njit(parallel=True, cache=True)
    def _mc_chain_sum_numba(
        Ws: np.ndarray,
        G: np.ndarray,
        t_contour: np.ndarray,
        dt: float,
        tmax: float,
        N_steps: int,
        tf: float,
        ti: float,
        s_samples: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        n_samples = s_samples.shape[0]
        m = s_samples.shape[1]
        dim = Ws.shape[0]

        per = np.zeros((n_samples, dim, dim), dtype=np.complex128)
        for n in prange(n_samples):
            s0 = s_samples[n, 0]
            temp = Ws @ _G_interp_numba(G, t_contour, dt, tmax, N_steps, s0, ti)
            for i in range(1, m):
                si = s_samples[n, i]
                sim1 = s_samples[n, i - 1]
                temp = Ws @ _G_interp_numba(G, t_contour, dt, tmax, N_steps, si, sim1) @ temp
            temp = Ws @ _G_interp_numba(G, t_contour, dt, tmax, N_steps, tf, s_samples[n, m - 1]) @ temp
            per[n, :, :] = weights[n] * temp

        out = np.zeros((dim, dim), dtype=np.complex128)
        for n in range(n_samples):
            out += per[n]
        return out

else:
    def _mc_chain_sum_numba(*args, **kwargs):  # type: ignore
        raise RuntimeError(
            "numba is not available. Install it (e.g. `pip install numba`) or disable numba backend."
        )
