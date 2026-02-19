import numpy as np
import math
from typing import List

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

try:
    from numba import njit, prange
except Exception:
    njit = None
    prange = range

try:
    from . import pairing as pr
except ImportError:
    import pairing as pr

class Inchworm:
    def __init__(
        self,
        Hs,
        Ws,
        tmax,
        N_steps,
        bath_corr,
        N_trunc=3,
        N_MC_samples=300,
        *,
        backend: str = "auto",
        numba_parallel: bool = True,
    ):
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
        self._backend = backend
        self._numba_parallel = bool(numba_parallel)
        self._linked_pairings = {}
        self._linked_pairings_arr = {}
        self._factorials = {}
        self._prepare_pairing_cache()

        if backend not in ("auto", "python", "numba"):
            raise ValueError("backend must be one of: 'auto', 'python', 'numba'")

        if backend == "python":
            self._use_numba = False
        elif backend == "numba":
            if njit is None:
                raise RuntimeError("numba is not available. Install it (e.g. `pip install numba`) or use backend='python'.")
            self._use_numba = True
        else:
            # Heuristic: numba kernels here avoid BLAS and favor small dim (e.g. 2-level systems).
            self._use_numba = (njit is not None) and (self.dim <= 8) and (self.N_MC_samples >= 128)


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


    def _bath_corr_sum_cached(self, times: np.ndarray, m: int):
        """Faster bath-correlation sum using cached linked-pairing indices.

        Uses a vectorized path when `bath_corr` supports ndarray inputs.
        Falls back to the original Python loops for maximum compatibility.
        """
        if times.size % 2 != 0:
            return 0

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

        phys_times = times.astype(float, copy=True)
        mask = phys_times > self.tmax
        phys_times[mask] = 2 * self.tmax - phys_times[mask]

        idx0 = linked_arr[:, :, 0]
        idx1 = linked_arr[:, :, 1]
        diffs = phys_times[idx1] - phys_times[idx0]

        try:
            corr = self.bath_corr(diffs)
            prod = np.prod(corr, axis=1)
            return np.sum(prod)
        except Exception:
            return self.bath_corr_sum(times, linked)


    def _bath_corr_sum_cached_samples(self, times_full: np.ndarray, m: int) -> np.ndarray:
        """Vectorized bath-correlation sums for many samples.

        times_full: (N_samples, m+1)
        Returns: (N_samples,)
        """
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

        phys = times_full.astype(float, copy=True)
        mask = phys > self.tmax
        phys[mask] = 2 * self.tmax - phys[mask]

        idx0 = linked_arr[:, :, 0]
        idx1 = linked_arr[:, :, 1]
        diffs = phys[:, idx1] - phys[:, idx0]

        try:
            corr = self.bath_corr(diffs)
            prod = np.prod(corr, axis=2)
            return np.sum(prod, axis=1)
        except Exception:
            # Compatibility fallback: loop samples.
            out = np.empty(phys.shape[0], dtype=np.complex128)
            for n in range(phys.shape[0]):
                out[n] = self.bath_corr_sum(times_full[n], linked)
            return out
    

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
            Lq_sum = self._bath_corr_sum_cached(np.append(s, tf), m)

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

        # Build full time arrays and compute weights (vectorized if bath_corr supports ndarray).
        times_full = np.empty((self.N_MC_samples, m + 1), dtype=float)
        times_full[:, :m] = s_samples
        times_full[:, m] = tf
        counts = np.sum(s_samples < self.tmax, axis=1).astype(np.int64)
        signs = np.where((counts % 2) == 0, 1.0, -1.0)
        Lq_sums = self._bath_corr_sum_cached_samples(times_full, m)
        weights = (signs * Lq_sums).astype(np.complex128)

        kernel = _mc_chain_sum_numba_parallel if self._numba_parallel else _mc_chain_sum_numba_serial
        out = kernel(
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

    
    def inchworm_step(self, Os, rhos, show_progress: bool = True):
        self.initialize()
        N = self.N_steps
        # N is N-, N+1 is N+
        total = (2 * N + 1) * (2 * N + 2) // 2
        pbar = None
        if show_progress and tqdm is not None:
            pbar = tqdm(total=total, desc="inchworm_step", unit="cell")

        try:
            for offset in range(1, 2 * N + 2):
                for j in range(offset, 2 * N + 2):
                    k = j - offset

                    if j == N + 1:
                        self.G[j, k] = Os @ self.G[N, k]
                        if pbar is not None:
                            pbar.update(1)
                        continue
                    if k == N:
                        self.G[j, k] = self.G[j, N + 1] @ Os
                        if pbar is not None:
                            pbar.update(1)
                        continue

                    # Heun's Step 1
                    int_temp = sum(
                        [
                            self.MC_int(m, self.t_contour[j - 1], self.t_contour[k])
                            for m in range(1, self.N_trunc + 1, 2)
                        ]
                    )
                    self.G[j, k] = self.G[j - 1, k] + self.sgn(j - 1) * self.dt * (
                        1j * self.Hs @ self.G[j - 1, k] + int_temp
                    )

                    # Heun's Step 2
                    int_temp = sum(
                        [
                            self.MC_int(m, self.t_contour[j], self.t_contour[k])
                            for m in range(1, self.N_trunc + 1, 2)
                        ]
                    )
                    self.G[j, k] = 0.5 * (self.G[j - 1, k] + self.G[j, k]) + 0.5 * self.sgn(j) * self.dt * (
                        1j * self.Hs @ self.G[j, k] + int_temp
                    )

                    if pbar is not None:
                        pbar.update(1)

        finally:
            if pbar is not None:
                pbar.close()
                
                # print(f"Computed G[{j}][{k}]")
                
        result = [None for _ in range(N+1)]
        for n in range(N+1):
            result[n] = np.trace(rhos @ self.G[N+1+n, N-n])

        return result


def _numba_available() -> bool:
    return njit is not None


if _numba_available():
    @njit(cache=True)
    def _matmul(A: np.ndarray, B: np.ndarray, out: np.ndarray) -> None:
        n = A.shape[0]
        for i in range(n):
            for j in range(n):
                s = 0.0 + 0.0j
                for k in range(n):
                    s += A[i, k] * B[k, j]
                out[i, j] = s


    @njit(cache=True)
    def _scale_inplace(A: np.ndarray, scalar: complex) -> None:
        n = A.shape[0]
        for i in range(n):
            for j in range(n):
                A[i, j] *= scalar


    @njit(cache=True)
    def _add_inplace(dst: np.ndarray, src: np.ndarray) -> None:
        n = dst.shape[0]
        for i in range(n):
            for j in range(n):
                dst[i, j] += src[i, j]

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
    def _G_interp_into(out: np.ndarray, G: np.ndarray, t_contour: np.ndarray, dt: float, tmax: float, N_steps: int, tf: float, ti: float) -> None:
        if tf < ti:
            raise ValueError("t_f must be greater than or equal to t_i.")
        dim = G.shape[2]
        if tf == ti:
            for i in range(dim):
                for j in range(dim):
                    out[i, j] = 0.0 + 0.0j
            for d in range(dim):
                out[d, d] = 1.0 + 0.0j
            return

        j = _contour_index(tf, tmax, dt, N_steps, False)
        k = _contour_index(ti, tmax, dt, N_steps, True)
        xi = (tf - t_contour[j]) / dt
        eta = (ti - t_contour[k]) / dt

        if xi > eta:
            w1 = 1.0 - xi
            w2 = xi - eta
            w3 = eta
            for a in range(dim):
                for b in range(dim):
                    out[a, b] = w1 * G[j, k, a, b] + w2 * G[j + 1, k, a, b] + w3 * G[j + 1, k + 1, a, b]
            return

        w1 = 1.0 - eta
        w2 = eta - xi
        w3 = xi
        if w3 < 1e-14:
            for a in range(dim):
                for b in range(dim):
                    out[a, b] = w1 * G[j, k, a, b] + w2 * G[j, k + 1, a, b]
            return

        for a in range(dim):
            for b in range(dim):
                out[a, b] = w1 * G[j, k, a, b] + w2 * G[j, k + 1, a, b] + w3 * G[j + 1, k + 1, a, b]


    @njit(cache=True)
    def _mc_chain_sum_numba_serial(
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
        out = np.zeros((dim, dim), dtype=np.complex128)
        Gi = np.empty((dim, dim), dtype=np.complex128)
        tmp = np.empty((dim, dim), dtype=np.complex128)
        temp = np.empty((dim, dim), dtype=np.complex128)

        for n in range(n_samples):
            s0 = s_samples[n, 0]
            _G_interp_into(Gi, G, t_contour, dt, tmax, N_steps, s0, ti)
            _matmul(Ws, Gi, temp)
            for i in range(1, m):
                si = s_samples[n, i]
                sim1 = s_samples[n, i - 1]
                _G_interp_into(Gi, G, t_contour, dt, tmax, N_steps, si, sim1)
                _matmul(Gi, temp, tmp)
                _matmul(Ws, tmp, temp)
            _G_interp_into(Gi, G, t_contour, dt, tmax, N_steps, tf, s_samples[n, m - 1])
            _matmul(Gi, temp, tmp)
            _matmul(Ws, tmp, temp)

            _scale_inplace(temp, weights[n])
            _add_inplace(out, temp)
        return out


    @njit(parallel=True, cache=True)
    def _mc_chain_sum_numba_parallel(
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
            Gi = np.empty((dim, dim), dtype=np.complex128)
            tmp = np.empty((dim, dim), dtype=np.complex128)
            temp = np.empty((dim, dim), dtype=np.complex128)

            s0 = s_samples[n, 0]
            _G_interp_into(Gi, G, t_contour, dt, tmax, N_steps, s0, ti)
            _matmul(Ws, Gi, temp)
            for i in range(1, m):
                si = s_samples[n, i]
                sim1 = s_samples[n, i - 1]
                _G_interp_into(Gi, G, t_contour, dt, tmax, N_steps, si, sim1)
                _matmul(Gi, temp, tmp)
                _matmul(Ws, tmp, temp)
            _G_interp_into(Gi, G, t_contour, dt, tmax, N_steps, tf, s_samples[n, m - 1])
            _matmul(Gi, temp, tmp)
            _matmul(Ws, tmp, temp)

            _scale_inplace(temp, weights[n])
            per[n, :, :] = temp

        out = np.zeros((dim, dim), dtype=np.complex128)
        for n in range(n_samples):
            out += per[n]
        return out

else:
    def _mc_chain_sum_numba_serial(*args, **kwargs):  # type: ignore
        raise RuntimeError(
            "numba is not available. Install it (e.g. `pip install numba`) or disable numba backend."
        )


    def _mc_chain_sum_numba_parallel(*args, **kwargs):  # type: ignore
        raise RuntimeError(
            "numba is not available. Install it (e.g. `pip install numba`) or disable numba backend."
        )
