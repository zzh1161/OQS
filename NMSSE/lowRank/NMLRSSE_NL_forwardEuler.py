import numpy as np
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple
from tqdm import tqdm

try:
    from numba import njit, prange  # type: ignore
except Exception:  # pragma: no cover
    njit = None
    prange = range

try:
    from ..utils.noise_generator import ColoredNoiseGenerator_Cholesky
    from ..utils.multi_index import MultiIndex
except ImportError:
    from utils.noise_generator import ColoredNoiseGenerator_Cholesky
    from utils.multi_index import MultiIndex


def _numba_available() -> bool:
    return njit is not None


if _numba_available():
    @njit(cache=True, nogil=True)
    def _matvec_add_scaled(A, x, y, scale):
        dim = A.shape[0]
        for i in range(dim):
            acc = 0.0 + 0.0j
            for j in range(dim):
                acc += A[i, j] * x[j]
            y[i] += scale * acc

    @njit(cache=True, nogil=True)
    def _vec_add_scaled(x, y, scale):
        dim = x.shape[0]
        for i in range(dim):
            y[i] += scale * x[i]

    @njit(cache=True, nogil=True)
    def _expectation_numba(psi, op):
        dim = psi.shape[0]
        den = 0.0 + 0.0j
        for i in range(dim):
            den += np.conj(psi[i]) * psi[i]
        if np.abs(den) == 0.0:
            return 0.0 + 0.0j

        num = 0.0 + 0.0j
        for i in range(dim):
            acc = 0.0 + 0.0j
            for j in range(dim):
                acc += op[i, j] * psi[j]
            num += np.conj(psi[i]) * acc
        return num / den

    @njit(cache=True, nogil=True)
    def _quadrature_trapezoidal_numba(dt, lam, U, Vn, L_avg, n):
        if n == 0:
            return 0.0 + 0.0j

        rank = lam.shape[0]
        integral = 0.0 + 0.0j

        for k in range(rank):
            integral += 0.5 * lam[k] * U[0, k] * Vn[k] * L_avg[0]
            for i in range(1, n):
                integral += lam[k] * U[i, k] * Vn[k] * L_avg[i]
            integral += 0.5 * lam[k] * U[n, k] * Vn[k] * L_avg[n]

        return integral * dt

    @njit(parallel=True, cache=True, nogil=True)
    def _psi_step_kernel_nl(
        psi_prev,
        psi_curr,
        A,
        L,
        L_dag,
        Vn,
        Un,
        lam,
        dt,
        idx_values,
        idx_minus,
        idx_plus,
        L_dag_avg_n,
    ):
        n_states, dim = psi_prev.shape
        rank = Vn.shape[0]

        for s in prange(n_states):
            for i in range(dim):
                acc = 0.0 + 0.0j
                for j in range(dim):
                    acc += A[i, j] * psi_prev[s, j]
                psi_curr[s, i] = acc

            for k in range(rank):
                s_minus = idx_minus[s, k]
                if s_minus >= 0:
                    scale_minus = dt * idx_values[s, k] * Vn[k]
                    _matvec_add_scaled(L, psi_prev[s_minus], psi_curr[s], scale_minus)

                s_plus = idx_plus[s, k]
                if s_plus >= 0:
                    scale_plus = -dt * lam[k] * Un[k]
                    _matvec_add_scaled(L_dag, psi_prev[s_plus], psi_curr[s], scale_plus)
                    scale_nl = dt * lam[k] * Un[k] * L_dag_avg_n
                    _vec_add_scaled(psi_prev[s_plus], psi_curr[s], scale_nl)

    @njit(cache=True, nogil=True)
    def _psi_step_kernel_nl_serial(
        psi_prev,
        psi_curr,
        A,
        L,
        L_dag,
        Vn,
        Un,
        lam,
        dt,
        idx_values,
        idx_minus,
        idx_plus,
        L_dag_avg_n,
    ):
        n_states, dim = psi_prev.shape
        rank = Vn.shape[0]

        for s in range(n_states):
            for i in range(dim):
                acc = 0.0 + 0.0j
                for j in range(dim):
                    acc += A[i, j] * psi_prev[s, j]
                psi_curr[s, i] = acc

            for k in range(rank):
                s_minus = idx_minus[s, k]
                if s_minus >= 0:
                    scale_minus = dt * idx_values[s, k] * Vn[k]
                    _matvec_add_scaled(L, psi_prev[s_minus], psi_curr[s], scale_minus)

                s_plus = idx_plus[s, k]
                if s_plus >= 0:
                    scale_plus = -dt * lam[k] * Un[k]
                    _matvec_add_scaled(L_dag, psi_prev[s_plus], psi_curr[s], scale_plus)
                    scale_nl = dt * lam[k] * Un[k] * L_dag_avg_n
                    _vec_add_scaled(psi_prev[s_plus], psi_curr[s], scale_nl)


def generate_complex_gaussian():
    n1 = np.random.normal()
    n2 = np.random.normal()
    return (n1 + 1j * n2) / np.sqrt(2)

def expectation(psi: np.ndarray, op: np.ndarray) -> complex:
    return np.vdot(psi, op @ psi) / np.vdot(psi, psi)

class NMLRSSE_NL_forwardEuler:
    def __init__(
        self,
        Hs,
        L,
        bath_corr,
        tmax,
        N_steps,
        rank,
        noise_generator = ColoredNoiseGenerator_Cholesky,
        max_layer: int | None = None,
        do_low_rank_decomposition: bool = True,
        noise_sample_Z: np.ndarray | None = None,
    ):
        self.dim = Hs.shape[0]
        self.Id = np.eye(self.dim)
        self._zero_state = np.zeros(self.dim, dtype=complex)
        self.Hs = Hs
        self.L = L
        self.L_dag = L.conj().T
        self.bath_corr = bath_corr
        self.tmax = tmax
        self.N_steps = N_steps
        self.rank = rank
        self.noise_sample_Z = noise_sample_Z
        if max_layer is None:
            self.max_layer = int(N_steps)
        else:
            self.max_layer = int(max_layer)
        if self.max_layer < 0:
            raise ValueError("max_layer must be non-negative")
        if self.max_layer > self.N_steps:
            self.max_layer = int(self.N_steps)
        
        self.dt = tmax / N_steps
        self.t_grid = np.linspace(0, tmax, N_steps+1)

        self.noise_generator = noise_generator(
            bath_corr, tmax, N_steps
        )

        self.psi_prev: Dict[Tuple[int, ...], np.ndarray] = {}
        self.psi_curr: Dict[Tuple[int, ...], np.ndarray] = {}

        self.idx_set = MultiIndex(r = self.rank, n = self.max_layer)
        self.idx_set.generate(order=1)
        self.idx_map: Dict[Tuple[int, ...], int] = {}
        self.initialize_index_map()
        self.initialize_psi()

        self._numba_idx_values = None
        self._numba_idx_minus = None
        self._numba_idx_plus = None
        self._zero_idx = self.idx_map[(0,) * self.rank]
        self._corr_matrix = None

        if do_low_rank_decomposition:
            timecost = self.compute_low_rank_decomposition()
            print(f"Low-rank decomposition done in {timecost:.2f} seconds.")

    def initialize_index_map(self):
        number = 0
        for k in range(self.max_layer + 1):
            for idx in self.idx_set.layer(k):
                self.idx_map[idx] = number
                number += 1
        return len(self.idx_map) == len(self.idx_set)
    
    def initialize_psi(self):
        for idx in self.idx_set:
            self.psi_prev[idx] = self._zero_state.copy()


    def compute_low_rank_decomposition(self):
        t0 = time.perf_counter()
        C = np.empty((self.N_steps + 1, self.N_steps + 1), dtype=complex)
        for i in range(self.N_steps + 1):
            for j in range(self.N_steps + 1):
                C[i, j] = self.bath_corr(self.t_grid[i] - self.t_grid[j])
        diag_weight = np.sqrt(self.dt) * np.ones(self.N_steps + 1)
        # diag_weight[0] = np.sqrt(0.5 * self.dt)
        # diag_weight[-1] = np.sqrt(0.5 * self.dt)
        W_half = np.diag(diag_weight)
        W_half_inv = np.diag(1.0 / diag_weight)
        U, s, V = np.linalg.svd(W_half @ C @ W_half)
        self.V = V[: self.rank, :] @ W_half_inv
        self.U = W_half_inv @ U[:, : self.rank]
        self.lam = s[: self.rank]
        dt = time.perf_counter() - t0
        return dt

    def set_SVD(self, lam, U, V):
        self.lam = lam
        self.U = U
        self.V = V

    def _build_numba_indexing(self):
        n_states = len(self.idx_set)
        idx_values = np.empty((n_states, self.rank), dtype=np.int64)
        idx_minus = -np.ones((n_states, self.rank), dtype=np.int64)
        idx_plus = -np.ones((n_states, self.rank), dtype=np.int64)

        for idx, pos in self.idx_map.items():
            idx_values[pos, :] = np.array(idx, dtype=np.int64)
            for k in range(self.rank):
                if idx[k] > 0:
                    idx_m = idx[:k] + (idx[k] - 1,) + idx[k + 1:]
                    idx_minus[pos, k] = self.idx_map.get(idx_m, -1)
                idx_p = idx[:k] + (idx[k] + 1,) + idx[k + 1:]
                idx_plus[pos, k] = self.idx_map.get(idx_p, -1)

        self._numba_idx_values = idx_values
        self._numba_idx_minus = idx_minus
        self._numba_idx_plus = idx_plus

    def _build_corr_matrix(self):
        if self._corr_matrix is not None:
            return
        C = np.empty((self.N_steps + 1, self.N_steps + 1), dtype=complex)
        for i in range(self.N_steps + 1):
            for j in range(self.N_steps + 1):
                C[i, j] = self.bath_corr(self.t_grid[i] - self.t_grid[j])
        self._corr_matrix = C
    

    def psi_step(self, z_tilde, L_dag_avg, n):
        if n < 0 or n >= self.N_steps:
            raise ValueError(f"n must satisfy 0 <= n < N_steps (got n={n})")
        
        def valid_psi(psi: Dict[Tuple[int, ...], np.ndarray], idx: Tuple[int, ...]) -> np.ndarray:
            return psi.get(idx, self._zero_state)
        
        A = self.Id - 1j * self.Hs * self.dt + z_tilde[n] * self.L * self.dt
        for idx in self.idx_set:
            self.psi_curr[idx] = A @ valid_psi(self.psi_prev, idx)
            for k in range(self.rank):
                idx_minus = idx[:k] + (idx[k]-1,) + idx[k+1:]
                self.psi_curr[idx] += self.dt*idx[k]*self.V[k,n]*self.L @ valid_psi(self.psi_prev, idx_minus)
            for k in range(self.rank):
                idx_plus = idx[:k] + (idx[k]+1,) + idx[k+1:]
                self.psi_curr[idx] -= self.dt*self.lam[k]*self.U[n,k]*self.L_dag @ valid_psi(self.psi_prev, idx_plus)
                self.psi_curr[idx] += self.dt*self.lam[k]*self.U[n,k]*L_dag_avg[n] * valid_psi(self.psi_prev, idx_plus)


    def quadrature_trapozoidal(self, dt, L_avg, n):
        integral = 0.0 + 0.0j
        if n == 0:
            return integral
        for k in range(self.rank):
            integral += 0.5 * self.lam[k] * self.U[0, k] * self.V[k, n] * L_avg[0]
            for i in range(1, n):
                integral += self.lam[k] * self.U[i, k] * self.V[k, n] * L_avg[i]
            integral += 0.5 * self.lam[k] * self.U[n, k] * self.V[k, n] * L_avg[n]
        return integral * dt


    def solve_python(
        self,
        N_traj: int,
        psi0: np.ndarray,
    ):
        self.initialize_psi()
        self.psi_curr.clear()
        psis = np.zeros((N_traj, self.N_steps+1, self.dim), dtype=complex)
        psis[:, 0, :] = psi0

        for traj in tqdm(range(N_traj), desc="Trajectories"):
            if self.noise_sample_Z is None:
                z = np.zeros(self.N_steps + 1, dtype=complex)
                for k in range(self.rank):
                    z[:] += generate_complex_gaussian() * np.sqrt(self.lam[k]) * self.V[k, :]
            else:
                z = self.noise_sample_Z[traj, :]
            self.psi_prev[(0,) * self.rank] = psi0.copy()

            z_tilde = np.zeros(self.N_steps + 1, dtype=complex)
            z_tilde[0] = z[0]

            L_dag_avg = np.zeros(self.N_steps + 1, dtype=complex)
            L_dag_avg[0] = expectation(psi0, self.L_dag)

            for n in range(self.N_steps):
                # 1. Compute psi_tilde
                self.psi_step(z_tilde, L_dag_avg, n)
                psi_curr_0 = self.psi_curr[(0,) * self.rank]
                psis[traj, n+1, :] = psi_curr_0

                # 2. Update L_dag_avg
                L_dag_avg[n+1] = expectation(psi_curr_0, self.L_dag)

                # 3. Update z_tilde
                z_tilde[n+1] = z[n+1] + self.quadrature_trapozoidal(self.dt, L_dag_avg, n+1)

                self.psi_prev, self.psi_curr = self.psi_curr, self.psi_prev
                self.psi_curr.clear()

        return psis

    def solve_numba(
        self,
        N_traj: int,
        psi0: np.ndarray,
        parallel_traj: bool = False,
        max_workers: int | None = None,
    ):
        if not _numba_available():
            raise RuntimeError("numba is not available")

        if self._numba_idx_values is None:
            self._build_numba_indexing()

        Id = np.ascontiguousarray(self.Id, dtype=np.complex128)
        Hs = np.ascontiguousarray(self.Hs, dtype=np.complex128)
        L = np.ascontiguousarray(self.L, dtype=np.complex128)
        L_dag = np.ascontiguousarray(self.L_dag, dtype=np.complex128)
        V = np.ascontiguousarray(self.V, dtype=np.complex128)
        U = np.ascontiguousarray(self.U, dtype=np.complex128)
        lam = np.ascontiguousarray(self.lam, dtype=np.float64)
        dt = float(self.dt)

        n_states = len(self.idx_set)
        N = self.N_steps + 1
        psis = np.zeros((N_traj, N, self.dim), dtype=np.complex128)
        psi0_arr = np.asarray(psi0, dtype=np.complex128)
        psis[:, 0, :] = psi0_arr

        if not parallel_traj:
            psi_prev = np.zeros((n_states, self.dim), dtype=np.complex128)
            psi_curr = np.zeros((n_states, self.dim), dtype=np.complex128)

            for traj in tqdm(range(N_traj), desc="Trajectories"):
                if self.noise_sample_Z is None:
                    z = np.zeros(self.N_steps + 1, dtype=complex)
                    for k in range(self.rank):
                        z[:] += generate_complex_gaussian() * np.sqrt(self.lam[k]) * self.V[k, :]
                else:
                    z = np.asarray(self.noise_sample_Z[traj, :], dtype=np.complex128)

                if z.shape[0] < N:
                    raise ValueError("noise length must be at least N_steps + 1")

                psi_prev[:, :] = 0.0
                psi_prev[self._zero_idx, :] = psi0_arr

                z_tilde = np.zeros(N, dtype=np.complex128)
                z_tilde[0] = z[0]

                L_dag_avg = np.zeros(N, dtype=np.complex128)
                L_dag_avg[0] = _expectation_numba(psi0_arr, L_dag)

                for n in range(self.N_steps):
                    A = Id - 1j * Hs * dt + z_tilde[n] * L * dt

                    _psi_step_kernel_nl(
                        psi_prev,
                        psi_curr,
                        A,
                        L,
                        L_dag,
                        V[:, n],
                        U[n, :],
                        lam,
                        dt,
                        self._numba_idx_values,
                        self._numba_idx_minus,
                        self._numba_idx_plus,
                        L_dag_avg[n],
                    )

                    psi_curr /= np.linalg.norm(psi_curr)

                    psis[traj, n + 1, :] = psi_curr[self._zero_idx, :]
                    L_dag_avg[n + 1] = _expectation_numba(psi_curr[self._zero_idx, :], L_dag)
                    z_tilde[n + 1] = z[n + 1] + _quadrature_trapezoidal_numba(
                        dt,
                        lam,
                        U,
                        V[:, n + 1],
                        L_dag_avg,
                        n + 1,
                    )

                    psi_prev, psi_curr = psi_curr, psi_prev

            return psis

        if max_workers is None:
            max_workers = os.cpu_count() or 1

        Z = self.noise_sample_Z
        if Z is None:
            Z = np.empty((N_traj, N), dtype=np.complex128)
            for traj in range(N_traj):
                z = np.zeros(self.N_steps + 1, dtype=complex)
                for k in range(self.rank):
                    z[:] += generate_complex_gaussian() * np.sqrt(self.lam[k]) * self.V[k, :]
                if z.shape[0] < N:
                    raise ValueError("noise length must be at least N_steps + 1")
                Z[traj, :] = z[:N]
        else:
            Z = np.asarray(Z, dtype=np.complex128)
            if Z.shape[1] < N:
                raise ValueError("noise_sample_Z must have at least N_steps + 1 columns")

        if self.N_steps > 0:
            _psi_prev0 = np.zeros((n_states, self.dim), dtype=np.complex128)
            _psi_curr0 = np.zeros((n_states, self.dim), dtype=np.complex128)
            A0 = Id - 1j * Hs * dt + Z[0, 0] * L * dt
            _psi_step_kernel_nl_serial(
                _psi_prev0,
                _psi_curr0,
                A0,
                L,
                L_dag,
                V[:, 0],
                U[0, :],
                lam,
                dt,
                self._numba_idx_values,
                self._numba_idx_minus,
                self._numba_idx_plus,
                0.0 + 0.0j,
            )

        def _run_one(traj: int):
            traj_psis = np.empty((N, self.dim), dtype=np.complex128)
            traj_psis[0, :] = psi0_arr

            psi_prev = np.zeros((n_states, self.dim), dtype=np.complex128)
            psi_curr = np.zeros((n_states, self.dim), dtype=np.complex128)
            psi_prev[self._zero_idx, :] = psi0_arr

            z_tilde = np.zeros(N, dtype=np.complex128)
            z_tilde[0] = Z[traj, 0]

            L_dag_avg = np.zeros(N, dtype=np.complex128)
            L_dag_avg[0] = _expectation_numba(psi0_arr, L_dag)

            for n in range(self.N_steps):
                A = Id - 1j * Hs * dt + z_tilde[n] * L * dt

                _psi_step_kernel_nl_serial(
                    psi_prev,
                    psi_curr,
                    A,
                    L,
                    L_dag,
                    V[:, n],
                    U[n, :],
                    lam,
                    dt,
                    self._numba_idx_values,
                    self._numba_idx_minus,
                    self._numba_idx_plus,
                    L_dag_avg[n],
                )

                psi_curr /= np.linalg.norm(psi_curr)

                traj_psis[n + 1, :] = psi_curr[self._zero_idx, :]
                L_dag_avg[n + 1] = _expectation_numba(psi_curr[self._zero_idx, :], L_dag)
                z_tilde[n + 1] = Z[traj, n + 1] + _quadrature_trapezoidal_numba(
                    dt,
                    lam,
                    U,
                    V[:, n + 1],
                    L_dag_avg,
                    n + 1,
                )

                psi_prev, psi_curr = psi_curr, psi_prev

            return traj, traj_psis

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_run_one, traj) for traj in range(N_traj)]
            for fut in tqdm(as_completed(futures), total=N_traj, desc="Trajectories"):
                traj, traj_psis = fut.result()
                psis[traj, :, :] = traj_psis

        return psis

    def solve(
        self,
        N_traj,
        psi0,
        backend: str = "auto",
        parallel_traj: bool = False,
        max_workers: int | None = None,
    ):
        if backend not in ("auto", "python", "numba"):
            raise ValueError("backend must be one of: 'auto', 'python', 'numba'")

        if backend in ("auto", "numba") and _numba_available():
            return self.solve_numba(
                N_traj,
                psi0,
                parallel_traj=parallel_traj,
                max_workers=max_workers,
            )
        return self.solve_python(N_traj, psi0)