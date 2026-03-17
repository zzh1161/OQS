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
    def _matvec_into(A, x, out):
        dim = A.shape[0]
        for i in range(dim):
            acc = 0.0 + 0.0j
            for j in range(dim):
                acc += A[i, j] * x[j]
            out[i] = acc

    @njit(cache=True, nogil=True)
    def _matvec_add_scaled(A, x, y, scale):
        dim = A.shape[0]
        for i in range(dim):
            acc = 0.0 + 0.0j
            for j in range(dim):
                acc += A[i, j] * x[j]
            y[i] += scale * acc

    @njit(cache=True, nogil=True)
    def _compute_rhs_ode(
        out,
        psi_in,
        L,
        L_dag,
        z_step,
        Vn,
        Un,
        lam,
        idx_values,
        idx_minus,
        idx_plus,
        state_idx,
    ):
        dim = L.shape[0]
        rank = Vn.shape[0]

        for i in range(dim):
            acc = 0.0 + 0.0j
            for j in range(dim):
                acc += (z_step * L[i, j]) * psi_in[state_idx, j]
            out[i] = acc

        for k in range(rank):
            s_minus = idx_minus[state_idx, k]
            if s_minus >= 0:
                scale = idx_values[state_idx, k] * Vn[k]
                _matvec_add_scaled(L, psi_in[s_minus], out, scale)

        for k in range(rank):
            s_plus = idx_plus[state_idx, k]
            if s_plus >= 0:
                scale = -lam[k] * Un[k]
                _matvec_add_scaled(L_dag, psi_in[s_plus], out, scale)

    @njit(parallel=True, cache=True, nogil=True)
    def _strang_step_kernel(
        psi_prev,
        psi_curr,
        psi_half,
        k1,
        psi_pred,
        P_half,
        L,
        L_dag,
        z_n,
        z_n1,
        Vn,
        Vn1,
        Un,
        Un1,
        lam,
        dt,
        idx_values,
        idx_minus,
        idx_plus,
    ):
        n_states, dim = psi_prev.shape

        for s in prange(n_states):
            _matvec_into(P_half, psi_prev[s], psi_half[s])

        for s in prange(n_states):
            _compute_rhs_ode(
                k1[s],
                psi_half,
                L,
                L_dag,
                z_n,
                Vn,
                Un,
                lam,
                idx_values,
                idx_minus,
                idx_plus,
                s,
            )

        for s in prange(n_states):
            for i in range(dim):
                psi_pred[s, i] = psi_half[s, i] + dt * k1[s, i]

        for s in prange(n_states):
            _compute_rhs_ode(
                psi_curr[s],
                psi_pred,
                L,
                L_dag,
                z_n1,
                Vn1,
                Un1,
                lam,
                idx_values,
                idx_minus,
                idx_plus,
                s,
            )

        for s in prange(n_states):
            for i in range(dim):
                psi_curr[s, i] = psi_half[s, i] + 0.5 * dt * (k1[s, i] + psi_curr[s, i])

        for s in prange(n_states):
            _matvec_into(P_half, psi_curr[s], psi_half[s])

        for s in prange(n_states):
            for i in range(dim):
                psi_curr[s, i] = psi_half[s, i]

    @njit(cache=True, nogil=True)
    def _strang_step_kernel_serial(
        psi_prev,
        psi_curr,
        psi_half,
        k1,
        psi_pred,
        P_half,
        L,
        L_dag,
        z_n,
        z_n1,
        Vn,
        Vn1,
        Un,
        Un1,
        lam,
        dt,
        idx_values,
        idx_minus,
        idx_plus,
    ):
        n_states, dim = psi_prev.shape

        for s in range(n_states):
            _matvec_into(P_half, psi_prev[s], psi_half[s])

        for s in range(n_states):
            _compute_rhs_ode(
                k1[s],
                psi_half,
                L,
                L_dag,
                z_n,
                Vn,
                Un,
                lam,
                idx_values,
                idx_minus,
                idx_plus,
                s,
            )

        for s in range(n_states):
            for i in range(dim):
                psi_pred[s, i] = psi_half[s, i] + dt * k1[s, i]

        for s in range(n_states):
            _compute_rhs_ode(
                psi_curr[s],
                psi_pred,
                L,
                L_dag,
                z_n1,
                Vn1,
                Un1,
                lam,
                idx_values,
                idx_minus,
                idx_plus,
                s,
            )

        for s in range(n_states):
            for i in range(dim):
                psi_curr[s, i] = psi_half[s, i] + 0.5 * dt * (k1[s, i] + psi_curr[s, i])

        for s in range(n_states):
            _matvec_into(P_half, psi_curr[s], psi_half[s])

        for s in range(n_states):
            for i in range(dim):
                psi_curr[s, i] = psi_half[s, i]


class NMLRSSE_Strang:
    def __init__(
        self,
        Hs,
        L,
        bath_corr,
        tmax,
        N_steps,
        rank,
        noise_generator=ColoredNoiseGenerator_Cholesky,
        max_layer: int | None = None,
        do_low_rank_decomposition: bool = True,
        noise_sample_Z: np.ndarray | None = None,
    ):
        self.dim = Hs.shape[0]
        self.Id = np.eye(self.dim)
        self._zero_state = np.zeros(self.dim, dtype=complex)
        self.Hs = Hs
        self.Hs_E, self.Hs_V = np.linalg.eigh(Hs)
        self.Hs_V_dag = self.Hs_V.conj().T
        self.Hs_propagator = lambda t: self.Hs_V @ np.diag(np.exp(-1j * self.Hs_E * t)) @ self.Hs_V_dag
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
        self.t_grid = np.linspace(0, tmax, N_steps + 1)
        self.P_half = self.Hs_propagator(0.5 * self.dt)

        self.noise_generator = noise_generator(bath_corr, tmax, N_steps)

        self.psi_prev: Dict[Tuple[int, ...], np.ndarray] = {}
        self.psi_curr: Dict[Tuple[int, ...], np.ndarray] = {}

        self.idx_set = MultiIndex(r=self.rank, n=self.max_layer)
        self.idx_set.generate(order=1)
        self.idx_map: Dict[Tuple[int, ...], int] = {}
        self.initialize_index_map()
        self.initialize_psi()

        self._numba_idx_values = None
        self._numba_idx_minus = None
        self._numba_idx_plus = None
        self._zero_idx = self.idx_map[(0,) * self.rank]

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
            self.psi_curr[idx] = self._zero_state.copy()

    def compute_low_rank_decomposition(self):
        t0 = time.perf_counter()
        C = np.empty((self.N_steps + 1, self.N_steps + 1), dtype=complex)
        for i in range(self.N_steps + 1):
            for j in range(self.N_steps + 1):
                C[i, j] = self.bath_corr(self.t_grid[i] - self.t_grid[j])
        diag_weight = np.sqrt(self.dt) * np.ones(self.N_steps + 1)
        diag_weight[0] = np.sqrt(0.5 * self.dt)
        diag_weight[-1] = np.sqrt(0.5 * self.dt)
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
                    idx_m = idx[:k] + (idx[k] - 1,) + idx[k + 1 :]
                    idx_minus[pos, k] = self.idx_map.get(idx_m, -1)
                idx_p = idx[:k] + (idx[k] + 1,) + idx[k + 1 :]
                idx_plus[pos, k] = self.idx_map.get(idx_p, -1)

        self._numba_idx_values = idx_values
        self._numba_idx_minus = idx_minus
        self._numba_idx_plus = idx_plus

    def valid_psi(self, psi: Dict[Tuple[int, ...], np.ndarray], idx: Tuple[int, ...]) -> np.ndarray:
        return psi.get(idx, self._zero_state)

    def ode_rhs(
        self,
        z: np.ndarray,
        step: int,
        psi: Dict[Tuple[int, ...], np.ndarray],
        idx: Tuple[int, ...],
    ) -> np.ndarray:
        base = self.valid_psi(psi, idx)
        acc = z[step] * (self.L @ base)
        for k in range(self.rank):
            idx_minus = idx[:k] + (idx[k] - 1,) + idx[k + 1 :]
            acc = acc + idx[k] * self.V[k, step] * (self.L @ self.valid_psi(psi, idx_minus))
        for k in range(self.rank):
            idx_plus = idx[:k] + (idx[k] + 1,) + idx[k + 1 :]
            acc = acc - (self.lam[k] * self.U[step, k]) * (self.L_dag @ self.valid_psi(psi, idx_plus))
        return acc

    def ode_step(
        self,
        psi_in: Dict[Tuple[int, ...], np.ndarray],
        psi_out: Dict[Tuple[int, ...], np.ndarray],
        z: np.ndarray,
        step: int,
        dt: float,
    ):
        k1: Dict[Tuple[int, ...], np.ndarray] = {}
        psi_pred: Dict[Tuple[int, ...], np.ndarray] = {}

        for idx in self.idx_set:
            psi_n = self.valid_psi(psi_in, idx)
            k1_idx = self.ode_rhs(z, step, psi_in, idx)
            k1[idx] = k1_idx
            psi_pred[idx] = psi_n + dt * k1_idx

        for idx in self.idx_set:
            psi_n = self.valid_psi(psi_in, idx)
            k2_idx = self.ode_rhs(z, step + 1, psi_pred, idx)
            psi_out[idx] = psi_n + 0.5 * dt * (k1[idx] + k2_idx)

    def exp_step(self, psi: Dict[Tuple[int, ...], np.ndarray], dt: float):
        H_prop = self.Hs_propagator(dt)
        for idx in self.idx_set:
            psi[idx] = H_prop @ psi[idx]

    def solve_python(self, N_traj: int, psi0: np.ndarray):
        self.initialize_psi()
        psis = np.zeros((N_traj, self.N_steps + 1, self.dim), dtype=complex)
        psis[:, 0, :] = psi0

        for traj in tqdm(range(N_traj), desc="Trajectories"):
            z = self.noise_sample_Z[traj, :] if self.noise_sample_Z is not None else self.noise_generator.sample_process()
            self.psi_prev[(0,) * self.rank] = psi0.copy()
            for step in range(self.N_steps):
                self.exp_step(self.psi_prev, 0.5 * self.dt)
                self.ode_step(self.psi_prev, self.psi_curr, z, step, self.dt)
                self.exp_step(self.psi_curr, 0.5 * self.dt)
                psis[traj, step + 1, :] = self.psi_curr[(0,) * self.rank]
                self.psi_prev, self.psi_curr = self.psi_curr, self.psi_prev

        return psis

    def solve_numba(self, N_traj, psi0, parallel_traj: bool = False, max_workers: int | None = None):
        if not _numba_available():
            raise RuntimeError("numba is not available")

        if self._numba_idx_values is None:
            self._build_numba_indexing()

        P_half = np.ascontiguousarray(self.P_half, dtype=np.complex128)
        L = np.ascontiguousarray(self.L, dtype=np.complex128)
        L_dag = np.ascontiguousarray(self.L_dag, dtype=np.complex128)
        V = np.ascontiguousarray(self.V, dtype=np.complex128)
        U = np.ascontiguousarray(self.U, dtype=np.complex128)
        lam = np.ascontiguousarray(self.lam, dtype=np.float64)
        dt = float(self.dt)

        n_states = len(self.idx_set)
        N = self.N_steps + 1
        psis = np.zeros((N_traj, N, self.dim), dtype=np.complex128)
        psis[:, 0, :] = np.asarray(psi0, dtype=np.complex128)

        if not parallel_traj:
            psi_prev = np.zeros((n_states, self.dim), dtype=np.complex128)
            psi_curr = np.zeros((n_states, self.dim), dtype=np.complex128)
            psi_half = np.zeros((n_states, self.dim), dtype=np.complex128)
            k1 = np.zeros((n_states, self.dim), dtype=np.complex128)
            psi_pred = np.zeros((n_states, self.dim), dtype=np.complex128)

            for traj in tqdm(range(N_traj), desc="Trajectories"):
                if self.noise_sample_Z is None:
                    z = np.asarray(self.noise_generator.sample_process(), dtype=np.complex128)
                else:
                    z = np.asarray(self.noise_sample_Z[traj, :], dtype=np.complex128)

                if z.shape[0] < N:
                    raise ValueError(f"noise length must be at least N_steps+1={N}, got {z.shape[0]}")

                psi_prev[:, :] = 0.0
                psi_prev[self._zero_idx, :] = psis[traj, 0, :]

                for n in range(self.N_steps):
                    _strang_step_kernel(
                        psi_prev,
                        psi_curr,
                        psi_half,
                        k1,
                        psi_pred,
                        P_half,
                        L,
                        L_dag,
                        z[n],
                        z[n + 1],
                        V[:, n],
                        V[:, n + 1],
                        U[n, :],
                        U[n + 1, :],
                        lam,
                        dt,
                        self._numba_idx_values,
                        self._numba_idx_minus,
                        self._numba_idx_plus,
                    )

                    psis[traj, n + 1, :] = psi_curr[self._zero_idx, :]
                    psi_prev, psi_curr = psi_curr, psi_prev

            return psis

        if max_workers is None:
            max_workers = os.cpu_count() or 1

        Z = self.noise_sample_Z
        if Z is None:
            Z = np.empty((N_traj, N), dtype=np.complex128)
            for traj in range(N_traj):
                z = np.asarray(self.noise_generator.sample_process(), dtype=np.complex128)
                if z.shape[0] < N:
                    raise ValueError(f"noise length must be at least N_steps+1={N}, got {z.shape[0]}")
                Z[traj, :] = z[:N]
        else:
            Z = np.asarray(Z, dtype=np.complex128)
            if Z.shape[1] < N:
                raise ValueError(f"noise_sample_Z second dimension must be at least N_steps+1={N}, got {Z.shape[1]}")

        if self.N_steps > 0:
            _psi_prev0 = np.zeros((n_states, self.dim), dtype=np.complex128)
            _psi_curr0 = np.zeros((n_states, self.dim), dtype=np.complex128)
            _psi_half0 = np.zeros((n_states, self.dim), dtype=np.complex128)
            _k10 = np.zeros((n_states, self.dim), dtype=np.complex128)
            _psi_pred0 = np.zeros((n_states, self.dim), dtype=np.complex128)

            _strang_step_kernel_serial(
                _psi_prev0,
                _psi_curr0,
                _psi_half0,
                _k10,
                _psi_pred0,
                P_half,
                L,
                L_dag,
                Z[0, 0],
                Z[0, 1],
                V[:, 0],
                V[:, 1],
                U[0, :],
                U[1, :],
                lam,
                dt,
                self._numba_idx_values,
                self._numba_idx_minus,
                self._numba_idx_plus,
            )

        psi0_arr = np.asarray(psi0, dtype=np.complex128)

        def _run_one(traj: int):
            traj_psis = np.empty((N, self.dim), dtype=np.complex128)
            traj_psis[0, :] = psi0_arr

            psi_prev = np.zeros((n_states, self.dim), dtype=np.complex128)
            psi_curr = np.zeros((n_states, self.dim), dtype=np.complex128)
            psi_half = np.zeros((n_states, self.dim), dtype=np.complex128)
            k1 = np.zeros((n_states, self.dim), dtype=np.complex128)
            psi_pred = np.zeros((n_states, self.dim), dtype=np.complex128)
            psi_prev[self._zero_idx, :] = psi0_arr

            z_traj = Z[traj]
            for n in range(self.N_steps):
                _strang_step_kernel_serial(
                    psi_prev,
                    psi_curr,
                    psi_half,
                    k1,
                    psi_pred,
                    P_half,
                    L,
                    L_dag,
                    z_traj[n],
                    z_traj[n + 1],
                    V[:, n],
                    V[:, n + 1],
                    U[n, :],
                    U[n + 1, :],
                    lam,
                    dt,
                    self._numba_idx_values,
                    self._numba_idx_minus,
                    self._numba_idx_plus,
                )

                traj_psis[n + 1, :] = psi_curr[self._zero_idx, :]
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
            return self.solve_numba(N_traj, psi0, parallel_traj=parallel_traj, max_workers=max_workers)
        else:
            return self.solve_python(N_traj, psi0)
