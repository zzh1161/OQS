import numpy as np
import time
from typing import Dict, Tuple
from tqdm import tqdm

try:
    from numba import njit, prange, cuda  # type: ignore
except Exception:  # pragma: no cover
    try:
        from numba import njit, prange  # type: ignore
    except Exception:  # pragma: no cover
        njit = None
        prange = range
    cuda = None

try:
    from ..utils.noise_generator import ColoredNoiseGenerator_Cholesky
    from ..utils.multi_index import MultiIndex
except ImportError:
    from utils.noise_generator import ColoredNoiseGenerator_Cholesky
    from utils.multi_index import MultiIndex


def _numba_available() -> bool:
    return njit is not None


def _numba_gpu_available() -> bool:
    if cuda is None:
        return False
    try:
        return bool(cuda.is_available())
    except Exception:
        return False


if _numba_gpu_available():
    @cuda.jit
    def _gpu_matmul_states(A, X, Y):
        s, i = cuda.grid(2)
        n_states = X.shape[0]
        dim = A.shape[0]
        if s < n_states and i < dim:
            acc = 0.0 + 0.0j
            for j in range(dim):
                acc += A[i, j] * X[s, j]
            Y[s, i] = acc

    @cuda.jit
    def _gpu_rhs_kernel(
        out,
        psi_in,
        L,
        L_dag,
        z_step,
        V,
        U,
        lam,
        step,
        idx_values,
        idx_minus,
        idx_plus,
    ):
        s, i = cuda.grid(2)
        n_states = psi_in.shape[0]
        dim = L.shape[0]
        if s < n_states and i < dim:
            acc = 0.0 + 0.0j
            for j in range(dim):
                acc += (z_step * L[i, j]) * psi_in[s, j]

            rank = lam.shape[0]
            for k in range(rank):
                s_minus = idx_minus[s, k]
                if s_minus >= 0:
                    scale_minus = idx_values[s, k] * V[k, step]
                    for j in range(dim):
                        acc += scale_minus * L[i, j] * psi_in[s_minus, j]

            for k in range(rank):
                s_plus = idx_plus[s, k]
                if s_plus >= 0:
                    scale_plus = -lam[k] * U[step, k]
                    for j in range(dim):
                        acc += scale_plus * L_dag[i, j] * psi_in[s_plus, j]

            out[s, i] = acc

    @cuda.jit
    def _gpu_linear2_kernel(out, x, y, ax, ay):
        s, i = cuda.grid(2)
        n_states, dim = out.shape
        if s < n_states and i < dim:
            out[s, i] = ax * x[s, i] + ay * y[s, i]

    @cuda.jit
    def _gpu_heun_combine_kernel(out, base, k1, k2, dt):
        s, i = cuda.grid(2)
        n_states, dim = out.shape
        if s < n_states and i < dim:
            out[s, i] = base[s, i] + 0.5 * dt * (k1[s, i] + k2[s, i])


class NMLRSSE_Strang_GPU:
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

    def solve_numba_gpu(self, N_traj: int, psi0: np.ndarray, threadsperblock: tuple[int, int] = (16, 8)):
        if not _numba_gpu_available():
            raise RuntimeError("numba cuda is not available")

        if self._numba_idx_values is None:
            self._build_numba_indexing()

        P_half = np.ascontiguousarray(self.P_half, dtype=np.complex128)
        L = np.ascontiguousarray(self.L, dtype=np.complex128)
        L_dag = np.ascontiguousarray(self.L_dag, dtype=np.complex128)
        V = np.ascontiguousarray(self.V, dtype=np.complex128)
        U = np.ascontiguousarray(self.U, dtype=np.complex128)
        lam = np.ascontiguousarray(self.lam, dtype=np.float64)

        idx_values = np.ascontiguousarray(self._numba_idx_values, dtype=np.int64)
        idx_minus = np.ascontiguousarray(self._numba_idx_minus, dtype=np.int64)
        idx_plus = np.ascontiguousarray(self._numba_idx_plus, dtype=np.int64)

        n_states = len(self.idx_set)
        N = self.N_steps + 1
        dt = float(self.dt)

        psi0_arr = np.asarray(psi0, dtype=np.complex128)
        psis = np.zeros((N_traj, N, self.dim), dtype=np.complex128)
        psis[:, 0, :] = psi0_arr

        d_P_half = cuda.to_device(P_half)
        d_L = cuda.to_device(L)
        d_L_dag = cuda.to_device(L_dag)
        d_V = cuda.to_device(V)
        d_U = cuda.to_device(U)
        d_lam = cuda.to_device(lam)
        d_idx_values = cuda.to_device(idx_values)
        d_idx_minus = cuda.to_device(idx_minus)
        d_idx_plus = cuda.to_device(idx_plus)

        d_prev = cuda.device_array((n_states, self.dim), dtype=np.complex128)
        d_half = cuda.device_array((n_states, self.dim), dtype=np.complex128)
        d_curr = cuda.device_array((n_states, self.dim), dtype=np.complex128)
        d_k1 = cuda.device_array((n_states, self.dim), dtype=np.complex128)
        d_pred = cuda.device_array((n_states, self.dim), dtype=np.complex128)

        tx, ty = threadsperblock
        blockspergrid = ((n_states + tx - 1) // tx, (self.dim + ty - 1) // ty)

        host_state = np.zeros((n_states, self.dim), dtype=np.complex128)
        host_zero = np.empty(self.dim, dtype=np.complex128)

        for traj in tqdm(range(N_traj), desc="Trajectories"):
            if self.noise_sample_Z is None:
                z = np.asarray(self.noise_generator.sample_process(), dtype=np.complex128)
            else:
                z = np.asarray(self.noise_sample_Z[traj, :], dtype=np.complex128)

            if z.shape[0] < N:
                raise ValueError(f"noise length must be at least N_steps+1={N}, got {z.shape[0]}")

            host_state[:, :] = 0.0
            host_state[self._zero_idx, :] = psi0_arr
            d_prev.copy_to_device(host_state)

            for n in range(self.N_steps):
                _gpu_matmul_states[blockspergrid, threadsperblock](d_P_half, d_prev, d_half)

                _gpu_rhs_kernel[blockspergrid, threadsperblock](
                    d_k1,
                    d_half,
                    d_L,
                    d_L_dag,
                    z[n],
                    d_V,
                    d_U,
                    d_lam,
                    n,
                    d_idx_values,
                    d_idx_minus,
                    d_idx_plus,
                )

                _gpu_linear2_kernel[blockspergrid, threadsperblock](d_pred, d_half, d_k1, 1.0 + 0.0j, dt + 0.0j)

                _gpu_rhs_kernel[blockspergrid, threadsperblock](
                    d_curr,
                    d_pred,
                    d_L,
                    d_L_dag,
                    z[n + 1],
                    d_V,
                    d_U,
                    d_lam,
                    n + 1,
                    d_idx_values,
                    d_idx_minus,
                    d_idx_plus,
                )

                _gpu_heun_combine_kernel[blockspergrid, threadsperblock](d_curr, d_half, d_k1, d_curr, dt)
                _gpu_matmul_states[blockspergrid, threadsperblock](d_P_half, d_curr, d_half)

                d_half[self._zero_idx, :].copy_to_host(host_zero)
                psis[traj, n + 1, :] = host_zero
                d_prev, d_half = d_half, d_prev

        cuda.synchronize()
        return psis

    def solve(self, N_traj: int, psi0: np.ndarray, backend: str = "auto"):
        if backend not in ("auto", "python", "numba", "numba-gpu"):
            raise ValueError("backend must be one of: 'auto', 'python', 'numba', 'numba-gpu'")

        if backend in ("numba", "numba-gpu"):
            return self.solve_numba_gpu(N_traj, psi0)

        if backend == "auto" and _numba_gpu_available():
            return self.solve_numba_gpu(N_traj, psi0)

        return self.solve_python(N_traj, psi0)