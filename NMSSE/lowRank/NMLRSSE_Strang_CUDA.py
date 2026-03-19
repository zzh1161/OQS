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
    def _gpu_matmul_states_batched(A, X, Y):
        bs, i = cuda.grid(2)
        n_batch = X.shape[0]
        n_states = X.shape[1]
        dim = A.shape[0]
        total_states = n_batch * n_states
        if bs < total_states and i < dim:
            b = bs // n_states
            s = bs - b * n_states
            acc = 0.0 + 0.0j
            for j in range(dim):
                acc += A[i, j] * X[b, s, j]
            Y[b, s, i] = acc

    @cuda.jit
    def _gpu_rhs_predict_kernel(
        out_k1,
        out_pred,
        psi_in,
        z,
        L,
        L_dag,
        V,
        U,
        lam,
        step,
        dt,
        minus_count,
        minus_k,
        minus_state,
        minus_value,
        plus_count,
        plus_k,
        plus_state,
    ):
        bs, i = cuda.grid(2)
        n_batch = psi_in.shape[0]
        n_states = psi_in.shape[1]
        dim = L.shape[0]
        total_states = n_batch * n_states
        if bs < total_states and i < dim:
            b = bs // n_states
            s = bs - b * n_states
            z_step = z[b, step]
            acc = 0.0 + 0.0j
            for j in range(dim):
                acc += (z_step * L[i, j]) * psi_in[b, s, j]

            m_count = minus_count[s]
            for m in range(m_count):
                k = minus_k[s, m]
                s_minus = minus_state[s, m]
                scale_minus = minus_value[s, m] * V[k, step]
                for j in range(dim):
                    acc += scale_minus * L[i, j] * psi_in[b, s_minus, j]

            p_count = plus_count[s]
            for m in range(p_count):
                k = plus_k[s, m]
                s_plus = plus_state[s, m]
                scale_plus = -lam[k] * U[step, k]
                for j in range(dim):
                    acc += scale_plus * L_dag[i, j] * psi_in[b, s_plus, j]

            out_k1[b, s, i] = acc
            out_pred[b, s, i] = psi_in[b, s, i] + dt * acc

    @cuda.jit
    def _gpu_rhs_heun_kernel(
        out,
        base,
        k1,
        psi_pred,
        z,
        L,
        L_dag,
        V,
        U,
        lam,
        step,
        dt,
        minus_count,
        minus_k,
        minus_state,
        minus_value,
        plus_count,
        plus_k,
        plus_state,
    ):
        bs, i = cuda.grid(2)
        n_batch = psi_pred.shape[0]
        n_states = psi_pred.shape[1]
        dim = L.shape[0]
        total_states = n_batch * n_states
        if bs < total_states and i < dim:
            b = bs // n_states
            s = bs - b * n_states
            z_step = z[b, step]
            acc = 0.0 + 0.0j
            for j in range(dim):
                acc += (z_step * L[i, j]) * psi_pred[b, s, j]

            m_count = minus_count[s]
            for m in range(m_count):
                k = minus_k[s, m]
                s_minus = minus_state[s, m]
                scale_minus = minus_value[s, m] * V[k, step]
                for j in range(dim):
                    acc += scale_minus * L[i, j] * psi_pred[b, s_minus, j]

            p_count = plus_count[s]
            for m in range(p_count):
                k = plus_k[s, m]
                s_plus = plus_state[s, m]
                scale_plus = -lam[k] * U[step, k]
                for j in range(dim):
                    acc += scale_plus * L_dag[i, j] * psi_pred[b, s_plus, j]

            out[b, s, i] = base[b, s, i] + 0.5 * dt * (k1[b, s, i] + acc)

    @cuda.jit
    def _gpu_store_zero_state_kernel(out_hist, states, zero_idx, step):
        b, i = cuda.grid(2)
        n_batch = states.shape[0]
        dim = states.shape[2]
        if b < n_batch and i < dim:
            out_hist[b, step, i] = states[b, zero_idx, i]

    @cuda.jit
    def _gpu_reset_states_kernel(states, zero_idx, psi0):
        bs, i = cuda.grid(2)
        n_batch = states.shape[0]
        n_states = states.shape[1]
        dim = states.shape[2]
        total_states = n_batch * n_states
        if bs < total_states and i < dim:
            b = bs // n_states
            s = bs - b * n_states
            if s == zero_idx:
                states[b, s, i] = psi0[i]
            else:
                states[b, s, i] = 0.0 + 0.0j


class NMLRSSE_Strang_CUDA:
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

        self._numba_minus_count = None
        self._numba_minus_k = None
        self._numba_minus_state = None
        self._numba_minus_value = None
        self._numba_plus_count = None
        self._numba_plus_k = None
        self._numba_plus_state = None
        self._zero_idx = self.idx_map[(0,) * self.rank]

        self._gpu_const_cache = None

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
        minus_count = np.zeros(n_states, dtype=np.int32)
        minus_k = np.zeros((n_states, self.rank), dtype=np.int32)
        minus_state = np.zeros((n_states, self.rank), dtype=np.int32)
        minus_value = np.zeros((n_states, self.rank), dtype=np.float64)

        plus_count = np.zeros(n_states, dtype=np.int32)
        plus_k = np.zeros((n_states, self.rank), dtype=np.int32)
        plus_state = np.zeros((n_states, self.rank), dtype=np.int32)

        for idx, pos in self.idx_map.items():
            m = 0
            p = 0
            for k in range(self.rank):
                if idx[k] > 0:
                    idx_m = idx[:k] + (idx[k] - 1,) + idx[k + 1 :]
                    s_minus = self.idx_map.get(idx_m, -1)
                    if s_minus >= 0:
                        minus_k[pos, m] = k
                        minus_state[pos, m] = s_minus
                        minus_value[pos, m] = float(idx[k])
                        m += 1
                idx_p = idx[:k] + (idx[k] + 1,) + idx[k + 1 :]
                s_plus = self.idx_map.get(idx_p, -1)
                if s_plus >= 0:
                    plus_k[pos, p] = k
                    plus_state[pos, p] = s_plus
                    p += 1
            minus_count[pos] = m
            plus_count[pos] = p

        self._numba_minus_count = minus_count
        self._numba_minus_k = minus_k
        self._numba_minus_state = minus_state
        self._numba_minus_value = minus_value
        self._numba_plus_count = plus_count
        self._numba_plus_k = plus_k
        self._numba_plus_state = plus_state

    def _ensure_gpu_constants(self):
        if self._gpu_const_cache is not None:
            return self._gpu_const_cache

        P_half = np.ascontiguousarray(self.P_half, dtype=np.complex128)
        L = np.ascontiguousarray(self.L, dtype=np.complex128)
        L_dag = np.ascontiguousarray(self.L_dag, dtype=np.complex128)
        V = np.ascontiguousarray(self.V, dtype=np.complex128)
        U = np.ascontiguousarray(self.U, dtype=np.complex128)
        lam = np.ascontiguousarray(self.lam, dtype=np.float64)

        minus_count = np.ascontiguousarray(self._numba_minus_count, dtype=np.int32)
        minus_k = np.ascontiguousarray(self._numba_minus_k, dtype=np.int32)
        minus_state = np.ascontiguousarray(self._numba_minus_state, dtype=np.int32)
        minus_value = np.ascontiguousarray(self._numba_minus_value, dtype=np.float64)
        plus_count = np.ascontiguousarray(self._numba_plus_count, dtype=np.int32)
        plus_k = np.ascontiguousarray(self._numba_plus_k, dtype=np.int32)
        plus_state = np.ascontiguousarray(self._numba_plus_state, dtype=np.int32)

        self._gpu_const_cache = {
            "d_P_half": cuda.to_device(P_half),
            "d_L": cuda.to_device(L),
            "d_L_dag": cuda.to_device(L_dag),
            "d_V": cuda.to_device(V),
            "d_U": cuda.to_device(U),
            "d_lam": cuda.to_device(lam),
            "d_minus_count": cuda.to_device(minus_count),
            "d_minus_k": cuda.to_device(minus_k),
            "d_minus_state": cuda.to_device(minus_state),
            "d_minus_value": cuda.to_device(minus_value),
            "d_plus_count": cuda.to_device(plus_count),
            "d_plus_k": cuda.to_device(plus_k),
            "d_plus_state": cuda.to_device(plus_state),
        }
        return self._gpu_const_cache
    

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

    def solve_numba_gpu(
        self,
        N_traj: int,
        psi0: np.ndarray,
        threadsperblock: tuple[int, int] = (128, 4),
        traj_batch_size: int = 64,
    ):
        if not _numba_gpu_available():
            raise RuntimeError("numba cuda is not available")

        if self._numba_minus_count is None:
            self._build_numba_indexing()

        gpu_const = self._ensure_gpu_constants()

        n_states = len(self.idx_set)
        N = self.N_steps + 1
        dt = float(self.dt)

        psi0_arr = np.asarray(psi0, dtype=np.complex128)
        psis = np.zeros((N_traj, N, self.dim), dtype=np.complex128)
        psis[:, 0, :] = psi0_arr

        tx, ty = threadsperblock
        if traj_batch_size <= 0:
            raise ValueError("traj_batch_size must be positive")
        if tx <= 0 or ty <= 0 or tx * ty > 1024:
            raise ValueError("threadsperblock must be positive and tx*ty <= 1024")

        if self.noise_sample_Z is None:
            Z = np.empty((N_traj, N), dtype=np.complex128)
            for traj in range(N_traj):
                z = np.asarray(self.noise_generator.sample_process(), dtype=np.complex128)
                if z.shape[0] < N:
                    raise ValueError(f"noise length must be at least N_steps+1={N}, got {z.shape[0]}")
                Z[traj, :] = z[:N]
        else:
            Z = np.asarray(self.noise_sample_Z, dtype=np.complex128)
            if Z.shape[0] < N_traj:
                raise ValueError(f"noise_sample_Z first dimension must be at least N_traj={N_traj}, got {Z.shape[0]}")
            if Z.shape[1] < N:
                raise ValueError(f"noise_sample_Z second dimension must be at least N_steps+1={N}, got {Z.shape[1]}")
            Z = Z[:N_traj, :N]

        max_nb = min(traj_batch_size, N_traj)
        d_z = cuda.device_array((max_nb, N), dtype=np.complex128)
        d_prev = cuda.device_array((max_nb, n_states, self.dim), dtype=np.complex128)
        d_half = cuda.device_array((max_nb, n_states, self.dim), dtype=np.complex128)
        d_curr = cuda.device_array((max_nb, n_states, self.dim), dtype=np.complex128)
        d_k1 = cuda.device_array((max_nb, n_states, self.dim), dtype=np.complex128)
        d_pred = cuda.device_array((max_nb, n_states, self.dim), dtype=np.complex128)
        d_hist = cuda.device_array((max_nb, N, self.dim), dtype=np.complex128)
        d_psi0 = cuda.to_device(psi0_arr)

        for batch_start in tqdm(range(0, N_traj, traj_batch_size), desc="Trajectory batches"):
            batch_end = min(batch_start + traj_batch_size, N_traj)
            nb = batch_end - batch_start

            z_batch = np.ascontiguousarray(Z[batch_start:batch_end, :], dtype=np.complex128)
            d_z[:nb, :].copy_to_device(z_batch)

            d_prev_b = d_prev[:nb, :, :]
            d_half_b = d_half[:nb, :, :]
            d_curr_b = d_curr[:nb, :, :]
            d_k1_b = d_k1[:nb, :, :]
            d_pred_b = d_pred[:nb, :, :]
            d_hist_b = d_hist[:nb, :, :]
            d_z_b = d_z[:nb, :]

            blocks_states = ((nb * n_states + tx - 1) // tx, (self.dim + ty - 1) // ty)
            blocks_batch = ((nb + tx - 1) // tx, (self.dim + ty - 1) // ty)

            _gpu_reset_states_kernel[blocks_states, threadsperblock](d_prev_b, self._zero_idx, d_psi0)

            _gpu_store_zero_state_kernel[blocks_batch, threadsperblock](d_hist_b, d_prev_b, self._zero_idx, 0)

            for n in range(self.N_steps):
                _gpu_matmul_states_batched[blocks_states, threadsperblock](gpu_const["d_P_half"], d_prev_b, d_half_b)

                _gpu_rhs_predict_kernel[blocks_states, threadsperblock](
                    d_k1_b,
                    d_pred_b,
                    d_half_b,
                    d_z_b,
                    gpu_const["d_L"],
                    gpu_const["d_L_dag"],
                    gpu_const["d_V"],
                    gpu_const["d_U"],
                    gpu_const["d_lam"],
                    n,
                    dt,
                    gpu_const["d_minus_count"],
                    gpu_const["d_minus_k"],
                    gpu_const["d_minus_state"],
                    gpu_const["d_minus_value"],
                    gpu_const["d_plus_count"],
                    gpu_const["d_plus_k"],
                    gpu_const["d_plus_state"],
                )

                _gpu_rhs_heun_kernel[blocks_states, threadsperblock](
                    d_curr_b,
                    d_half_b,
                    d_k1_b,
                    d_pred_b,
                    d_z_b,
                    gpu_const["d_L"],
                    gpu_const["d_L_dag"],
                    gpu_const["d_V"],
                    gpu_const["d_U"],
                    gpu_const["d_lam"],
                    n + 1,
                    dt,
                    gpu_const["d_minus_count"],
                    gpu_const["d_minus_k"],
                    gpu_const["d_minus_state"],
                    gpu_const["d_minus_value"],
                    gpu_const["d_plus_count"],
                    gpu_const["d_plus_k"],
                    gpu_const["d_plus_state"],
                )

                _gpu_matmul_states_batched[blocks_states, threadsperblock](gpu_const["d_P_half"], d_curr_b, d_half_b)
                _gpu_store_zero_state_kernel[blocks_batch, threadsperblock](d_hist_b, d_half_b, self._zero_idx, n + 1)
                d_prev_b, d_half_b = d_half_b, d_prev_b

            psis[batch_start:batch_end, :, :] = d_hist_b.copy_to_host()

        cuda.synchronize()
        return psis

    def solve(
        self,
        N_traj: int,
        psi0: np.ndarray,
        backend: str = "auto",
        threadsperblock: tuple[int, int] = (128, 4),
        traj_batch_size: int = 64,
    ):
        if backend not in ("auto", "python", "numba", "numba-gpu"):
            raise ValueError("backend must be one of: 'auto', 'python', 'numba', 'numba-gpu'")

        if backend in ("numba", "numba-gpu"):
            return self.solve_numba_gpu(
                N_traj,
                psi0,
                threadsperblock=threadsperblock,
                traj_batch_size=traj_batch_size,
            )

        if backend == "auto" and _numba_gpu_available():
            return self.solve_numba_gpu(
                N_traj,
                psi0,
                threadsperblock=threadsperblock,
                traj_batch_size=traj_batch_size,
            )

        return self.solve_python(N_traj, psi0)