import numpy as np
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple
from tqdm import tqdm
from enum import Enum

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

class ODEScheme(Enum):
    pass

class Order1Scheme(ODEScheme):
    Forward_Euler = "forward_euler"

class Order2Scheme(ODEScheme):
    Heun = "heun"
    Midpoint = "midpoint"

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


    @njit(parallel=True, cache=True, nogil=True)
    def _psi_step_forward_euler_kernel(
        psi_prev,
        psi_curr,
        same_idx,
        minus_idx,
        plus_idx,
        idx_vals,
        Hs,
        L,
        L_dag,
        dt,
        z_n,
        Vcol,
        Ucol,
        lam,
    ):
        """Compute one time step for all multi-indices <= n+1.

        Indices are represented via integer lookup tables into psi_prev.
        -1 means 'missing' => treated as zero vector.
        """
        M_out, dim = psi_curr.shape
        r = Vcol.shape[0]
        for i in prange(M_out):
            for d in range(dim):
                psi_curr[i, d] = 0.0 + 0.0j

            j_same = same_idx[i]
            if j_same != -1:
                base = psi_prev[j_same]
                for d in range(dim):
                    psi_curr[i, d] += base[d]
                _matvec_add_scaled(Hs, base, psi_curr[i], (-1j) * dt)
                _matvec_add_scaled(L, base, psi_curr[i], z_n * dt)

            for k in range(r):
                j_minus = minus_idx[i, k]
                if j_minus != -1:
                    _matvec_add_scaled(
                        L,
                        psi_prev[j_minus],
                        psi_curr[i],
                        dt * idx_vals[i, k] * Vcol[k]
                    )

            for k in range(r):
                j_plus = plus_idx[i, k]
                if j_plus != -1:
                    _matvec_add_scaled(
                        L_dag,
                        psi_prev[j_plus],
                        psi_curr[i],
                        -dt * lam[k] * Ucol[k],
                    )


    @njit(parallel=True, cache=True, nogil=True)
    def _psi_step_heun_kernel(
        psi_prev,
        psi_curr,
        same_prev,
        minus_prev,
        plus_prev,
        same_out,
        minus_out,
        plus_out,
        idx_vals,
        Hs,
        L,
        L_dag,
        dt,
        z_n,
        z_np1,
        Vcol_n,
        Vcol_np1,
        Urow_n,
        Urow_np1,
        lam,
    ):
        """Heun (explicit trapezoidal) one-step update for all out indices.

        Uses two RHS evaluations:
        - k1 at (t_n, psi_prev)
        - k2 at (t_{n+1}, psi_pred) where psi_pred = psi_prev + dt*k1

        Lookup tables:
        - *_prev map indices to rows in psi_prev (missing => -1 => zero)
        - *_out map indices to rows in psi_pred (i.e. current out index set)
        """
        M_out, dim = psi_curr.shape
        r = Vcol_n.shape[0]

        psi_base = np.empty((M_out, dim), dtype=np.complex128)
        k1 = np.empty((M_out, dim), dtype=np.complex128)
        psi_pred = np.empty((M_out, dim), dtype=np.complex128)

        for i in prange(M_out):
            # psi_base = psi_n for this index (or zero)
            for d in range(dim):
                psi_base[i, d] = 0.0 + 0.0j

            j_same = same_prev[i]
            if j_same != -1:
                for d in range(dim):
                    psi_base[i, d] = psi_prev[j_same, d]

            # k1 = rhs(t_n, psi_prev)
            for d in range(dim):
                k1[i, d] = 0.0 + 0.0j

            _matvec_add_scaled(Hs, psi_base[i], k1[i], (-1j))
            _matvec_add_scaled(L, psi_base[i], k1[i], z_n)

            for kk in range(r):
                j_minus = minus_prev[i, kk]
                if j_minus != -1:
                    _matvec_add_scaled(L, psi_prev[j_minus], k1[i], idx_vals[i, kk] * Vcol_n[kk])

            for kk in range(r):
                j_plus = plus_prev[i, kk]
                if j_plus != -1:
                    _matvec_add_scaled(L_dag, psi_prev[j_plus], k1[i], -lam[kk] * Urow_n[kk])

            # predictor
            for d in range(dim):
                psi_pred[i, d] = psi_base[i, d] + dt * k1[i, d]

        # corrector (k2 uses psi_pred and step n+1 coefficients)
        for i in prange(M_out):
            # reuse psi_curr as accumulator for k2
            for d in range(dim):
                psi_curr[i, d] = 0.0 + 0.0j

            _matvec_add_scaled(Hs, psi_pred[i], psi_curr[i], (-1j))
            _matvec_add_scaled(L, psi_pred[i], psi_curr[i], z_np1)

            for kk in range(r):
                j_minus = minus_out[i, kk]
                if j_minus != -1:
                    _matvec_add_scaled(L, psi_pred[j_minus], psi_curr[i], idx_vals[i, kk] * Vcol_np1[kk])

            for kk in range(r):
                j_plus = plus_out[i, kk]
                if j_plus != -1:
                    _matvec_add_scaled(L_dag, psi_pred[j_plus], psi_curr[i], -lam[kk] * Urow_np1[kk])

            # final update: psi_{n+1} = psi_n + dt/2*(k1+k2)
            for d in range(dim):
                psi_curr[i, d] = psi_base[i, d] + 0.5 * dt * (k1[i, d] + psi_curr[i, d])


    @njit(cache=True, nogil=True)
    def _psi_step_forward_euler_kernel_serial(
        psi_prev,
        psi_curr,
        same_idx,
        minus_idx,
        plus_idx,
        idx_vals,
        Hs,
        L,
        L_dag,
        dt,
        z_n,
        Vcol,
        Ucol,
        lam,
    ):
        """Serial variant used when trajectories are parallelized in Python."""
        M_out, dim = psi_curr.shape
        r = Vcol.shape[0]
        for i in range(M_out):
            for d in range(dim):
                psi_curr[i, d] = 0.0 + 0.0j

            j_same = same_idx[i]
            if j_same != -1:
                base = psi_prev[j_same]
                for d in range(dim):
                    psi_curr[i, d] += base[d]
                _matvec_add_scaled(Hs, base, psi_curr[i], (-1j) * dt)
                _matvec_add_scaled(L, base, psi_curr[i], z_n * dt)

            for k in range(r):
                j_minus = minus_idx[i, k]
                if j_minus != -1:
                    _matvec_add_scaled(L, psi_prev[j_minus], psi_curr[i], dt * idx_vals[i, k] * Vcol[k])

            for k in range(r):
                j_plus = plus_idx[i, k]
                if j_plus != -1:
                    _matvec_add_scaled(
                        L_dag,
                        psi_prev[j_plus],
                        psi_curr[i],
                        -dt * lam[k] * Ucol[k],
                    )


    @njit(cache=True, nogil=True)
    def _psi_step_heun_kernel_serial(
        psi_prev,
        psi_curr,
        same_prev,
        minus_prev,
        plus_prev,
        same_out,
        minus_out,
        plus_out,
        idx_vals,
        Hs,
        L,
        L_dag,
        dt,
        z_n,
        z_np1,
        Vcol_n,
        Vcol_np1,
        Urow_n,
        Urow_np1,
        lam,
    ):
        """Serial Heun kernel used when trajectories are parallelized in Python."""
        M_out, dim = psi_curr.shape
        r = Vcol_n.shape[0]

        psi_base = np.empty((M_out, dim), dtype=np.complex128)
        k1 = np.empty((M_out, dim), dtype=np.complex128)
        psi_pred = np.empty((M_out, dim), dtype=np.complex128)

        for i in range(M_out):
            for d in range(dim):
                psi_base[i, d] = 0.0 + 0.0j

            j_same = same_prev[i]
            if j_same != -1:
                for d in range(dim):
                    psi_base[i, d] = psi_prev[j_same, d]

            for d in range(dim):
                k1[i, d] = 0.0 + 0.0j

            _matvec_add_scaled(Hs, psi_base[i], k1[i], (-1j))
            _matvec_add_scaled(L, psi_base[i], k1[i], z_n)

            for kk in range(r):
                j_minus = minus_prev[i, kk]
                if j_minus != -1:
                    _matvec_add_scaled(L, psi_prev[j_minus], k1[i], idx_vals[i, kk] * Vcol_n[kk])

            for kk in range(r):
                j_plus = plus_prev[i, kk]
                if j_plus != -1:
                    _matvec_add_scaled(L_dag, psi_prev[j_plus], k1[i], -lam[kk] * Urow_n[kk])

            for d in range(dim):
                psi_pred[i, d] = psi_base[i, d] + dt * k1[i, d]

        for i in range(M_out):
            for d in range(dim):
                psi_curr[i, d] = 0.0 + 0.0j

            _matvec_add_scaled(Hs, psi_pred[i], psi_curr[i], (-1j))
            _matvec_add_scaled(L, psi_pred[i], psi_curr[i], z_np1)

            for kk in range(r):
                j_minus = minus_out[i, kk]
                if j_minus != -1:
                    _matvec_add_scaled(L, psi_pred[j_minus], psi_curr[i], idx_vals[i, kk] * Vcol_np1[kk])

            for kk in range(r):
                j_plus = plus_out[i, kk]
                if j_plus != -1:
                    _matvec_add_scaled(L_dag, psi_pred[j_plus], psi_curr[i], -lam[kk] * Urow_np1[kk])

            for d in range(dim):
                psi_curr[i, d] = psi_base[i, d] + 0.5 * dt * (k1[i, d] + psi_curr[i, d])


class NMLRSSE:
    def __init__(
        self,
        Hs,
        L,
        bath_corr,
        tmax,
        N_steps,
        rank,
        noise_generator = ColoredNoiseGenerator_Cholesky,
        max_order: int | None = None,
        scheme: ODEScheme = Order1Scheme.Forward_Euler,
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
        self.scheme = scheme
        # Hierarchy truncation (multi-index total order cutoff).
        # Default keeps original behavior: max_order == N_steps.
        if max_order is None:
            self.max_order = int(N_steps)
        else:
            self.max_order = int(max_order)
        if self.max_order < 0:
            raise ValueError("max_order must be non-negative")
        if self.max_order > self.N_steps:
            self.max_order = int(self.N_steps)
        
        self.dt = tmax / N_steps
        self.t_grid = np.linspace(0, tmax, N_steps+1)

        self.noise_generator = noise_generator(
            bath_corr, tmax, N_steps
        )

        self.psi_prev: Dict[Tuple[int, ...], np.ndarray] = {}
        self.psi_curr: Dict[Tuple[int, ...], np.ndarray] = {}


    def initialize(self):
        self.idx_set = MultiIndex(r = self.rank, n = self.max_order)
        if isinstance(self.scheme, Order1Scheme):
            self.idx_set.generate(order=1)
        elif isinstance(self.scheme, Order2Scheme):
            self.idx_set.generate(order=2)
        else:
            raise ValueError(f"Unsupported ODE scheme: {self.scheme}")

        time_cost = self.compute_low_rank_decomposition()
        print(f"Low-rank decomposition computed in {time_cost:.2f} seconds.")


    def compute_low_rank_decomposition(self):
        N = self.N_steps
        if self.scheme == Order2Scheme.Midpoint:
            N = 2 * self.N_steps

        t0 = time.perf_counter()
        C = np.empty((N+1, N+1), dtype=complex)
        for i in range(N+1):
            for j in range(N+1):
                C[i,j] = self.bath_corr(self.t_grid[i]-self.t_grid[j])
        U, s, V = np.linalg.svd(C)
        self.V = V[:self.rank, :]
        self.U = U[:, :self.rank]
        self.lam = s[:self.rank]
        dt = time.perf_counter() - t0
        return dt


    def set_Hamiltonian(self, Hs):
        self.Hs = Hs
    
    def set_L(self, L):
        self.L = L
        self.L_dag = L.conj().T
    
    def set_bath_corr(self, bath_corr):
        self.bath_corr = bath_corr
        self.compute_low_rank_decomposition()
    
    def set_scheme(self, scheme: ODEScheme):
        self.scheme = scheme
        self.initialize()


    def valid_psi(self, psi:Dict[Tuple[int, ...], np.ndarray], idx:Tuple[int, ...]) -> np.ndarray:
        if psi.get(idx) is not None:
            return psi[idx]
        else:
            # Safe to share: callers only read from this vector.
            return self._zero_state
        
        
    def psi_step_forward_euler(self, z, n):
        A = self.Id - 1j * self.dt * self.Hs + z[n] * self.L * self.dt
        # With hierarchy truncation, we only keep indices with |k| <= max_order.
        # Missing higher-order neighbors are treated as zero by `valid_psi`.
        for idx in self.idx_set.iter_leq(min(n + 1, self.max_order)):
            self.psi_curr[idx] = A @ self.valid_psi(self.psi_prev, idx)
            for k in range(self.rank):
                idx_minus = idx[:k] + (idx[k]-1,) + idx[k+1:]
                self.psi_curr[idx] += self.dt*idx[k]*self.V[k,n]*self.L @ self.valid_psi(self.psi_prev, idx_minus)
            for k in range(self.rank):
                idx_plus = idx[:k] + (idx[k]+1,) + idx[k+1:]
                self.psi_curr[idx] -= self.dt*self.lam[k]*self.U[n,k]*self.L_dag @ self.valid_psi(self.psi_prev, idx_plus)


    def psi_step_heun(self, z, n):
        """One Heun (explicit trapezoidal) step.

        This is a 2nd-order predictor-corrector method:
        - Predictor: Euler step using rhs at (t_n, psi_n)
        - Corrector: psi_{n+1} = psi_n + dt/2 * (k1 + k2)

        Notes
        -----
        - This is the simple Python implementation (no numba).
        - We evaluate the second slope using coefficients at step n+1:
          z[n+1], V[:, n+1], U[n+1, :].
        """

        if n < 0 or n >= self.N_steps:
            raise ValueError(f"n must satisfy 0 <= n < N_steps (got n={n})")

        max_layer = min(n + 1, self.max_order)
        out_indices = list(self.idx_set.iter_leq(max_layer))

        def _rhs(step: int, psi: Dict[Tuple[int, ...], np.ndarray], idx: Tuple[int, ...]) -> np.ndarray:
            base = self.valid_psi(psi, idx)
            acc = (-1j) * (self.Hs @ base) + z[step] * (self.L @ base)

            for k in range(self.rank):
                idx_minus = idx[:k] + (idx[k] - 1,) + idx[k + 1 :]
                acc = acc + idx[k] * self.V[k, step] * (self.L @ self.valid_psi(psi, idx_minus))

            for k in range(self.rank):
                idx_plus = idx[:k] + (idx[k] + 1,) + idx[k + 1 :]
                acc = acc - (self.lam[k] * self.U[step, k]) * (self.L_dag @ self.valid_psi(psi, idx_plus))

            return acc

        k1: Dict[Tuple[int, ...], np.ndarray] = {}
        psi_pred: Dict[Tuple[int, ...], np.ndarray] = {}

        for idx in out_indices:
            psi_n = self.valid_psi(self.psi_prev, idx)
            k1_idx = _rhs(n, self.psi_prev, idx)
            k1[idx] = k1_idx
            psi_pred[idx] = psi_n + self.dt * k1_idx

        for idx in out_indices:
            psi_n = self.valid_psi(self.psi_prev, idx)
            k2_idx = _rhs(n + 1, psi_pred, idx)
            self.psi_curr[idx] = psi_n + 0.5 * self.dt * (k1[idx] + k2_idx)


    def _psi_step_numba_tables(self, prev_map: Dict[Tuple[int, ...], int], out_indices):
        """Build integer lookup tables for one step.

        Returns (same_idx, minus_idx, plus_idx, idx_vals) for use by the numba kernel.
        """
        M_out = len(out_indices)
        same_idx = np.empty(M_out, dtype=np.int64)
        minus_idx = np.empty((M_out, self.rank), dtype=np.int64)
        plus_idx = np.empty((M_out, self.rank), dtype=np.int64)
        idx_vals = np.empty((M_out, self.rank), dtype=np.int64)

        for i, idx in enumerate(out_indices):
            same_idx[i] = prev_map.get(idx, -1)
            for k in range(self.rank):
                idx_vals[i, k] = idx[k]
                idx_minus = idx[:k] + (idx[k] - 1,) + idx[k + 1 :]
                minus_idx[i, k] = prev_map.get(idx_minus, -1)
                idx_plus = idx[:k] + (idx[k] + 1,) + idx[k + 1 :]
                plus_idx[i, k] = prev_map.get(idx_plus, -1)

        return same_idx, minus_idx, plus_idx, idx_vals

    def _prepare_numba_tables(self):
        """Precompute neighbor lookup tables for all steps.

        The tables depend only on `rank`, `N_steps`, and the deterministic ordering of `MultiIndex`.
        This makes per-trajectory propagation much faster.
        """
        if hasattr(self, "_numba_tables") and self._numba_tables is not None:
            return

        zero_idx = (0,) * self.rank
        prev_indices = [zero_idx]
        prev_map: Dict[Tuple[int, ...], int] = {zero_idx: 0}

        tables = []
        tail_table = None
        for n in range(self.N_steps):
            # After reaching `max_order`, no new layer is added.
            # The topology is frozen, so we can reuse the same neighbor table.
            if n + 1 <= self.max_order:
                new_layer = self.idx_set.layer(n + 1)
                out_indices = prev_indices + new_layer
                # Tables mapping indices -> psi_prev (previous index set)
                same_idx, minus_idx, plus_idx, idx_vals = self._psi_step_numba_tables(prev_map, out_indices)

                # Tables mapping indices -> psi_out (current out index set)
                out_map: Dict[Tuple[int, ...], int] = {idx: i for i, idx in enumerate(out_indices)}
                same_out, minus_out, plus_out, _ = self._psi_step_numba_tables(out_map, out_indices)
                tables.append(
                    {
                        "same": same_idx,
                        "minus": minus_idx,
                        "plus": plus_idx,
                        "same_out": same_out,
                        "minus_out": minus_out,
                        "plus_out": plus_out,
                        "idx_vals": idx_vals,
                        "out_len": len(out_indices),
                    }
                )

                prev_len = len(prev_indices)
                for i, idx in enumerate(new_layer):
                    prev_map[idx] = prev_len + i
                prev_indices = out_indices
            else:
                if tail_table is None:
                    out_indices = prev_indices
                    same_idx, minus_idx, plus_idx, idx_vals = self._psi_step_numba_tables(prev_map, out_indices)
                    out_map: Dict[Tuple[int, ...], int] = {idx: i for i, idx in enumerate(out_indices)}
                    same_out, minus_out, plus_out, _ = self._psi_step_numba_tables(out_map, out_indices)
                    tail_table = {
                        "same": same_idx,
                        "minus": minus_idx,
                        "plus": plus_idx,
                        "same_out": same_out,
                        "minus_out": minus_out,
                        "plus_out": plus_out,
                        "idx_vals": idx_vals,
                        "out_len": len(out_indices),
                    }
                tables.append(tail_table)

        self._numba_tables = tables


    def solve_numba(self, N_traj, psi0, parallel_traj: bool = False, max_workers: int | None = None):
        if not _numba_available():
            raise ImportError(
                "numba is not available. Install it (e.g. `pip install numba`) or use backend='python'."
            )

        self._prepare_numba_tables()
        
        Hs = np.ascontiguousarray(self.Hs, dtype=np.complex128)
        L = np.ascontiguousarray(self.L, dtype=np.complex128)
        L_dag = np.ascontiguousarray(self.L_dag, dtype=np.complex128)
        V = np.ascontiguousarray(self.V, dtype=np.complex128)
        U = np.ascontiguousarray(self.U, dtype=np.complex128)
        lam = np.ascontiguousarray(self.lam, dtype=np.float64)
        dt = float(self.dt)

        N = self.N_steps + 1
        psis = np.zeros((N_traj, N, self.dim), dtype=np.complex128)
        psis[:, 0, :] = np.asarray(psi0, dtype=np.complex128)

        if not parallel_traj:
            for traj in tqdm(range(N_traj), desc="Trajectories"):
                z = np.asarray(self.noise_generator.sample_process(), dtype=np.complex128)

                psi_prev = np.zeros((1, self.dim), dtype=np.complex128)
                psi_prev[0, :] = psis[traj, 0, :]

                for n in tqdm(
                    range(self.N_steps),
                    desc="Steps",
                    leave=False,
                ):
                    tab = self._numba_tables[n]
                    psi_curr = np.empty((tab["out_len"], self.dim), dtype=np.complex128)

                    if self.scheme == Order1Scheme.Forward_Euler:
                        _psi_step_forward_euler_kernel(
                            psi_prev,
                            psi_curr,
                            tab["same"],
                            tab["minus"],
                            tab["plus"],
                            tab["idx_vals"],
                            Hs,
                            L,
                            L_dag,
                            dt,
                            z[n],
                            V[:, n],
                            U[n, :],
                            lam,
                        )
                    elif self.scheme == Order2Scheme.Heun:
                        _psi_step_heun_kernel(
                            psi_prev,
                            psi_curr,
                            tab["same"],
                            tab["minus"],
                            tab["plus"],
                            tab["same_out"],
                            tab["minus_out"],
                            tab["plus_out"],
                            tab["idx_vals"],
                            Hs,
                            L,
                            L_dag,
                            dt,
                            z[n],
                            z[n + 1],
                            V[:, n],
                            V[:, n + 1],
                            U[n, :],
                            U[n + 1, :],
                            lam,
                        )
                    else:
                        raise NotImplementedError(f"numba backend does not implement scheme={self.scheme}")

                    psis[traj, n + 1, :] = psi_curr[0, :]
                    psi_prev = psi_curr

            return psis

        # Parallelize across trajectories (each trajectory remains sequential in time).
        # Note: we use the *serial* kernel to avoid nested parallelism.
        if max_workers is None:
            max_workers = os.cpu_count() or 1

        # Pre-sample noises in the main thread for deterministic ordering and
        # to avoid oversubscribing FFT/LAPACK threads across workers.
        # Heun needs z[n+1], so store N_steps+1 samples.
        Z = np.empty((N_traj, self.N_steps + 1), dtype=np.complex128)
        for traj in range(N_traj):
            z = np.asarray(self.noise_generator.sample_process(), dtype=np.complex128)
            Z[traj, :] = z[: self.N_steps + 1]

        # Trigger compilation once (avoids multiple threads compiling simultaneously).
        if self.N_steps > 0:
            tab0 = self._numba_tables[0]
            _psi_prev0 = np.zeros((1, self.dim), dtype=np.complex128)
            _psi_curr0 = np.empty((tab0["out_len"], self.dim), dtype=np.complex128)
            if self.scheme == Order1Scheme.Forward_Euler:
                _psi_step_forward_euler_kernel_serial(
                    _psi_prev0,
                    _psi_curr0,
                    tab0["same"],
                    tab0["minus"],
                    tab0["plus"],
                    tab0["idx_vals"],
                    Hs,
                    L,
                    L_dag,
                    dt,
                    Z[0, 0],
                    V[:, 0],
                    U[0, :],
                    lam,
                )
            elif self.scheme == Order2Scheme.Heun:
                _psi_step_heun_kernel_serial(
                    _psi_prev0,
                    _psi_curr0,
                    tab0["same"],
                    tab0["minus"],
                    tab0["plus"],
                    tab0["same_out"],
                    tab0["minus_out"],
                    tab0["plus_out"],
                    tab0["idx_vals"],
                    Hs,
                    L,
                    L_dag,
                    dt,
                    Z[0, 0],
                    Z[0, 1],
                    V[:, 0],
                    V[:, 1],
                    U[0, :],
                    U[1, :],
                    lam,
                )
            else:
                raise NotImplementedError(f"numba backend does not implement scheme={self.scheme}")

        psi0_arr = np.asarray(psi0, dtype=np.complex128)

        def _run_one(traj: int):
            traj_psis = np.empty((N, self.dim), dtype=np.complex128)
            traj_psis[0, :] = psi0_arr

            psi_prev = np.zeros((1, self.dim), dtype=np.complex128)
            psi_prev[0, :] = psi0_arr

            z_traj = Z[traj]
            for n in range(self.N_steps):
                tab = self._numba_tables[n]
                psi_curr = np.empty((tab["out_len"], self.dim), dtype=np.complex128)

                if self.scheme == Order1Scheme.Forward_Euler:
                    _psi_step_forward_euler_kernel_serial(
                        psi_prev,
                        psi_curr,
                        tab["same"],
                        tab["minus"],
                        tab["plus"],
                        tab["idx_vals"],
                        Hs,
                        L,
                        L_dag,
                        dt,
                        z_traj[n],
                        V[:, n],
                        U[n, :],
                        lam,
                    )
                elif self.scheme == Order2Scheme.Heun:
                    _psi_step_heun_kernel_serial(
                        psi_prev,
                        psi_curr,
                        tab["same"],
                        tab["minus"],
                        tab["plus"],
                        tab["same_out"],
                        tab["minus_out"],
                        tab["plus_out"],
                        tab["idx_vals"],
                        Hs,
                        L,
                        L_dag,
                        dt,
                        z_traj[n],
                        z_traj[n + 1],
                        V[:, n],
                        V[:, n + 1],
                        U[n, :],
                        U[n + 1, :],
                        lam,
                    )
                else:
                    raise NotImplementedError(f"numba backend does not implement scheme={self.scheme}")

                traj_psis[n + 1, :] = psi_curr[0, :]
                psi_prev = psi_curr

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
        self.psi_prev.clear()
        self.psi_curr.clear()
        self.initialize()

        if backend not in ("auto", "python", "numba"):
            raise ValueError("backend must be one of: 'auto', 'python', 'numba'")

        if backend in ("auto", "numba") and _numba_available():
            # Numba backend supports Forward Euler and Heun.
            if self.scheme in (Order1Scheme.Forward_Euler, Order2Scheme.Heun):
                return self.solve_numba(N_traj, psi0, parallel_traj=parallel_traj, max_workers=max_workers)
            if backend == "numba":
                raise NotImplementedError(f"backend='numba' does not implement scheme={self.scheme}")
            # backend == 'auto' => fall back to python implementation

        psis = np.zeros((N_traj, self.N_steps + 1, self.dim), dtype=complex)
        psis[:, 0, :] = psi0

        for traj in tqdm(range(N_traj), desc="Trajectories"):
            z = self.noise_generator.sample_process()
            self.psi_prev[(0,) * self.rank] = psi0.copy()
            for n in tqdm(
                range(1, self.N_steps + 1),
                desc="Steps",
                leave=False,
            ):
                if self.scheme == Order1Scheme.Forward_Euler:
                    self.psi_step_forward_euler(z, n - 1)
                elif self.scheme == Order2Scheme.Heun:
                    self.psi_step_heun(z, n - 1)
                else:
                    raise NotImplementedError(f"Python backend does not implement scheme={self.scheme}")
                psis[traj, n, :] = self.psi_curr[(0,) * self.rank]
                self.psi_prev, self.psi_curr = self.psi_curr, self.psi_prev
                self.psi_curr.clear()

        return psis