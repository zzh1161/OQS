import numpy as np
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple
from tqdm import tqdm

try:
    from numba import njit, prange  # type: ignore
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    njit = None
    prange = range
    NUMBA_AVAILABLE = False


def generate_white_noise(N):
    # u = np.random.rand(N)
    # u = np.clip(u, np.finfo(float).tiny, 1.0)
    # return np.sqrt(-np.log(u)) * np.exp(2.j*np.pi*np.random.rand(N))
    rng = np.random.default_rng()
    return (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2)

class noise_generator:
    def __init__(self, alpha, t_stop, N_steps):
        self.alpha = alpha
        self.t_stop = t_stop
        self.N_steps = N_steps
        ts = np.linspace(0, self.t_stop, self.N_steps+1)
        correlations = np.empty((self.N_steps+1, self.N_steps+1), dtype=complex)

        for i in range(self.N_steps+1):
            for j in range(self.N_steps+1):
                correlations[i,j] = self.alpha(ts[j], ts[i])

        # Hermitianize the correlation matrix
        correlations = 0.5 * (correlations + correlations.conj().T)
        # Add jitter to the diagonal for numerical stability
        jitter = 1e-10 * np.eye(self.N_steps+1)
        correlations += jitter

        self.L = np.linalg.cholesky(correlations)
        # err = np.linalg.norm(self.L @ self.L.conj().T - correlations)
        # print(f"Cholesky decomposition error: {err:e}")

    def sample_process(self):
        eta = generate_white_noise(self.N_steps+1)
        z = self.L @ eta
        if not np.all(np.isfinite(z)):
            raise FloatingPointError("Non-finite values encountered in generated noise process")
        return z


# Numba-optimized functions for parallel execution
def _numba_available() -> bool:
    return NUMBA_AVAILABLE


if NUMBA_AVAILABLE:
    @njit(cache=True, nogil=True)
    def _matvec_into(A, x, out):
        """Compute A @ x and store in out"""
        dim = A.shape[0]
        for i in range(dim):
            acc = 0.0 + 0.0j
            for j in range(dim):
                acc += A[i, j] * x[j]
            out[i] = acc

    @njit(cache=True, nogil=True)
    def _matvec_add_scaled(A, x, y, scale):
        """Compute y += scale * (A @ x)"""
        dim = A.shape[0]
        for i in range(dim):
            acc = 0.0 + 0.0j
            for j in range(dim):
                acc += A[i, j] * x[j]
            y[i] += scale * acc

    @njit(cache=True, nogil=True)
    def _ode_rhs_numba(
        out,
        z_step,
        L,
        L_dag,
        V_step,
        V_conj_step,
        lamda,
        psi_in,
        max_layer,
        dim,
        zero_state,
        idx,
    ):
        """Compute ODE RHS: d/dt psi_idx"""
        # out = z * L @ psi[idx]
        if idx >= 0 and idx < max_layer:
            _matvec_into(L, psi_in[idx], out)
            for i in range(dim):
                out[i] = z_step * out[i]
        else:
            for i in range(dim):
                out[i] = 0.0 + 0.0j

        # Add: idx * V[step] * L @ psi[idx-1]
        if idx > 0:
            _matvec_add_scaled(L, psi_in[idx - 1], out, float(idx) * V_step)

        # Subtract: lamda * V_conj[step] * L_dag @ psi[idx+1]
        if idx + 1 < max_layer:
            _matvec_add_scaled(L_dag, psi_in[idx + 1], out, -lamda * V_conj_step)

    @njit(cache=True, nogil=True)
    def _heun_step_kernel_serial(
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
        V_n,
        V_conj_n,
        V_n1,
        V_conj_n1,
        lamda,
        dt,
        max_layer,
        dim,
    ):
        """Serial Heun's method step: P(0.5*dt) @ ODE(dt) @ P(0.5*dt)"""
        zero_state = np.zeros((max_layer, dim), dtype=np.complex128)

        # Forward half-step: apply exp_step P(0.5*dt)
        for idx in range(max_layer):
            _matvec_into(P_half, psi_prev[idx], psi_half[idx])

        # ODE step
        for idx in range(max_layer):
            _ode_rhs_numba(
                k1[idx],
                z_n,
                L,
                L_dag,
                V_n,
                V_conj_n,
                lamda,
                psi_half,
                max_layer,
                dim,
                zero_state,
                idx,
            )
            for i in range(dim):
                psi_pred[idx, i] = psi_half[idx, i] + dt * k1[idx, i]

        for idx in range(max_layer):
            _ode_rhs_numba(
                psi_curr[idx],
                z_n1,
                L,
                L_dag,
                V_n1,
                V_conj_n1,
                lamda,
                psi_pred,
                max_layer,
                dim,
                zero_state,
                idx,
            )
            for i in range(dim):
                psi_curr[idx, i] = psi_half[idx, i] + 0.5 * dt * (k1[idx, i] + psi_curr[idx, i])

        # Backward half-step: apply exp_step P(0.5*dt)
        for idx in range(max_layer):
            _matvec_into(P_half, psi_curr[idx], psi_half[idx])

        for idx in range(max_layer):
            for i in range(dim):
                psi_curr[idx, i] = psi_half[idx, i]

    @njit(parallel=True, cache=True, nogil=True)
    def _heun_step_kernel(
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
        V_n,
        V_conj_n,
        V_n1,
        V_conj_n1,
        lamda,
        dt,
        max_layer,
        dim,
    ):
        """Parallel Heun's method step"""
        zero_state = np.zeros((max_layer, dim), dtype=np.complex128)

        # Forward half-step
        for idx in prange(max_layer):
            _matvec_into(P_half, psi_prev[idx], psi_half[idx])

        # ODE first stage
        for idx in prange(max_layer):
            _ode_rhs_numba(
                k1[idx],
                z_n,
                L,
                L_dag,
                V_n,
                V_conj_n,
                lamda,
                psi_half,
                max_layer,
                dim,
                zero_state,
                idx,
            )
            for i in range(dim):
                psi_pred[idx, i] = psi_half[idx, i] + dt * k1[idx, i]

        # ODE second stage
        for idx in prange(max_layer):
            _ode_rhs_numba(
                psi_curr[idx],
                z_n1,
                L,
                L_dag,
                V_n1,
                V_conj_n1,
                lamda,
                psi_pred,
                max_layer,
                dim,
                zero_state,
                idx,
            )
            for i in range(dim):
                psi_curr[idx, i] = psi_half[idx, i] + 0.5 * dt * (k1[idx, i] + psi_curr[idx, i])

        # Backward half-step
        for idx in prange(max_layer):
            _matvec_into(P_half, psi_curr[idx], psi_half[idx])

        for idx in prange(max_layer):
            for i in range(dim):
                psi_curr[idx, i] = psi_half[idx, i]


class NMLRSSE_Test_SVD:
    def __init__(
        self,
        Hs,
        L,
        V,
        lamda,
        tmax,
        N_steps,
        noise_generator=noise_generator,
        max_layer: int | None = None,
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

        self.V_func = V
        self.lamda = lamda
        self.bath_corr = lambda t,s: self.lamda * np.conj(self.V_func(t)) * self.V_func(s)

        self.tmax = tmax
        self.N_steps = N_steps
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
        self.V = np.array([self.V_func(t) for t in self.t_grid], dtype=complex)
        self.V_conj = np.conj(self.V)
        self.P_half = self.Hs_propagator(0.5 * self.dt)

        self.noise_generator = noise_generator(self.bath_corr, tmax, N_steps)

        self.psi_prev: np.ndarray = np.zeros((self.max_layer, self.dim), dtype=complex)
        self.psi_curr: np.ndarray = np.zeros((self.max_layer, self.dim), dtype=complex)


    def valid_psi(self, psi: np.ndarray, idx: int):
        if idx >= 0 and idx < self.max_layer:
            return psi[idx]
        else:
            return self._zero_state
        
    def ode_rhs(
        self,
        z: np.ndarray,
        step: int,
        psi: np.ndarray,
        idx: int,
    ):
        base = self.valid_psi(psi, idx)
        acc = z[step] * self.L @ base
        acc += idx * self.V[step] * self.L @ self.valid_psi(psi, idx-1)
        acc -= self.lamda * self.V_conj[step] * self.L_dag @ self.valid_psi(psi, idx+1)
        return acc
    
    def ode_step(
        self,
        psi_in: np.ndarray,
        psi_out: np.ndarray,
        z: np.ndarray,
        step: int,
        dt: float,
    ):
        k1 = np.zeros((self.max_layer, self.dim), dtype=complex)
        psi_pred = np.zeros((self.max_layer, self.dim), dtype=complex)

        for idx in range(self.max_layer):
            k1[idx] = self.ode_rhs(z, step, psi_in, idx)
            psi_pred[idx] = psi_in[idx] + dt * k1[idx]
        for idx in range(self.max_layer):
            k2_idx = self.ode_rhs(z, step+1, psi_pred, idx)
            psi_out[idx] = psi_in[idx] + 0.5 * dt * (k1[idx] + k2_idx)

    def exp_step(
        self,
        psi: np.ndarray,
        dt: float,
    ):
        H_prop = self.Hs_propagator(dt)
        for idx in range(self.max_layer):
            psi[idx] = H_prop @ psi[idx]

    def solve_python(
        self,
        N_traj: int,
        psi0: np.ndarray,
    ):
        self.psi_prev.fill(0)
        self.psi_curr.fill(0)
        psis = np.zeros((N_traj, self.N_steps + 1, self.dim), dtype=complex)
        psis[:, 0, :] = psi0
        
        for traj in tqdm(range(N_traj), desc="Trajectories"):
            if self.noise_sample_Z is not None:
                z = self.noise_sample_Z[traj]
            else:
                z = self.noise_generator.sample_process()
            self.psi_prev[0] = psi0
            for step in range(self.N_steps):
                self.exp_step(self.psi_prev, 0.5 * self.dt)
                self.ode_step(self.psi_prev, self.psi_curr, z, step, self.dt)
                self.exp_step(self.psi_curr, 0.5 * self.dt)
                psis[traj, step+1, :] = self.psi_curr[0]
                # swap buffers
                self.psi_prev, self.psi_curr = self.psi_curr, self.psi_prev
        return psis

    def solve_numba(
        self,
        N_traj: int,
        psi0: np.ndarray,
        parallel_traj: bool = False,
        max_workers: int | None = None,
    ):
        """
        Solve using Numba JIT compilation.
        
        Args:
            N_traj: Number of trajectories
            psi0: Initial state
            parallel_traj: Whether to parallelize over trajectories
            max_workers: Maximum number of threads for trajectory parallelization
            
        Returns:
            Array of shape (N_traj, N_steps+1, dim) with trajectories
        """
        if not _numba_available():
            raise RuntimeError("Numba is not available. Install numba to use this method.")

        # Prepare data
        P_half = np.ascontiguousarray(self.P_half, dtype=np.complex128)
        L = np.ascontiguousarray(self.L, dtype=np.complex128)
        L_dag = np.ascontiguousarray(self.L_dag, dtype=np.complex128)
        V = np.ascontiguousarray(self.V, dtype=np.complex128)
        V_conj = np.ascontiguousarray(self.V_conj, dtype=np.complex128)
        lamda = complex(self.lamda)
        dt = float(self.dt)

        N = self.N_steps + 1
        psis = np.zeros((N_traj, N, self.dim), dtype=np.complex128)
        psis[:, 0, :] = np.asarray(psi0, dtype=np.complex128)

        # Non-parallel trajectory execution
        if not parallel_traj:
            psi_prev = np.zeros((self.max_layer, self.dim), dtype=np.complex128)
            psi_curr = np.zeros((self.max_layer, self.dim), dtype=np.complex128)
            psi_half = np.zeros((self.max_layer, self.dim), dtype=np.complex128)
            k1 = np.zeros((self.max_layer, self.dim), dtype=np.complex128)
            psi_pred = np.zeros((self.max_layer, self.dim), dtype=np.complex128)

            for traj in tqdm(range(N_traj), desc="Trajectories"):
                if self.noise_sample_Z is None:
                    z = np.asarray(self.noise_generator.sample_process(), dtype=np.complex128)
                else:
                    z = np.asarray(self.noise_sample_Z[traj, :], dtype=np.complex128)

                if z.shape[0] < N:
                    raise ValueError(f"noise length must be at least N_steps+1={N}, got {z.shape[0]}")

                psi_prev.fill(0)
                psi_prev[0, :] = psis[traj, 0, :]

                for n in range(self.N_steps):
                    _heun_step_kernel_serial(
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
                        V[n],
                        V_conj[n],
                        V[n + 1],
                        V_conj[n + 1],
                        lamda,
                        dt,
                        self.max_layer,
                        self.dim,
                    )

                    psis[traj, n + 1, :] = psi_curr[0, :]
                    psi_prev, psi_curr = psi_curr, psi_prev

            return psis

        # Parallel trajectory execution with ThreadPoolExecutor
        if max_workers is None:
            max_workers = os.cpu_count() or 1

        # Generate or prepare noise samples
        Z = self.noise_sample_Z
        if Z is None:
            Z = np.empty((N_traj, N), dtype=np.complex128)
            for traj in tqdm(range(N_traj), desc="Generating noise samples"):
                z = np.asarray(self.noise_generator.sample_process(), dtype=np.complex128)
                if z.shape[0] < N:
                    raise ValueError(f"noise length must be at least N_steps+1={N}, got {z.shape[0]}")
                Z[traj, :] = z[:N]
        else:
            Z = np.asarray(Z, dtype=np.complex128)
            if Z.shape[1] < N:
                raise ValueError(f"noise_sample_Z second dimension must be at least N_steps+1={N}, got {Z.shape[1]}")

        # Warm up numba with one trajectory
        if self.N_steps > 0:
            _psi_prev0 = np.zeros((self.max_layer, self.dim), dtype=np.complex128)
            _psi_curr0 = np.zeros((self.max_layer, self.dim), dtype=np.complex128)
            _psi_half0 = np.zeros((self.max_layer, self.dim), dtype=np.complex128)
            _k10 = np.zeros((self.max_layer, self.dim), dtype=np.complex128)
            _psi_pred0 = np.zeros((self.max_layer, self.dim), dtype=np.complex128)

            _heun_step_kernel_serial(
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
                V[0],
                V_conj[0],
                V[1],
                V_conj[1],
                lamda,
                dt,
                self.max_layer,
                self.dim,
            )

        psi0_arr = np.asarray(psi0, dtype=np.complex128)

        def _run_one_trajectory(traj: int):
            """Run a single trajectory"""
            traj_psis = np.empty((N, self.dim), dtype=np.complex128)
            traj_psis[0, :] = psi0_arr

            psi_prev = np.zeros((self.max_layer, self.dim), dtype=np.complex128)
            psi_curr = np.zeros((self.max_layer, self.dim), dtype=np.complex128)
            psi_half = np.zeros((self.max_layer, self.dim), dtype=np.complex128)
            k1 = np.zeros((self.max_layer, self.dim), dtype=np.complex128)
            psi_pred = np.zeros((self.max_layer, self.dim), dtype=np.complex128)
            psi_prev[0, :] = psi0_arr

            z_traj = Z[traj]
            for n in range(self.N_steps):
                _heun_step_kernel_serial(
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
                    V[n],
                    V_conj[n],
                    V[n + 1],
                    V_conj[n + 1],
                    lamda,
                    dt,
                    self.max_layer,
                    self.dim,
                )

                traj_psis[n + 1, :] = psi_curr[0, :]
                psi_prev, psi_curr = psi_curr, psi_prev

            return traj, traj_psis

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_run_one_trajectory, traj) for traj in range(N_traj)]
            for fut in tqdm(as_completed(futures), total=N_traj, desc="Trajectories"):
                traj, traj_psis = fut.result()
                psis[traj, :, :] = traj_psis

        return psis

    def solve(
        self,
        N_traj: int,
        psi0: np.ndarray,
        backend: str = "auto",
        parallel_traj: bool = False,
        max_workers: int | None = None,
    ):
        """
        Unified solver interface.
        
        Args:
            N_traj: Number of trajectories
            psi0: Initial state
            backend: "python", "numba", or "auto" (auto tries numba, falls back to python)
            parallel_traj: Whether to parallelize over trajectories (numba only)
            max_workers: Maximum threads for trajectory parallelization (numba only)
            
        Returns:
            Array of shape (N_traj, N_steps+1, dim) with trajectories
        """
        if backend not in ("auto", "python", "numba"):
            raise ValueError("backend must be one of: 'auto', 'python', 'numba'")

        if backend in ("auto", "numba") and _numba_available():
            return self.solve_numba(N_traj, psi0, parallel_traj=parallel_traj, max_workers=max_workers)
        else:
            return self.solve_python(N_traj, psi0)