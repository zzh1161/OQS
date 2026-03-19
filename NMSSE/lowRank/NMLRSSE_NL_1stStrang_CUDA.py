import numpy as np
import time
import math
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


def generate_complex_gaussian():
	n1 = np.random.normal()
	n2 = np.random.normal()
	return (n1 + 1j * n2) / np.sqrt(2)


def expectation(psi: np.ndarray, op: np.ndarray) -> complex:
	den = np.vdot(psi, psi)
	if den == 0:
		return 0.0 + 0.0j
	return np.vdot(psi, op @ psi) / den


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
	def _gpu_nl_1st_step_kernel(
		out,
		psi_half,
		z_step,
		L_dag_avg_step,
		L,
		L_dag,
		V,
		U,
		lam,
		step,
		dt,
		idx_values,
		idx_minus,
		idx_plus,
	):
		bs, i = cuda.grid(2)
		n_batch = psi_half.shape[0]
		n_states = psi_half.shape[1]
		dim = psi_half.shape[2]
		rank = lam.shape[0]
		total_states = n_batch * n_states

		if bs < total_states and i < dim:
			b = bs // n_states
			s = bs - b * n_states

			z_n = z_step[b]
			avg_n = L_dag_avg_step[b]

			acc = psi_half[b, s, i]

			tmp = 0.0 + 0.0j
			for j in range(dim):
				tmp += L[i, j] * psi_half[b, s, j]
			acc += dt * z_n * tmp

			for k in range(rank):
				s_minus = idx_minus[s, k]
				if s_minus >= 0:
					tmp_m = 0.0 + 0.0j
					for j in range(dim):
						tmp_m += L[i, j] * psi_half[b, s_minus, j]
					acc += dt * idx_values[s, k] * V[k, step] * tmp_m

				s_plus = idx_plus[s, k]
				if s_plus >= 0:
					tmp_p = 0.0 + 0.0j
					for j in range(dim):
						tmp_p += L_dag[i, j] * psi_half[b, s_plus, j]

					uk = U[step, k]
					lk = lam[k]
					acc -= dt * lk * uk * tmp_p
					acc += dt * lk * uk * avg_n * psi_half[b, s_plus, i]

			out[b, s, i] = acc

	@cuda.jit
	def _gpu_norm_kernel(norm_out, states):
		b = cuda.grid(1)
		n_batch = states.shape[0]
		n_states = states.shape[1]
		dim = states.shape[2]
		if b < n_batch:
			acc = 0.0
			for s in range(n_states):
				for i in range(dim):
					v = states[b, s, i]
					acc += (v.real * v.real + v.imag * v.imag)
			norm_out[b] = math.sqrt(acc)

	@cuda.jit
	def _gpu_scale_kernel(states, norms):
		bs, i = cuda.grid(2)
		n_batch = states.shape[0]
		n_states = states.shape[1]
		dim = states.shape[2]
		total_states = n_batch * n_states
		if bs < total_states and i < dim:
			b = bs // n_states
			s = bs - b * n_states
			nrm = norms[b]
			if nrm > 0.0:
				states[b, s, i] = states[b, s, i] / nrm

	@cuda.jit
	def _gpu_expectation_zero_state_kernel(out_exp, states, op, zero_idx):
		b = cuda.grid(1)
		n_batch = states.shape[0]
		dim = states.shape[2]
		if b < n_batch:
			den = 0.0
			for i in range(dim):
				v = states[b, zero_idx, i]
				den += (v.real * v.real + v.imag * v.imag)

			if den == 0.0:
				out_exp[b] = 0.0 + 0.0j
				return

			num = 0.0 + 0.0j
			for i in range(dim):
				acc = 0.0 + 0.0j
				for j in range(dim):
					acc += op[i, j] * states[b, zero_idx, j]
				v = states[b, zero_idx, i]
				num += (v.real - 1j * v.imag) * acc
			out_exp[b] = num / den

	@cuda.jit
	def _gpu_update_S_kernel(S, L_curr, L_next, U, step):
		b, k = cuda.grid(2)
		n_batch = S.shape[0]
		rank = S.shape[1]
		if b < n_batch and k < rank:
			S[b, k] += 0.5 * U[step, k] * L_curr[b] + 0.5 * U[step + 1, k] * L_next[b]

	@cuda.jit
	def _gpu_update_z_tilde_kernel(z_next, Z, S, V, lam, step, dt):
		b = cuda.grid(1)
		n_batch = z_next.shape[0]
		rank = lam.shape[0]
		if b < n_batch:
			acc = 0.0 + 0.0j
			for k in range(rank):
				acc += lam[k] * V[k, step] * S[b, k]
			z_next[b] = Z[b, step] + dt * acc

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


class NMLRSSE_NL_1stStrang_CUDA:
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
		self.Hs_prop_dt = self.Hs_V @ np.diag(np.exp(-1j * self.Hs_E * self.dt)) @ self.Hs_V_dag

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
		idx_values = np.empty((n_states, self.rank), dtype=np.float64)
		idx_minus = -np.ones((n_states, self.rank), dtype=np.int32)
		idx_plus = -np.ones((n_states, self.rank), dtype=np.int32)

		for idx, pos in self.idx_map.items():
			idx_values[pos, :] = np.array(idx, dtype=np.float64)
			for k in range(self.rank):
				if idx[k] > 0:
					idx_m = idx[:k] + (idx[k] - 1,) + idx[k + 1 :]
					idx_minus[pos, k] = self.idx_map.get(idx_m, -1)
				idx_p = idx[:k] + (idx[k] + 1,) + idx[k + 1 :]
				idx_plus[pos, k] = self.idx_map.get(idx_p, -1)

		self._numba_idx_values = idx_values
		self._numba_idx_minus = idx_minus
		self._numba_idx_plus = idx_plus

	def _ensure_gpu_constants(self):
		if self._gpu_const_cache is not None:
			return self._gpu_const_cache

		if self._numba_idx_values is None:
			self._build_numba_indexing()

		P = np.ascontiguousarray(self.Hs_prop_dt, dtype=np.complex128)
		L = np.ascontiguousarray(self.L, dtype=np.complex128)
		L_dag = np.ascontiguousarray(self.L_dag, dtype=np.complex128)
		V = np.ascontiguousarray(self.V, dtype=np.complex128)
		U = np.ascontiguousarray(self.U, dtype=np.complex128)
		lam = np.ascontiguousarray(self.lam, dtype=np.float64)

		idx_values = np.ascontiguousarray(self._numba_idx_values, dtype=np.float64)
		idx_minus = np.ascontiguousarray(self._numba_idx_minus, dtype=np.int32)
		idx_plus = np.ascontiguousarray(self._numba_idx_plus, dtype=np.int32)

		self._gpu_const_cache = {
			"d_P": cuda.to_device(P),
			"d_L": cuda.to_device(L),
			"d_L_dag": cuda.to_device(L_dag),
			"d_V": cuda.to_device(V),
			"d_U": cuda.to_device(U),
			"d_lam": cuda.to_device(lam),
			"d_idx_values": cuda.to_device(idx_values),
			"d_idx_minus": cuda.to_device(idx_minus),
			"d_idx_plus": cuda.to_device(idx_plus),
		}
		return self._gpu_const_cache

	def _normalize_psi_dict(self):
		norm_sq = 0.0
		for idx in self.idx_set:
			vec = self.psi_curr[idx]
			norm_sq += np.real(np.vdot(vec, vec))
		if norm_sq > 0.0:
			scale = 1.0 / np.sqrt(norm_sq)
			for idx in self.idx_set:
				self.psi_curr[idx] *= scale

	def psi_step(self, z_tilde, L_dag_avg, n):
		if n < 0 or n >= self.N_steps:
			raise ValueError(f"n must satisfy 0 <= n < N_steps (got n={n})")

		def valid_psi(psi: Dict[Tuple[int, ...], np.ndarray], idx: Tuple[int, ...]) -> np.ndarray:
			return psi.get(idx, self._zero_state)

		for idx in self.idx_set:
			self.psi_curr[idx] = self.Hs_prop_dt @ valid_psi(self.psi_prev, idx)
		self.psi_curr, self.psi_prev = self.psi_prev, self.psi_curr

		for idx in self.idx_set:
			self.psi_curr[idx] = (self.Id + z_tilde[n] * self.L * self.dt) @ valid_psi(self.psi_prev, idx)
			for k in range(self.rank):
				idx_minus = idx[:k] + (idx[k] - 1,) + idx[k + 1 :]
				self.psi_curr[idx] += self.dt * idx[k] * self.V[k, n] * self.L @ valid_psi(self.psi_prev, idx_minus)
			for k in range(self.rank):
				idx_plus = idx[:k] + (idx[k] + 1,) + idx[k + 1 :]
				self.psi_curr[idx] -= self.dt * self.lam[k] * self.U[n, k] * self.L_dag @ valid_psi(self.psi_prev, idx_plus)
				self.psi_curr[idx] += (
					self.dt * self.lam[k] * self.U[n, k] * L_dag_avg[n] * valid_psi(self.psi_prev, idx_plus)
				)

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

	def solve_python(self, N_traj: int, psi0: np.ndarray):
		self.initialize_psi()
		self.psi_curr.clear()
		psis = np.zeros((N_traj, self.N_steps + 1, self.dim), dtype=complex)
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
				self.psi_step(z_tilde, L_dag_avg, n)
				self._normalize_psi_dict()
				psi_curr_0 = self.psi_curr[(0,) * self.rank]
				psis[traj, n + 1, :] = psi_curr_0

				L_dag_avg[n + 1] = expectation(psi_curr_0, self.L_dag)
				z_tilde[n + 1] = z[n + 1] + self.quadrature_trapozoidal(self.dt, L_dag_avg, n + 1)

				self.psi_prev, self.psi_curr = self.psi_curr, self.psi_prev
				self.psi_curr.clear()
				for idx in self.idx_set:
					self.psi_curr[idx] = self._zero_state.copy()

		return psis

	def solve_numba_gpu(
		self,
		N_traj: int,
		psi0: np.ndarray,
		threadsperblock: tuple[int, int] = (256, 2),
		traj_batch_size: int | None = None,
	):
		if not _numba_gpu_available():
			raise RuntimeError("numba cuda is not available")

		gpu_const = self._ensure_gpu_constants()

		n_states = len(self.idx_set)
		N = self.N_steps + 1
		dt = float(self.dt)

		psi0_arr = np.asarray(psi0, dtype=np.complex128)
		psis = np.zeros((N_traj, N, self.dim), dtype=np.complex128)
		psis[:, 0, :] = psi0_arr

		tx, ty = threadsperblock
		if tx <= 0 or ty <= 0 or tx * ty > 1024:
			raise ValueError("threadsperblock must be positive and tx*ty <= 1024")

		if self.noise_sample_Z is None:
			Z = np.empty((N_traj, N), dtype=np.complex128)
			for traj in range(N_traj):
				z = np.zeros(N, dtype=np.complex128)
				for k in range(self.rank):
					z += generate_complex_gaussian() * np.sqrt(self.lam[k]) * self.V[k, :N]
				Z[traj, :] = z
		else:
			Z = np.asarray(self.noise_sample_Z, dtype=np.complex128)
			if Z.shape[0] < N_traj:
				raise ValueError(f"noise_sample_Z first dimension must be at least N_traj={N_traj}, got {Z.shape[0]}")
			if Z.shape[1] < N:
				raise ValueError(f"noise_sample_Z second dimension must be at least N_steps+1={N}, got {Z.shape[1]}")
			Z = Z[:N_traj, :N]

		if traj_batch_size is None:
			traj_batch_size = self._suggest_batch_size_gpu(N_traj)
		if traj_batch_size <= 0:
			raise ValueError("traj_batch_size must be positive")

		max_nb = min(traj_batch_size, N_traj)
		d_Z = cuda.device_array((max_nb, N), dtype=np.complex128)
		d_prev = cuda.device_array((max_nb, n_states, self.dim), dtype=np.complex128)
		d_half = cuda.device_array((max_nb, n_states, self.dim), dtype=np.complex128)
		d_curr = cuda.device_array((max_nb, n_states, self.dim), dtype=np.complex128)
		d_hist = cuda.device_array((max_nb, N, self.dim), dtype=np.complex128)
		d_norm = cuda.device_array((max_nb,), dtype=np.float64)
		d_L_curr = cuda.device_array((max_nb,), dtype=np.complex128)
		d_L_next = cuda.device_array((max_nb,), dtype=np.complex128)
		d_z_curr = cuda.device_array((max_nb,), dtype=np.complex128)
		d_z_next = cuda.device_array((max_nb,), dtype=np.complex128)
		d_S = cuda.device_array((max_nb, self.rank), dtype=np.complex128)
		d_psi0 = cuda.to_device(psi0_arr)

		t1 = 32

		for batch_start in tqdm(range(0, N_traj, traj_batch_size), desc="Trajectory batches"):
			batch_end = min(batch_start + traj_batch_size, N_traj)
			nb = batch_end - batch_start

			Z_batch = np.ascontiguousarray(Z[batch_start:batch_end, :], dtype=np.complex128)
			d_Z[:nb, :].copy_to_device(Z_batch)

			d_Z_b = d_Z[:nb, :]
			d_prev_b = d_prev[:nb, :, :]
			d_half_b = d_half[:nb, :, :]
			d_curr_b = d_curr[:nb, :, :]
			d_hist_b = d_hist[:nb, :, :]
			d_norm_b = d_norm[:nb]
			d_L_curr_b = d_L_curr[:nb]
			d_L_next_b = d_L_next[:nb]
			d_z_curr_b = d_z_curr[:nb]
			d_z_next_b = d_z_next[:nb]
			d_S_b = d_S[:nb, :]

			blocks_states = ((nb * n_states + tx - 1) // tx, (self.dim + ty - 1) // ty)
			blocks_batch_2d = ((nb + tx - 1) // tx, (self.dim + ty - 1) // ty)
			blocks_batch_1d = (nb + t1 - 1) // t1
			blocks_rank_2d = ((nb + tx - 1) // tx, (self.rank + ty - 1) // ty)

			_gpu_reset_states_kernel[blocks_states, threadsperblock](d_prev_b, self._zero_idx, d_psi0)
			_gpu_store_zero_state_kernel[blocks_batch_2d, threadsperblock](d_hist_b, d_prev_b, self._zero_idx, 0)

			L0 = expectation(psi0_arr, self.L_dag)
			d_L_curr_b.copy_to_device(np.full(nb, L0, dtype=np.complex128))
			d_z_curr_b.copy_to_device(np.ascontiguousarray(Z_batch[:, 0], dtype=np.complex128))
			d_S_b.copy_to_device(np.zeros((nb, self.rank), dtype=np.complex128))

			for n in range(self.N_steps):
				_gpu_matmul_states_batched[blocks_states, threadsperblock](gpu_const["d_P"], d_prev_b, d_half_b)

				_gpu_nl_1st_step_kernel[blocks_states, threadsperblock](
					d_curr_b,
					d_half_b,
					d_z_curr_b,
					d_L_curr_b,
					gpu_const["d_L"],
					gpu_const["d_L_dag"],
					gpu_const["d_V"],
					gpu_const["d_U"],
					gpu_const["d_lam"],
					n,
					dt,
					gpu_const["d_idx_values"],
					gpu_const["d_idx_minus"],
					gpu_const["d_idx_plus"],
				)

				_gpu_norm_kernel[blocks_batch_1d, t1](d_norm_b, d_curr_b)
				_gpu_scale_kernel[blocks_states, threadsperblock](d_curr_b, d_norm_b)

				_gpu_store_zero_state_kernel[blocks_batch_2d, threadsperblock](d_hist_b, d_curr_b, self._zero_idx, n + 1)
				_gpu_expectation_zero_state_kernel[blocks_batch_1d, t1](d_L_next_b, d_curr_b, gpu_const["d_L_dag"], self._zero_idx)
				_gpu_update_S_kernel[blocks_rank_2d, threadsperblock](d_S_b, d_L_curr_b, d_L_next_b, gpu_const["d_U"], n)
				_gpu_update_z_tilde_kernel[blocks_batch_1d, t1](
					d_z_next_b,
					d_Z_b,
					d_S_b,
					gpu_const["d_V"],
					gpu_const["d_lam"],
					n + 1,
					dt,
				)

				d_L_curr_b, d_L_next_b = d_L_next_b, d_L_curr_b
				d_z_curr_b, d_z_next_b = d_z_next_b, d_z_curr_b
				d_prev_b, d_curr_b = d_curr_b, d_prev_b

			psis[batch_start:batch_end, :, :] = d_hist_b.copy_to_host()

		cuda.synchronize()
		return psis

	def _suggest_batch_size_gpu(self, N_traj: int) -> int:
		if not _numba_gpu_available():
			return min(64, N_traj)

		try:
			free_bytes, _ = cuda.current_context().get_memory_info()
		except Exception:
			return min(256, N_traj)

		n_states = len(self.idx_set)
		bytes_complex = 16
		bytes_float = 8
		N = self.N_steps + 1

		# Per-trajectory device footprint used by batch buffers.
		bytes_per_traj = 0
		bytes_per_traj += 3 * n_states * self.dim * bytes_complex  # prev, half, curr
		bytes_per_traj += N * bytes_complex  # Z
		bytes_per_traj += N * self.dim * bytes_complex  # history
		bytes_per_traj += 4 * bytes_complex  # L_curr, L_next, z_curr, z_next
		bytes_per_traj += self.rank * bytes_complex  # S
		bytes_per_traj += bytes_float  # norm

		# Keep a conservative safety margin for driver/runtime allocations.
		safe_budget = int(0.70 * free_bytes)
		if bytes_per_traj <= 0:
			return min(256, N_traj)

		est = max(1, safe_budget // bytes_per_traj)
		# Clamp to avoid overly large launches while still favoring big batches on A10.
		est = min(est, 4096, N_traj)
		return int(est)

	def solve(
		self,
		N_traj: int,
		psi0: np.ndarray,
		backend: str = "auto",
		threadsperblock: tuple[int, int] = (256, 2),
		traj_batch_size: int | None = None,
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
