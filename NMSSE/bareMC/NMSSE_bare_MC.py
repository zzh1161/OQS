import numpy as np
import math
from itertools import combinations
from ..utils.noise_generator import ColoredNoiseGenerator_FourierFiltering

def generate_pairings(indices):
    """
    Generate all possible pairings of the given indices.
    
    Parameters
    ----------
    indices : list
        List of indices to be paired
    """
    if len(indices) % 2 != 0:
        raise ValueError("Number of indices must be even to form pairings.")
    if len(indices) == 0:
        return [[]]
    
    first = indices[0]
    pairings = []
    for i in range(1, len(indices)):
        second = indices[i]
        remaining = indices[1:i] + indices[i+1:]
        for sub_pairing in generate_pairings(remaining):
            pairings.append([(first, second)] + sub_pairing)
    
    return pairings

def get_combinations(numbers, m):
    return list(combinations(numbers, m))

class NMSSE_Linear_Bare_MC:
    def __init__(self, Hs, L, tmax, bath_corr, N_trunc, N_grid):
        """
        Parameters
        ----------
        Hs: np.ndarray
            System Hamiltonian. Must be a Hermitian matrix
        L : np.ndarray
            System-bath coupling operator. Must be the same shape as the system Hamiltonian
        tmax : float
            Maximum time for the simulation statring at t=0
        bath_corr : callable
            Bath correlation function
        N_trunc : int
            Truncation order for diagrammatic expansion
        N_grid : int
            Number of grid points for the stochastic sampling
        """
        self.dim = Hs.shape[0] # dimension of the physical hilbert space
        self.Hs = Hs
        self.L = L
        E, V = np.linalg.eigh(Hs)
        self.propagator = lambda t: V @ np.diag(np.exp(-1j * E * t)) @ V.conj().T
        self.Lint = lambda t: self.propagator(-t) @ L @ self.propagator(t)

        self.tmax = tmax
        self.bath_corr = bath_corr
        self.N_trunc = N_trunc
        self.N_grid = N_grid
        self.t_grid = np.linspace(0, tmax, N_grid+1)

        self.noise_generator = ColoredNoiseGenerator_FourierFiltering(
            bath_corr, tmax, N_grid
        )

    def noise_interp(self, z, t):
        """
        Interpolates the noise process at time t given the sampled noise z on the grid
        
        Parameters
        ----------
        z : np.ndarray
            Sampled noise process on the grid of shape (N_grid,)
        t : float
            Time at which to interpolate the noise process
        """
        assert(t >= 0 and t <= self.tmax)
        dt = self.t_grid[1] - self.t_grid[0]
        index = int(t / dt)
        t0 = self.t_grid[index]
        z0 = z[index]
        z1 = z[index + 1] if index + 1 < len(z) else z[index]
        return z0 + (z1 - z0) * (t - t0) / dt
    
    def compute_realization(
            self,
            N_traj,
            N_MC_samples = 500,
            psi0 = np.array([1.0, 0.0], dtype=complex),
            data_path = None                
        ):
        """
        Parameters
        ----------
        N_traj : int
            Number of trajectories
        psi0 : np.ndarray
            Initial state vector of the system
        data_path : str
            Path to save the trajectory data. If None, data is not saved.

        Returns
        -------
        np.ndarray
            array of shape (N_traj, dim) of dtype complex containing the physical state \Psi_t^{(k=0)}
            for discrete times t. Is only returned if data_path == None.
        """
        psis = np.empty((N_traj, self.dim), dtype=complex)
        for traj in range(N_traj):
            psi = psi0.copy()
            z = self.noise_generator.sample_process()
            for n in range(1, self.N_trunc+1):
                indices = list(range(n))
                for m in range(0, n+1, 2):
                    combs = get_combinations(indices, m)
                    for comb in combs:
                        pairings = generate_pairings(list(comb))
                        for pairing in pairings:
                            psi += self.MC_int(N_MC_samples, psi0, z, indices, pairing)
                            
            psis[traj] = self.propagator(self.tmax) @ psi
        
        return psis

    def MC_int(self, N_MC_samples, psi0, z, indices, pairing):
        n = len(indices)
        conjugate_flag = np.zeros(n, dtype=bool)
        for pair in pairing:
            conjugate_flag[pair[1]] = True
        arc_indices = [i for pair in pairing for i in pair]
        cross_indices = [i for i in indices if i not in arc_indices]
        cpsi = np.zeros((self.dim,), dtype=complex)
        for sample in range(N_MC_samples):
            psi_temp = psi0.copy()
            tspan = np.sort(np.random.uniform(0, self.tmax, n))
            constant = (-1) ** (len(arc_indices) // 2)
            for i in cross_indices:
                constant *= self.noise_interp(z, tspan[i])
            for pair in pairing:
                constant *= self.bath_corr(tspan[pair[1]] - tspan[pair[0]])
            
            for i in indices:
                if conjugate_flag[i]:
                    psi_temp = self.Lint(tspan[i]).conj().T @ psi_temp
                else:
                    psi_temp = self.Lint(tspan[i]) @ psi_temp
            psi_temp *= constant
            
            cpsi += psi_temp
            
        cpsi *= (self.tmax ** n) / (N_MC_samples * math.factorial(n))
        return cpsi