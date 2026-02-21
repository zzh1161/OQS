import numpy as np
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple
from tqdm import tqdm

try:
    from numba import njit, prange
except Exception:
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

class NMSSE_Direct_Diagram:
    def __init__(
        self,
        Hs,
        L,
        bath_corr,
        tmax,
        N_steps,
        noise_generator=ColoredNoiseGenerator_Cholesky
    ):
        self.dim = Hs.shape[0]
        self.Id = np.eye(self.dim)
        self.Hs = Hs
        self.L = L
        self.L_dag = L.conj().T
        self.bath_corr = bath_corr
        self.tmax = tmax
        self.N_steps = N_steps
        self.dt = tmax / N_steps
        self.t_grid = np.linspace(0, tmax, N_steps+1)

        self.noise_generator = noise_generator(bath_corr, tmax, N_steps)

        