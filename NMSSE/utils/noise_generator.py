import numpy as np

def generate_white_noise(N):
    # u = np.random.rand(N)
    # u = np.clip(u, np.finfo(float).tiny, 1.0)
    # return np.sqrt(-np.log(u)) * np.exp(2.j*np.pi*np.random.rand(N))
    rng = np.random.default_rng()
    return (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2)

class ColoredNoiseGenerator_FourierFiltering:
    '''
    Generate a noise of length N_steps + 1
    '''
    def __init__(self, alpha, t_stop, N_steps):
        self.alpha = alpha
        self.t_stop = t_stop
        self.N_steps = N_steps
        ts = np.linspace(0, self.t_stop, self.N_steps+1)
        correlations = np.empty(2*self.N_steps+1, dtype=complex)
        correlations[:self.N_steps+1] = self.alpha(ts)
        correlations[self.N_steps+1:] = np.conj(correlations[self.N_steps:0:-1])
        # compute spectral density by IFT and normalizing properly
        self.sqrtJ = np.sqrt(np.fft.fft(correlations))
    
    def sample_process(self):
        eta = generate_white_noise(2*self.N_steps+1)
        z = np.fft.ifft(np.fft.fft(eta) * self.sqrtJ)[0:self.N_steps+1]
        if not np.all(np.isfinite(z)):
            raise FloatingPointError("Non-finite values encountered in generated noise process")
        return z
    
class ColoredNoiseGenerator_Cholesky:
    '''
    Generate a noise of length N_steps + 1
    E(z(t)z*(s)) = alpha*(t-s) = alpha(s-t)
    '''
    def __init__(self, alpha, t_stop, N_steps):
        self.alpha = alpha
        self.t_stop = t_stop
        self.N_steps = N_steps
        ts = np.linspace(0, self.t_stop, self.N_steps+1)
        correlations = np.empty((self.N_steps+1, self.N_steps+1), dtype=complex)

        for i in range(self.N_steps+1):
            for j in range(self.N_steps+1):
                correlations[i,j] = self.alpha(ts[j]-ts[i])

        # Hermitianize the correlation matrix
        correlations = 0.5 * (correlations + correlations.conj().T)
        # Add jitter to the diagonal for numerical stability
        jitter = 1e-14 * np.eye(self.N_steps+1)
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