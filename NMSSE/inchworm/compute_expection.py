import numpy as np

def compute_expectation_value(psis, op):
    """
    Computes the expectation value of a given operator for a stochastic density matrix
    computed from an ensemble of states sampled using the ***linear*** HOPS
    
    \rho = \mathbb{E}[ |\Psi_t(z)><\Psi_t(z)| ]
    
    Parameters
    ----------
    psis : np.ndarray
        array of shape (N_samples, N_steps, dim) and dtype complex. All HOPS or HOMPS
        have a function compute_realizations() that outputs a np.ndarray in this shape
    op : np.ndarray
        the operator we want to compute the expectation value of. Has to be of
        shape (dim, dim).
    
    Returns
    -------
    np.ndarray
        array of shape (N_steps,) containing the expectation value of the given operator
        at all discrete times, averaged over the samples
    """
    N_steps = psis.shape[1]
    result = np.empty(N_steps)
    for i in range(N_steps):
        num = np.dot(np.conj((op@psis[:, i, :].T)).flatten(), psis[:, i, :].T.flatten())
        denom = np.dot(np.conj(psis[:, i, :]).flatten(), psis[:, i, :].flatten())
        result[i] = np.real_if_close(num/denom)
    return result