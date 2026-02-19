import numpy as np

def compute_exp_val(psis, op):
    
    num = np.dot(np.conj(op@psis.T).flatten(), psis.T.flatten())
    denom = np.dot(np.conj(psis).flatten(), psis.flatten())
    return np.real_if_close(num/denom)