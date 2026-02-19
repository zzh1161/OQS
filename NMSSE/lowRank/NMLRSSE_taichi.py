import taichi as ti

@ti.data_oriented
class NMLRSSE_taichi:
    def __init__(
        self,
        Hs,
        L,
        bath_corr,
        tmax,
        N_steps,
        rank,
        noise_generator,
    ):
        self.dim = Hs.shape[0]
        
        self.prev_psi = ti.Vector.field(self.dim, dtype=ti.complex128)