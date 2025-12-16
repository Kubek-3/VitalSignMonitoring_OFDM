import numpy as np
from src.config import c


def compute_phase(d, f):
    lam = c / f
    return (2 * np.pi / lam) * d
