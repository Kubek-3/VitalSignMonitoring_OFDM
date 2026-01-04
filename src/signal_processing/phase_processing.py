import numpy as np
from src.config import C


def compute_phase(d, f):
    lam = C / f
    return (2 * np.pi / lam) * d
