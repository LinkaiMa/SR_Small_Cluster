"""Synthetic two-cluster matrices used in the paper tests."""

import numpy as np


def make_synthetic_matrix(n: int, d: int, k: int, sigma_top: float,
                          sigma_tail: float, seed: int = 1) -> np.ndarray:
    """Random n x d matrix with k singular values sigma_top and d-k sigma_tail."""
    matrix_seed = np.random.SeedSequence(seed).spawn(2)[0]
    rng = np.random.default_rng(matrix_seed)
    U, _ = np.linalg.qr(rng.standard_normal((n, d)))
    V, _ = np.linalg.qr(rng.standard_normal((d, d)))
    sv = np.array([sigma_top] * k + [sigma_tail] * (d - k))
    return U @ np.diag(sv) @ V.T
