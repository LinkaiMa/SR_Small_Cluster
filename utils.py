"""Stochastic-rounding quantizer and the theorem quantities.

The experiment only needs: quantize a matrix, compute its singular values,
and compare the tail-energy increase T with the leading term
sum_{j>k} nu_j^downarrow. The nu_j are computed while preparing the quantizer.
"""

import numpy as np


def quantization_resolution(B: int, R: float) -> float:
    """Resolution rho = 2R / (2^B - 1) of a B-bit quantizer on [-R, R]."""
    return 2 * R / ((2 ** B) - 1)


def prepare_quantizer(A: np.ndarray, B: int, R: float):
    """Precompute the deterministic part of the stochastic quantizer.

    Returns (lower, prob, nu): per-entry lower grid endpoint, per-entry
    probability of rounding up, and the column variance parameters
    nu_j = rho^2 * sum_i p_ij (1 - p_ij). Shared by all Monte Carlo trials
    at a given bit width.

    Endpoints are evaluated in A's stored dtype so the sampled quantizer
    stays unbiased after dtype rounding.
    """
    rho = quantization_resolution(B, R)
    stored_rho = np.asarray(rho, dtype=A.dtype)
    clipped = np.clip(A, -R, R)
    lower = np.floor((clipped + R) / rho) * rho - R
    lower[lower > clipped] -= stored_rho
    lower[lower + stored_rho < clipped] += stored_rho
    upper = lower + stored_rho
    prob = np.clip((clipped - lower) / (upper - lower), 0, 1)
    nu = float(stored_rho) ** 2 * np.sum(
        prob * (1 - prob), axis=0, dtype=np.float64
    )
    return lower, prob, nu


def sample_quantizer(lower: np.ndarray, prob: np.ndarray, rho: float,
                     rng: np.random.Generator) -> np.ndarray:
    """Draw one stochastically rounded matrix from a prepared quantizer."""
    stored_rho = np.asarray(rho, dtype=lower.dtype)
    return lower + stored_rho * (
        rng.random(prob.shape, dtype=np.float32) < prob
    )


def _gram_eigenvalues(A: np.ndarray) -> np.ndarray:
    """Ascending eigenvalues of A^T A, accumulated in float64."""
    A64 = np.asarray(A, dtype=np.float64)
    return np.linalg.eigvalsh(A64.T @ A64)


def tail_energy(A: np.ndarray, k: int) -> float:
    """Tail energy sum_{i>k} sigma_i(A)^2."""
    eigenvalues = _gram_eigenvalues(A)
    return float(np.sum(eigenvalues[: A.shape[1] - k]))


def singular_values(A: np.ndarray) -> np.ndarray:
    """All singular values of A in decreasing order."""
    eigenvalues = _gram_eigenvalues(A)[::-1]
    return np.sqrt(np.maximum(eigenvalues, 0.0))


def leading_term(nu: np.ndarray, k: int) -> float:
    """Leading theorem term: sum of the d-k smallest column variances."""
    return float(np.sum(np.sort(nu)[::-1][k:]))


def summarize(samples: np.ndarray):
    """Return (mean, sample std, standard error) of Monte Carlo samples."""
    samples = np.asarray(samples, dtype=float)
    mean = float(np.mean(samples))
    sample_std = float(np.std(samples, ddof=1)) if samples.size > 1 else 0.0
    standard_error = sample_std / np.sqrt(samples.size)
    return mean, sample_std, standard_error
