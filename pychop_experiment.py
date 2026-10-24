"""FP8 experiment: stochastic rounding of the token embeddings with PyChop.

Rounds the FP32 token embedding of ``python main.py gpt2small`` (or
gpt2medium) to the FP8 formats E5M2 (5 exponent bits, 2 stored mantissa
bits) and E4M3 (4, 3) with PyChop's proportional stochastic rounding, and
compares the Monte Carlo tail-energy increase T with the leading term
sum_{j>k} nu_j^downarrow, exactly as main.py does for the uniform quantizer.

For each entry A_ij the two adjacent representable values q^-_ij <= A_ij <=
q^+_ij give the exact, nonuniform variance Var(E_ij) = (A_ij - q^-_ij)
(q^+_ij - A_ij), from which the column variances nu_j are summed; the
theorem's global resolution rho is the largest local spacing. The runner
checks that every rounded entry returned by PyChop is one of the two
neighbours.

Examples:
    python pychop_experiment.py gpt2small
    python pychop_experiment.py gpt2medium --trials 50 --formats e4m3
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pychop import Chop

from plotting import save_pychop_results
from real_data import MODEL_SPECS, load_real_matrix
from utils import leading_term, singular_values, summarize, tail_energy


@dataclass(frozen=True)
class Format:
    label: str
    exponent_bits: int
    mantissa_bits: int  # stored mantissa bits


FORMATS = {
    "e5m2": Format("E5M2", 5, 2),
    "e4m3": Format("E4M3", 4, 3),
}


def float_neighbors(A: np.ndarray, exponent_bits: int, mantissa_bits: int):
    """Adjacent representable values of a binary floating-point format.

    The format has ``exponent_bits`` exponent bits (bias 2^(exponent_bits-1)-1,
    subnormals included) and ``mantissa_bits`` stored mantissa bits. Returns
    (lower, upper, phase, rho_max): lower <= A <= upper are the two neighbours
    in A's dtype (equal where A is exactly representable), phase is
    (A - lower) / (upper - lower), and rho_max is the largest spacing.
    """
    A64 = np.asarray(A, dtype=np.float64)
    emin = 2 - 2 ** (exponent_bits - 1)
    emax = 2 ** (exponent_bits - 1) - 1
    if np.max(np.abs(A64)) >= 2.0 ** (emax + 1):
        raise ValueError("matrix entries overflow the target format")
    _, exponent = np.frexp(np.abs(A64))  # |x| = m * 2^exponent, m in [0.5, 1)
    exponent = np.maximum(exponent - 1, emin)  # floor(log2|x|), subnormals clamp
    spacing = np.exp2(exponent - mantissa_bits)
    lower = np.floor(A64 / spacing) * spacing
    exact = lower == A64
    spacing = np.where(exact, 0.0, spacing)
    upper = lower + spacing
    phase = np.where(exact, 0.0, (A64 - lower) / np.where(exact, 1.0, spacing))
    return lower.astype(A.dtype), upper.astype(A.dtype), phase, float(np.max(spacing))


def column_variances(A, lower, upper):
    """Exact stochastic-rounding column variances for nonuniform spacings."""
    A64 = np.asarray(A, dtype=np.float64)
    lower64 = np.asarray(lower, dtype=np.float64)
    upper64 = np.asarray(upper, dtype=np.float64)
    return np.sum((A64 - lower64) * (upper64 - A64), axis=0, dtype=np.float64)


def pychop_seed(base_seed: int, fmt: Format, trial: int) -> int:
    """A trial stream independent of the order in which formats are run."""
    sequence = np.random.SeedSequence(
        [base_seed, 1000 * fmt.exponent_bits + fmt.mantissa_bits, trial]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def run_format(A: np.ndarray, fmt: Format, *, k: int, trials: int, seed: int):
    """Monte Carlo estimate of T for one target format."""
    n, d = A.shape
    lower, upper, phase, rho_max = float_neighbors(
        A, fmt.exponent_bits, fmt.mantissa_bits
    )
    exact_fraction = float(np.mean(lower == upper))
    if exact_fraction > 0.99:
        raise ValueError(
            f"{fmt.label}: {100 * exact_fraction:.1f}% of the entries are "
            "exactly representable, nothing to round"
        )
    nu = column_variances(A, lower, upper)
    leading = leading_term(nu, k)
    tail_A = tail_energy(A, k)

    samples = np.empty(trials)
    for trial in range(trials):
        quantizer = Chop(
            exp_bits=fmt.exponent_bits, sig_bits=fmt.mantissa_bits, rmode=5,
            subnormal=True, random_state=pychop_seed(seed, fmt, trial),
        )
        A_quant = np.asarray(quantizer(A), dtype=A.dtype)
        if not np.all((A_quant == lower) | (A_quant == upper)):
            raise RuntimeError(f"PyChop produced non-adjacent values for {fmt.label}")
        samples[trial] = tail_energy(A_quant, k) - tail_A

    T, sample_std, standard_error = summarize(samples)
    sv = singular_values(A)
    ratio = T / leading
    print(
        f"FP32 -> {fmt.label}: rho_max={rho_max:.3g}  T={T:.5g}  "
        f"SE={standard_error:.3g}  leading={leading:.5g}  T/leading={ratio:.5f}"
    )
    return dict(
        format=fmt.label, exponent_bits=fmt.exponent_bits,
        mantissa_bits=fmt.mantissa_bits, n=n, d=d, k=k, trials=trials,
        seed=seed, rho_max=rho_max, exact_fraction=exact_fraction, T=T,
        leading=leading, difference=T - leading, ratio=ratio,
        sample_std=sample_std, standard_error=standard_error,
        sigma_1=float(sv[0]), sigma_k=float(sv[k - 1]),
        sigma_k_plus_1=float(sv[k]), samples=samples,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare T with its leading term under FP8 stochastic "
                    "rounding of a token embedding (PyChop)."
    )
    parser.add_argument("source", choices=sorted(MODEL_SPECS))
    parser.add_argument(
        "--formats", nargs="+", choices=sorted(FORMATS),
        default=["e5m2", "e4m3"], help="target formats (default: both)",
    )
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026,
                        help="base seed of the PyChop streams")
    parser.add_argument(
        "--weights", type=Path, default=None,
        help="override the registered model's default local weight path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.trials < 2:
        raise ValueError("trials must be at least 2 for a standard error")
    A, name = load_real_matrix(args.source, args.weights)
    k = 1
    print(f"Loaded {name}: shape={A.shape}")
    results = [
        run_format(A, FORMATS[key], k=k, trials=args.trials, seed=args.seed)
        for key in args.formats
    ]
    summary_path, sample_path = save_pychop_results(results, name=name)
    print(f"Summary saved to {summary_path}")
    print(f"Raw Monte Carlo samples saved to {sample_path}")


if __name__ == "__main__":
    main()
