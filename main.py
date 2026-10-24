"""Uniform-quantizer experiment: sweep the bit width B and compare the Monte
Carlo tail-energy increase T with the leading theorem term
sum_{j>k} nu_j^downarrow.

Examples:
    python main.py synthetic
    python main.py synthetic --n 20 1000 100000 --d 20 --k 2
    python main.py gpt2small
    python main.py gpt2medium --trials 50 --bits 8 10 12
"""

import argparse
from pathlib import Path

import numpy as np

from plotting import (
    plot_singular_spectrum,
    plot_tail_energy_sweep,
    save_sweep_results,
)
from real_data import MODEL_SPECS, load_real_matrix
from synthetic_data import make_synthetic_matrix
from utils import (
    leading_term,
    prepare_quantizer,
    quantization_resolution,
    sample_quantizer,
    singular_values,
    summarize,
    tail_energy,
)


def sweep_B(A: np.ndarray, k: int, bits, trials: int, seed: int):
    """Estimate T for each bit width and compare with the leading term.

    Each bit width uses its own RNG stream from SeedSequence([seed, B]), so
    a bit-width result is reproducible regardless of which other bit widths
    are run.
    """
    n, d = A.shape
    R = float(np.max(np.abs(A)))
    tail_A = tail_energy(A, k)
    sv = singular_values(A)

    results = []
    for B in bits:
        rng = np.random.default_rng(np.random.SeedSequence([seed, B]))
        rho = quantization_resolution(B, R)
        lower, prob, nu = prepare_quantizer(A, B, R)

        samples = np.array([
            tail_energy(sample_quantizer(lower, prob, rho, rng), k) - tail_A
            for _ in range(trials)
        ])
        T, sample_std, standard_error = summarize(samples)
        leading = leading_term(nu, k)
        ratio = T / leading if leading != 0 else np.nan
        results.append(dict(
            B=B, rho=rho, T=T, leading=leading, difference=T - leading,
            ratio=ratio, sample_std=sample_std,
            standard_error=standard_error, samples=samples,
            n=n, d=d, k=k, sigma_1=float(sv[0]), sigma_k=float(sv[k - 1]),
            sigma_k_plus_1=float(sv[k]),
        ))
        print(
            f"B={B:2d}  rho={rho:.5f}  T={T:.5f}  SE={standard_error:.5f}  "
            f"leading={leading:.5f}  T-leading={T - leading:.5f}  "
            f"T/leading={ratio:.5f}"
        )
    return results


def run_experiment(A: np.ndarray, name: str, k: int, bits, trials: int,
                   seed: int, confidence_interval: bool = False):
    n, d = A.shape
    results = sweep_B(A, k, bits, trials, seed)
    plot_path = plot_tail_energy_sweep(
        results, name=name, n=n, d=d, k=k,
        confidence_interval=confidence_interval, bootstrap_seed=seed,
    )
    summary_path, sample_path = save_sweep_results(results, name=name)
    print(f"Plot saved to {plot_path}")
    print(f"Summary saved to {summary_path}")
    print(f"Raw Monte Carlo samples saved to {sample_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare T with its leading term over a sweep of bit widths."
    )
    parser.add_argument(
        "source",
        choices=["synthetic"] + sorted(MODEL_SPECS),
        help="matrix source: synthetic or a registered token embedding",
    )
    parser.add_argument(
        "--trials", type=int, default=100,
        help="Monte Carlo trials per bit width (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="base seed; bit width B uses stream SeedSequence([seed, B]) "
             "(default: 1 synthetic, 2026 real)",
    )
    parser.add_argument(
        "--bits", type=int, nargs="+", default=list(range(8, 15)),
        help="quantization bit widths (default: 8 through 14)",
    )
    parser.add_argument(
        "--no-confidence-interval", dest="confidence_interval",
        action="store_false",
        help="plot the Monte Carlo mean of T without error bars (default: "
             "draw its 95 percent bootstrap confidence interval)",
    )

    synthetic = parser.add_argument_group("synthetic options")
    synthetic.add_argument(
        "--n", type=int, nargs="+", default=[1000],
        help="row counts; one full sweep is run per value (default: 1000)",
    )
    synthetic.add_argument("--d", type=int, default=10, help="columns")
    synthetic.add_argument("--k", type=int, default=3, help="spectral cutoff")
    synthetic.add_argument(
        "--sigma-top", type=float, default=1e3,
        help="the k leading singular values",
    )
    synthetic.add_argument(
        "--sigma-tail", type=float, default=1e-3,
        help="the d-k tail singular values",
    )

    real = parser.add_argument_group("real-data options")
    real.add_argument(
        "--weights", type=Path, default=None,
        help="override the registered model's default local weight path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    is_synthetic = args.source == "synthetic"
    trials = args.trials
    seed = args.seed if args.seed is not None else (1 if is_synthetic else 2026)
    ci = args.confidence_interval

    if is_synthetic:
        for n in args.n:
            A = make_synthetic_matrix(
                n, args.d, args.k, args.sigma_top, args.sigma_tail, seed=seed
            )
            name = f"synthetic_n{n}_d{args.d}_k{args.k}"
            print(f"\n--- {name} ---")
            run_experiment(A, name, args.k, args.bits, trials, seed, ci)
        return

    A, name = load_real_matrix(args.source, args.weights)
    spec = MODEL_SPECS[args.source]
    k = 1
    sv = singular_values(A)
    print(f"Loaded {name}: shape={A.shape}")
    print(f"Tensor: {spec.tensor_name}; SHA-256 verified")
    print(f"Top singular values: {sv[:10]}")
    print(
        f"Selected k={k}: sigma_1/sigma_2={sv[0] / sv[1]:.6f}, "
        f"g/lambda_1={1 - sv[1] ** 2 / sv[0] ** 2:.6f}"
    )
    spectrum_path = plot_singular_spectrum(sv, name=name, highlight_k=k)
    print(f"Spectrum plot saved to {spectrum_path}")
    run_experiment(A, name, k, args.bits, trials, seed, ci)


if __name__ == "__main__":
    main()
