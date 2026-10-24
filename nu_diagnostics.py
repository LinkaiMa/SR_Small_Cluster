"""Compute fixed-rho normalized variance nu across experiment datasets.

For the uniform quantizer used by ``main.py``,

    nu = min_j sum_i Var(E_ij) / (n rho^2).

If fractional positions inside quantization cells behave like independent
Uniform(0, 1) variables, then E[p(1-p)] = 1/6.  This script reports both the
theorem's minimum-over-columns statistic and the across-column average.
Computation is chunked so the large embedding matrices do not require
additional full-size temporary arrays.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from real_data import MODEL_SPECS, load_real_matrix
from synthetic_data import make_synthetic_matrix
from utils import prepare_quantizer, quantization_resolution


EXPECTED_NU = 1.0 / 6.0


def uniform_nu_statistics(A: np.ndarray, B: int, chunk_rows: int):
    """Compute normalized column variances for one matrix and bit width."""
    n, d = A.shape
    R = float(np.max(np.abs(A)))
    rho = quantization_resolution(B, R)
    column_variances = np.zeros(d, dtype=np.float64)
    exact_counts = np.zeros(d, dtype=np.int64)

    for start in range(0, n, chunk_rows):
        chunk = A[start:start + chunk_rows]
        _, probability, chunk_variances = prepare_quantizer(chunk, B, R)
        column_variances += chunk_variances
        exact_counts += np.sum(
            (probability == 0.0) | (probability == 1.0),
            axis=0,
            dtype=np.int64,
        )

    normalized = column_variances / (n * rho ** 2)
    worst_column = int(np.argmin(normalized))
    return {
        "B": B,
        "rho": rho,
        "nu_theorem": float(np.min(normalized)),
        "nu_q05": float(np.quantile(normalized, 0.05)),
        "nu_median": float(np.median(normalized)),
        "nu_mean": float(np.mean(normalized)),
        "nu_q95": float(np.quantile(normalized, 0.95)),
        "nu_max": float(np.max(normalized)),
        "worst_column": worst_column,
        "exact_fraction": float(np.sum(exact_counts) / (n * d)),
        "worst_column_exact_fraction": float(
            exact_counts[worst_column] / n
        ),
        "mean_minus_one_sixth": float(np.mean(normalized) - EXPECTED_NU),
        "max_column_deviation": float(np.max(np.abs(
            normalized - EXPECTED_NU
        ))),
    }


def dataset_rows(name: str, A: np.ndarray, bits, chunk_rows: int):
    """Return one fixed-rho detailed row per bit width for a dataset."""
    print(f"\n--- fixed-rho uniform grid: {name}, shape={A.shape} ---")
    rows = []
    for B in bits:
        statistics = uniform_nu_statistics(A, B, chunk_rows)
        row = {
            "dataset": name,
            "quantizer": "uniform fixed-rho",
            "n": A.shape[0],
            "d": A.shape[1],
            **statistics,
        }
        rows.append(row)
        print(
            f"B={B:2d}: nu={row['nu_theorem']:.6f}, "
            f"column mean={row['nu_mean']:.6f}, "
            f"mean-1/6={row['mean_minus_one_sixth']:+.3e}"
        )
    return rows


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def save_tables(rows, output_dir="results"):
    """Save detailed CSV, dataset summary CSV, and a LaTeX table."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "fixed_rho_nu_diagnostics.csv"
    summary_path = output_dir / "fixed_rho_nu_diagnostics_summary.csv"
    latex_path = output_dir / "fixed_rho_nu_diagnostics_table.tex"

    detail = pd.DataFrame(rows)
    detail.to_csv(detail_path, index=False)

    summary_rows = []
    for dataset, group in detail.groupby("dataset", sort=False):
        summary_rows.append({
            "dataset": dataset,
            "quantizer": "uniform fixed-rho",
            "n": int(group["n"].iloc[0]),
            "d": int(group["d"].iloc[0]),
            "nu_theorem_min": float(group["nu_theorem"].min()),
            "nu_theorem_max": float(group["nu_theorem"].max()),
            "nu_median_min": float(group["nu_median"].min()),
            "nu_median_max": float(group["nu_median"].max()),
            "nu_mean_min": float(group["nu_mean"].min()),
            "nu_mean_max": float(group["nu_mean"].max()),
            "max_abs_mean_minus_one_sixth": float(
                group["mean_minus_one_sixth"].abs().max()
            ),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_path, index=False)

    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        (
            r"Fixed-$\rho$ dataset & Shape & Theorem $\nu$ range "
            r"& Mean normalized variance range \\"
        ),
        r"\midrule",
    ]
    for row in summary_rows:
        lines.append(
            f"{latex_escape(row['dataset'])} & "
            f"${row['n']}\\times {row['d']}$ & "
            f"${row['nu_theorem_min']:.4f}$--"
            f"${row['nu_theorem_max']:.4f}$ & "
            f"${row['nu_mean_min']:.4f}$--"
            f"${row['nu_mean_max']:.4f}$ \\\\"
        )
    lines.extend([
        r"\midrule",
        r"Uniform-cell prediction & -- & -- & $1/6 \approx 0.1667$ \\",
        r"\bottomrule",
        r"\end{tabular}",
    ])
    latex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return detail_path, summary_path, latex_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute fixed-rho uniform-grid normalized variance nu."
    )
    parser.add_argument("--bits", type=int, nargs="+",
                        default=list(range(8, 15)))
    parser.add_argument("--synthetic-n", type=int, nargs="+",
                        default=[20, 50, 100, 1000, 10000, 100000])
    parser.add_argument("--d", type=int, default=10)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--sigma-top", type=float, default=1e3)
    parser.add_argument("--sigma-tail", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--models",
        nargs="*",
        choices=sorted(MODEL_SPECS),
        default=sorted(MODEL_SPECS),
    )
    parser.add_argument("--chunk-rows", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for n in args.synthetic_n:
        A = make_synthetic_matrix(
            n,
            args.d,
            args.k,
            args.sigma_top,
            args.sigma_tail,
            seed=args.seed,
        )
        rows.extend(dataset_rows(
            f"Synthetic n={n}",
            A,
            args.bits,
            args.chunk_rows,
        ))

    for model in args.models:
        A, name = load_real_matrix(model)
        rows.extend(dataset_rows(name, A, args.bits, args.chunk_rows))
        del A

    detail_path, summary_path, latex_path = save_tables(rows)
    print(f"Detailed table saved to {detail_path}")
    print(f"Summary table saved to {summary_path}")
    print(f"LaTeX table saved to {latex_path}")


if __name__ == "__main__":
    main()
