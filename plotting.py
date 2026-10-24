"""Plotting and CSV serialization for the tail-energy experiments."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def slugify(name: str) -> str:
    """Filesystem-safe version of a dataset name."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return slug or "matrix"


def plot_singular_spectrum(singular_values: np.ndarray, *, name: str,
                           highlight_k: int | None = None,
                           output_dir="plots") -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    indices = np.arange(1, len(singular_values) + 1)
    ax.semilogy(indices, singular_values, color="C0", linewidth=1.5)
    if highlight_k is not None:
        ax.axvline(
            highlight_k,
            color="C3",
            linestyle="--",
            label=rf"selected gap $k={highlight_k}$",
        )
        ax.legend(loc="best")
    ax.set_xlabel("singular-value index")
    ax.set_ylabel(r"$\sigma_i$")
    ax.set_title(f"{name}\nsingular-value spectrum")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"spectrum_{slugify(name)}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_tail_energy_sweep(results, *, name: str, n: int, d: int, k: int,
                           output_dir="plots", confidence_interval=False,
                           bootstrap_seed=0, n_boot=1000) -> Path:
    """Plot the Monte Carlo estimate of T against the leading theorem term.

    The T curve is the Monte Carlo mean of the tail-energy increase; with
    confidence_interval=True (the default of main.py) a 95% bootstrap CI of
    that mean is drawn as capped error bars (following the plotting pattern
    in github.com/LinkaiMa/SMW).
    """
    rows = []
    for result in results:
        rows.extend(
            {"rho": result["rho"], "value": float(sample), "metric": "T"}
            for sample in result["samples"]
        )
        rows.append(
            {"rho": result["rho"], "value": result["leading"],
             "metric": "leading"}
        )
    plot_data = pd.DataFrame(rows)

    marker_styles = {"T": "o", "leading": "s"}
    line_styles = {"T": (), "leading": (4, 2)}
    palette = {"T": "C0", "leading": "C1"}
    t_label = (
        r"$\mathcal{T}$ (Monte Carlo mean, 95% bootstrap CI)"
        if confidence_interval
        else r"$\mathcal{T}$ (Monte Carlo mean)"
    )
    label_map = {
        "T": t_label,
        "leading": r"leading term $\sum_{j=k+1}^d \nu_j^\downarrow$",
    }

    if confidence_interval:
        errorbar_kwargs = dict(
            errorbar=("ci", 95),
            err_style="bars",
            err_kws={"capsize": 4, "elinewidth": 1.5, "capthick": 1.5},
            n_boot=n_boot,
            seed=bootstrap_seed,
        )
    else:
        errorbar_kwargs = dict(errorbar=None)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.lineplot(
        data=plot_data,
        x="rho",
        y="value",
        hue="metric",
        style="metric",
        markers=marker_styles,
        dashes=line_styles,
        palette=palette,
        estimator="mean",
        ax=ax,
        **errorbar_kwargs,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\rho$ (quantization resolution)")
    ax.set_ylabel(r"$\mathcal{T}$")
    ax.set_title(
        rf"{name}: tail energy vs. $\rho$" "\n" rf"$(n={n},\,d={d},\,k={k})$"
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        [label_map.get(label, label) for label in labels],
        loc="best",
    )
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"sweep_B_{slugify(name)}.png"
    fig.savefig(output_path, dpi=150)
    fig.savefig(output_path.with_suffix(".pdf"))  # vector copy for the paper
    plt.close(fig)
    return output_path


def save_sweep_results(results, *, name: str, output_dir="results"):
    """Save summary statistics and raw Monte Carlo samples as CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(name)
    summary_path = output_dir / f"sweep_B_{slug}_summary.csv"
    sample_path = output_dir / f"sweep_B_{slug}_samples.csv"

    summary_columns = (
        "B", "rho", "T", "leading", "difference", "ratio",
        "sample_std", "standard_error",
        "n", "d", "k", "sigma_1", "sigma_k", "sigma_k_plus_1",
    )
    pd.DataFrame(
        [{key: result[key] for key in summary_columns} for result in results]
    ).to_csv(summary_path, index=False)

    sample_rows = [
        {"B": result["B"], "rho": result["rho"], "T_sample": float(sample)}
        for result in results
        for sample in result["samples"]
    ]
    pd.DataFrame(sample_rows).to_csv(sample_path, index=False)
    return summary_path, sample_path


def save_pychop_results(results, *, name: str, output_dir="results"):
    """Save the FP8 summaries and raw Monte Carlo samples as CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(name)
    summary_path = output_dir / f"fp8_{slug}_summary.csv"
    sample_path = output_dir / f"fp8_{slug}_samples.csv"

    summary_columns = (
        "format", "exponent_bits", "mantissa_bits", "n", "d", "k", "trials",
        "seed", "rho_max", "exact_fraction", "T", "leading", "difference",
        "ratio", "sample_std", "standard_error",
        "sigma_1", "sigma_k", "sigma_k_plus_1",
    )
    pd.DataFrame(
        [{key: result[key] for key in summary_columns} for result in results]
    ).to_csv(summary_path, index=False)

    sample_rows = [
        {"format": result["format"], "trial": trial, "T_sample": float(sample)}
        for result in results
        for trial, sample in enumerate(result["samples"])
    ]
    pd.DataFrame(sample_rows).to_csv(sample_path, index=False)
    return summary_path, sample_path
