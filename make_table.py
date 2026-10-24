"""Build the paper table from the saved FP8 results: one row per (model,
format) with the Monte Carlo mean of T and its standard error, the leading
term L, and T/L.

    python make_table.py

Reads results/fp8_<model>_summary.csv for every model in
real_data.MODEL_SPECS and writes results/fp8_table.tex.
"""

import math
from pathlib import Path

import pandas as pd

from plotting import slugify
from real_data import MODEL_SPECS

RESULTS = Path("results")


def decimals(standard_error: float) -> int:
    """Number of decimals that shows two significant digits of the s.e."""
    if not math.isfinite(standard_error) or standard_error <= 0:
        return 3
    return int(max(0, min(7, 1 - math.floor(math.log10(standard_error)))))


def main():
    rows = []
    for key in MODEL_SPECS:
        name = MODEL_SPECS[key].name
        path = RESULTS / f"fp8_{slugify(name)}_summary.csv"
        if not path.exists():
            print(f"skipping {name}: {path} not found")
            continue
        for _, row in pd.read_csv(path).iterrows():
            rows.append({
                "model": name,
                "conversion": f"FP32 to {row['format']}",
                "T": row["T"],
                "se": row["standard_error"],
                "L": row["leading"],
                "T/L": row["ratio"],
            })
    table = pd.DataFrame(rows)

    lines = [
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"model & conversion & $\mathcal{T}$ (mean $\pm$ s.e.) & $L$ & "
        r"$\mathcal{T}/L$ \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        n = decimals(row["se"])
        lines.append(
            f"{row['model']} & {row['conversion']} & "
            f"${row['T']:.{n}f} \\pm {row['se']:.{n}f}$ & "
            f"{row['L']:.{n}f} & {row['T/L']:.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    output = RESULTS / "fp8_table.tex"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    pd.set_option("display.width", 200)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"LaTeX table saved to {output}")


if __name__ == "__main__":
    main()
