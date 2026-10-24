# Numerical experiments for *Stochastic Rounding Increases Small Singular Values*

Code and results for the experiments in

> L. Ma, T. Yu, and P. Drineas, *Stochastic Rounding Increases Small Singular
> Values*, arXiv:2606.00312, 2026. https://arxiv.org/abs/2606.00312

Each experiment estimates the expected excess tail energy $\mathcal{T}$ of a
stochastically rounded matrix by Monte Carlo and compares it with the leading
term $L=\sum_{j>k}\nu_j^\downarrow$ of the paper's small-cluster bound.

## Setup

Python 3.10 or later with the packages in `requirements.txt`:

```
python -m pip install -r requirements.txt
```

## Data

The pretrained matrices are the token embeddings of GPT-2 small and GPT-2
medium. Download the two tensors into `data_cache/` (their checksums are
verified when loaded):

```powershell
New-Item -ItemType Directory -Force data_cache
curl.exe -L --range 287223763-441613266 "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors" -o "data_cache/gpt2-wte-f32.bin"
curl.exe -L --range 1314132290-1519984961 "https://huggingface.co/openai-community/gpt2-medium/resolve/main/model.safetensors" -o "data_cache/gpt2-medium-wte-f32.bin"
```

On Linux or macOS use `mkdir -p data_cache` and `curl`.

## Usage

Uniform quantizer, bit widths $B=8,\dots,14$ (plots in `plots/`, CSVs in
`results/`):

```
python main.py synthetic                            # tall synthetic matrix, 1000 x 10
python main.py synthetic --n 1000 --d 1000 --k 3    # square synthetic matrix
python main.py gpt2small
python main.py gpt2medium
```

FP8 stochastic rounding (E5M2, E4M3) with PyChop, and the LaTeX table
`results/fp8_table.tex`:

```
python pychop_experiment.py gpt2small
python pychop_experiment.py gpt2medium
python make_table.py
```

Normalized minimum column variance $\nu$ of the uniform quantizer
(`results/fixed_rho_nu_diagnostics*`):

```
python nu_diagnostics.py --synthetic-n 1000
```

The Monte Carlo scripts accept `--trials` (default 100) and `--seed`; run a
script with `--help` for the remaining options. Every Monte Carlo run saves
a summary CSV ($\mathcal{T}$ with its standard error, $L$, $\mathcal{T}/L$)
and the raw samples.

## Reproducing everything

```
bash run_all.sh      # Linux, macOS
.\run_all.ps1        # Windows PowerShell
```

runs the commands above in order (about 45 minutes on a 6-core desktop).

## Citation

```bibtex
@misc{ma2026stochastic,
  title         = {Stochastic Rounding Increases Small Singular Values},
  author        = {Ma, Linkai and Yu, Tingzhou and Drineas, Petros},
  year          = {2026},
  eprint        = {2606.00312},
  archivePrefix = {arXiv},
  primaryClass  = {math.NA},
  url           = {https://arxiv.org/abs/2606.00312}
}
```
