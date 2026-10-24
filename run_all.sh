#!/usr/bin/env bash
# Reproduce every file in results/ and plots/. Run from the repository root
# in an environment with the packages of requirements.txt installed and the
# two tensors in data_cache/ (see README.md). Add "--trials 10" to the first
# six commands for a quick check; the defaults (100 trials) take about
# 45 minutes on a 6-core desktop.
set -euo pipefail

# 1. Uniform quantizer: synthetic matrices (tall 1000 x 10 and square
#    1000 x 1000, both k = 3) and the two token embeddings (plots)
python main.py synthetic
python main.py synthetic --n 1000 --d 1000 --k 3
python main.py gpt2small
python main.py gpt2medium

# 2. FP8 stochastic rounding of the two token embeddings with PyChop (table)
python pychop_experiment.py gpt2small
python pychop_experiment.py gpt2medium
python make_table.py

# 3. Fixed-rho normalized column variance nu for the matrices of step 1
python nu_diagnostics.py --synthetic-n 1000
