#!/usr/bin/env bash
# Cloud Agent install for ssl-lab.
#
# The repo normally uses conda (environment.yml), but conda is not present on the
# default Cloud Agent image, so we reproduce the same dependency set with pip.
# CPU is the repo's local default (see environment.yml), so we install the CPU
# PyTorch wheels explicitly to avoid pulling the much larger default CUDA wheels.
#
# Idempotent and non-interactive: safe to run repeatedly against cached state.
set -euo pipefail

python3 -m pip install --upgrade pip

# CPU-only PyTorch/torchvision. Installing these first means the editable install
# below sees torch>=2.1 / torchvision>=0.16 already satisfied and does not fall
# back to the CUDA wheels from the default PyPI index.
python3 -m pip install --index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.1" "torchvision>=0.16"

# Editable install of the ssllab package plus the dev extra (pytest).
# The heavy single-cell perturbation stack is optional and processing-time only;
# install it on demand with:  python3 -m pip install -e ".[perturb]"
python3 -m pip install -e ".[dev]"
