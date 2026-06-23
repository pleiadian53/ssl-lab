#!/usr/bin/env bash
# Pod entrypoint: Stage A — process Norman 2019 (once) and pretrain the cell encoder.
#
# Self-contained, like the MNIST pipeline: the data is processed ON the pod onto a
# persistent path, so no separate data-staging step is needed. If an output root is
# given (arg 1) it is symlinked to ./output so artifacts land on the network volume
# and rsync back. Point DATA_DIR at the volume too so the ~6 GB cache persists and
# later pods skip reprocessing.
#
# Usage:
#   bash examples/perturbation_response/run_stage_a_pod.sh [OUTPUT_ROOT]
# Env overrides:
#   EXPERIMENT (default norman_stage_a), DATA_DIR (default data), ARTIFACT (norman2019),
#   EPOCHS (50), N_HVG (5000)
set -euo pipefail

OUT_ROOT="${1:-output}"
EXP="${EXPERIMENT:-norman_stage_a}"
DATA_DIR="${DATA_DIR:-data}"
ARTIFACT="${ARTIFACT:-norman2019}"
if [ "$OUT_ROOT" != "output" ]; then
  mkdir -p "$OUT_ROOT"
  ln -sfn "$OUT_ROOT" output
fi

mkdir -p "output/$EXP/logs"
exec > >(tee -a "output/$EXP/logs/train.log") 2>&1

: "${EPOCHS:=50}"
: "${N_HVG:=5000}"

echo "[run_stage_a_pod] exp=$EXP data=$DATA_DIR artifact=$ARTIFACT epochs=$EPOCHS n_hvg=$N_HVG"

# Process Norman onto the (persistent) data dir, only if the cache is absent.
# The heavy single-cell stack (pertpy/scanpy) is installed ONLY for processing —
# it upgrades NumPy, which breaks the NGC image's torch (`torch.from_numpy:
# "Numpy is not available"`). So we capture the torch-built NumPy first and
# restore it before training. When the cache is already present, we skip all of
# this and train in the clean base env.
if [ ! -f "$DATA_DIR/$ARTIFACT/manifest.json" ]; then
  echo "[run_stage_a_pod] processing Norman -> $DATA_DIR/$ARTIFACT (first run)"
  # pertpy upgrades the whole scientific stack (numpy/scipy/sklearn) to NumPy-2.x-era
  # versions; torch (NGC, built vs numpy 1.26) and sklearn then break each other. Capture
  # the base versions and restore ALL of them after processing, so training is clean.
  NPV="$(python -c 'import numpy; print(numpy.__version__)')"
  SPV="$(python -c 'import scipy; print(scipy.__version__)' 2>/dev/null || true)"
  SKV="$(python -c 'import sklearn; print(sklearn.__version__)' 2>/dev/null || true)"
  echo "[run_stage_a_pod] base stack: numpy=$NPV scipy=$SPV sklearn=$SKV (restored after processing)"
  pip install anndata scanpy scikit-misc pertpy
  python examples/perturbation_response/00_process_norman.py \
    --source pertpy --data-dir "$DATA_DIR" --artifact "$ARTIFACT" --n-hvg "$N_HVG"
  pip install --no-deps "numpy==$NPV" ${SPV:+"scipy==$SPV"} ${SKV:+"scikit-learn==$SKV"}
else
  echo "[run_stage_a_pod] cache present at $DATA_DIR/$ARTIFACT — skipping processing + heavy install"
fi

A=(--experiment "$EXP" --data-dir "$DATA_DIR" --artifact "$ARTIFACT")
python examples/perturbation_response/01_pretrain_stage_a.py "${A[@]}" --epochs "$EPOCHS" --reg-coef 0.04
python examples/perturbation_response/02_probe_cell_encoder.py "${A[@]}"

echo "[run_stage_a_pod] done. artifacts under output/$EXP/"
