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

# The base pod setup installs only core deps; processing Norman needs the heavy
# single-cell stack. (pertpy pulls a large tree — one-time cost; the cache persists.)
pip install anndata scanpy scikit-misc pertpy

# Process Norman onto the (persistent) data dir, only if the cache is absent.
if [ ! -f "$DATA_DIR/$ARTIFACT/manifest.json" ]; then
  echo "[run_stage_a_pod] processing Norman -> $DATA_DIR/$ARTIFACT (first run)"
  python examples/perturbation_response/00_process_norman.py \
    --source pertpy --data-dir "$DATA_DIR" --artifact "$ARTIFACT" --n-hvg "$N_HVG"
else
  echo "[run_stage_a_pod] cache present at $DATA_DIR/$ARTIFACT — skipping processing"
fi

A=(--experiment "$EXP" --data-dir "$DATA_DIR" --artifact "$ARTIFACT")
python examples/perturbation_response/01_pretrain_stage_a.py "${A[@]}" --epochs "$EPOCHS" --reg-coef 0.04
python examples/perturbation_response/02_probe_cell_encoder.py "${A[@]}"

echo "[run_stage_a_pod] done. artifacts under output/$EXP/"
