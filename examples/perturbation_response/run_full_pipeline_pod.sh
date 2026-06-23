#!/usr/bin/env bash
# Pod entrypoint: the FULL perturbation pipeline in one session —
#   Stage A (01 encoder) -> probe (02) -> Stage C (03 NB decoder)
#   -> Stage B (04 conditional flow) -> effect-size eval (06).
#
# This is the "real numbers" run: a strong 256-dim encoder on all combo-train
# cells, then the generative head + the held-out-cells effect-size test that
# answers the core question (does the method recover effect size?).
#
# Designed to run in the CLEAN NGC base env: if the processed cache is already
# present at $DATA_DIR/$ARTIFACT (e.g. staged via SkyPilot file_mounts), we NEVER
# install pertpy/scanpy — that install pollutes numpy/scipy/sklearn and breaks
# torch + sklearn (the two failures of 2026-06-22). The pertpy-process-and-restore
# path is kept ONLY as a fallback for when no cache is staged.
#
# Usage:
#   DATA_DIR=/cache bash examples/perturbation_response/run_full_pipeline_pod.sh [OUTPUT_ROOT]
# Env overrides:
#   EXPERIMENT (norman_stage_a), DATA_DIR (data), ARTIFACT (norman2019),
#   EPOCHS (50), N_HVG (5000), DEC_EPOCHS (30), FLOW_EPOCHS (60), TOP_K (20), EVAL_N (200)
set -uo pipefail

OUT_ROOT="${1:-output}"
EXP="${EXPERIMENT:-norman_stage_a}"
DATA_DIR="${DATA_DIR:-data}"
ARTIFACT="${ARTIFACT:-norman2019}"
if [ "$OUT_ROOT" != "output" ]; then
  mkdir -p "$OUT_ROOT"
  ln -sfn "$OUT_ROOT" output
fi

mkdir -p "output/$EXP/logs"
exec > >(tee -a "output/$EXP/logs/full_pipeline.log") 2>&1

: "${EPOCHS:=50}"
: "${N_HVG:=5000}"
: "${DEC_EPOCHS:=30}"
: "${FLOW_EPOCHS:=60}"
: "${TOP_K:=20}"
: "${EVAL_N:=200}"

echo "[run_full_pipeline] exp=$EXP data=$DATA_DIR artifact=$ARTIFACT epochs=$EPOCHS"
echo "[run_full_pipeline] dec_epochs=$DEC_EPOCHS flow_epochs=$FLOW_EPOCHS top_k=$TOP_K eval_n=$EVAL_N"

# --- Data: only process on-pod if the cache is absent (fallback path). -------
if [ ! -f "$DATA_DIR/$ARTIFACT/manifest.json" ]; then
  echo "[run_full_pipeline] cache ABSENT -> processing Norman on-pod (installs pertpy; restores base stack after)"
  NPV="$(python -c 'import numpy; print(numpy.__version__)')"
  SPV="$(python -c 'import scipy; print(scipy.__version__)' 2>/dev/null || true)"
  SKV="$(python -c 'import sklearn; print(sklearn.__version__)' 2>/dev/null || true)"
  echo "[run_full_pipeline] base stack: numpy=$NPV scipy=$SPV sklearn=$SKV (restored after processing)"
  pip install anndata scanpy scikit-misc pertpy
  python examples/perturbation_response/00_process_norman.py \
    --source pertpy --data-dir "$DATA_DIR" --artifact "$ARTIFACT" --n-hvg "$N_HVG"
  pip install --no-deps "numpy==$NPV" ${SPV:+"scipy==$SPV"} ${SKV:+"scikit-learn==$SKV"}
else
  echo "[run_full_pipeline] cache present at $DATA_DIR/$ARTIFACT — clean base env, no pertpy install"
fi

A=(--experiment "$EXP" --data-dir "$DATA_DIR" --artifact "$ARTIFACT")

# Per-stage runner: log a banner, run, record status; a failed downstream stage
# must not discard the artifacts already written (encoder.pt etc.) on the volume.
status=0
stage() {  # stage <fatal:0|1> <label> <cmd...>
  local fatal="$1" label="$2"; shift 2
  echo "=================================================================="
  echo "[run_full_pipeline] >>> $label"
  echo "=================================================================="
  if "$@"; then
    echo "[run_full_pipeline] <<< $label OK"
  else
    echo "[run_full_pipeline] !!! $label FAILED (exit $?)"
    status=1
    if [ "$fatal" = "1" ]; then
      echo "[run_full_pipeline] fatal stage failed — stopping; partial artifacts remain on the volume."
      exit 1
    fi
  fi
}

# Stage A — intra-cell JEPA encoder (combo split: no SSL leakage of test combos).
stage 1 "Stage A: 01_pretrain_stage_a" \
  python examples/perturbation_response/01_pretrain_stage_a.py "${A[@]}" \
    --split combo --epochs "$EPOCHS" --reg-coef 0.04
# Probe is diagnostic only — never let it sink the run.
stage 0 "probe: 02_probe_cell_encoder" \
  python examples/perturbation_response/02_probe_cell_encoder.py "${A[@]}"

# Stage C — NB count decoder on frozen latents (cells split: effect-size test).
stage 1 "Stage C: 03_train_count_decoder" \
  python examples/perturbation_response/03_train_count_decoder.py "${A[@]}" \
    --split cells --epochs "$DEC_EPOCHS"
# Stage B — conditional flow over cell latents (cells split).
stage 1 "Stage B: 04_train_cond_flow" \
  python examples/perturbation_response/04_train_cond_flow.py "${A[@]}" \
    --split cells --epochs "$FLOW_EPOCHS"
# Eval — sample per perturbation and score the Pearson Δ-correlation on top-DE genes.
stage 1 "Eval: 06_eval_effect_size" \
  python examples/perturbation_response/06_eval_effect_size.py "${A[@]}" \
    --split cells --n "$EVAL_N" --top-k "$TOP_K"

echo "[run_full_pipeline] done (overall status=$status). artifacts under output/$EXP/"
exit "$status"
