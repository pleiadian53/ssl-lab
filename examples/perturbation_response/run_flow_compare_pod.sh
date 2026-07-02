#!/usr/bin/env bash
# Pod entrypoint: gaussian-prior vs control-transport flow, head-to-head on held-out combos.
#
# Reuses the SAME Stage-A encoder AND the SAME NB count decoder (both staged via
# file_mounts) for both flow modes, so the only variable is --flow-base:
#   gaussian : noise -> outcome, condition fuses (z_b, z_p)     [current default]
#   control  : transport control latent -> outcome, condition = z_p only  [the reformulation]
# Each mode trains only Stage B (04, geneset, combo split) + evals (06, combo), writing
# to its own experiment dir. Decoder is condition-free, so sharing it is fair.
#
# Usage:
#   DATA_DIR=/cache ENCODER=/cache/encoder.pt DECODER=/cache/count_decoder.pt \
#     bash examples/perturbation_response/run_flow_compare_pod.sh [OUTPUT_ROOT]
set -uo pipefail

OUT_ROOT="${1:-output}"
DATA_DIR="${DATA_DIR:-data}"
ARTIFACT="${ARTIFACT:-norman2019}"
ENCODER="${ENCODER:-output/norman_combo/checkpoints/encoder.pt}"
DECODER="${DECODER:-output/norman_combo/checkpoints/count_decoder.pt}"
if [ "$OUT_ROOT" != "output" ]; then mkdir -p "$OUT_ROOT"; ln -sfn "$OUT_ROOT" output; fi

: "${FLOW_EPOCHS:=60}"
: "${COMPOSE:=additive}"
: "${EVAL_N:=200}"
: "${TOP_K:=20}"

if [ ! -f "$DATA_DIR/$ARTIFACT/manifest.json" ]; then
  echo "[run_flow_compare] ERROR: cache absent at $DATA_DIR/$ARTIFACT — stage it via file_mounts." >&2
  exit 2
fi
echo "[run_flow_compare] cache present — clean base env; flow_epochs=$FLOW_EPOCHS compose=$COMPOSE"

run_mode() {  # run_mode <flow_base>
  local base="$1" exp="norman_flow_$1"
  mkdir -p "output/$exp/checkpoints" "output/$exp/logs"
  cp -f "$ENCODER" "output/$exp/checkpoints/encoder.pt"
  cp -f "$DECODER" "output/$exp/checkpoints/count_decoder.pt"
  echo "=================================================================="
  echo "[run_flow_compare] >>> flow_base=$base  (exp=$exp)"
  python examples/perturbation_response/04_train_cond_flow.py \
    --experiment "$exp" --data-dir "$DATA_DIR" --artifact "$ARTIFACT" \
    --split combo --cond-type geneset --compose "$COMPOSE" --flow-base "$base" --epochs "$FLOW_EPOCHS" \
    || { echo "[run_flow_compare] !!! 04 ($base) FAILED"; return 1; }
  python examples/perturbation_response/06_eval_effect_size.py \
    --experiment "$exp" --data-dir "$DATA_DIR" --artifact "$ARTIFACT" \
    --split combo --n "$EVAL_N" --top-k "$TOP_K" \
    || { echo "[run_flow_compare] !!! 06 ($base) FAILED"; return 1; }
  echo "[run_flow_compare] <<< flow_base=$base done -> output/$exp/reports/effect_size.json"
}

status=0
run_mode gaussian || status=1
run_mode control  || status=1
echo "[run_flow_compare] done (status=$status). Compare:"
echo "  gaussian: output/norman_flow_gaussian/reports/effect_size.json"
echo "  control : output/norman_flow_control/reports/effect_size.json"
exit "$status"
