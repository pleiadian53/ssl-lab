#!/usr/bin/env bash
# Pod entrypoint: held-out-combo generalization + the NB-VAE baseline.
#
# Reuses the Stage-A encoder (staged via file_mounts -> $ENCODER), so NO Stage A
# retrain. On the combo split (test = 20 unseen 2-gene combos, both singles seen):
#   03 NB decoder (combo) -> 04 conditional flow (combo, GENESET embedding)
#   -> 06 effect-size eval (combo)            [the method on held-out combos]
#   08 NB-VAE baseline (combo) -> 09 eval     [the from-scratch control]
# Writes to experiment $EXP (default norman_combo) so it never clobbers the
# cells-split (in-distribution) results under norman_stage_a.
#
# Clean NGC base env: when the cache is staged (manifest present) pertpy is never
# installed. Usage:
#   DATA_DIR=/cache ENCODER=/cache/encoder.pt \
#     bash examples/perturbation_response/run_combo_generalization_pod.sh [OUTPUT_ROOT]
set -uo pipefail

OUT_ROOT="${1:-output}"
EXP="${EXPERIMENT:-norman_combo}"
DATA_DIR="${DATA_DIR:-data}"
ARTIFACT="${ARTIFACT:-norman2019}"
ENCODER="${ENCODER:-output/norman_stage_a/checkpoints/encoder.pt}"
if [ "$OUT_ROOT" != "output" ]; then
  mkdir -p "$OUT_ROOT"
  ln -sfn "$OUT_ROOT" output
fi
mkdir -p "output/$EXP/checkpoints" "output/$EXP/logs"
# Reused encoder must live in the experiment's checkpoints dir for 06's loader.
cp -f "$ENCODER" "output/$EXP/checkpoints/encoder.pt"
exec > >(tee -a "output/$EXP/logs/combo_generalization.log") 2>&1

: "${DEC_EPOCHS:=30}"
: "${FLOW_EPOCHS:=60}"
: "${VAE_EPOCHS:=60}"
: "${COMPOSE:=additive}"
: "${EVAL_N:=200}"
: "${TOP_K:=20}"

echo "[run_combo] exp=$EXP data=$DATA_DIR artifact=$ARTIFACT encoder=$ENCODER compose=$COMPOSE"
echo "[run_combo] dec_epochs=$DEC_EPOCHS flow_epochs=$FLOW_EPOCHS vae_epochs=$VAE_EPOCHS eval_n=$EVAL_N"

if [ ! -f "$DATA_DIR/$ARTIFACT/manifest.json" ]; then
  echo "[run_combo] ERROR: cache absent at $DATA_DIR/$ARTIFACT — stage it via file_mounts (no pertpy on a training pod)." >&2
  exit 2
fi
echo "[run_combo] cache present — clean base env, no pertpy install"

A=(--experiment "$EXP" --data-dir "$DATA_DIR" --artifact "$ARTIFACT")
ENC=(--encoder "output/$EXP/checkpoints/encoder.pt")
status=0
stage() {  # stage <fatal> <label> <cmd...>
  local fatal="$1" label="$2"; shift 2
  echo "=================================================================="
  echo "[run_combo] >>> $label"
  if "$@"; then echo "[run_combo] <<< $label OK"; else
    echo "[run_combo] !!! $label FAILED (exit $?)"; status=1
    [ "$fatal" = "1" ] && { echo "[run_combo] fatal — stopping; partial artifacts remain."; exit 1; }
  fi
}

# --- The method on held-out combos (gene-compositional condition) ---
stage 1 "Stage C: 03 count decoder (combo)" \
  python examples/perturbation_response/03_train_count_decoder.py "${A[@]}" "${ENC[@]}" \
    --split combo --epochs "$DEC_EPOCHS"
stage 1 "Stage B: 04 conditional flow (combo, geneset/$COMPOSE)" \
  python examples/perturbation_response/04_train_cond_flow.py "${A[@]}" "${ENC[@]}" \
    --split combo --cond-type geneset --compose "$COMPOSE" --epochs "$FLOW_EPOCHS"
stage 1 "Eval: 06 effect size (combo, held-out combos)" \
  python examples/perturbation_response/06_eval_effect_size.py "${A[@]}" \
    --split combo --n "$EVAL_N" --top-k "$TOP_K"

# --- The from-scratch baseline (no JEPA, no flow) ---
stage 1 "Baseline: 08 NB-VAE (combo)" \
  python examples/perturbation_response/08_train_cvae_baseline.py "${A[@]}" \
    --split combo --compose "$COMPOSE" --epochs "$VAE_EPOCHS"
stage 1 "Baseline eval: 09 NB-VAE effect size (combo)" \
  python examples/perturbation_response/09_eval_cvae_baseline.py "${A[@]}" \
    --split combo --n "$EVAL_N" --top-k "$TOP_K"

echo "[run_combo] done (status=$status). reports: effect_size.json (flow) + effect_size_cvae.json (baseline)"
exit "$status"
