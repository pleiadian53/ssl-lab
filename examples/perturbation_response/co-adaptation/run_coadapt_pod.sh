#!/usr/bin/env bash
# Round 12 — co-adaptation Phase 1: does training the encoder and the decoder together raise the
# stack's readout ceiling?
#
# Arms, three seeds each, all with the corrected likelihood (library size over the modeled genes):
#   dec            decoder only on the frozen encoder: round 11's norman_r11_dec_hvg_s{s} (same
#                  trainer, same recipe and flags), re-read here
#   joint          encoder (from norman_stage_a) and a fresh decoder trained together
#   joint_anchor   joint, while maintaining the JEPA predictive objective (anchor weight by rule)
# Each moved encoder gets its own transport flow, retrained with round 1's recipe (the flow is not
# co-trained). The protocol is in 26_score_round12.py's docstring.
#
# Usage (on the pod, from the repo root):
#   bash examples/perturbation_response/co-adaptation/run_coadapt_pod.sh
#   SEEDS=0 EPOCHS=1 FLOW_EPOCHS=1 LIMIT=2000 REQUIRE_CUDA=0 DEV19=cpu bash ...   # smoke

set -uo pipefail

VOL="${VOL:-/runpod-volume/ssl-lab}"
DATA="${DATA:-$VOL/data}"
OUT="${OUT:-$VOL/output}"
ENC="${ENC:-$OUT/norman_stage_a/checkpoints/encoder.pt}"
REF="${REF:-$OUT/norman_r11_dec_hvg_s0/checkpoints/count_decoder.pt}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-30}"
FLOW_EPOCHS="${FLOW_EPOCHS:-60}"
PARALLEL="${PARALLEL:-3}"
LIMIT="${LIMIT:-}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
DEV19="${DEV19:-cuda}"
RIDGE_MAX="${RIDGE_MAX:-}"          # smoke only: cap the ridge's training cells
DE_MD5="${DE_MD5:-9844c2c25d2903872e1c0a92c722f105}"
LOGS="$OUT/round12_logs"
mkdir -p "$LOGS"
limit_flag=(); [ -n "$LIMIT" ] && limit_flag=(--limit "$LIMIT")
ridge_flag=(); [ -n "$RIDGE_MAX" ] && ridge_flag=(--ridge-max-cells "$RIDGE_MAX")
EP=examples/perturbation_response

fail=0
if [ "$REQUIRE_CUDA" = "1" ]; then
  python -c "import torch; assert torch.cuda.is_available(); print('cuda:', torch.cuda.get_device_name(0))" || { echo "FATAL: no CUDA"; fail=1; }
fi
m=$(md5sum "$DATA/norman2019/de_genes.json" 2>/dev/null | cut -d' ' -f1); [ "$m" = "$DE_MD5" ] || { echo "FATAL: de_genes md5"; fail=1; }
m=$(md5sum "$ENC" 2>/dev/null | cut -d' ' -f1); [ "${m:0:8}" = "38f51067" ] || { echo "FATAL: encoder md5 ${m:0:8}"; fail=1; }
[ -f "$REF" ] || { echo "FATAL: missing reference decoder $REF"; fail=1; }
for s in $SEEDS; do
  [ -f "$OUT/norman_r11_dec_hvg_s$s/checkpoints/count_decoder.pt" ] || { echo "FATAL: missing dec arm s$s"; fail=1; }
  [ -f "$OUT/norman_ceiling_r11_hvg_s$s/reports/ceiling.json" ] || { echo "FATAL: missing round-11 ceiling s$s"; fail=1; }
  [ -f "$OUT/norman_sweep_control_s$s/checkpoints/cond_flow.pt" ] || { echo "FATAL: missing flow s$s"; fail=1; }
done
[ "$fail" -eq 1 ] && { echo "preflight FAILED; nothing was run."; exit 1; }
echo "preflight OK"

run_parallel() {   # runs the given commands, PARALLEL at a time
  local running=0
  for job in "$@"; do
    eval "$job" &
    running=$((running + 1)); if [ "$running" -ge "$PARALLEL" ]; then wait -n; running=$((running - 1)); fi
  done
  wait
}

# ---- 1. train joint and joint_anchor ------------------------------------------------------- #
train() {   # $1 = mode, $2 = seed
  python $EP/co-adaptation/02_coadapt_decoder.py --experiment "norman_r12_$1_s$2" --mode "$1" --encoder "$ENC" \
    --ref-decoder "$REF" --output-root "$OUT" --data-dir "$DATA" --split combo --epochs "$EPOCHS" --seed "$2" \
    "${limit_flag[@]}" > "$LOGS/train_$1_s$2.log" 2>&1 && echo "trained $1 s$2" || echo "TRAIN FAILED: $1 s$2"
}
jobs=(); for s in $SEEDS; do jobs+=("train joint $s" "train joint_anchor $s"); done
run_parallel "${jobs[@]}"

# ---- 2. readout ceilings, probes, and the flows ----------------------------------------------- #
ceiling() {   # $1 = arm, $2 = seed, $3 = encoder, $4 = decoder
  python $EP/14_ceiling_analysis.py --encoder "$3" --decoder "$4" --split combo --arms identity roundtrip linear \
    --save-profiles "${ridge_flag[@]}" --experiment "norman_ceiling_r12_$1_s$2" --output-root "$OUT" --data-dir "$DATA" \
    > "$LOGS/ceiling_$1_s$2.log" 2>&1 || echo "CEILING FAILED: $1 s$2"
  python $EP/02_probe_cell_encoder.py --encoder "$3" --experiment "norman_ceiling_r12_$1_s$2" --output-root "$OUT" \
    --data-dir "$DATA" > "$LOGS/probe_$1_s$2.log" 2>&1 || echo "PROBE FAILED: $1 s$2"
  echo "read $1 s$2"
}
pipeline() {   # $1 = arm, $2 = seed, $3 = encoder, $4 = decoder, $5 = existing flow (or "train")
  local exp="norman_r12_flow_$1_s$2"
  mkdir -p "$OUT/$exp/checkpoints"
  if [ "$5" = train ]; then
    python $EP/04_train_cond_flow.py --experiment "$exp" --encoder "$3" --output-root "$OUT" --data-dir "$DATA" \
      --split combo --cond-type geneset --flow-base control --coupling independent --epochs "$FLOW_EPOCHS" \
      --seed "$2" "${limit_flag[@]}" > "$LOGS/flow_$1_s$2.log" 2>&1 || { echo "FLOW FAILED: $1 s$2"; return; }
  else
    cp "$5" "$OUT/$exp/checkpoints/cond_flow.pt"
  fi
  cp "$4" "$OUT/$exp/checkpoints/count_decoder.pt"
  python $EP/06_eval_effect_size.py --experiment "$exp" --split combo --output-root "$OUT" --data-dir "$DATA" \
    > "$LOGS/eff_$exp.log" 2>&1 || echo "EFFECT FAILED: $exp"
  python $EP/19_score_interaction.py --experiment "$exp" --kind flow --output-root "$OUT" --data-dir "$DATA" \
    --device "$DEV19" > "$LOGS/int_$exp.log" 2>&1 || echo "19 FAILED: $exp"
  python $EP/10_eval_calibration.py --experiment "$exp" --model flow --split combo --output-root "$OUT" \
    --data-dir "$DATA" > "$LOGS/pop_$exp.log" 2>&1 || echo "POP FAILED: $exp"
  echo "pipeline $1 s$2"
}
jobs=()
for s in $SEEDS; do
  D="$OUT/norman_r11_dec_hvg_s$s/checkpoints/count_decoder.pt"
  jobs+=("ceiling dec $s $ENC $D" "pipeline dec $s $ENC $D $OUT/norman_sweep_control_s$s/checkpoints/cond_flow.pt")
  for mode in joint joint_anchor; do
    E="$OUT/norman_r12_${mode}_s$s/checkpoints/encoder.pt"; D2="$OUT/norman_r12_${mode}_s$s/checkpoints/count_decoder.pt"
    [ -f "$E" ] || { echo "skip $mode s$s (no encoder)"; continue; }
    jobs+=("ceiling $mode $s $E $D2" "pipeline $mode $s $E $D2 train")
  done
done
run_parallel "${jobs[@]}"
echo "done. fetch narrowly: reports of norman_r12_* and norman_ceiling_r12_*, the encoder.pt of norman_r12_{joint,joint_anchor}_s*, and $LOGS"
