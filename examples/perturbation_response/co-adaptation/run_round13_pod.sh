#!/usr/bin/env bash
# Round 13 — does the JEPA pretraining contribute to round 12's gain? Joint training from a random start.
#
# Arms, three seeds each, all --mode joint with the corrected likelihood (library size over the
# modeled genes); the protocol is in 27_score_round13.py's docstring:
#   joint         round 12's pretrained start, norman_r12_joint_s{s}: re-read, not retrained
#   random_lr4    --init random, encoder learning rate 1e-4 (round 12's recipe exactly)
#   random_lr3    --init random, encoder learning rate 1e-3 (the decoder's)
# Each random encoder gets its readout ceiling, its perturbation probe, and a transport flow
# retrained with round 1's recipe, read out through its own decoder.
#
# One pod job, in order:
#   preflight   every input this round reads is on the volume; CUDA; md5s
#   0. smoke    the whole round at tiny settings (seed 0, one epoch, 2,000 cells) in a separate
#               output directory built from copies of the inputs, then 27 --smoke; any failure stops
#   1. full     train, read, retrain flows, grade
#   2. compare  27_score_round13.py -> $OUT/round13_compare.json
#
# Usage (on the pod, from the repo root):
#   bash examples/perturbation_response/co-adaptation/run_round13_pod.sh

set -uo pipefail

VOL="${VOL:-/runpod-volume/ssl-lab}"
DATA="${DATA:-$VOL/data}"
OUT="${OUT:-$VOL/output}"
EP=examples/perturbation_response
DE_MD5="${DE_MD5:-9844c2c25d2903872e1c0a92c722f105}"
SMOKE="$VOL/smoke_r13/output"

# ---- preflight ----------------------------------------------------------------------------- #
fail=0
python -c "import torch; assert torch.cuda.is_available(); print('cuda:', torch.cuda.get_device_name(0))" || { echo "FATAL: no CUDA"; fail=1; }
m=$(md5sum "$DATA/norman2019/de_genes.json" 2>/dev/null | cut -d' ' -f1); [ "$m" = "$DE_MD5" ] || { echo "FATAL: de_genes md5"; fail=1; }
m=$(md5sum "$OUT/norman_stage_a/checkpoints/encoder.pt" 2>/dev/null | cut -d' ' -f1); [ "${m:0:8}" = "38f51067" ] || { echo "FATAL: encoder md5 ${m:0:8}"; fail=1; }
need=(norman_additive/reports/effect_size_additive.json norman_additive/reports/profiles_additive.npz)
for s in 0 1 2; do
  need+=("norman_r12_joint_s$s/checkpoints/encoder.pt" "norman_r12_joint_s$s/checkpoints/count_decoder.pt"
         "norman_r12_joint_s$s/reports/coadapt.json"
         "norman_ceiling_r12_joint_s$s/reports/ceiling.json" "norman_ceiling_r12_joint_s$s/reports/ceiling_profiles.npz"
         "norman_ceiling_r12_joint_s$s/reports/stage_a_probe.json"
         "norman_ceiling_r12_dec_s$s/reports/ceiling.json" "norman_ceiling_r12_dec_s$s/reports/ceiling_profiles.npz"
         "norman_r12_flow_joint_s$s/reports/effect_size.json" "norman_r12_flow_joint_s$s/reports/profiles_flow.npz"
         "norman_r12_flow_joint_s$s/reports/calibration_flow.json"
         "norman_r12_flow_dec_s$s/reports/effect_size.json" "norman_r12_flow_dec_s$s/reports/profiles_flow.npz"
         "norman_r11_vae_hvg_s$s/reports/effect_size_cvae.json" "norman_r11_vae_hvg_s$s/reports/profiles_vae.npz")
done
for f in "${need[@]}"; do [ -f "$OUT/$f" ] || { echo "FATAL: missing $OUT/$f"; fail=1; }; done
[ "$fail" -eq 1 ] && { echo "PREFLIGHT FAILED; nothing was run."; exit 1; }
echo "preflight OK"

run_parallel() {   # runs the given commands, PARALLEL at a time
  local running=0
  for job in "$@"; do
    eval "$job" &
    running=$((running + 1)); if [ "$running" -ge "$PARALLEL" ]; then wait -n; running=$((running - 1)); fi
  done
  wait
}

train() {   # $1 = arm, $2 = encoder lr, $3 = seed
  python $EP/co-adaptation/02_coadapt_decoder.py --experiment "norman_r13_$1_s$3" --mode joint --init random \
    --encoder "$O/norman_stage_a/checkpoints/encoder.pt" --encoder-lr "$2" --output-root "$O" --data-dir "$DATA" \
    --split combo --epochs "$EPOCHS" --seed "$3" "${limit_flag[@]}" > "$LOGS/train_$1_s$3.log" 2>&1 \
    && echo "trained $1 s$3" || echo "TRAIN FAILED: $1 s$3"
}
ceiling() {   # $1 = arm, $2 = seed
  local E="$O/norman_r13_$1_s$2/checkpoints/encoder.pt" D="$O/norman_r13_$1_s$2/checkpoints/count_decoder.pt"
  python $EP/14_ceiling_analysis.py --encoder "$E" --decoder "$D" --split combo --arms identity roundtrip linear \
    --save-profiles "${ridge_flag[@]}" --experiment "norman_ceiling_r13_$1_s$2" --output-root "$O" --data-dir "$DATA" \
    > "$LOGS/ceiling_$1_s$2.log" 2>&1 || echo "CEILING FAILED: $1 s$2"
  python $EP/02_probe_cell_encoder.py --encoder "$E" --experiment "norman_ceiling_r13_$1_s$2" --output-root "$O" \
    --data-dir "$DATA" > "$LOGS/probe_$1_s$2.log" 2>&1 || echo "PROBE FAILED: $1 s$2"
  echo "read $1 s$2"
}
pipeline() {   # $1 = arm, $2 = seed
  local E="$O/norman_r13_$1_s$2/checkpoints/encoder.pt" exp="norman_r13_flow_$1_s$2"
  mkdir -p "$O/$exp/checkpoints"
  python $EP/04_train_cond_flow.py --experiment "$exp" --encoder "$E" --output-root "$O" --data-dir "$DATA" \
    --split combo --cond-type geneset --flow-base control --coupling independent --epochs "$FLOW_EPOCHS" \
    --seed "$2" "${limit_flag[@]}" > "$LOGS/flow_$1_s$2.log" 2>&1 || { echo "FLOW FAILED: $1 s$2"; return; }
  cp "$O/norman_r13_$1_s$2/checkpoints/count_decoder.pt" "$O/$exp/checkpoints/count_decoder.pt"
  python $EP/06_eval_effect_size.py --experiment "$exp" --split combo --output-root "$O" --data-dir "$DATA" \
    > "$LOGS/eff_$exp.log" 2>&1 || echo "EFFECT FAILED: $exp"
  python $EP/19_score_interaction.py --experiment "$exp" --kind flow --output-root "$O" --data-dir "$DATA" \
    --device cuda > "$LOGS/int_$exp.log" 2>&1 || echo "19 FAILED: $exp"
  python $EP/10_eval_calibration.py --experiment "$exp" --model flow --split combo --output-root "$O" \
    --data-dir "$DATA" > "$LOGS/pop_$exp.log" 2>&1 || echo "POP FAILED: $exp"
  echo "pipeline $1 s$2"
}

round13() {   # trains and reads every random arm under the current O, SEEDS, EPOCHS, ...
  LOGS="$O/round13_logs"; mkdir -p "$LOGS"
  limit_flag=(); [ -n "$LIMIT" ] && limit_flag=(--limit "$LIMIT")
  ridge_flag=(); [ -n "$RIDGE_MAX" ] && ridge_flag=(--ridge-max-cells "$RIDGE_MAX")
  local jobs=()
  for s in $SEEDS; do jobs+=("train random_lr4 1e-4 $s" "train random_lr3 1e-3 $s"); done
  run_parallel "${jobs[@]}"
  jobs=()
  for s in $SEEDS; do
    for a in random_lr4 random_lr3; do
      [ -f "$O/norman_r13_${a}_s$s/checkpoints/encoder.pt" ] || { echo "skip $a s$s (no encoder)"; continue; }
      jobs+=("ceiling $a $s" "pipeline $a $s")
    done
  done
  run_parallel "${jobs[@]}"
}

# ---- 0. smoke ------------------------------------------------------------------------------ #
echo "==== 0. smoke on the pod ===="
rm -rf "$VOL/smoke_r13"; mkdir -p "$SMOKE"
for d in norman_stage_a/checkpoints norman_additive/reports norman_r12_joint_s0/checkpoints norman_r12_joint_s0/reports \
         norman_ceiling_r12_joint_s0/reports norman_ceiling_r12_dec_s0/reports norman_r12_flow_joint_s0/reports \
         norman_r12_flow_dec_s0/reports norman_r11_vae_hvg_s0/reports; do
  mkdir -p "$SMOKE/$d"; cp -r "$OUT/$d/." "$SMOKE/$d/"
done
O="$SMOKE" SEEDS=0 EPOCHS=1 FLOW_EPOCHS=1 LIMIT=2000 RIDGE_MAX=3000 PARALLEL=4
round13 > "$VOL/smoke_r13/round13.out" 2>&1
python $EP/27_score_round13.py --output-root "$SMOKE" --data-dir "$DATA" --seeds 0 --smoke --out "$VOL/smoke_r13/r13.json" \
  > "$VOL/smoke_r13/compare.out" 2>&1 || echo "SMOKE: 27 FAILED"
bad=$(grep -lE "FAILED|Traceback" "$VOL/smoke_r13"/*.out "$SMOKE"/round13_logs/*.log 2>/dev/null)
if [ -n "$bad" ] || [ ! -f "$VOL/smoke_r13/r13.json" ]; then
  echo "SMOKE FAILED; nothing full was run. Problem files:"; echo "$bad"; exit 1
fi
echo "smoke passed"

# ---- 1. full ------------------------------------------------------------------------------- #
echo "==== 1. round 13 ===="
O="$OUT" SEEDS="0 1 2" EPOCHS=30 FLOW_EPOCHS=60 LIMIT="" RIDGE_MAX="" PARALLEL=3
round13

# ---- 2. compare ---------------------------------------------------------------------------- #
echo "==== 2. comparison ===="
python $EP/27_score_round13.py --output-root "$OUT" --data-dir "$DATA" --out "$OUT/round13_compare.json" \
  > "$OUT/round13_logs/compare.log" 2>&1 || echo "27 FAILED"
echo "done. fetch: $OUT/round13_compare.json $OUT/round13_logs and the JSON reports of norman_r13_* and norman_ceiling_r13_*"
