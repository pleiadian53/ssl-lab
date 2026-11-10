#!/usr/bin/env bash
# Re-baseline after round 11: every published model, read under the corrected conventions.
#
# Round 11 found the count decoder's offset came from the library size in its likelihood, and that
# the benchmark's sampled populations were normalized differently from the data. Both corrections
# are now the defaults. This run reads every model on the published scoreboard under them; it tests
# no hypothesis and makes no verdict (25_rebaseline.py writes the tables).
#
#   flow family   Gaussian flow, transport flow, transport + OT, deterministic and stochastic action
#                 operators (round-1 and round-3 runs), each seed read through round 11's
#                 corrected decoder of the same seed (norman_r11_dec_hvg_s{s}); latents unchanged
#   cVAE          conditioned (round 11's norman_r11_vae_hvg_s{s}) and condition-blind (retrained
#                 here with the corrected likelihood, norman_rb_vae_blind_s{s})
#   ladder        readout ceiling and mean-latent rung through each corrected decoder
#
# Population quality and calibration use the defaults: sampled cells normalized like the data, at
# the median real depth over the modeled genes. Variance decompositions (11) on seed 0.
#
# Usage (on the pod, from the repo root):
#   bash examples/perturbation_response/run_rebaseline_pod.sh

set -uo pipefail

VOL="${VOL:-/runpod-volume/ssl-lab}"
DATA="${DATA:-$VOL/data}"
OUT="${OUT:-$VOL/output}"
ENC="${ENC:-$OUT/norman_stage_a/checkpoints/encoder.pt}"
SEEDS="${SEEDS:-0 1 2}"
VAE_EPOCHS="${VAE_EPOCHS:-60}"
PARALLEL="${PARALLEL:-3}"
LIMIT="${LIMIT:-}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
DEV19="${DEV19:-cuda}"
LOGS="$OUT/rebaseline_logs"
mkdir -p "$LOGS"
limit_flag=(); [ -n "$LIMIT" ] && limit_flag=(--limit "$LIMIT")

fail=0
if [ "$REQUIRE_CUDA" = "1" ]; then
  python -c "import torch; assert torch.cuda.is_available(); print('cuda:', torch.cuda.get_device_name(0))" || { echo "FATAL: no CUDA"; fail=1; }
fi
m=$(md5sum "$ENC" 2>/dev/null | cut -d' ' -f1); [ "${m:0:8}" = "38f51067" ] || { echo "FATAL: encoder md5 ${m:0:8}"; fail=1; }
for s in $SEEDS; do
  for f in norman_r11_dec_hvg_s$s/checkpoints/count_decoder.pt norman_r11_vae_hvg_s$s/checkpoints/cvae_baseline.pt \
           norman_sweep_gaussian_s$s/checkpoints/cond_flow.pt norman_sweep_control_s$s/checkpoints/cond_flow.pt \
           norman_sweep_control_ot_s$s/checkpoints/cond_flow.pt norman_operator_s$s/checkpoints/operator.pt \
           norman_operator_stoch_s$s/checkpoints/operator.pt; do
    [ -f "$OUT/$f" ] || { echo "FATAL: missing $f"; fail=1; }
  done
done
[ "$fail" -eq 1 ] && { echo "preflight FAILED; nothing was run."; exit 1; }
echo "preflight OK"

# ---- 1. train the condition-blind cVAE with the corrected likelihood ------------------------- #
running=0
for s in $SEEDS; do
  ( python examples/perturbation_response/08_train_cvae_baseline.py --experiment "norman_rb_vae_blind_s$s" \
      --output-root "$OUT" --data-dir "$DATA" --split combo --epochs "$VAE_EPOCHS" --seed "$s" "${limit_flag[@]}" \
      > "$LOGS/train_norman_rb_vae_blind_s$s.log" 2>&1 && echo "trained vae_blind s$s" || echo "TRAIN FAILED: vae_blind s$s" ) &
  running=$((running + 1)); if [ "$running" -ge "$PARALLEL" ]; then wait -n; running=$((running - 1)); fi
done
wait

# ---- 2. read every model ---------------------------------------------------------------------- #
read_stageb() {   # $1 = arm, $2 = source experiment, $3 = stage-b (flow|operator), $4 = seed
  local arm="$1" src="$2" sb="$3" s="$4" exp="norman_rb_$1_s$4"
  local ck=cond_flow.pt; [ "$sb" = operator ] && ck=operator.pt
  mkdir -p "$OUT/$exp/checkpoints"
  cp "$OUT/$src/checkpoints/$ck" "$OUT/$exp/checkpoints/"
  cp "$OUT/norman_r11_dec_hvg_s$s/checkpoints/count_decoder.pt" "$OUT/$exp/checkpoints/"
  python examples/perturbation_response/06_eval_effect_size.py --experiment "$exp" --stage-b "$sb" --split combo \
    --output-root "$OUT" --data-dir "$DATA" > "$LOGS/eff_$exp.log" 2>&1 || echo "EFFECT FAILED: $exp"
  python examples/perturbation_response/19_score_interaction.py --experiment "$exp" --kind "$sb" --output-root "$OUT" \
    --data-dir "$DATA" --device "$DEV19" > "$LOGS/int_$exp.log" 2>&1 || echo "19 FAILED: $exp"
  python examples/perturbation_response/10_eval_calibration.py --experiment "$exp" --model flow --stage-b "$sb" \
    --split combo --output-root "$OUT" --data-dir "$DATA" > "$LOGS/pop_$exp.log" 2>&1 || echo "POP FAILED: $exp"
  if [ "$s" = 0 ]; then
    python examples/perturbation_response/11_diagnose_variance.py --experiment "$exp" --model flow --stage-b "$sb" \
      --split combo --output-root "$OUT" --data-dir "$DATA" > "$LOGS/var_$exp.log" 2>&1 || echo "VAR FAILED: $exp"
  fi
  echo "read $exp"
}
read_vae() {   # $1 = experiment, $2 = seed, $3 = grade effect size too (1/0)
  local exp="$1" s="$2"
  if [ "$3" = 1 ]; then
    python examples/perturbation_response/09_eval_cvae_baseline.py --experiment "$exp" --split combo --output-root "$OUT" \
      --data-dir "$DATA" > "$LOGS/eff_$exp.log" 2>&1 || echo "EFFECT FAILED: $exp"
    python examples/perturbation_response/19_score_interaction.py --experiment "$exp" --kind vae --output-root "$OUT" \
      --data-dir "$DATA" --device "$DEV19" > "$LOGS/int_$exp.log" 2>&1 || echo "19 FAILED: $exp"
  fi
  python examples/perturbation_response/10_eval_calibration.py --experiment "$exp" --model vae --split combo \
    --output-root "$OUT" --data-dir "$DATA" > "$LOGS/pop_$exp.log" 2>&1 || echo "POP FAILED: $exp"
  if [ "$s" = 0 ]; then
    python examples/perturbation_response/11_diagnose_variance.py --experiment "$exp" --model vae --split combo \
      --output-root "$OUT" --data-dir "$DATA" > "$LOGS/var_$exp.log" 2>&1 || echo "VAR FAILED: $exp"
  fi
  echo "read $exp"
}
ladder() {   # $1 = seed
  python examples/perturbation_response/14_ceiling_analysis.py --encoder "$ENC" \
    --decoder "$OUT/norman_r11_dec_hvg_s$1/checkpoints/count_decoder.pt" --split combo \
    --arms identity roundtrip latent_mean --save-profiles --experiment "norman_rb_ceiling_s$1" \
    --output-root "$OUT" --data-dir "$DATA" > "$LOGS/ceiling_s$1.log" 2>&1 && echo "ladder s$1" || echo "CEILING FAILED: s$1"
}
jobs_list=()
for s in $SEEDS; do
  jobs_list+=("read_stageb gaussian norman_sweep_gaussian_s$s flow $s" "read_stageb transport norman_sweep_control_s$s flow $s"
              "read_stageb ot norman_sweep_control_ot_s$s flow $s" "read_stageb operator norman_operator_s$s operator $s"
              "read_stageb operator_stoch norman_operator_stoch_s$s operator $s"
              "read_vae norman_r11_vae_hvg_s$s $s 0" "read_vae norman_rb_vae_blind_s$s $s 1" "ladder $s")
done
running=0
for job in "${jobs_list[@]}"; do
  $job &
  running=$((running + 1)); if [ "$running" -ge "$PARALLEL" ]; then wait -n; running=$((running - 1)); fi
done
wait
echo "done. fetch narrowly: reports of norman_rb_*, the new reports of norman_r11_vae_hvg_s*, and $LOGS"
