#!/usr/bin/env bash
# Round 11 — decoder calibration: the library size in the NB likelihood.
#
# One change: the total the NB likelihood multiplies the rate profile by, all measured genes
# (the published convention) or the 5,000 modeled genes. Everything else is identical.
#
#   dec_all  / dec_hvg   decoder only on the frozen encoder (co-adaptation/02 --mode dec)
#   vae_all  / vae_hvg   conditioned NB-VAE (vae_all = the existing round-6 runs; vae_hvg retrained)
#
# Flows are not retrained: their latents do not change, so the transport flows
# norman_sweep_control_s{s} are read through each decoder. Every sampled population is normalized
# like the data (each sampled cell by its own total over the modeled genes) and sampled at the
# same two depths in every arm. The protocol is in 24_score_round11.py's docstring.
#
# Usage (on the pod, from the repo root):
#   bash examples/perturbation_response/run_round11_pod.sh
#   SEEDS=0 EPOCHS=1 VAE_EPOCHS=1 LIMIT=2000 bash examples/perturbation_response/run_round11_pod.sh   # smoke

set -uo pipefail

VOL="${VOL:-/runpod-volume/ssl-lab}"
DATA="${DATA:-$VOL/data}"
OUT="${OUT:-$VOL/output}"
ENC="${ENC:-$OUT/norman_stage_a/checkpoints/encoder.pt}"
DEC0="${DEC0:-$OUT/norman_flow_control/checkpoints/count_decoder.pt}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-30}"
VAE_EPOCHS="${VAE_EPOCHS:-60}"
PARALLEL="${PARALLEL:-3}"
LIMIT="${LIMIT:-}"
DEPTHS="${DEPTHS:-4832 13921}"
DE_MD5="${DE_MD5:-9844c2c25d2903872e1c0a92c722f105}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"     # 0 only for a local CPU smoke
DEV19="${DEV19:-cuda}"               # device for 19_score_interaction.py
LOGS="$OUT/round11_logs"
mkdir -p "$LOGS"
limit_flag=(); [ -n "$LIMIT" ] && limit_flag=(--limit "$LIMIT")

echo "=============================================================="
echo " ROUND 11 — library size in the NB likelihood"
echo " seeds: $SEEDS  decoder epochs: $EPOCHS  cVAE epochs: $VAE_EPOCHS  depths: $DEPTHS"
echo "=============================================================="

fail=0
if [ "$REQUIRE_CUDA" = "1" ]; then
  python -c "import torch; assert torch.cuda.is_available(); print('cuda:', torch.cuda.get_device_name(0))" || { echo "FATAL: no CUDA"; fail=1; }
fi
m=$(md5sum "$DATA/norman2019/de_genes.json" 2>/dev/null | cut -d' ' -f1); [ "$m" = "$DE_MD5" ] || { echo "FATAL: de_genes md5 $m"; fail=1; }
m=$(md5sum "$ENC" 2>/dev/null | cut -d' ' -f1); [ "${m:0:8}" = "38f51067" ] || { echo "FATAL: encoder md5 ${m:0:8}"; fail=1; }
m=$(md5sum "$DEC0" 2>/dev/null | cut -d' ' -f1); [ "${m:0:8}" = "52a44337" ] || { echo "FATAL: decoder md5 ${m:0:8}"; fail=1; }
for s in $SEEDS; do
  [ -f "$OUT/norman_sweep_control_s$s/checkpoints/cond_flow.pt" ] || { echo "FATAL: missing flow s$s"; fail=1; }
  [ -f "$OUT/norman_vae_cond_s$s/checkpoints/cvae_baseline.pt" ] || { echo "FATAL: missing cVAE s$s"; fail=1; }
done
[ "$fail" -eq 1 ] && { echo "preflight FAILED; nothing was run."; exit 1; }
echo "preflight OK"

# ---- gate: the existing decoder's readout ceiling reproduces round 10 / the stored ceiling ---- #
python examples/perturbation_response/14_ceiling_analysis.py --encoder "$ENC" --decoder "$DEC0" --split combo \
  --arms identity roundtrip --experiment norman_ceiling_r11_gate --output-root "$OUT" --data-dir "$DATA" \
  > "$LOGS/gate.log" 2>&1 || { echo "GATE RUN FAILED"; exit 1; }
python - "$OUT/norman_ceiling_r11_gate/reports/ceiling.json" <<'PY' || { echo "GATE FAILED; stopping before training."; exit 1; }
import json, sys
v = json.load(open(sys.argv[1]))["arms"]["roundtrip"]["mean_delta_r"]
print(f"existing decoder roundtrip {v:.5f} (stored 0.67916)")
sys.exit(0 if abs(v - 0.67916) <= 1e-4 else 1)
PY

# ---- 1. train: 6 decoders and 3 cVAEs ------------------------------------------------------ #
train() {   # $1 = kind (dec|vae), $2 = library size, $3 = seed
  if [ "$1" = dec ]; then
    exp="norman_r11_dec_$2_s$3"
    python examples/perturbation_response/co-adaptation/02_coadapt_decoder.py --experiment "$exp" --mode dec \
      --library-size "$2" --encoder "$ENC" --output-root "$OUT" --data-dir "$DATA" --split combo \
      --epochs "$EPOCHS" --seed "$3" "${limit_flag[@]}" > "$LOGS/train_$exp.log" 2>&1
  else
    exp="norman_r11_vae_$2_s$3"
    python examples/perturbation_response/08_train_cvae_baseline.py --experiment "$exp" --output-root "$OUT" \
      --data-dir "$DATA" --split combo --epochs "$VAE_EPOCHS" --seed "$3" --encoder-cond --library-size "$2" \
      "${limit_flag[@]}" > "$LOGS/train_$exp.log" 2>&1
  fi && echo "trained $exp" || echo "TRAIN FAILED: $exp"
}
running=0
for s in $SEEDS; do
  for job in "dec all" "dec hvg" "vae hvg"; do
    train $job "$s" &
    running=$((running + 1)); if [ "$running" -ge "$PARALLEL" ]; then wait -n; running=$((running - 1)); fi
  done
done
wait

# ---- 2. read: ceilings, flows through each decoder, cVAE grading ---------------------------- #
pop() {   # $1 = experiment, $2 = model, extra args...; published + both matched depths
  local exp="$1" model="$2"; shift 2
  python examples/perturbation_response/10_eval_calibration.py --experiment "$exp" --model "$model" --split combo \
    --output-root "$OUT" --data-dir "$DATA" "$@" > "$LOGS/pop_${exp}_published.log" 2>&1 || echo "POP FAILED: $exp published"
  for d in $DEPTHS; do
    python examples/perturbation_response/10_eval_calibration.py --experiment "$exp" --model "$model" --split combo \
      --output-root "$OUT" --data-dir "$DATA" --normalize data --depth "$d" \
      --report-name "calibration_${model}_d${d}_data" "$@" > "$LOGS/pop_${exp}_d${d}.log" 2>&1 || echo "POP FAILED: $exp d$d"
  done
}
read_dec() {   # $1 = library size, $2 = seed
  local dexp="norman_r11_dec_$1_s$2" fexp="norman_r11_flow_$1_s$2"
  local dec="$OUT/$dexp/checkpoints/count_decoder.pt"
  [ -f "$dec" ] || { echo "skip $dexp (no decoder)"; return; }
  python examples/perturbation_response/14_ceiling_analysis.py --encoder "$ENC" --decoder "$dec" --split combo \
    --arms identity roundtrip --save-profiles --experiment "norman_ceiling_r11_$1_s$2" --output-root "$OUT" \
    --data-dir "$DATA" > "$LOGS/ceiling_$1_s$2.log" 2>&1 || echo "CEILING FAILED: $1 s$2"
  mkdir -p "$OUT/$fexp/checkpoints"
  cp "$OUT/norman_sweep_control_s$2/checkpoints/cond_flow.pt" "$OUT/$fexp/checkpoints/"
  cp "$dec" "$OUT/$fexp/checkpoints/count_decoder.pt"
  python examples/perturbation_response/06_eval_effect_size.py --experiment "$fexp" --split combo --output-root "$OUT" \
    --data-dir "$DATA" > "$LOGS/eff_$fexp.log" 2>&1 || echo "EFFECT FAILED: $fexp"
  python examples/perturbation_response/19_score_interaction.py --experiment "$fexp" --kind flow --output-root "$OUT" \
    --data-dir "$DATA" --device "$DEV19" > "$LOGS/int_$fexp.log" 2>&1 || echo "19 FAILED: $fexp"
  pop "$fexp" flow
  echo "read $dexp"
}
read_vae() {   # $1 = experiment
  local exp="$1"
  [ -f "$OUT/$exp/checkpoints/cvae_baseline.pt" ] || { echo "skip $exp (no checkpoint)"; return; }
  if [ "${exp#norman_r11_}" != "$exp" ]; then
    python examples/perturbation_response/09_eval_cvae_baseline.py --experiment "$exp" --split combo --output-root "$OUT" \
      --data-dir "$DATA" > "$LOGS/eff_$exp.log" 2>&1 || echo "EFFECT FAILED: $exp"
    python examples/perturbation_response/19_score_interaction.py --experiment "$exp" --kind vae --output-root "$OUT" \
      --data-dir "$DATA" --device "$DEV19" > "$LOGS/int_$exp.log" 2>&1 || echo "19 FAILED: $exp"
  fi
  for d in $DEPTHS; do
    python examples/perturbation_response/10_eval_calibration.py --experiment "$exp" --model vae --split combo \
      --output-root "$OUT" --data-dir "$DATA" --normalize data --depth "$d" \
      --report-name "calibration_vae_d${d}_data" > "$LOGS/pop_${exp}_d${d}.log" 2>&1 || echo "POP FAILED: $exp d$d"
  done
  echo "read $exp"
}
running=0
for s in $SEEDS; do
  for job in "read_dec all $s" "read_dec hvg $s" "read_vae norman_r11_vae_hvg_s$s" "read_vae norman_vae_cond_s$s"; do
    $job &
    running=$((running + 1)); if [ "$running" -ge "$PARALLEL" ]; then wait -n; running=$((running - 1)); fi
  done
done
wait
echo "done. fetch narrowly (reports + decoder/cVAE checkpoints of norman_r11_*, norman_ceiling_r11_*,"
echo "the matched-depth reports of norman_vae_cond_s*, and $LOGS); then run 24_score_round11.py locally."
