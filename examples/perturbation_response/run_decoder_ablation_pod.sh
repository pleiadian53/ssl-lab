#!/usr/bin/env bash
# Pod entrypoint: decoder ablation (chapter-8 readout levers B1/B2) on held-out combos.
#
# Reuses the SAME encoder AND the SAME transport flow (cond_flow.pt, flow_base=control)
# across all four arms, so the ONLY variable is the decoder config. Stage B (the flow) is
# trained on latents alone and is decoder-independent, so sharing it is fair. Each arm
# retrains only Stage C (03, the count decoder) on the combo split -- whose training side
# excludes the held-out combos, so no leakage -- then evals effect size (06, mean rate ->
# B1's axis) and calibration (10, sampled counts -> B2's axis). Compare against the NB-VAE
# reference already in norman_combo (no VAE retrain needed).
#
#   baseline    : bare softmax mean head + constant per-gene kappa   [current default]
#   anchmean    : B1 identity-anchored mean head            (--anchored-mean)
#   statedisp   : B2 state-aware, anchored dispersion       (--state-dispersion)
#   both        : B1 + B2
#
# Usage:
#   DATA_DIR=/cache ENCODER=/cache/encoder.pt FLOW=/cache/cond_flow.pt \
#     bash examples/perturbation_response/run_decoder_ablation_pod.sh [OUTPUT_ROOT]
#
# Knobs (env): DEC_EPOCHS (30), ANCHOR_WEIGHT (0.1), EVAL_N (200), TOP_K (20), SEED (0).
set -uo pipefail

# Inputs default to the persistent RunPod volume (survives pod teardown); override any via env.
VOL="${VOL:-/runpod-volume/ssl-lab}"
OUT_ROOT="${1:-$VOL/output}"
DATA_DIR="${DATA_DIR:-$VOL/data}"
ARTIFACT="${ARTIFACT:-norman2019}"
ENCODER="${ENCODER:-$VOL/output/norman_flow_control/checkpoints/encoder.pt}"
FLOW="${FLOW:-$VOL/output/norman_flow_control/checkpoints/cond_flow.pt}"

: "${DEC_EPOCHS:=30}"
: "${ANCHOR_WEIGHT:=0.1}"
: "${EVAL_N:=200}"
: "${TOP_K:=20}"
: "${SEED:=0}"

# --- Preflight: verify EVERY input on the volume before any training, with a specific fix ---
# --- per missing input, so a launch never half-runs and the fix is unambiguous.          ---
miss=0
if [ ! -f "$DATA_DIR/$ARTIFACT/manifest.json" ]; then
  echo "[preflight] MISSING cache       : $DATA_DIR/$ARTIFACT/manifest.json" >&2
  echo "            fix: build it -> python examples/perturbation_response/00_process_norman.py" >&2
  echo "                 stage it -> python examples/ops/ops_stage_data.py data/norman2019" >&2
  miss=1
fi
if [ ! -f "$ENCODER" ]; then
  echo "[preflight] MISSING encoder      : $ENCODER  (Stage A output)" >&2
  echo "            fix: if absent locally you skipped Stage A -> ops_run_pipeline.py --execute --gpu a40 -- \\" >&2
  echo "                     bash examples/perturbation_response/run_stage_a_pod.sh" >&2
  echo "                 then stage -> python examples/ops/ops_stage_data.py output/norman_flow_control/checkpoints" >&2
  miss=1
fi
if [ ! -f "$FLOW" ]; then
  echo "[preflight] MISSING transport flow: $FLOW  (Stage B output)" >&2
  echo "            fix: if absent locally you skipped Stage B -> ops_run_pipeline.py --execute --gpu a40 -- \\" >&2
  echo "                     bash examples/perturbation_response/run_flow_compare_pod.sh" >&2
  echo "                 then stage -> python examples/ops/ops_stage_data.py output/norman_flow_control/checkpoints" >&2
  miss=1
fi
if [ "$miss" -ne 0 ]; then
  echo "[preflight] Not launching the A/B (fix each missing input above, then re-launch)." >&2
  exit 2
fi
[ "$OUT_ROOT" != "output" ] && { mkdir -p "$OUT_ROOT"; ln -sfn "$OUT_ROOT" output; }
python -c 'import torch; print("[preflight] cuda:", torch.cuda.is_available())' 2>/dev/null || true
echo "[preflight] OK  cache=$DATA_DIR/$ARTIFACT  encoder+flow present  out=$OUT_ROOT  dec_epochs=$DEC_EPOCHS anchor=$ANCHOR_WEIGHT seed=$SEED"

run_arm() {  # run_arm <name> <extra 03 flags...>
  local name="$1"; shift
  local exp="norman_dec_$name"
  mkdir -p "output/$exp/checkpoints" "output/$exp/logs"
  cp -f "$ENCODER" "output/$exp/checkpoints/encoder.pt"     # for completeness; 03 uses --encoder
  cp -f "$FLOW"    "output/$exp/checkpoints/cond_flow.pt"   # shared transport flow, read by 06/10
  echo "=================================================================="
  echo "[decoder_ablation] >>> arm=$name  (exp=$exp)  03-flags: ${*:-none}"
  python examples/perturbation_response/03_train_count_decoder.py \
    --experiment "$exp" --data-dir "$DATA_DIR" --artifact "$ARTIFACT" \
    --split combo --encoder "$ENCODER" --epochs "$DEC_EPOCHS" --seed "$SEED" "$@" \
    || { echo "[decoder_ablation] !!! 03 ($name) FAILED"; return 1; }
  python examples/perturbation_response/06_eval_effect_size.py \
    --experiment "$exp" --data-dir "$DATA_DIR" --artifact "$ARTIFACT" \
    --split combo --n "$EVAL_N" --top-k "$TOP_K" --seed "$SEED" \
    || { echo "[decoder_ablation] !!! 06 ($name) FAILED"; return 1; }
  python examples/perturbation_response/10_eval_calibration.py \
    --experiment "$exp" --data-dir "$DATA_DIR" --artifact "$ARTIFACT" \
    --model flow --split combo --n "$EVAL_N" --top-k "$TOP_K" --seed "$SEED" \
    || { echo "[decoder_ablation] !!! 10 ($name) FAILED"; return 1; }
  echo "[decoder_ablation] <<< arm=$name done -> output/$exp/reports/{effect_size,calibration_flow}.json"
}

status=0
run_arm baseline                                                              || status=1
run_arm anchmean  --anchored-mean                                             || status=1
run_arm statedisp --state-dispersion --anchor-weight "$ANCHOR_WEIGHT"         || status=1
run_arm both      --anchored-mean --state-dispersion --anchor-weight "$ANCHOR_WEIGHT" || status=1

echo "=================================================================="
echo "[decoder_ablation] collating (arms + NB-VAE reference)..."
python - <<'PY'
import json
def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None
def fmt(x, p=3):
    return f"{x:.{p}f}" if isinstance(x, (int, float)) else "  -  "

rows = []
for a in ["baseline", "anchmean", "statedisp", "both"]:
    es = load(f"output/norman_dec_{a}/reports/effect_size.json") or {}
    cal = load(f"output/norman_dec_{a}/reports/calibration_flow.json") or {}
    rows.append((a, es.get("mean_delta_r"), cal.get("mean_coverage"), cal.get("mean_energy")))
ves = load("output/norman_combo/reports/effect_size_cvae.json") or {}
vcal = load("output/norman_combo/reports/calibration_vae.json") or {}
rows.append(("NB-VAE (ref)", ves.get("mean_delta_r"), vcal.get("mean_coverage"), vcal.get("mean_energy")))

print(f"\n{'arm':<14}{'delta_r ↑':>12}{'coverage':>12}{'energy ↓':>12}")
print("-" * 50)
for a, dr, cov, en in rows:
    print(f"{a:<14}{fmt(dr):>12}{fmt(cov):>12}{fmt(en, 4):>12}")
print("\nread: coverage nominal 0.80 (1.00 = over-dispersed); VAE energy ~0.032, delta_r ~0.633;")
print("      B1 (anchmean) targets delta_r, B2 (statedisp) targets coverage/energy.")
PY
echo "[decoder_ablation] done (status=$status)."
exit "$status"
