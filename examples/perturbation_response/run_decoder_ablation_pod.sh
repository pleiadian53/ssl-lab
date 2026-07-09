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

OUT_ROOT="${1:-output}"
DATA_DIR="${DATA_DIR:-data}"
ARTIFACT="${ARTIFACT:-norman2019}"
ENCODER="${ENCODER:-output/norman_flow_control/checkpoints/encoder.pt}"
FLOW="${FLOW:-output/norman_flow_control/checkpoints/cond_flow.pt}"
if [ "$OUT_ROOT" != "output" ]; then mkdir -p "$OUT_ROOT"; ln -sfn "$OUT_ROOT" output; fi

: "${DEC_EPOCHS:=30}"
: "${ANCHOR_WEIGHT:=0.1}"
: "${EVAL_N:=200}"
: "${TOP_K:=20}"
: "${SEED:=0}"

if [ ! -f "$DATA_DIR/$ARTIFACT/manifest.json" ]; then
  echo "[decoder_ablation] ERROR: cache absent at $DATA_DIR/$ARTIFACT -- stage it via file_mounts." >&2
  exit 2
fi
[ -f "$FLOW" ]    || { echo "[decoder_ablation] ERROR: transport flow absent at $FLOW" >&2; exit 2; }
[ -f "$ENCODER" ] || { echo "[decoder_ablation] ERROR: encoder absent at $ENCODER" >&2; exit 2; }
echo "[decoder_ablation] cache+flow+encoder present; dec_epochs=$DEC_EPOCHS anchor=$ANCHOR_WEIGHT seed=$SEED"

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
