#!/usr/bin/env bash
# Pod entrypoint: ECG rhythm study v1 PILOT (repeat 0, validation only; design D25).
#
# HISTORICAL RECORD: this is how the pilot ran on 2026-10-11 under the base protocol
# (configs/study_v1.json, grid {3e-4, 1e-3}). It is not meant to be rerun: 02 now refuses
# to overwrite the saved subsets, and the learning-rate selection moved to
# run_ecg_lrsearch_pod.sh under amendment 1. Its artifacts (runs/pilot/, lr_selection.json)
# are preserved untouched.
#
#   02  draw, check and save the labelled-patient subsets (exits 2 on a core support failure)
#   03  span-masked JEPA pretraining, repeat 0 (training folds only)
#   04  learning-rate grid at budget 10% for scratch / finetune / probe
#   05  choose each arm's rate on validation, then freeze it
#   04  scratch / finetune / probe at budget 100% with the chosen rates
#   04  shuffled-label sanity check (scratch, 10%)
#   06  paired patient-cluster bootstrap on VALIDATION (machinery check, not results)
#   07  pilot report + full-grid runtime/cost estimate
#
# No test predictions are produced. Inputs default to the persistent RunPod volume.
#
# Usage (on the pod, after staging the cache: python examples/ops/ops_stage_data.py data/ptbxl100):
#   bash examples/ecg_rhythm/run_ecg_pilot_pod.sh [OUTPUT_ROOT]
# Local plumbing check (CPU, tiny settings, seconds to minutes):
#   SMOKE=1 VOL=. DATA_DIR=data bash examples/ecg_rhythm/run_ecg_pilot_pod.sh /tmp/ecg_smoke
#
# Env: VOL (/runpod-volume/ssl-lab), DATA_DIR ($VOL/data), EXPERIMENT (ecg_rhythm_v1),
#      PYTHON (python), SMOKE (0), USD_PER_HR (0.63, A40)
set -euo pipefail

VOL="${VOL:-/runpod-volume/ssl-lab}"
OUT_ROOT="${1:-$VOL/output}"
DATA_DIR="${DATA_DIR:-$VOL/data}"
EXP="${EXPERIMENT:-ecg_rhythm_v1}"
PY="${PYTHON:-python}"
SMOKE="${SMOKE:-0}"
USD_PER_HR="${USD_PER_HR:-0.63}"
D=examples/ecg_rhythm

mkdir -p "$OUT_ROOT/$EXP/logs"
exec > >(tee -a "$OUT_ROOT/$EXP/logs/pilot.log") 2>&1
echo "[pilot] exp=$EXP data=$DATA_DIR out=$OUT_ROOT smoke=$SMOKE $(date -u +%FT%TZ)"

# --- Preflight -----------------------------------------------------------------------
miss=0
if [ ! -f "$DATA_DIR/ptbxl100/manifest.json" ]; then
  echo "[preflight] MISSING cache: $DATA_DIR/ptbxl100/manifest.json" >&2
  echo "            fix (local): python examples/ecg_rhythm/00_process_ptbxl.py ... then" >&2
  echo "                         python examples/ops/ops_stage_data.py data/ptbxl100" >&2
  miss=1
fi
if [ "$SMOKE" != "1" ]; then
  if ! "$PY" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    echo "[preflight] no CUDA device visible (did the launch request a GPU?)" >&2
    miss=1
  fi
fi
[ "$miss" = 0 ] || { echo "[preflight] FAILED; nothing was trained" >&2; exit 1; }
"$PY" -c 'import torch; print("[preflight] torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")'

C=(--data-dir "$DATA_DIR" --experiment "$EXP" --output-root "$OUT_ROOT")
PRE=(); DS=(); DEV=()
if [ "$SMOKE" = "1" ]; then PRE=(--epochs 2 --limit 1024); DS=(--max-steps 40 --eval-every 20); DEV=(--device cpu); fi
# bash 3.2 (macOS) treats empty arrays as unset under `set -u`; expand defensively.
pre() { echo ${PRE[@]+"${PRE[@]}"}; }
X() { "$PY" "$@"; }

GRID=$("$PY" -c 'import json; print(" ".join(str(x) for x in json.load(open("examples/ecg_rhythm/configs/study_v1.json"))["downstream"]["lr_grid"]))')
echo "[pilot] lr grid: $GRID"

X "$D/02_make_subsets.py" "${C[@]}"
X "$D/03_pretrain_jepa.py" --repeat 0 "${C[@]}" ${PRE[@]+"${PRE[@]}"} ${DEV[@]+"${DEV[@]}"}

for arm in scratch finetune probe; do
  for lr in $GRID; do
    X "$D/04_train_downstream.py" --arm "$arm" --repeat 0 --budget 0.1 --lr "$lr" --tag pilot \
      "${C[@]}" ${DS[@]+"${DS[@]}"} ${DEV[@]+"${DEV[@]}"}
  done
done

SEL=$(X "$D/05_select_lr.py" --tag pilot --experiment "$EXP" --output-root "$OUT_ROOT" | tail -1)
echo "[pilot] selected: $SEL"
lr_of() { echo "$SEL" | tr ' ' '\n' | sed -n "s/^$1=//p"; }

for arm in scratch finetune probe; do
  X "$D/04_train_downstream.py" --arm "$arm" --repeat 0 --budget 1.0 --lr "$(lr_of "$arm")" --tag pilot \
    "${C[@]}" ${DS[@]+"${DS[@]}"} ${DEV[@]+"${DEV[@]}"}
done
X "$D/04_train_downstream.py" --arm scratch --repeat 0 --budget 0.1 --lr "$(lr_of scratch)" --tag pilot --shuffle-labels \
  "${C[@]}" ${DS[@]+"${DS[@]}"} ${DEV[@]+"${DEV[@]}"}

X "$D/06_compare_arms.py" --tag pilot --split val "${C[@]}"
X "$D/07_pilot_report.py" --tag pilot --experiment "$EXP" --output-root "$OUT_ROOT" --usd-per-hr "$USD_PER_HR"
echo "[pilot] done $(date -u +%FT%TZ). Reports under $OUT_ROOT/$EXP/reports/"
