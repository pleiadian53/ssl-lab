#!/usr/bin/env bash
# Pod entrypoint: ECG rhythm study v1 — learning-rate search for the protocol in force.
# Trains only the rates the amendment adds ("new_rates"; all grid rates if undeclared); 05
# then selects over the full grid, reusing compatible runs from the tags the amendment lists.
# (Amendment 1 ran all four rates; amendment 2 adds 1e-2 only.)
#
# Every arm (scratch, finetune, probe) is trained at repeat 0, budget 10%, at every rate
# of the amended shared grid {1e-4, 3e-4, 1e-3, 3e-3}; 05 then picks each arm's rate on
# VALIDATION by the unchanged rule and freezes it in lr_selection_<tag>.json. All four
# rates are rerun (the pilot runs are preserved, not reused for selection). Reused, after
# compatibility checks recorded in each report: the saved subsets (byte-identical, pinned)
# and the repeat-0 JEPA checkpoint (pinned). No test predictions are produced.
#
# Usage (on the pod; cache + subsets + repeat-0 checkpoint already on the volume):
#   bash examples/ecg_rhythm/run_ecg_lrsearch_pod.sh [OUTPUT_ROOT]
# Local plumbing check:
#   SMOKE=1 VOL=. DATA_DIR=data bash examples/ecg_rhythm/run_ecg_lrsearch_pod.sh <scratch-output-root>
set -euo pipefail

VOL="${VOL:-/runpod-volume/ssl-lab}"
OUT_ROOT="${1:-$VOL/output}"
DATA_DIR="${DATA_DIR:-$VOL/data}"
EXP="${EXPERIMENT:-ecg_rhythm_v1}"
PY="${PYTHON:-python}"
SMOKE="${SMOKE:-0}"
D=examples/ecg_rhythm

mkdir -p "$OUT_ROOT/$EXP/logs"
exec > >(tee -a "$OUT_ROOT/$EXP/logs/lrsearch.log") 2>&1
TAG=$("$PY" -c 'import sys; sys.path.insert(0,"examples/ecg_rhythm"); import _study as S; print(S.selection_tag(S.load_study()))')
GRID=$("$PY" -c 'import sys; sys.path.insert(0,"examples/ecg_rhythm"); import _study as S; print(" ".join(str(x) for x in S.new_rates(S.load_study())))')
echo "[lrsearch] exp=$EXP tag=$TAG new rates=$GRID data=$DATA_DIR out=$OUT_ROOT smoke=$SMOKE $(date -u +%FT%TZ)"

# --- Preflight: every reused input verified before any training ------------------------
"$PY" - "$OUT_ROOT" "$EXP" "$SMOKE" <<'PYEOF'
import sys
sys.path.insert(0, "examples/ecg_rhythm")
import _study as S
out, exp, smoke = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
s = S.load_study()
sub = S.load_subsets(s, out, exp)
print("[preflight] subsets:", sub["_reuse_reason"])
ok, why = S.pretrain_complete(s, out, exp, 0)
print("[preflight] pretrain r0:", why)
if not ok and not smoke:
    sys.exit("[preflight] repeat-0 pretraining unusable; nothing was trained")
PYEOF
if [ ! -f "$DATA_DIR/ptbxl100/manifest.json" ]; then echo "[preflight] MISSING $DATA_DIR/ptbxl100" >&2; exit 1; fi
if [ "$SMOKE" != "1" ]; then
  "$PY" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || { echo "[preflight] no CUDA" >&2; exit 1; }
fi

C=(--data-dir "$DATA_DIR" --experiment "$EXP" --output-root "$OUT_ROOT")
DS=(); DEV=()
if [ "$SMOKE" = "1" ]; then DS=(--max-steps 40 --eval-every 20); DEV=(--device cpu); fi

for arm in scratch finetune probe; do
  for lr in $GRID; do
    "$PY" "$D/04_train_downstream.py" --arm "$arm" --repeat 0 --budget 0.1 --lr "$lr" --tag "$TAG" \
      "${C[@]}" ${DS[@]+"${DS[@]}"} ${DEV[@]+"${DEV[@]}"}
  done
done
if [ "$SMOKE" = "1" ]; then echo "[lrsearch] smoke: skipping selection (smoke runs are never selectable)"; exit 0; fi
"$PY" "$D/05_select_lr.py" --experiment "$EXP" --output-root "$OUT_ROOT"
echo "[lrsearch] done $(date -u +%FT%TZ). Selection: $OUT_ROOT/$EXP/lr_selection_$TAG.json"
