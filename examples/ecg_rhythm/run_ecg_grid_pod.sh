#!/usr/bin/env bash
# Pod entrypoint: ECG rhythm study v1 (amendment 1) — the full comparison grid.
#
# For each repeat r (outer loop, so a partial run still leaves whole repeats):
#   03  JEPA pretraining for repeat r (skipped when complete and compatible; repeat 0 is the
#       pinned pilot checkpoint; an unfinished run restarts from scratch, D28)
#   04  scratch / finetune / probe at every budget, at each arm's FROZEN rate from
#       lr_selection_<tag>.json, with test predictions (--eval-test). Finished, compatible
#       runs are skipped (--skip-if-complete); unfinished ones rerun from the beginning.
# Optionally (RUN_COMPARE=1) 06 on the test split; by default the bootstrap runs locally
# after fetching predictions, at no pod cost.
#
# Usage (on the pod; cache, subsets, repeat-0 checkpoint and the frozen selection on the volume):
#   bash examples/ecg_rhythm/run_ecg_grid_pod.sh [OUTPUT_ROOT]
# Env: VOL, DATA_DIR, EXPERIMENT, PYTHON, REPEATS (default: protocol n; reduce only for compute),
#      RUN_COMPARE (0), SMOKE (0)
set -euo pipefail

VOL="${VOL:-/runpod-volume/ssl-lab}"
OUT_ROOT="${1:-$VOL/output}"
DATA_DIR="${DATA_DIR:-$VOL/data}"
EXP="${EXPERIMENT:-ecg_rhythm_v1}"
PY="${PYTHON:-python}"
SMOKE="${SMOKE:-0}"
RUN_COMPARE="${RUN_COMPARE:-0}"
D=examples/ecg_rhythm

mkdir -p "$OUT_ROOT/$EXP/logs"
exec > >(tee -a "$OUT_ROOT/$EXP/logs/grid.log") 2>&1

# --- Preflight: protocol, subsets, frozen selection --------------------------------------
eval "$("$PY" - "$OUT_ROOT" "$EXP" <<'PYEOF'
import json, sys
sys.path.insert(0, "examples/ecg_rhythm")
import _study as S
out, exp = sys.argv[1], sys.argv[2]
s = S.load_study()
sub = S.load_subsets(s, out, exp)
tag = S.selection_tag(s)
sp = S.selection_path(out, exp, tag)
if not sp.exists():
    sys.exit(f"[preflight] frozen selection missing: {sp}")
sel = json.loads(sp.read_text())
why = S.check_compatible(sel, s, "run", str(sp))
if sel["grid"] != s["downstream"]["lr_grid"]:
    sys.exit(f"[preflight] selection grid {sel['grid']} != protocol grid {s['downstream']['lr_grid']}")
print(f'echo "[preflight] subsets: {sub["_reuse_reason"]}"')
print(f'echo "[preflight] selection {sp.name}: {why}"')
print(f'N_PROTOCOL={s["repeats"]["n"]}')
print("BUDGETS='" + " ".join(f"{b:g}" for b in s["budgets"]) + "'")
for arm, v in sel["arms"].items():
    print(f"LR_{arm}={v['lr']}")
PYEOF
)"
REPEATS="${REPEATS:-$N_PROTOCOL}"
echo "[grid] exp=$EXP repeats=$REPEATS (protocol $N_PROTOCOL) budgets=$BUDGETS lr: scratch=$LR_scratch finetune=$LR_finetune probe=$LR_probe smoke=$SMOKE $(date -u +%FT%TZ)"
if [ "$REPEATS" != "$N_PROTOCOL" ]; then echo "[grid] NOTE: repeats reduced from $N_PROTOCOL to $REPEATS (must be a compute decision)"; fi
if [ ! -f "$DATA_DIR/ptbxl100/manifest.json" ]; then echo "[preflight] MISSING $DATA_DIR/ptbxl100" >&2; exit 1; fi
if [ "$SMOKE" != "1" ]; then
  "$PY" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || { echo "[preflight] no CUDA" >&2; exit 1; }
fi

C=(--data-dir "$DATA_DIR" --experiment "$EXP" --output-root "$OUT_ROOT")
PRE=(); DS=(); DEV=()
if [ "$SMOKE" = "1" ]; then PRE=(--epochs 1 --limit 512); DS=(--max-steps 20 --eval-every 10); DEV=(--device cpu); fi

for r in $(seq 0 $((REPEATS - 1))); do
  echo "[grid] ===== repeat $r ====="
  "$PY" "$D/03_pretrain_jepa.py" --repeat "$r" "${C[@]}" ${PRE[@]+"${PRE[@]}"} ${DEV[@]+"${DEV[@]}"}
  for b in $BUDGETS; do
    for arm in scratch finetune probe; do
      lrv="LR_$arm"
      "$PY" "$D/04_train_downstream.py" --arm "$arm" --repeat "$r" --budget "$b" --lr "${!lrv}" --tag main \
        --eval-test --skip-if-complete "${C[@]}" ${DS[@]+"${DS[@]}"} ${DEV[@]+"${DEV[@]}"}
    done
  done
done

if [ "$RUN_COMPARE" = "1" ]; then
  "$PY" "$D/06_compare_arms.py" --tag main --split test "${C[@]}"
fi
echo "[grid] done $(date -u +%FT%TZ). Runs under $OUT_ROOT/$EXP/runs/main/"
