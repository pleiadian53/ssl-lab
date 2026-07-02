#!/usr/bin/env bash
# Pod entrypoint: seed sweep of three flow configs on held-out combos, for CIs.
#
# Configs (all geneset, combo split, reusing the staged encoder + NB decoder):
#   gaussian    : flow_base=gaussian                  (noise->outcome, current default)
#   control     : flow_base=control  coupling=indep   (transport, random pairing)
#   control_ot  : flow_base=control  coupling=ot       (transport, minibatch-OT pairing)
# Each x seeds {0,1,2}. Eval seed fixed (0) so the spread reflects TRAINING variance.
# Aggregate locally: mean per-combo over seeds -> paired bootstrap between configs.
#
# Usage:
#   DATA_DIR=/cache ENCODER=/cache/encoder.pt DECODER=/cache/count_decoder.pt \
#     bash examples/perturbation_response/run_seed_sweep_pod.sh [OUTPUT_ROOT]
set -uo pipefail

OUT_ROOT="${1:-output}"
DATA_DIR="${DATA_DIR:-data}"
ARTIFACT="${ARTIFACT:-norman2019}"
ENCODER="${ENCODER:-/cache/encoder.pt}"
DECODER="${DECODER:-/cache/count_decoder.pt}"
[ "$OUT_ROOT" != "output" ] && { mkdir -p "$OUT_ROOT"; ln -sfn "$OUT_ROOT" output; }

: "${FLOW_EPOCHS:=60}"
: "${EVAL_N:=200}"
: "${SEEDS:=0 1 2}"

[ -f "$DATA_DIR/$ARTIFACT/manifest.json" ] || { echo "[sweep] ERROR: cache absent at $DATA_DIR/$ARTIFACT" >&2; exit 2; }
python -c 'import torch; print("[sweep] cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")'
echo "[sweep] cache present; seeds=$SEEDS epochs=$FLOW_EPOCHS eval_n=$EVAL_N"

run_one() {  # label flow_base coupling seed
  local label="$1" fb="$2" cp="$3" seed="$4" exp="norman_sweep_${1}_s${4}"
  mkdir -p "output/$exp/checkpoints"
  cp -f "$DECODER" "output/$exp/checkpoints/count_decoder.pt"
  echo "================ $label seed=$seed (exp=$exp) ================"
  python examples/perturbation_response/04_train_cond_flow.py --experiment "$exp" \
    --data-dir "$DATA_DIR" --artifact "$ARTIFACT" --encoder "$ENCODER" \
    --split combo --cond-type geneset --flow-base "$fb" --coupling "$cp" --seed "$seed" --epochs "$FLOW_EPOCHS" \
  && python examples/perturbation_response/06_eval_effect_size.py --experiment "$exp" \
    --data-dir "$DATA_DIR" --artifact "$ARTIFACT" --split combo --n "$EVAL_N" --seed 0 \
  || echo "[sweep] !!! FAILED $label seed $seed"
}

for s in $SEEDS; do
  run_one gaussian   gaussian independent "$s"
  run_one control    control  independent "$s"
  run_one control_ot control  ot          "$s"
done
echo "[sweep] done. Reports under output/norman_sweep_*/reports/effect_size.json"
