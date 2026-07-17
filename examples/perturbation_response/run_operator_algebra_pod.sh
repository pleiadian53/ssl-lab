#!/usr/bin/env bash
# The operator-ALGEBRA round — one generator per gene, combinations compose in the group so the
# Lie bracket [M_A, M_B] carries epistasis. See:
#   dev/planning/action_operator/03-the-operator-algebra-composition-and-epistasis.md
#
# This entrypoint only TRAINS (16). The evals are run locally after fetch, against the current DE
# selection, exactly as round 3 did: the DE list is an eval-time scoring seam, and grading locally
# sidesteps the stale-de_genes-on-the-volume footgun that clobbered reports last round.
#
# Reuses the frozen encoder and retrains only Stage B, per the frozen-encoder invariant.

set -uo pipefail

VOL="${VOL:-/runpod-volume/ssl-lab}"
DATA="${DATA:-$VOL/data}"
OUT="${OUT:-$VOL/output}"
ENCODER="${ENCODER:-$OUT/norman_stage_a/checkpoints/encoder.pt}"
EPOCHS="${EPOCHS:-60}"
SEEDS="${SEEDS:-0}"                      # one seed first to read the correlation; 3 later if green
ACTION_WEIGHT="${ACTION_WEIGHT:-1e-4}"

echo "=============================================================="
echo " OPERATOR-ALGEBRA round"
echo " volume : $VOL"
echo " encoder: $ENCODER"
echo " seeds  : $SEEDS      epochs: $EPOCHS      action_weight: $ACTION_WEIGHT"
echo "=============================================================="

# ---- preflight: fail in seconds, not after 20 minutes of training ---------------------- #
fail=0
python -c "import torch; assert torch.cuda.is_available(); print('cuda:', torch.cuda.get_device_name(0))" \
  || { echo "FATAL: no CUDA. This is GPU work; check the pod."; fail=1; }

if [ ! -d "$DATA/norman2019" ]; then
  echo "FATAL: cache missing at $DATA/norman2019"
  echo "  fix: stage it once from your laptop ->"
  echo "       python examples/ops/ops_stage_data.py data/norman2019"
  fail=1
fi
if [ ! -f "$ENCODER" ]; then
  echo "FATAL: frozen encoder missing at $ENCODER"
  echo "  fix: stage it once from your laptop ->"
  echo "       python examples/ops/ops_stage_data.py output/norman_stage_a/checkpoints"
  echo "  or point ENCODER= at an existing encoder.pt on the volume."
  fail=1
fi
[ "$fail" -eq 1 ] && { echo "preflight FAILED; nothing was trained."; exit 1; }
echo "preflight OK"

# ---- train ---------------------------------------------------------------------------- #
for s in $SEEDS; do
  exp="norman_operator_algebra_s${s}"
  echo ""
  echo "---- $exp ----"
  python examples/perturbation_response/16_train_operator_algebra.py \
    --experiment "$exp" --output-root "$OUT" --data-dir "$DATA" \
    --encoder "$ENCODER" --split combo --epochs "$EPOCHS" \
    --action-weight "$ACTION_WEIGHT" --seed "$s" \
    || { echo "TRAIN FAILED: $exp"; continue; }
done

echo ""
echo "=============================================================="
echo " done. fetch the checkpoints and grade them LOCALLY against the"
echo " current DE selection:"
echo "   rsync -Pavz -u <cluster>:$OUT/norman_operator_algebra_s0/ output/norman_operator_algebra_s0/"
echo "   # primary endpoint — does the bracket predict epistasis?"
echo "   python examples/perturbation_response/17_eval_bracket_epistasis.py \\"
echo "       --experiment norman_operator_algebra_s0"
echo "   # standing benchmark for the ledger (reuse the canonical decoder):"
echo "   python examples/perturbation_response/06_eval_effect_size.py \\"
echo "       --experiment norman_operator_algebra_s0 --stage-b operator_algebra --split combo \\"
echo "       --decoder output/norman_flow_control/checkpoints/count_decoder.pt"
echo "=============================================================="
