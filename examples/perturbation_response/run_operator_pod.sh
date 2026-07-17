#!/usr/bin/env bash
# Lever A — the action operator. Two arms, trained on the frozen encoder, graded by the
# same harness as the flow and the VAE.
#
#   arm 1  operator          deterministic. A_p = exp(M(e(p))) transports the control cloud.
#                            A deterministic operator is a diffeomorphism, so it CANNOT widen
#                            the cloud (verified: it preserves the variance exactly). This arm
#                            is therefore a pure EFFECT-SIZE play, which is the primary metric.
#   arm 2  operator_stoch    alpha ~ Gaussian + a residual displacement, so the perturbation
#                            induces a MIXTURE of operators. This is the only variant that can
#                            grow sigma^2_bio, i.e. the only one that can touch CALIBRATION.
#
# Reuses the frozen encoder and retrains only Stage B, per the frozen-encoder invariant.
# Nothing here reads de_genes.json: the DE list is an eval-time scoring seam, so the arms are
# fetched back and graded locally against the current gene selection.

set -uo pipefail

VOL="${VOL:-/runpod-volume/ssl-lab}"
DATA="${DATA:-$VOL/data}"
OUT="${OUT:-$VOL/output}"
ENCODER="${ENCODER:-$OUT/norman_stage_a/checkpoints/encoder.pt}"
EPOCHS="${EPOCHS:-60}"
SEEDS="${SEEDS:-0 1 2}"
ARMS="${ARMS:-operator operator_stoch}"

echo "=============================================================="
echo " LEVER A — action operator"
echo " volume : $VOL"
echo " encoder: $ENCODER"
echo " arms   : $ARMS      seeds: $SEEDS      epochs: $EPOCHS"
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
  echo "  fix: Stage A was never trained (or never staged). Run:"
  echo "       bash examples/perturbation_response/run_stage_a_pod.sh"
  echo "  or point ENCODER= at an existing encoder.pt on the volume."
  fail=1
fi
[ "$fail" -eq 1 ] && { echo "preflight FAILED; nothing was trained."; exit 1; }
echo "preflight OK"

# ---- train ---------------------------------------------------------------------------- #
for arm in $ARMS; do
  case "$arm" in
    operator)        FLAGS="" ;;
    operator_stoch)  FLAGS="--stochastic --residual-scale 1.0" ;;
    *) echo "unknown arm '$arm'"; exit 2 ;;
  esac
  for s in $SEEDS; do
    exp="norman_${arm}_s${s}"
    echo ""
    echo "---- $exp ----"
    python examples/perturbation_response/13_train_operator.py \
      --experiment "$exp" --output-root "$OUT" --data-dir "$DATA" \
      --encoder "$ENCODER" --split combo --epochs "$EPOCHS" --seed "$s" \
      $FLAGS || { echo "TRAIN FAILED: $exp"; continue; }
  done
done

echo ""
echo "=============================================================="
echo " done. fetch the checkpoints and grade them locally against the"
echo " current DE selection:"
echo "   rsync -Pavz <cluster>:$OUT/norman_operator*/ output/"
echo "   python examples/perturbation_response/06_eval_effect_size.py \\"
echo "       --experiment norman_operator_s0 --stage-b operator --split combo"
echo "=============================================================="
