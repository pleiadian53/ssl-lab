#!/usr/bin/env bash
# Sweep the least-action penalty. The dense run at 1e-4 let the generators wander to ||M||~12, far
# outside the near-identity regime the bracket-is-epistasis equivalence needs, so its null endpoint
# could not test the claim. Find a weight that keeps the operator near identity.
set -uo pipefail
VOL="${VOL:-/runpod-volume/ssl-lab}"; DATA="$VOL/data"; OUT="$VOL/output"
ENCODER="$OUT/norman_stage_a/checkpoints/encoder.pt"
for aw in 1e-3 1e-2 1e-1; do
  exp="norman_opalg_aw${aw}_s0"
  echo ""; echo "===== action_weight=$aw -> $exp ====="
  python examples/perturbation_response/16_train_operator_algebra.py \
    --experiment "$exp" --output-root "$OUT" --data-dir "$DATA" \
    --encoder "$ENCODER" --split combo --epochs 60 --action-weight "$aw" --seed 0 \
    || { echo "TRAIN FAILED: $exp"; continue; }
done
