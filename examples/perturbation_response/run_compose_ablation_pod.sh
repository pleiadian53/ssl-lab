#!/usr/bin/env bash
# Composition ablation — can a non-additive gene-set embedding learn interactions?
#
# Every learned model so far embeds a combination as the SUM of its genes' vectors,
# z_p(A+B) = e(A) + e(B) (GeneSetEmbedding, compose="additive"). The DeepSets composition,
# z_p = phi(psi(e(A)) + psi(e(B))), is nonlinear, so a pair's embedding no longer has to be
# the sum of its singles'. This round trains the DeepSets twin of the two strongest learned
# arms; the ONLY change from the existing additive runs is --compose deepsets.
#
#   arm  flow_ds      transport flow, geneset condition, independent coupling, --compose deepsets
#                     twin of norman_sweep_control_s{0,1,2} (the 0.648 transport sweep)
#   arm  vae_cond_ds  conditioned NB-VAE, q(z | x, c), --compose deepsets
#                     twin of norman_vae_cond_s{0,1,2} (the 0.858 conditioned cVAE, round 6)
#
# The additive twins are reused, not retrained: no training code changed since they ran
# (git log), their reports reproduce when re-graded, and the frozen encoder here is the
# same file they used (md5 38f51067).
#
# This script ONLY trains. Grading happens locally after fetch, against the current DE
# selection, so no number exists before the round's endpoint is fixed in writing.
#
# Usage (on the pod, from the repo root):
#   bash examples/perturbation_response/run_compose_ablation_pod.sh
#   ARMS=vae_cond_ds SEEDS=0 bash examples/perturbation_response/run_compose_ablation_pod.sh

set -uo pipefail

VOL="${VOL:-/runpod-volume/ssl-lab}"
DATA="${DATA:-$VOL/data}"
OUT="${OUT:-$VOL/output}"
ENCODER="${ENCODER:-$VOL/output/norman_stage_a/checkpoints/encoder.pt}"
EPOCHS="${EPOCHS:-60}"
SEEDS="${SEEDS:-0 1 2}"
ARMS="${ARMS:-vae_cond_ds flow_ds}"
LIMIT="${LIMIT:-}"          # smoke only: cap the training cells
DEVICE="${DEVICE:-auto}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"

echo "=============================================================="
echo " COMPOSITION ABLATION — gene-set embedding: additive vs DeepSets"
echo " volume : $VOL"
echo " arms   : $ARMS      seeds: $SEEDS      epochs: $EPOCHS"
echo "=============================================================="

# ---- preflight: fail in seconds, not after an hour of training ------------------------- #
fail=0
if [ "$REQUIRE_CUDA" = "1" ]; then
  python -c "import torch; assert torch.cuda.is_available(); print('cuda:', torch.cuda.get_device_name(0))" \
    || { echo "FATAL: no CUDA. This is GPU work; check the pod."; fail=1; }
fi
if [ ! -f "$DATA/norman2019/manifest.json" ]; then
  echo "FATAL: cache missing at $DATA/norman2019"
  echo "  fix: stage it once from your laptop -> python examples/ops/ops_stage_data.py data/norman2019"
  fail=1
fi
case " $ARMS " in *" flow_ds "*)
  [ -f "$ENCODER" ] || {
    echo "FATAL: encoder missing at $ENCODER"
    echo "  fix: python examples/ops/ops_stage_data.py output/norman_stage_a/checkpoints"
    fail=1; } ;;
esac
[ "$fail" -eq 1 ] && { echo "preflight FAILED; nothing was trained."; exit 1; }
echo "preflight OK"

limit_flag=()
[ -n "$LIMIT" ] && limit_flag=(--limit "$LIMIT")

# ---- train ---------------------------------------------------------------------------- #
for arm in $ARMS; do
  for s in $SEEDS; do
    case "$arm" in
      flow_ds)
        exp="norman_sweep_control_ds_s${s}"
        cmd=(python examples/perturbation_response/04_train_cond_flow.py
             --experiment "$exp" --output-root "$OUT" --data-dir "$DATA" --encoder "$ENCODER"
             --split combo --cond-type geneset --compose deepsets
             --flow-base control --coupling independent) ;;
      vae_cond_ds)
        exp="norman_vae_cond_ds_s${s}"
        cmd=(python examples/perturbation_response/08_train_cvae_baseline.py
             --experiment "$exp" --output-root "$OUT" --data-dir "$DATA"
             --split combo --encoder-cond --compose deepsets) ;;
      *) echo "unknown arm '$arm'"; exit 2 ;;
    esac
    echo ""
    echo "---- $exp ----"
    "${cmd[@]}" --epochs "$EPOCHS" --seed "$s" --device "$DEVICE" "${limit_flag[@]}" \
      || echo "TRAIN FAILED: $exp"
  done
done

echo ""
echo "=============================================================="
echo " done. fetch the checkpoints and grade them locally:"
echo "   rsync -Pavz -u <cluster>:$OUT/norman_sweep_control_ds_s* <cluster>:$OUT/norman_vae_cond_ds_s* output/"
echo " grade only after the round's endpoint is fixed in writing (ledger rule: pre-register)."
echo "=============================================================="
