#!/usr/bin/env bash
# Round 10 — the control reference: re-read every encoder's readout ceiling with control profiles.
#
# Nothing is trained. 14_ceiling_analysis.py is rerun with --save-profiles, which now also saves
# each readout's prediction for the control cells, so 23_score_round10.py can score a predicted
# change against the readout's own control as well as against the measured control mean.
#
#   norman_ceiling_r10                    frozen encoder norman_stage_a, arms identity roundtrip
#                                         latent_mean linear (the primary endpoint's readouts)
#   norman_ceiling_r10_{arm}_s{seed}      the nine round-9 encoders, arms identity linear
#
# Encoding every training cell for the ridge takes ~18 min per encoder on a laptop CPU, which is
# why this runs on the pod. The learned models (flows, NB-VAEs, operators) are rescored locally
# with 19_score_interaction.py, where their stored reports were computed. The protocol and its
# gates are in 23_score_round10.py's docstring.
#
# Usage (on the pod, from the repo root):
#   bash examples/perturbation_response/run_round10_ceiling_pod.sh

set -uo pipefail

VOL="${VOL:-/runpod-volume/ssl-lab}"
DATA="${DATA:-$VOL/data}"
OUT="${OUT:-$VOL/output}"
DECODER="${DECODER:-$OUT/norman_flow_control/checkpoints/count_decoder.pt}"
DE_MD5="${DE_MD5:-9844c2c25d2903872e1c0a92c722f105}"
PARALLEL="${PARALLEL:-3}"
LOGS="$OUT/round10_logs"
mkdir -p "$LOGS"

echo "=============================================================="
echo " ROUND 10 — readout ceilings with control profiles"
echo "=============================================================="

fail=0
python -c "import torch; assert torch.cuda.is_available(); print('cuda:', torch.cuda.get_device_name(0))" \
  || { echo "FATAL: no CUDA"; fail=1; }
got_md5=$(md5sum "$DATA/norman2019/de_genes.json" 2>/dev/null | cut -d' ' -f1)
[ "$got_md5" = "$DE_MD5" ] || { echo "FATAL: de_genes.json md5 $got_md5 != $DE_MD5"; fail=1; }
enc_md5=$(md5sum "$OUT/norman_stage_a/checkpoints/encoder.pt" 2>/dev/null | cut -d' ' -f1)
[ "${enc_md5:0:8}" = "38f51067" ] || { echo "FATAL: encoder md5 ${enc_md5:0:8} != 38f51067"; fail=1; }
dec_md5=$(md5sum "$DECODER" 2>/dev/null | cut -d' ' -f1)
[ "${dec_md5:0:8}" = "52a44337" ] || { echo "FATAL: decoder md5 ${dec_md5:0:8} != 52a44337"; fail=1; }
for arm in vicreg sigreg_tok sigreg_pool; do for s in 0 1 2; do
  [ -f "$OUT/norman_stage_a_${arm}_s${s}/checkpoints/encoder.pt" ] || { echo "FATAL: missing encoder $arm s$s"; fail=1; }
done; done
[ "$fail" -eq 1 ] && { echo "preflight FAILED; nothing was run."; exit 1; }
echo "preflight OK"

ceiling() {   # $1 = encoder, $2 = experiment, $3.. = arms
  local enc="$1" exp="$2"; shift 2
  python examples/perturbation_response/14_ceiling_analysis.py \
    --encoder "$enc" --decoder "$DECODER" --split combo --arms "$@" --save-profiles \
    --experiment "$exp" --output-root "$OUT" --data-dir "$DATA" \
    > "$LOGS/$exp.log" 2>&1 && echo "done $exp" || echo "CEILING FAILED: $exp"
}

running=0
ceiling "$OUT/norman_stage_a/checkpoints/encoder.pt" norman_ceiling_r10 identity roundtrip latent_mean linear &
running=1
for arm in vicreg sigreg_tok sigreg_pool; do for s in 0 1 2; do
  ceiling "$OUT/norman_stage_a_${arm}_s${s}/checkpoints/encoder.pt" "norman_ceiling_r10_${arm}_s${s}" identity linear &
  running=$((running + 1))
  if [ "$running" -ge "$PARALLEL" ]; then wait -n; running=$((running - 1)); fi
done; done
wait
echo "fetch narrowly: rsync -az <cluster>:$OUT/norman_ceiling_r10* (reports only) and $LOGS"
