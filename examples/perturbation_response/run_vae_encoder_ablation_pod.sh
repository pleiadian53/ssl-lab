#!/usr/bin/env bash
# Baseline ablation — does the NB-VAE bar change if its encoder sees the condition?
#
# The NB-VAE that every round is measured against uses a condition-blind encoder, q(z | x):
# the perturbation reaches it only through the decoder. The textbook conditional VAE also
# conditions the encoder, q(z | x, c). This round trains BOTH, same seeds, same data, same
# everything else, so the bar the method must beat is the stronger of the two.
#
#   arm 1  vae_blind   q(z | x)      the current baseline, re-trained in this round so the
#                                     contrast is within-round (ledger rule)
#   arm 2  vae_cond    q(z | x, c)   --encoder-cond
#
# Primary endpoint, fixed before any numbers: delta_r on the 20 held-out combinations,
# contrast vae_cond - vae_blind, 3 seeds per arm, joint bootstrap via 12_compare_arms.py.
# Calibration is secondary (intervals only, no verdict).
#
# Needs no encoder: the VAE is the no-JEPA control. Nothing here reads de_genes.json; the
# checkpoints are fetched back and graded locally against the current gene selection.

set -uo pipefail

VOL="${VOL:-/runpod-volume/ssl-lab}"
DATA="${DATA:-$VOL/data}"
OUT="${OUT:-$VOL/output}"
EPOCHS="${EPOCHS:-60}"
SEEDS="${SEEDS:-0 1 2}"
ARMS="${ARMS:-vae_blind vae_cond}"

echo "=============================================================="
echo " BASELINE ABLATION — NB-VAE encoder: q(z|x) vs q(z|x,c)"
echo " volume : $VOL"
echo " arms   : $ARMS      seeds: $SEEDS      epochs: $EPOCHS"
echo "=============================================================="

# ---- preflight: fail in seconds, not after an hour of training ------------------------- #
fail=0
python -c "import torch; assert torch.cuda.is_available(); print('cuda:', torch.cuda.get_device_name(0))" \
  || { echo "FATAL: no CUDA. This is GPU work; check the pod."; fail=1; }
if [ ! -d "$DATA/norman2019" ]; then
  echo "FATAL: cache missing at $DATA/norman2019"
  echo "  fix: stage it once from your laptop ->"
  echo "       python examples/ops/ops_stage_data.py data/norman2019"
  fail=1
fi
python -c "import sys; sys.path.insert(0, 'src'); \
from ssllab.generative.cvae import ConditionalNBVAE; import inspect; \
assert 'encoder_cond' in inspect.signature(ConditionalNBVAE).parameters" \
  || { echo "FATAL: this checkout's ConditionalNBVAE has no encoder_cond; sync the code."; fail=1; }
[ "$fail" -eq 1 ] && { echo "preflight FAILED; nothing was trained."; exit 1; }
echo "preflight OK"

# ---- train ---------------------------------------------------------------------------- #
for arm in $ARMS; do
  case "$arm" in
    vae_blind) FLAGS="" ;;
    vae_cond)  FLAGS="--encoder-cond" ;;
    *) echo "unknown arm '$arm'"; exit 2 ;;
  esac
  for s in $SEEDS; do
    exp="norman_${arm}_s${s}"
    echo ""
    echo "---- $exp ----"
    python examples/perturbation_response/08_train_cvae_baseline.py \
      --experiment "$exp" --output-root "$OUT" --data-dir "$DATA" \
      --split combo --epochs "$EPOCHS" --seed "$s" \
      $FLAGS || { echo "TRAIN FAILED: $exp"; continue; }
  done
done

echo ""
echo "=============================================================="
echo " done. fetch the checkpoints and grade them locally:"
echo "   rsync -Pavz <cluster>:$OUT/norman_vae_blind_s* <cluster>:$OUT/norman_vae_cond_s* output/"
echo "   for e in output/norman_vae_{blind,cond}_s{0,1,2}; do"
echo "     x=\$(basename \$e)"
echo "     python examples/perturbation_response/09_eval_cvae_baseline.py --experiment \$x --split combo"
echo "     python examples/perturbation_response/10_eval_calibration.py   --experiment \$x --split combo --model vae"
echo "   done"
echo "   python examples/perturbation_response/12_compare_arms.py \\"
echo "       --arm 'blind=norman_vae_blind_s{s}:vae:0,1,2' --arm 'cond=norman_vae_cond_s{s}:vae:0,1,2' \\"
echo "       --contrast cond-blind"
echo "=============================================================="
