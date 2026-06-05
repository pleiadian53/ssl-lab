#!/usr/bin/env bash
# Pod entrypoint: train the full generative-JEPA vertical for real epochs.
#
# Runs 01 (JEPA) -> 02 (probe) -> 03 (decoder) -> 04 (prior) -> 05 (sample) with
# a consistent --experiment so the chain shares artifacts. Volume-agnostic: if an
# output root is given (arg 1), symlink ./output to it so all artifacts land
# there (e.g. a persistent network volume) and rsync back cleanly.
#
# Usage:
#   bash examples/generative_jepa/run_pod_pipeline.sh [OUTPUT_ROOT]
# Env overrides:
#   EXPERIMENT (default jepa_mnist), EPOCHS_JEPA/EPOCHS_DECODER/EPOCHS_PRIOR
set -euo pipefail

OUT_ROOT="${1:-output}"
EXP="${EXPERIMENT:-jepa_mnist}"
if [ "$OUT_ROOT" != "output" ]; then
  mkdir -p "$OUT_ROOT"
  ln -sfn "$OUT_ROOT" output
fi

# Capture the full run log into the experiment's logs/ dir (on the volume via the
# symlink above), so it rsyncs back alongside checkpoints/samples/reports — no
# manual sky_logs fetch needed. tee keeps console output for sky log streaming.
mkdir -p "output/$EXP/logs"
exec > >(tee -a "output/$EXP/logs/train.log") 2>&1

: "${EPOCHS_JEPA:=100}"
: "${EPOCHS_DECODER:=30}"
: "${EPOCHS_PRIOR:=100}"

echo "[run_pod_pipeline] experiment=$EXP out=$OUT_ROOT jepa=$EPOCHS_JEPA decoder=$EPOCHS_DECODER prior=$EPOCHS_PRIOR"
A=(--experiment "$EXP")

python examples/jepa_basics/01_train_jepa_mnist.py     "${A[@]}" --epochs "$EPOCHS_JEPA" --reg-coef 0.04
python examples/jepa_basics/02_linear_probe.py         "${A[@]}"
python examples/generative_jepa/03_train_decoder.py    "${A[@]}" --epochs "$EPOCHS_DECODER"
python examples/generative_jepa/04_train_flow_prior.py "${A[@]}" --epochs "$EPOCHS_PRIOR"
python examples/generative_jepa/05_sample_and_decode.py "${A[@]}" --n 64 --steps 100

echo "[run_pod_pipeline] done. artifacts under output/$EXP/"
