#!/usr/bin/env bash
# Training-speed benchmark for joint Geneformer + decoder training (12_joint_benchmark.py). Pod only.
# Nothing is trained for a result; the output is the cost of each configuration and an equivalence check.
#
# Usage (on the pod, from the repo root):
#   bash examples/perturbation_response/foundation-encoder/run_joint_benchmark_pod.sh

set -uo pipefail
VOL="${VOL:-/runpod-volume/ssl-lab}"; DATA="${DATA:-$VOL/data}"; OUT="${OUT:-$VOL/output}"
GF="${GF:-$VOL/models/geneformer}"
GF_REV="${GF_REV:-1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5}"
FE=examples/perturbation_response/foundation-encoder
LOGS="$OUT/gf_bench_logs"; mkdir -p "$LOGS"

CONSTRAINTS="$LOGS/constraints.txt"
printf '%s\n' "numpy<2" "pandas<3" "pyarrow<22" > "$CONSTRAINTS"
pip install -q -c "$CONSTRAINTS" "numpy<2" "pandas<3" "pyarrow<22" anndata h5py huggingface_hub > "$LOGS/install.log" 2>&1 \
  || { echo "FATAL: base install"; tail -20 "$LOGS/install.log"; exit 1; }
[ -f "$GF/Geneformer-V2-104M/model.safetensors" ] || { echo "FATAL: Geneformer V2-104M weights missing under $GF"; exit 1; }
pip install -q -r "$GF/requirements.txt" -c "$CONSTRAINTS" >> "$LOGS/install.log" 2>&1 \
  || { echo "FATAL: Geneformer requirements"; tail -20 "$LOGS/install.log"; exit 1; }
for f in "$OUT/geneformer_full/tokens_V2.dataset" "$OUT/geneformer_full/stores/V2-104M_cell/manifest.json"; do
  [ -e "$f" ] || { echo "FATAL: missing $f (round 14)"; exit 1; }
done

python $FE/12_joint_benchmark.py --gf-root "$GF" --data-dir "$DATA" --tokens "$OUT/geneformer_full/tokens_V2.dataset" \
  --store "$OUT/geneformer_full/stores/V2-104M_cell" --out "$OUT/geneformer_joint_benchmark.json" > "$LOGS/bench.log" 2>&1 \
  || { echo "BENCHMARK FAILED"; tail -40 "$LOGS/bench.log"; exit 1; }
grep -E "^(equivalence|speed)" "$LOGS/bench.log"
echo "[bench] done. fetch: $OUT/geneformer_joint_benchmark.json $LOGS"
