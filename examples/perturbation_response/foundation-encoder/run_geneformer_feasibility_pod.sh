#!/usr/bin/env bash
# R2.7 feasibility: can pretrained Geneformer models embed the Norman cells, at what cost?
#
# Embeds a random sample of cells (default 2,000) with each Geneformer model and reports, per
# model and embedding mode: time per 1,000 cells and the projected time for all 109,737 cells,
# peak GPU memory, embedding width and effective rank, and how many genes per cell the
# tokenizer keeps. No comparison with our encoders happens here; that needs a protocol first.
#
# Inputs on the volume (stage with examples/ops/ops_stage_data.py):
#   $DATA/norman2019/raw/norman_2019.h5ad          raw counts over all genes
#   $DATA/norman2019/foundation/cell_index.npz     from 01_build_cell_index.py
# Geneformer code, dictionaries and weights are downloaded once to $GF at a pinned commit.
#
# Usage (on the pod, from the repo root):
#   bash examples/perturbation_response/foundation-encoder/run_geneformer_feasibility_pod.sh
#   N_CELLS=500 MODELS="V1-10M" bash ...      # smaller run

set -uo pipefail

VOL="${VOL:-/runpod-volume/ssl-lab}"
DATA="${DATA:-$VOL/data}"
OUT="${OUT:-$VOL/output}"
GF="${GF:-$VOL/models/geneformer}"
GF_REPO="ctheodoris/Geneformer"
GF_REV="${GF_REV:-1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5}"   # main on 2026-09-12, the code read for 02
MODELS="${MODELS:-V1-10M V2-104M V2-104M_CLcancer V2-316M}"
N_CELLS="${N_CELLS:-2000}"
BATCH="${BATCH:-16}"
RAW_MD5="${RAW_MD5:-e05c04a7eb410ea542f55ce4a5fa1ab3}"
SCRIPT="examples/perturbation_response/foundation-encoder/02_embed_geneformer.py"
LOGS="$OUT/geneformer_logs"
mkdir -p "$LOGS"

echo "=============================================================="
echo " R2.7 Geneformer feasibility  models: $MODELS  cells: $N_CELLS"
echo "=============================================================="

# ---- scientific stack: the image's torch is compiled against numpy 1.x ---------------------- #
# Installing recent anndata/pandas pulls numpy 2, after which torch<->numpy calls fail
# ("A module that was compiled using NumPy 1.x cannot be run in NumPy 2"). pyarrow 22 and
# later also need numpy 2 (21.0.0 verified with numpy 1.26.4 and datasets 5.0.1). Every pip
# call below uses these constraints, and the first one repairs a pod where numpy 2 got in.
CONSTRAINTS="$LOGS/constraints.txt"
printf '%s\n' "numpy<2" "pandas<3" "pyarrow<22" > "$CONSTRAINTS"
pip install -q -c "$CONSTRAINTS" "numpy<2" "pandas<3" "pyarrow<22" anndata h5py huggingface_hub > "$LOGS/install.log" 2>&1 \
  || { echo "FATAL: base install (see $LOGS/install.log)"; tail -20 "$LOGS/install.log"; exit 1; }
python -c "import numpy, torch; torch.from_numpy(numpy.zeros(2)); print('numpy', numpy.__version__, 'torch', torch.__version__, 'interop OK')" \
  || { echo "FATAL: numpy/torch interop"; exit 1; }

# ---- preflight ------------------------------------------------------------------------------ #
fail=0
python -c "import torch; assert torch.cuda.is_available(); print('cuda:', torch.cuda.get_device_name(0))" || { echo "FATAL: no CUDA"; fail=1; }
m=$(md5sum "$DATA/norman2019/raw/norman_2019.h5ad" 2>/dev/null | cut -d' ' -f1)
[ "$m" = "$RAW_MD5" ] || { echo "FATAL: raw h5ad missing or md5 '$m' (stage data/norman2019/raw)"; fail=1; }
[ -f "$DATA/norman2019/foundation/cell_index.npz" ] || { echo "FATAL: missing foundation/cell_index.npz (stage data/norman2019/foundation)"; fail=1; }
[ "$fail" -eq 1 ] && { echo "preflight FAILED; nothing was run."; exit 1; }
echo "preflight OK"

# ---- install: Geneformer source + weights at the pinned commit, then its requirements -------- #
GF="$GF" GF_REPO="$GF_REPO" GF_REV="$GF_REV" MODELS="$MODELS" python - <<'PY' >> "$LOGS/install.log" 2>&1 || { echo "FATAL: download (see $LOGS/install.log)"; exit 1; }
import os, sys
from huggingface_hub import snapshot_download
import importlib.util
spec = importlib.util.spec_from_file_location("e", "examples/perturbation_response/foundation-encoder/02_embed_geneformer.py")
e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
dirs = [e.MODELS[m]["dir"] for m in os.environ["MODELS"].split()]
pats = ["geneformer/*", "requirements.txt", "setup.py", "MANIFEST.in"]
for d in dirs:
    pats += [f"{d}/config.json", f"{d}/generation_config.json", f"{d}/model.safetensors"]
p = snapshot_download(os.environ["GF_REPO"], revision=os.environ["GF_REV"], local_dir=os.environ["GF"], allow_patterns=pats)
print("downloaded to", p)
PY
pip install -q -r "$GF/requirements.txt" -c "$CONSTRAINTS" >> "$LOGS/install.log" 2>&1 \
  || { echo "FATAL: Geneformer requirements (see $LOGS/install.log)"; tail -20 "$LOGS/install.log"; exit 1; }
python -c "import sys; sys.path.insert(0, '$GF'); import geneformer, transformers, torch; import numpy; print('geneformer OK, transformers', transformers.__version__, 'torch', torch.__version__, 'numpy', numpy.__version__)" \
  || { echo "FATAL: geneformer import"; exit 1; }

# ---- smoke: 64 cells, the smallest V1 and V2 models, separate output ------------------------ #
python "$SCRIPT" --gf-root "$GF" --data-dir "$DATA" --out-dir "$OUT/geneformer_smoke" --n-cells 64 \
  --models V1-10M V2-104M --batch-size "$BATCH" > "$LOGS/smoke.log" 2>&1 || { echo "SMOKE FAILED"; tail -40 "$LOGS/smoke.log"; exit 1; }
python - "$OUT/geneformer_smoke/feasibility.json" <<'PY' || { echo "SMOKE CHECK FAILED"; tail -40 "$LOGS/smoke.log"; exit 1; }
import json, sys
r = json.load(open(sys.argv[1]))["models"]
bad = {k: v["error"] for k, v in r.items() if "error" in v}
print("smoke models:", sorted(r), "errors:", bad)
sys.exit(1 if bad or len(r) < 3 else 0)
PY
echo "smoke OK"

# ---- feasibility sample --------------------------------------------------------------------- #
# shellcheck disable=SC2086
python "$SCRIPT" --gf-root "$GF" --data-dir "$DATA" --out-dir "$OUT/geneformer_feasibility" --n-cells "$N_CELLS" \
  --models $MODELS --batch-size "$BATCH" > "$LOGS/feasibility.log" 2>&1 || { echo "FEASIBILITY RUN FAILED"; tail -40 "$LOGS/feasibility.log"; }
cat "$OUT/geneformer_feasibility/feasibility.json" 2>/dev/null || echo "no feasibility.json"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "[geneformer] done"
