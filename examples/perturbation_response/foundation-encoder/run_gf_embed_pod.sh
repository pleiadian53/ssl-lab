#!/usr/bin/env bash
# R2.7, pod job 1: every embedding store the Geneformer comparison reads.
#
# Writes, under $OUT:
#   geneformer_full/stores/<model>_<mode>/     Geneformer V1-10M (gene mean), V2-104M and
#                                              V2-104M_CLcancer (<cls> token and gene mean), all
#                                              109,737 processed cells, second-to-last layer
#   geneformer_full/stores_p256/<model>_<mode>/ the four V2 stores projected to 256 dimensions,
#                                              fit on combo training cells only
#   embeddings/norman_jepa_stage_a/            our frozen JEPA encoder's vectors, same contract
#   geneformer_mechanism/mechanism.json        optional descriptive checks (08); a failure there
#                                              does not fail the job
# and checks gate (d) of the protocol: every store covers every processed cell, was computed on
# this cache, and its files match its manifest. The protocol is in 07_score_comparison.py.
#
# Inputs on the volume:
#   $DATA/norman2019/raw/norman_2019.h5ad, foundation/cell_index.npz, tokens_meta.npz, manifest.json
#   $OUT/norman_stage_a/checkpoints/encoder.pt
#   $OUT/geneformer_feasibility/tokens_V2.dataset   (for the optional checks, from the feasibility run)
# Geneformer is pinned to the Hugging Face commit the feasibility run used.
#
# About 11 GPU-hours on one A40 (feasibility measurement: V1 0.23 h, each V2 model and mode ~2.6 h).
#
# Usage (on the pod, from the repo root):
#   bash examples/perturbation_response/foundation-encoder/run_gf_embed_pod.sh

set -uo pipefail

VOL="${VOL:-/runpod-volume/ssl-lab}"
DATA="${DATA:-$VOL/data}"
OUT="${OUT:-$VOL/output}"
GF="${GF:-$VOL/models/geneformer}"
GF_REPO="ctheodoris/Geneformer"
GF_REV="${GF_REV:-1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5}"
MODELS="${MODELS:-V1-10M V2-104M V2-104M_CLcancer}"
BATCH="${BATCH:-16}"
RAW_MD5="${RAW_MD5:-e05c04a7eb410ea542f55ce4a5fa1ab3}"
FE=examples/perturbation_response/foundation-encoder
LOGS="$OUT/gf_embed_logs"
mkdir -p "$LOGS"

echo "=============================================================="
echo " R2.7 embedding stores  models: $MODELS  all cells"
echo "=============================================================="

# ---- scientific stack, as in the feasibility run (the image's torch needs numpy 1.x) -------- #
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
[ "$m" = "$RAW_MD5" ] || { echo "FATAL: raw h5ad missing or md5 '$m'"; fail=1; }
for f in foundation/cell_index.npz tokens_meta.npz manifest.json; do
  [ -f "$DATA/norman2019/$f" ] || { echo "FATAL: missing $DATA/norman2019/$f"; fail=1; }
done
m=$(md5sum "$OUT/norman_stage_a/checkpoints/encoder.pt" 2>/dev/null | cut -d' ' -f1); [ "${m:0:8}" = "38f51067" ] || { echo "FATAL: encoder md5 ${m:0:8}"; fail=1; }
[ "$fail" -eq 1 ] && { echo "preflight FAILED; nothing was run."; exit 1; }
echo "preflight OK"

# ---- install: Geneformer source + weights at the pinned commit (cached on the volume) ------- #
GF="$GF" GF_REPO="$GF_REPO" GF_REV="$GF_REV" MODELS="$MODELS" python - <<'PY' >> "$LOGS/install.log" 2>&1 || { echo "FATAL: download (see $LOGS/install.log)"; exit 1; }
import os, importlib.util
from huggingface_hub import snapshot_download
spec = importlib.util.spec_from_file_location("e", "examples/perturbation_response/foundation-encoder/02_embed_geneformer.py")
e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
dirs = [e.MODELS[m]["dir"] for m in os.environ["MODELS"].split()]
pats = ["geneformer/*", "requirements.txt", "setup.py", "MANIFEST.in"]
for d in dirs:
    pats += [f"{d}/config.json", f"{d}/generation_config.json", f"{d}/model.safetensors"]
print("downloaded to", snapshot_download(os.environ["GF_REPO"], revision=os.environ["GF_REV"], local_dir=os.environ["GF"], allow_patterns=pats))
PY
pip install -q -r "$GF/requirements.txt" -c "$CONSTRAINTS" >> "$LOGS/install.log" 2>&1 \
  || { echo "FATAL: Geneformer requirements (see $LOGS/install.log)"; tail -20 "$LOGS/install.log"; exit 1; }
python -c "import sys; sys.path.insert(0, '$GF'); import geneformer, transformers; print('geneformer OK, transformers', transformers.__version__)" \
  || { echo "FATAL: geneformer import"; exit 1; }

# SKIP_EMBED=1 resumes after the Geneformer step when its stores are already on the volume.
if [ "${SKIP_EMBED:-0}" = 1 ]; then
  [ -f "$OUT/geneformer_full/feasibility.json" ] || { echo "FATAL: SKIP_EMBED=1 but no geneformer_full/feasibility.json"; exit 1; }
  echo "skip smoke and Geneformer embedding: stores already written"
else
# ---- smoke: 64 cells, V1 and V2, stores must load ------------------------------------------- #
python $FE/02_embed_geneformer.py --gf-root "$GF" --data-dir "$DATA" --out-dir "$OUT/geneformer_smoke_stores" --n-cells 64 \
  --models V1-10M V2-104M --batch-size "$BATCH" --gf-revision "$GF_REV" > "$LOGS/smoke.log" 2>&1 \
  || { echo "SMOKE FAILED"; tail -40 "$LOGS/smoke.log"; exit 1; }
python - "$OUT/geneformer_smoke_stores" <<'PY' || { echo "SMOKE CHECK FAILED"; exit 1; }
import json, sys
from pathlib import Path
from ssllab.embeddings import load_store
root = Path(sys.argv[1])
r = json.load(open(root / "feasibility.json"))["models"]
bad = {k: v["error"] for k, v in r.items() if "error" in v}
for k in r:
    if "error" not in r[k]:
        s = load_store(root / "stores" / k, verify_content=True); print(k, s.n, s.width)
sys.exit(1 if bad or len(r) < 3 else 0)
PY
echo "smoke OK"

# ---- 1. Geneformer, every cell ---------------------------------------------------------------- #
# Four tokenizer workers, not eight: tokenizing every cell with eight ran a 46 GB pod out of memory.
rm -rf "$OUT/geneformer_full"
# shellcheck disable=SC2086
python $FE/02_embed_geneformer.py --gf-root "$GF" --data-dir "$DATA" --out-dir "$OUT/geneformer_full" --n-cells 0 \
  --models $MODELS --batch-size "$BATCH" --nproc "${NPROC:-4}" --gf-revision "$GF_REV" > "$LOGS/full.log" 2>&1 \
  || { echo "FULL EMBEDDING FAILED"; tail -40 "$LOGS/full.log"; exit 1; }
echo "geneformer stores written"
fi

# ---- 2. our JEPA encoder, same contract --------------------------------------------------------- #
# On the CPU: GPU kernels make the latents depend on the batch shape by up to ~2e-3 (see 04).
python $FE/04_export_jepa_latents.py --data-dir "$DATA" --encoder "$OUT/norman_stage_a/checkpoints/encoder.pt" \
  --out "$OUT/embeddings/norman_jepa_stage_a" --device cpu > "$LOGS/jepa_export.log" 2>&1 \
  || { echo "JEPA EXPORT FAILED"; tail -20 "$LOGS/jepa_export.log"; exit 1; }
echo "jepa store written"

# ---- 3. projections of the V2 stores ------------------------------------------------------------ #
python $FE/05_project_store.py --data-dir "$DATA" --out-root "$OUT/geneformer_full/stores_p256" \
  --parents "$OUT"/geneformer_full/stores/V2-104M_cls "$OUT"/geneformer_full/stores/V2-104M_cell \
            "$OUT"/geneformer_full/stores/V2-104M_CLcancer_cls "$OUT"/geneformer_full/stores/V2-104M_CLcancer_cell \
  > "$LOGS/project.log" 2>&1 || { echo "PROJECTION FAILED"; tail -20 "$LOGS/project.log"; exit 1; }
echo "projections written"

# ---- gate (d): every store covers every cell, on this cache, and matches its manifest ---------- #
python - "$OUT" "$DATA/norman2019" <<'PY' > "$LOGS/gate_d.log" 2>&1 || { echo "GATE (d) FAILED"; cat "$LOGS/gate_d.log"; exit 1; }
import json, sys
from pathlib import Path
import numpy as np
from ssllab.data.perturbseq import cache_fingerprint_from_dir
from ssllab.embeddings import load_store
out, cache = Path(sys.argv[1]), sys.argv[2]
fp = cache_fingerprint_from_dir(cache)
dirs = sorted((out / "geneformer_full/stores").iterdir()) + sorted(p for p in (out / "geneformer_full/stores_p256").iterdir() if p.is_dir())
dirs.append(out / "embeddings/norman_jepa_stage_a")
summary = {}
for d in dirs:
    s = load_store(d, expect_fingerprint=fp, verify_content=True)
    assert s.n == 109_737 and np.array_equal(s.row_idx, np.arange(109_737)), f"{d}: covers {s.n} rows"
    summary[str(d.relative_to(out))] = {k: s.manifest.get(k) for k in ("source", "model", "mode", "width", "n", "content_md5",
                                                                     "space_key", "explained_variance_share")}
    print("ok", d.name, s.n, s.width)
(out / "geneformer_full/stores_summary.json").write_text(json.dumps(summary, indent=2))
PY
echo "gate (d) passed"

# ---- 4. optional mechanism checks (descriptive; failures do not fail the job) ----------------- #
if [ -d "$OUT/geneformer_feasibility/tokens_V2.dataset" ]; then
  python $FE/08_mechanism_diagnostics.py --gf-root "$GF" --tokens "$OUT/geneformer_feasibility/tokens_V2.dataset" \
    --out-dir "$OUT/geneformer_mechanism" --batch-size "$BATCH" > "$LOGS/mechanism.log" 2>&1 \
    && echo "mechanism checks written" || echo "MECHANISM CHECKS FAILED (optional; see $LOGS/mechanism.log)"
else
  echo "skip mechanism checks: no feasibility tokens_V2.dataset"
fi

nvidia-smi --query-gpu=name,memory.total --format=csv
echo "[gf-embed] done. fetch: $OUT/geneformer_full/stores_summary.json $OUT/geneformer_full/feasibility.json $OUT/geneformer_full/stores_p256/projection_summary.json $OUT/embeddings/norman_jepa_stage_a/export_check.json $OUT/geneformer_mechanism/mechanism.json $LOGS"
