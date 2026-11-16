#!/usr/bin/env bash
# Round 15, pod job: the four variant stores, their projections, the coverage gate, then the comparison.
#
# Writes, under $OUT/geneformer_r15:
#   stores/v2rand_cell, v2panel_cell, v2cut_cell, v2big_cell   10_embed_variants.py, all 109,737 cells
#   stores_p256/<arm>                                          each projected to 256 (05), fit on training cells
#   stores_summary.json                                        gate (d): coverage, cache, contents
# then, unless COMPARE=0, runs ROUND=15 run_gf_compare_pod.sh: decoders, ceilings, projected flows, scoring
# (11_score_round15.py, the protocol) -> $OUT/gf15_compare.json.
#
# Inputs on the volume: the raw h5ad and cache (as for round 14), round 14's stores, decoders, ceilings and
# flows, and round 14's full V2 tokenized dataset (geneformer_full/tokens_V2.dataset; re-tokenized if gone).
#
# About 11 GPU-hours of embedding on one A40 (V2-316M alone about 5.8), then about 2 hours of comparison.
#
# Usage (on the pod, from the repo root):
#   bash examples/perturbation_response/foundation-encoder/run_gf15_embed_pod.sh

set -uo pipefail

VOL="${VOL:-/runpod-volume/ssl-lab}"
DATA="${DATA:-$VOL/data}"
OUT="${OUT:-$VOL/output}"
GF="${GF:-$VOL/models/geneformer}"
GF_REPO="ctheodoris/Geneformer"
GF_REV="${GF_REV:-1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5}"
BATCH="${BATCH:-16}"
RAW_MD5="${RAW_MD5:-e05c04a7eb410ea542f55ce4a5fa1ab3}"
FE=examples/perturbation_response/foundation-encoder
R15="$OUT/geneformer_r15"
LOGS="$OUT/gf15_embed_logs"
mkdir -p "$LOGS"

echo "=============================================================="
echo " Round 15 variant stores: random, panel, cut2048, big (V2-316M)"
echo "=============================================================="

# ---- scientific stack, as in round 14's embedding job ----------------------------------------- #
CONSTRAINTS="$LOGS/constraints.txt"
printf '%s\n' "numpy<2" "pandas<3" "pyarrow<22" > "$CONSTRAINTS"
pip install -q -c "$CONSTRAINTS" "numpy<2" "pandas<3" "pyarrow<22" anndata h5py huggingface_hub > "$LOGS/install.log" 2>&1 \
  || { echo "FATAL: base install (see $LOGS/install.log)"; tail -20 "$LOGS/install.log"; exit 1; }

# ---- preflight ------------------------------------------------------------------------------ #
fail=0
python -c "import torch; assert torch.cuda.is_available(); print('cuda:', torch.cuda.get_device_name(0))" || { echo "FATAL: no CUDA"; fail=1; }
m=$(md5sum "$DATA/norman2019/raw/norman_2019.h5ad" 2>/dev/null | cut -d' ' -f1)
[ "$m" = "$RAW_MD5" ] || { echo "FATAL: raw h5ad missing or md5 '$m'"; fail=1; }
for f in "$OUT/gf_compare.json" "$OUT/geneformer_full/stores/V2-104M_cell/manifest.json" \
         "$OUT/geneformer_full/stores_p256/V2-104M_cell/manifest.json" "$OUT/embeddings/norman_jepa_stage_a/manifest.json"; do
  [ -f "$f" ] || { echo "FATAL: missing $f (round 14)"; fail=1; }
done
[ -d "$OUT/geneformer_full/tokens_V2.dataset" ] && TOKENS=(--tokens "$OUT/geneformer_full/tokens_V2.dataset") \
  || { echo "note: round 14's tokens_V2.dataset is gone; the variants re-tokenize every cell"; TOKENS=(); }
[ "$fail" -eq 1 ] && { echo "preflight FAILED; nothing was run."; exit 1; }
echo "preflight OK"

# ---- Geneformer source and weights at the pinned commit, V2-104M and V2-316M -------------------- #
GF="$GF" GF_REPO="$GF_REPO" GF_REV="$GF_REV" python - <<'PY' >> "$LOGS/install.log" 2>&1 || { echo "FATAL: download (see $LOGS/install.log)"; exit 1; }
import os
from huggingface_hub import snapshot_download
pats = ["geneformer/*", "requirements.txt", "setup.py", "MANIFEST.in"]
for d in ("Geneformer-V2-104M", "Geneformer-V2-316M"):
    pats += [f"{d}/config.json", f"{d}/generation_config.json", f"{d}/model.safetensors"]
print("downloaded to", snapshot_download(os.environ["GF_REPO"], revision=os.environ["GF_REV"], local_dir=os.environ["GF"], allow_patterns=pats))
PY
pip install -q -r "$GF/requirements.txt" -c "$CONSTRAINTS" >> "$LOGS/install.log" 2>&1 \
  || { echo "FATAL: Geneformer requirements (see $LOGS/install.log)"; tail -20 "$LOGS/install.log"; exit 1; }
python -c "import sys; sys.path.insert(0, '$GF'); import geneformer, transformers; print('geneformer OK, transformers', transformers.__version__)" \
  || { echo "FATAL: geneformer import"; exit 1; }

# ---- smoke: 64 cells, every variant, stores must load --------------------------------------------- #
rm -rf "$OUT/geneformer_r15_smoke"
python $FE/10_embed_variants.py --gf-root "$GF" --data-dir "$DATA" --out-dir "$OUT/geneformer_r15_smoke" --n-cells 64 \
  "${TOKENS[@]}" --batch-size "$BATCH" --gf-revision "$GF_REV" > "$LOGS/smoke.log" 2>&1 \
  || { echo "SMOKE FAILED"; tail -40 "$LOGS/smoke.log"; exit 1; }
python - "$OUT/geneformer_r15_smoke" <<'PY' || { echo "SMOKE CHECK FAILED"; exit 1; }
import sys
from pathlib import Path
from ssllab.embeddings import load_store
for arm in ("v2rand_cell", "v2panel_cell", "v2cut_cell", "v2big_cell"):
    s = load_store(Path(sys.argv[1]) / "stores" / arm, verify_content=True); print(arm, s.n, s.width)
    assert s.n == 64, f"{arm}: {s.n} of 64 cells"
PY
echo "smoke OK"

# ---- 1. the four variant stores, every cell ------------------------------------------------------ #
rm -rf "$R15"
python $FE/10_embed_variants.py --gf-root "$GF" --data-dir "$DATA" --out-dir "$R15" --n-cells 0 \
  "${TOKENS[@]}" --batch-size "$BATCH" --nproc "${NPROC:-4}" --gf-revision "$GF_REV" > "$LOGS/full.log" 2>&1 \
  || { echo "VARIANT EMBEDDING FAILED"; tail -40 "$LOGS/full.log"; exit 1; }
echo "variant stores written"

# ---- 2. projections to 256, fit on training cells -------------------------------------------------- #
python $FE/05_project_store.py --data-dir "$DATA" --out-root "$R15/stores_p256" \
  --parents "$R15"/stores/v2rand_cell "$R15"/stores/v2panel_cell "$R15"/stores/v2cut_cell "$R15"/stores/v2big_cell \
  > "$LOGS/project.log" 2>&1 || { echo "PROJECTION FAILED"; tail -20 "$LOGS/project.log"; exit 1; }
echo "projections written"

# ---- gate (d) ------------------------------------------------------------------------------------ #
python - "$R15" "$DATA/norman2019" <<'PY' > "$LOGS/gate_d.log" 2>&1 || { echo "GATE (d) FAILED"; cat "$LOGS/gate_d.log"; exit 1; }
import json, sys
from pathlib import Path
import numpy as np
from ssllab.data.perturbseq import cache_fingerprint_from_dir
from ssllab.embeddings import load_store
r15 = Path(sys.argv[1]); fp = cache_fingerprint_from_dir(sys.argv[2]); summary = {}
for d in sorted((r15 / "stores").iterdir()) + sorted(p for p in (r15 / "stores_p256").iterdir() if p.is_dir()):
    s = load_store(d, expect_fingerprint=fp, verify_content=True)
    assert s.n == 109_737 and np.array_equal(s.row_idx, np.arange(109_737)), f"{d}: covers {s.n} rows"
    summary[str(d.relative_to(r15))] = {k: s.manifest.get(k) for k in ("model", "variant", "width", "n", "content_md5",
                                                                     "explained_variance_share")}
    print("ok", d, s.n, s.width)
(r15 / "stores_summary.json").write_text(json.dumps(summary, indent=2))
PY
echo "gate (d) passed"
nvidia-smi --query-gpu=name,memory.total --format=csv

# ---- 3. the comparison ---------------------------------------------------------------------------- #
if [ "${COMPARE:-1}" = "1" ]; then
  ROUND=15 bash $FE/run_gf_compare_pod.sh || { echo "ROUND 15 COMPARISON FAILED"; exit 1; }
fi
echo "[gf15] done. fetch: $R15/stores_summary.json $R15/variants.json $R15/stores_p256/projection_summary.json $OUT/gf15_compare.json $LOGS $OUT/gf_logs"
