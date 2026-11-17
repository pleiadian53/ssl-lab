#!/usr/bin/env bash
# R2.7, pod job 2: the comparison. Geneformer cell embeddings against our JEPA encoder, frozen.
#
# Every arm reads its cell vectors from an embedding store written by run_gf_embed_pod.sh; the
# protocol, arms and decision rule are in 07_score_comparison.py's docstring. Per arm and seed
# (0, 1, 2):
#   decoder    co-adaptation/02_coadapt_decoder.py --mode dec --latents <store> (round-11 recipe)
#   ceiling    14_ceiling_analysis.py --latents <store>: identity, roundtrip, linear, profiles saved
#   pipeline   04_train_cond_flow.py --latents <store> (round-1 transport flow, hidden = width),
#              read through the same seed's decoder: 06, 19, 10
#
# One job, in order:
#   preflight  inputs on the volume; CUDA; md5s; every store loads and matches its manifest
#   gate (e)   the JEPA store reproduces the published round-11 ceilings through the new path
#   0. smoke   the whole job at tiny settings (seed 0, one epoch, 2,000 cells) in a separate output
#              directory that links the stores, then 07 --smoke; any failure stops
#   1. full    train, read, grade
#   2. compare 07_score_comparison.py -> $OUT/gf_compare.json
#
# FLOWS_ONLY="<arms>" replaces steps 0 and 1 with a rerun of only those arms' flows and their
# readouts (06, 19, 10) at full settings; used for amendments 1 and 2 (the four 768-wide arms).
#
# LOCAL=1 runs the same steps on a laptop CPU (no CUDA check, md5 via `md5 -q`), for the code smoke.
#
# Usage (on the pod, from the repo root):
#   bash examples/perturbation_response/foundation-encoder/run_gf_compare_pod.sh

set -uo pipefail

VOL="${VOL:-/runpod-volume/ssl-lab}"
DATA="${DATA:-$VOL/data}"
OUT="${OUT:-$VOL/output}"
LOCAL="${LOCAL:-0}"
DEV="${DEV:-cuda}"
EP=examples/perturbation_response
FE=$EP/foundation-encoder
DE_MD5="${DE_MD5:-9844c2c25d2903872e1c0a92c722f105}"
SMOKE_ROOT="${SMOKE_ROOT:-$VOL/smoke_gf}"
SMOKE="$SMOKE_ROOT/output"
ONLY_SMOKE="${ONLY_SMOKE:-0}"

md5of() { if command -v md5sum >/dev/null 2>&1; then md5sum "$1" 2>/dev/null | cut -d' ' -f1; else md5 -q "$1" 2>/dev/null; fi; }

# arm -> store under the output root ; arm -> flow hidden width (decision D19). Functions rather
# than associative arrays, so the script also runs under the bash 3.2 macOS ships (LOCAL=1).
store_of() {
  case "$1" in
    jepa) echo embeddings/norman_jepa_stage_a ;;
    v1_cell) echo geneformer_full/stores/V1-10M_cell ;;
    v2_cls) echo geneformer_full/stores/V2-104M_cls ;;
    v2_cell) echo geneformer_full/stores/V2-104M_cell ;;
    v2cl_cls) echo geneformer_full/stores/V2-104M_CLcancer_cls ;;
    v2cl_cell) echo geneformer_full/stores/V2-104M_CLcancer_cell ;;
    v2_cls_p) echo geneformer_full/stores_p256/V2-104M_cls ;;
    v2_cell_p) echo geneformer_full/stores_p256/V2-104M_cell ;;
    v2cl_cls_p) echo geneformer_full/stores_p256/V2-104M_CLcancer_cls ;;
    v2cl_cell_p) echo geneformer_full/stores_p256/V2-104M_CLcancer_cell ;;
    v2rand_cell|v2panel_cell|v2cut_cell|v2big_cell) echo "geneformer_r15/stores/$1" ;;           # round 15
    v2rand_cell_p|v2panel_cell_p|v2cut_cell_p|v2big_cell_p) echo "geneformer_r15/stores_p256/${1%_p}" ;;
    *) echo "unknown arm $1" >&2; return 1 ;;
  esac
}
hidden_of() { case "$1" in v2_cls|v2_cell|v2cl_cls|v2cl_cell|v2rand_cell|v2panel_cell|v2cut_cell) echo 768 ;;
                           v2big_cell) echo 1152 ;; *) echo 256 ;; esac; }
# Amendment 2 (07's docstring): the 768-wide flows train at learning rate 3e-4; the others at 04's default.
lr_of() { case "$1" in v2_cls|v2_cell|v2cl_cls|v2cl_cell) echo 3e-4 ;; *) echo "" ;; esac; }
# ROUND picks the arms and the scorer. Round 15 (11_score_round15.py) trains the new arms only and re-reads
# round 14's v2_cell, jepa and v2_cell_p; its smoke pass trains those too, so the scorer can run on it.
ROUND="${ROUND:-14}"
if [ "$ROUND" = "15" ]; then
  ARMS="${ARMS:-v2rand_cell v2panel_cell v2cut_cell v2big_cell v2rand_cell_p v2panel_cell_p v2cut_cell_p v2big_cell_p}"
  SMOKE_ARMS="${SMOKE_ARMS:-jepa v2_cell v2_cell_p $ARMS}"
  PIPELINE_ARMS="${PIPELINE_ARMS:-jepa v2_cell_p v2rand_cell_p v2panel_cell_p v2cut_cell_p v2big_cell_p}"
  SCORER="$FE/11_score_round15.py"; SCORE_JSON=gf15_compare.json; SCORE_LOG=compare15.log
else
  ARMS="${ARMS:-jepa v1_cell v2_cls v2_cell v2cl_cls v2cl_cell v2_cls_p v2_cell_p v2cl_cls_p v2cl_cell_p}"
  SMOKE_ARMS="${SMOKE_ARMS:-$ARMS}"
  PIPELINE_ARMS="${PIPELINE_ARMS:-$ARMS}"
  SCORER="$FE/07_score_comparison.py"; SCORE_JSON=gf_compare.json; SCORE_LOG=compare.log
fi
in_list() { case " $2 " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

# ---- preflight ------------------------------------------------------------------------------ #
fail=0
if [ "$LOCAL" != "1" ]; then
  python -c "import torch; assert torch.cuda.is_available(); print('cuda:', torch.cuda.get_device_name(0))" || { echo "FATAL: no CUDA"; fail=1; }
fi
[ "$(md5of "$DATA/norman2019/de_genes.json")" = "$DE_MD5" ] || { echo "FATAL: de_genes md5"; fail=1; }
[ -f "$DATA/norman2019/foundation/visible_hvg.json" ] || { echo "FATAL: missing foundation/visible_hvg.json (06_visible_genes.py)"; fail=1; }
need=(norman_additive/reports/effect_size_additive.json norman_additive/reports/profiles_additive.npz)
for s in ${SEEDS:-0 1 2}; do   # SEEDS: local tests only
  need+=("norman_r11_dec_hvg_s$s/checkpoints/count_decoder.pt"
         "norman_r11_vae_hvg_s$s/reports/effect_size_cvae.json" "norman_r11_vae_hvg_s$s/reports/profiles_vae.npz")
done
for f in "${need[@]}"; do [ -f "$OUT/$f" ] || { echo "FATAL: missing $OUT/$f"; fail=1; }; done
for a in $SMOKE_ARMS; do [ -f "$OUT/$(store_of $a)/manifest.json" ] || { echo "FATAL: missing store $OUT/$(store_of $a)"; fail=1; }; done
[ "$fail" -eq 1 ] && { echo "PREFLIGHT FAILED; nothing was run."; exit 1; }
stores=(); for a in $SMOKE_ARMS; do stores+=("$OUT/$(store_of $a)"); done
python - "$DATA/norman2019" "${stores[@]}" <<'PY' || { echo "PREFLIGHT FAILED: a store does not load or match its manifest"; exit 1; }
import sys
from ssllab.data.perturbseq import cache_fingerprint_from_dir
from ssllab.embeddings import load_store
fp = cache_fingerprint_from_dir(sys.argv[1])
for d in sys.argv[2:]:
    s = load_store(d, expect_fingerprint=fp, verify_content=True); print("store ok", d, s.n, s.width)
PY
echo "preflight OK"

# ---- gate (e): the JEPA store through the new path reproduces the published ceilings ----------- #
G="$OUT/norman_gf_gate2"; mkdir -p "$G"
for s in ${SEEDS:-0 1 2}; do   # SEEDS: local tests only
  python $EP/14_ceiling_analysis.py --latents "$OUT/$(store_of jepa)" --decoder "$OUT/norman_r11_dec_hvg_s$s/checkpoints/count_decoder.pt" \
    --split combo --arms identity roundtrip --experiment "norman_gf_gate2_s$s" --output-root "$OUT" --data-dir "$DATA" \
    --device "$DEV" > "$G/gate2_s$s.log" 2>&1 || { echo "GATE (e) FAILED: ceiling s$s"; exit 1; }
done
python - "$OUT" <<'PY' || { echo "GATE (e) FAILED"; exit 1; }
import json, sys
from pathlib import Path
out = Path(sys.argv[1]); ref = {0: 0.86695, 1: 0.86338, 2: 0.86502}
got = {s: json.load(open(out / f"norman_gf_gate2_s{s}/reports/ceiling.json"))["arms"]["roundtrip"]["mean_delta_r"] for s in ref}
ok = all(abs(got[s] - ref[s]) <= 1e-4 for s in ref)
(out / "norman_gf_gate2/gate2.json").write_text(json.dumps({"roundtrip": {str(s): v for s, v in got.items()},
                                                            "reference": {str(s): v for s, v in ref.items()}, "passed": ok}, indent=2))
print("gate (e):", got, "passed" if ok else "FAILED"); sys.exit(0 if ok else 1)
PY
echo "gate (e) passed"

run_parallel() {   # runs the given commands, at most PARALLEL at a time
  # Counts the jobs still running instead of trusting `wait -n`, which can return at once for a job
  # that finished earlier; round 15's first attempt ran four memory-heavy ceilings together that way,
  # and the pod's container was killed. `jobs -rp` and `sleep` also work in bash 3.2.
  for job in "$@"; do
    while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 2; done
    eval "$job" &
  done
  wait
}

train() {   # $1 = arm, $2 = seed
  python $EP/co-adaptation/02_coadapt_decoder.py --experiment "norman_gf_dec_$1_s$2" --mode dec --latents "$O/$(store_of $1)" \
    --library-size hvg --split combo --epochs "$EPOCHS" --seed "$2" --output-root "$O" --data-dir "$DATA" --device "$DEV" \
    "${limit_flag[@]}" > "$LOGS/train_$1_s$2.log" 2>&1 && echo "trained $1 s$2" || echo "TRAIN FAILED: $1 s$2"
}
ceiling() {   # $1 = arm, $2 = seed
  python $EP/14_ceiling_analysis.py --latents "$O/$(store_of $1)" --decoder "$O/norman_gf_dec_$1_s$2/checkpoints/count_decoder.pt" \
    --split combo --arms identity roundtrip linear --save-profiles "${ridge_flag[@]}" --experiment "norman_gf_ceiling_$1_s$2" \
    --output-root "$O" --data-dir "$DATA" --device "$DEV" > "$LOGS/ceiling_$1_s$2.log" 2>&1 \
    && echo "read $1 s$2" || echo "CEILING FAILED: $1 s$2"
}
pipeline() {   # $1 = arm, $2 = seed
  local exp="norman_gf_flow_$1_s$2" lr
  local lr_flag=(); lr=$(lr_of "$1"); [ -n "$lr" ] && lr_flag=(--lr "$lr")
  rm -rf "${O:?}/$exp"; mkdir -p "$O/$exp/checkpoints"
  python $EP/04_train_cond_flow.py --experiment "$exp" --latents "$O/$(store_of $1)" --output-root "$O" --data-dir "$DATA" \
    --split combo --cond-type geneset --flow-base control --coupling independent --epochs "$FLOW_EPOCHS" \
    --hidden "$(hidden_of $1)" ${lr_flag[@]+"${lr_flag[@]}"} --seed "$2" --device "$DEV" "${limit_flag[@]}" > "$LOGS/flow_$1_s$2.log" 2>&1 \
    || { echo "FLOW FAILED: $1 s$2"; return; }
  cp "$O/norman_gf_dec_$1_s$2/checkpoints/count_decoder.pt" "$O/$exp/checkpoints/count_decoder.pt"
  python $EP/06_eval_effect_size.py --experiment "$exp" --split combo --output-root "$O" --data-dir "$DATA" --device "$DEV" \
    > "$LOGS/eff_$exp.log" 2>&1 || echo "EFFECT FAILED: $exp"
  python $EP/19_score_interaction.py --experiment "$exp" --kind flow --output-root "$O" --data-dir "$DATA" \
    --device "$DEV" > "$LOGS/int_$exp.log" 2>&1 || echo "19 FAILED: $exp"
  python $EP/10_eval_calibration.py --experiment "$exp" --model flow --split combo --output-root "$O" \
    --data-dir "$DATA" --device "$DEV" > "$LOGS/pop_$exp.log" 2>&1 || echo "POP FAILED: $exp"
  echo "pipeline $1 s$2"
}

compare_round() {   # trains and reads every arm of RUN_ARMS under the current O, SEEDS, EPOCHS, ...
  LOGS="$O/gf_logs"; mkdir -p "$LOGS"
  limit_flag=(); [ -n "$LIMIT" ] && limit_flag=(--limit "$LIMIT")
  ridge_flag=(); [ -n "$RIDGE_MAX" ] && ridge_flag=(--ridge-max-cells "$RIDGE_MAX")
  # SKIP_DONE=1 resumes an interrupted run: a step whose final output is already on the volume is not
  # rerun (decoder checkpoint; ceiling.json; a pipeline's calibration_flow.json, its last report).
  local skip="${SKIP_DONE:-0}" jobs=()
  for s in $SEEDS; do for a in $RUN_ARMS; do
    [ "$skip" = 1 ] && [ -f "$O/norman_gf_dec_${a}_s$s/checkpoints/count_decoder.pt" ] && { echo "done: train $a s$s"; continue; }
    jobs+=("train $a $s")
  done; done
  [ ${#jobs[@]} -gt 0 ] && run_parallel "${jobs[@]}"
  jobs=()
  for s in $SEEDS; do
    for a in $RUN_ARMS; do
      [ -f "$O/norman_gf_dec_${a}_s$s/checkpoints/count_decoder.pt" ] || { echo "skip $a s$s (no decoder)"; continue; }
      if [ "$skip" = 1 ] && [ -f "$O/norman_gf_ceiling_${a}_s$s/reports/ceiling.json" ]; then echo "done: ceiling $a s$s"
      else jobs+=("ceiling $a $s"); fi
      in_list "$a" "$PIPELINE_ARMS" || continue
      if [ "$skip" = 1 ] && [ -f "$O/norman_gf_flow_${a}_s$s/reports/calibration_flow.json" ]; then echo "done: pipeline $a s$s"
      else jobs+=("pipeline $a $s"); fi
    done
  done
  [ ${#jobs[@]} -gt 0 ] || return 0
  # Each full ceiling fits its ridge readout in float64 on every training cell, about 10 GB of host
  # memory, so the readouts run READ_PARALLEL at a time (default 2) within the pod's 46.6 GB container.
  PARALLEL="${READ_PARALLEL:-2}" run_parallel "${jobs[@]}"
}

if [ -n "${FLOWS_ONLY:-}" ]; then
# ---- FLOWS_ONLY="<arms>": rerun only these arms' flows and their readouts, at full settings ------ #
# Decoders and ceilings stay as they are. The previous attempt's logs are kept in gf_logs/<attempt>.
echo "==== flows only: $FLOWS_ONLY ===="
O="$OUT" FLOW_EPOCHS="${FLOW_EPOCHS:-60}" PARALLEL="${PARALLEL:-3}"; LOGS="$O/gf_logs"
limit_flag=(); [ -n "${LIMIT:-}" ] && limit_flag=(--limit "$LIMIT")   # LIMIT and FLOW_EPOCHS: local tests only
KEEP="$LOGS/${FLOWS_ATTEMPT:-attempt1}"; mkdir -p "$KEEP"
jobs=()
for s in ${SEEDS:-0 1 2}; do   # SEEDS: local tests only
  for a in $FLOWS_ONLY; do
    [ -f "$O/norman_gf_dec_${a}_s$s/checkpoints/count_decoder.pt" ] || { echo "FATAL: no decoder for $a s$s"; exit 1; }
    for f in "flow_${a}_s$s" "eff_norman_gf_flow_${a}_s$s" "int_norman_gf_flow_${a}_s$s" "pop_norman_gf_flow_${a}_s$s"; do
      [ -f "$LOGS/$f.log" ] && mv "$LOGS/$f.log" "$KEEP/"
    done
    jobs+=("pipeline $a $s")
  done
done
run_parallel "${jobs[@]}"
else
# ---- 0. smoke ------------------------------------------------------------------------------ #
echo "==== 0. smoke ===="
rm -rf "$SMOKE_ROOT"; mkdir -p "$SMOKE"
ln -s "$OUT/embeddings" "$SMOKE/embeddings"; ln -s "$OUT/geneformer_full" "$SMOKE/geneformer_full"
ln -s "$OUT/norman_gf_gate2" "$SMOKE/norman_gf_gate2"
[ -d "$OUT/geneformer_r15" ] && ln -s "$OUT/geneformer_r15" "$SMOKE/geneformer_r15"
for d in norman_additive/reports norman_r11_vae_hvg_s0/reports; do mkdir -p "$SMOKE/$d"; cp -r "$OUT/$d/." "$SMOKE/$d/"; done
O="$SMOKE" SEEDS=0 EPOCHS=1 FLOW_EPOCHS=1 LIMIT=2000 RIDGE_MAX=3000 PARALLEL="${SMOKE_PARALLEL:-4}" RUN_ARMS="$SMOKE_ARMS"
compare_round > "$SMOKE_ROOT/compare_round.out" 2>&1
python "$SCORER" --output-root "$SMOKE" --data-dir "$DATA" --seeds 0 --smoke --n-boot 500 \
  --out "$SMOKE_ROOT/gf_compare_smoke.json" > "$SMOKE_ROOT/compare.out" 2>&1 || echo "SMOKE: scorer FAILED"
bad=$(grep -lE "FAILED|Traceback" "$SMOKE_ROOT"/*.out "$SMOKE"/gf_logs/*.log 2>/dev/null)
if [ -n "$bad" ] || [ ! -f "$SMOKE_ROOT/gf_compare_smoke.json" ]; then
  echo "SMOKE FAILED; nothing full was run. Problem files:"; echo "$bad"; exit 1
fi
echo "smoke passed"
[ "$ONLY_SMOKE" = "1" ] && { echo "ONLY_SMOKE=1: stopping after the smoke"; exit 0; }

# ---- 1. full ------------------------------------------------------------------------------- #
echo "==== 1. comparison ===="
O="$OUT" SEEDS="0 1 2" EPOCHS=30 FLOW_EPOCHS=60 LIMIT="" RIDGE_MAX="" PARALLEL="${PARALLEL:-3}" RUN_ARMS="$ARMS"
compare_round
fi

# ---- 2. compare ---------------------------------------------------------------------------- #
echo "==== 2. scoring ===="
mkdir -p "$OUT/gf_logs"
python "$SCORER" --output-root "$OUT" --data-dir "$DATA" --out "$OUT/$SCORE_JSON" \
  > "$OUT/gf_logs/$SCORE_LOG" 2>&1 || echo "SCORER FAILED"
head -60 "$OUT/gf_logs/$SCORE_LOG"
echo "done. fetch: $OUT/$SCORE_JSON $OUT/gf_logs and the JSON reports of norman_gf_*"
