#!/usr/bin/env bash
# Regularizer swap — Stage A with SIGReg (LeJEPA) in place of VICReg.
#
# Trains nine Stage A encoders (three arms x seeds 0, 1, 2), then reads each one with the
# encoder-only diagnostics, then runs the frozen comparison. Nothing downstream (decoder,
# flow) is retrained here; that happens only if an encoder passes (22's decision rule).
#
#   arm  vicreg       VICReg on the context-token embeddings, weight 0.04 (the existing recipe)
#   arm  sigreg_tok   SIGReg on the context-token embeddings, 0.95*pred + 0.05*SIGReg
#   arm  sigreg_pool  SIGReg on the cell vector (mean of the context tokens), same weight
#
# Steps
#   0. gate: re-score the existing VICReg encoder (norman_stage_a, md5 38f51067) with the
#      linear readout and check it reproduces 0.85214 to 1e-4. Stops here if it does not, so
#      no GPU hours are spent on a harness that has drifted.
#   1. train: 01_pretrain_stage_a.py, PARALLEL runs at a time on the one GPU.
#   2. read:  14_ceiling_analysis.py (identity + linear, profiles saved) and
#             02_probe_cell_encoder.py for every new encoder.
#   3. compare: 22_compare_encoders.py -> $OUT/sigreg_compare.json (protocol in its docstring).
#
# Usage (on the pod, from the repo root):
#   bash examples/perturbation_response/run_sigreg_pod.sh
#   ARMS=sigreg_pool SEEDS=0 EPOCHS=1 LIMIT=3000 SKIP_COMPARE=1 bash examples/perturbation_response/run_sigreg_pod.sh  # smoke

set -uo pipefail

VOL="${VOL:-/runpod-volume/ssl-lab}"
DATA="${DATA:-$VOL/data}"
OUT="${OUT:-$VOL/output}"
EXISTING="${EXISTING:-$OUT/norman_stage_a/checkpoints/encoder.pt}"
DECODER="${DECODER:-$OUT/norman_flow_control/checkpoints/count_decoder.pt}"   # 14 loads it; the linear arm does not use it
EPOCHS="${EPOCHS:-50}"
SEEDS="${SEEDS:-0 1 2}"
ARMS="${ARMS:-vicreg sigreg_tok sigreg_pool}"
PARALLEL="${PARALLEL:-3}"
LIMIT="${LIMIT:-}"
SKIP_COMPARE="${SKIP_COMPARE:-0}"
DE_MD5="${DE_MD5:-9844c2c25d2903872e1c0a92c722f105}"   # data/norman2019/de_genes.json used for every graded number
LOGS="$OUT/sigreg_logs"
mkdir -p "$LOGS"

echo "=============================================================="
echo " REGULARIZER SWAP — Stage A: VICReg vs SIGReg"
echo " volume : $VOL"
echo " arms   : $ARMS      seeds: $SEEDS      epochs: $EPOCHS   parallel: $PARALLEL"
echo "=============================================================="

# ---- preflight ------------------------------------------------------------------------- #
fail=0
python -c "import torch; assert torch.cuda.is_available(); print('cuda:', torch.cuda.get_device_name(0))" \
  || { echo "FATAL: no CUDA"; fail=1; }
[ -f "$DATA/norman2019/manifest.json" ] || { echo "FATAL: cache missing at $DATA/norman2019"; fail=1; }
got_md5=$(md5sum "$DATA/norman2019/de_genes.json" 2>/dev/null | cut -d' ' -f1)
[ "$got_md5" = "$DE_MD5" ] || { echo "FATAL: de_genes.json md5 $got_md5 != $DE_MD5 (stage the current cache)"; fail=1; }
enc_md5=$(md5sum "$EXISTING" 2>/dev/null | cut -d' ' -f1)
[ "${enc_md5:0:8}" = "38f51067" ] || { echo "FATAL: existing encoder md5 ${enc_md5:0:8} != 38f51067"; fail=1; }
[ -f "$DECODER" ] || { echo "FATAL: decoder missing at $DECODER"; fail=1; }
[ "$fail" -eq 1 ] && { echo "preflight FAILED; nothing was run."; exit 1; }
echo "preflight OK"

read_encoder() {   # $1 = encoder checkpoint, $2 = arm, $3 = seed
  python examples/perturbation_response/14_ceiling_analysis.py \
    --encoder "$1" --decoder "$DECODER" --split combo --arms identity linear --save-profiles \
    --experiment "norman_ceiling_$2_s$3" --output-root "$OUT" --data-dir "$DATA" \
    > "$LOGS/ceiling_$2_s$3.log" 2>&1 || { echo "CEILING FAILED: $2 s$3"; return 1; }
  python examples/perturbation_response/02_probe_cell_encoder.py \
    --encoder "$1" --experiment "norman_stage_a_$2_s$3" --output-root "$OUT" --data-dir "$DATA" \
    > "$LOGS/probe_$2_s$3.log" 2>&1 || { echo "PROBE FAILED: $2 s$3"; return 1; }
}

# ---- 0. gate: the existing encoder must re-score as stored ------------------------------- #
echo ""
echo "---- step 0: re-score the existing VICReg encoder ----"
read_encoder "$EXISTING" vicreg 42 || exit 1
python - "$OUT/norman_ceiling_vicreg_s42/reports/ceiling.json" <<'EOF' || { echo "GATE FAILED; stopping before training."; exit 1; }
import json, sys
v = json.load(open(sys.argv[1]))["arms"]["linear"]["mean_delta_r"]
print(f"existing encoder linear readout {v:.5f} (stored 0.85214)")
sys.exit(0 if abs(v - 0.85214) <= 1e-4 else 1)
EOF

# ---- 1. train ------------------------------------------------------------------------------ #
limit_flag=()
[ -n "$LIMIT" ] && limit_flag=(--limit "$LIMIT")
train_one() {   # $1 = arm, $2 = seed
  case "$1" in
    vicreg)      reg=(--reg-kind vicreg --reg-coef 0.04) ;;
    sigreg_tok)  reg=(--reg-kind sigreg --reg-target tokens --reg-coef 0.05) ;;
    sigreg_pool) reg=(--reg-kind sigreg --reg-target pooled --reg-coef 0.05) ;;
    *) echo "unknown arm '$1'"; return 2 ;;
  esac
  python examples/perturbation_response/01_pretrain_stage_a.py \
    --experiment "norman_stage_a_$1_s$2" --output-root "$OUT" --data-dir "$DATA" \
    --split combo --epochs "$EPOCHS" --seed "$2" "${reg[@]}" "${limit_flag[@]}" \
    > "$LOGS/train_$1_s$2.log" 2>&1 && echo "trained $1 s$2" || echo "TRAIN FAILED: $1 s$2"
}
echo ""
echo "---- step 1: train ($PARALLEL at a time) ----"
running=0
for s in $SEEDS; do
  for arm in $ARMS; do
    train_one "$arm" "$s" &
    running=$((running + 1))
    if [ "$running" -ge "$PARALLEL" ]; then wait -n; running=$((running - 1)); fi
  done
done
wait

# ---- 2. read every new encoder ----------------------------------------------------------- #
echo ""
echo "---- step 2: linear readout and probe ----"
for s in $SEEDS; do
  for arm in $ARMS; do
    enc="$OUT/norman_stage_a_${arm}_s${s}/checkpoints/encoder.pt"
    [ -f "$enc" ] || { echo "skip $arm s$s (no encoder)"; continue; }
    read_encoder "$enc" "$arm" "$s" && echo "read $arm s$s"
  done
done

# ---- 3. compare ---------------------------------------------------------------------------- #
if [ "$SKIP_COMPARE" != "1" ]; then
  echo ""
  echo "---- step 3: frozen comparison ----"
  python examples/perturbation_response/22_compare_encoders.py \
    --output-root "$OUT" --data-dir "$DATA" --out "$OUT/sigreg_compare.json" 2>&1 | tee "$LOGS/compare.log"
fi

echo ""
echo "=============================================================="
echo " done. fetch narrowly:"
echo "   rsync -Pavz -u <cluster>:$OUT/sigreg_compare.json <cluster>:$OUT/sigreg_logs output/"
echo "   (encoders: <cluster>:$OUT/norman_stage_a_{vicreg,sigreg_tok,sigreg_pool}_s*)"
echo "=============================================================="
