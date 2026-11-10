#!/usr/bin/env bash
# One pod job: the re-baseline after round 11, then round 12 (co-adaptation Phase 1), then both
# comparisons, all on the pod. Only the comparison JSONs and logs need fetching.
#
#   0. smoke: both runners and both comparisons at tiny settings (one seed, one epoch, 2,000 cells)
#      in a separate output directory built from copies of the inputs; any failure stops the job
#   1. run_rebaseline_pod.sh                    (full)
#   2. co-adaptation/run_coadapt_pod.sh         (full; protocol in 26_score_round12.py)
#   3. 19 on the additive reference (control profile), 25_rebaseline.py, 26_score_round12.py
#
# Usage (on the pod, from the repo root):
#   bash examples/perturbation_response/run_rebaseline_round12_pod.sh

set -uo pipefail
VOL="${VOL:-/runpod-volume/ssl-lab}"
DATA="${DATA:-$VOL/data}"
OUT="${OUT:-$VOL/output}"
EP=examples/perturbation_response
SMOKE="$VOL/smoke_r12/output"

echo "==== 0. smoke on the pod ===="
rm -rf "$VOL/smoke_r12"; mkdir -p "$SMOKE"
for d in norman_stage_a/checkpoints norman_r11_dec_hvg_s0/checkpoints norman_r11_vae_hvg_s0/checkpoints \
         norman_r11_vae_hvg_s0/reports norman_sweep_gaussian_s0/checkpoints norman_sweep_control_s0/checkpoints \
         norman_sweep_control_ot_s0/checkpoints norman_operator_s0/checkpoints norman_operator_stoch_s0/checkpoints \
         norman_additive/reports norman_ceiling_r10/reports norman_ceiling_r11_hvg_s0/reports; do
  mkdir -p "$SMOKE/$d"; cp -r "$OUT/$d/." "$SMOKE/$d/"
done
smoke_env=(OUT="$SMOKE" DATA="$DATA" SEEDS=0 EPOCHS=1 VAE_EPOCHS=1 FLOW_EPOCHS=1 LIMIT=2000 RIDGE_MAX=3000 PARALLEL=4)
env "${smoke_env[@]}" bash $EP/run_rebaseline_pod.sh > "$VOL/smoke_r12/rebaseline.out" 2>&1
env "${smoke_env[@]}" bash $EP/co-adaptation/run_coadapt_pod.sh > "$VOL/smoke_r12/coadapt.out" 2>&1
python $EP/19_score_interaction.py --experiment norman_additive --kind additive --output-root "$SMOKE" --data-dir "$DATA" \
  > "$VOL/smoke_r12/additive.out" 2>&1
python $EP/25_rebaseline.py --output-root "$SMOKE" --data-dir "$DATA" --seeds 0 --out "$VOL/smoke_r12/rb.json" \
  > "$VOL/smoke_r12/compare25.out" 2>&1 || echo "SMOKE: 25 FAILED"
python $EP/26_score_round12.py --output-root "$SMOKE" --data-dir "$DATA" --seeds 0 --out "$VOL/smoke_r12/r12.json" \
  > "$VOL/smoke_r12/compare26.out" 2>&1 || echo "SMOKE: 26 FAILED"
bad=$(grep -lE "FAILED|Traceback" "$VOL/smoke_r12"/*.out "$SMOKE"/rebaseline_logs/*.log "$SMOKE"/round12_logs/*.log 2>/dev/null)
if [ -n "$bad" ] || [ ! -f "$VOL/smoke_r12/rb.json" ] || [ ! -f "$VOL/smoke_r12/r12.json" ]; then
  echo "SMOKE FAILED; nothing full was run. Problem files:"; echo "$bad"; exit 1
fi
echo "smoke passed"

echo "==== 1. re-baseline ===="
bash $EP/run_rebaseline_pod.sh
echo "==== 2. round 12 ===="
bash $EP/co-adaptation/run_coadapt_pod.sh
echo "==== 3. comparisons ===="
python $EP/19_score_interaction.py --experiment norman_additive --kind additive --output-root "$OUT" --data-dir "$DATA" \
  > "$OUT/rebaseline_logs/int_norman_additive.log" 2>&1 || echo "19 FAILED: additive"
python $EP/25_rebaseline.py --output-root "$OUT" --data-dir "$DATA" --out "$OUT/rebaseline_compare.json" \
  > "$OUT/rebaseline_logs/compare.log" 2>&1 || echo "25 FAILED"
python $EP/26_score_round12.py --output-root "$OUT" --data-dir "$DATA" --out "$OUT/round12_compare.json" \
  > "$OUT/round12_logs/compare.log" 2>&1 || echo "26 FAILED"
echo "done. fetch: $OUT/rebaseline_compare.json $OUT/round12_compare.json $OUT/rebaseline_logs $OUT/round12_logs"
