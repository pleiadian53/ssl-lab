#!/usr/bin/env bash
# Why do the 768-wide flows of round 14 diverge? A training-stability probe, nothing scored.
#
# Round 14's flows on the full 768-dimensional V2 embeddings (hidden width 768, learning rate 1e-3)
# blew up between epochs 5 and 31, with or without the gradient clipped at 1.0, while every flow on
# a 256-wide embedding (hidden width 256) trained smoothly. 09_coordinate_stats.py found no
# coordinate that the flow's per-coordinate standardization turns extreme, so the input's scale is
# not the explanation. This probe separates the remaining candidates: the learning rate, the hidden
# width and the input width. Every run is the round-1 transport flow recipe on seed 0 for the full
# 60 epochs, and only its per-epoch loss and gradient norms are read (04 reports them).
#
#   probe            store                      hidden  lr      asks
#   lr3e-4           V2-104M gene mean (768)    768     3e-4    does a smaller step stabilize it?
#   lr1e-4           V2-104M gene mean (768)    768     1e-4
#   hid256           V2-104M gene mean (768)    256     1e-3    is the 768-wide hidden layer needed to fail?
#   p256_hid768      the same, projected (256)  768     1e-3    is the 768-wide hidden layer enough to fail?
#   jepa_hid768      our JEPA encoder (256)     768     1e-3    ... on a store that is not Geneformer's
#
# Usage (on the pod, from the repo root):
#   bash examples/perturbation_response/foundation-encoder/run_flow_stability_probe_pod.sh

set -uo pipefail
VOL="${VOL:-/runpod-volume/ssl-lab}"; DATA="${DATA:-$VOL/data}"; OUT="${OUT:-$VOL/output}"
EP=examples/perturbation_response
LOGS="$OUT/gf_probe_logs"; mkdir -p "$LOGS"
PARALLEL="${PARALLEL:-3}"

probe() {   # $1 = name, $2 = store under the output root, $3 = hidden width, $4 = learning rate
  local exp="norman_gf_probe_$1"
  rm -rf "${OUT:?}/$exp"; mkdir -p "$OUT/$exp/checkpoints"
  python $EP/04_train_cond_flow.py --experiment "$exp" --latents "$OUT/$2" --output-root "$OUT" --data-dir "$DATA" \
    --split combo --cond-type geneset --flow-base control --coupling independent --epochs 60 \
    --hidden "$3" --lr "$4" --seed 0 --device cuda > "$LOGS/$1.log" 2>&1 && echo "probe $1 done" || echo "PROBE FAILED: $1"
}

V2=geneformer_full/stores/V2-104M_cell
V2P=geneformer_full/stores_p256/V2-104M_cell
JEPA=embeddings/norman_jepa_stage_a
running=0
for spec in "lr3e-4 $V2 768 3e-4" "lr1e-4 $V2 768 1e-4" "hid256 $V2 256 1e-3" "p256_hid768 $V2P 768 1e-3" "jepa_hid768 $JEPA 768 1e-3"; do
  # shellcheck disable=SC2086
  probe $spec &
  running=$((running + 1)); [ "$running" -ge "$PARALLEL" ] && { wait -n; running=$((running - 1)); }
done
wait

python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1]); rows = {}
for d in sorted(out.glob("norman_gf_probe_*")):
    r = json.loads((d / "reports/stage_b_flow.json").read_text()); h = r["history"]; loss = h["cfm_loss"]
    over = [i + 1 for i, v in enumerate(loss) if v > 2 * loss[0]]
    rows[d.name[len("norman_gf_probe_"):]] = {
        "dim": r["dim"], "hidden": r["hidden"], "lr": r["lr"], "first_loss": loss[0], "final_loss": loss[-1],
        "max_over_first": max(loss) / loss[0], "first_epoch_above_2x": over[0] if over else None,
        "grad_norm_max": max(h["grad_norm_max"]), "grad_norm_median_last": h["grad_norm_median"][-1], "cfm_loss": loss}
    x = rows[d.name[len("norman_gf_probe_"):]]
    print(f"{d.name:32s} dim {x['dim']:3d} hidden {x['hidden']:3d} lr {x['lr']:.0e}  first {x['first_loss']:.3f} "
          f"final {x['final_loss']:.3f} max/first {x['max_over_first']:.3g} first>2x {x['first_epoch_above_2x']} "
          f"grad max {x['grad_norm_max']:.3g}")
(out / "gf_flow_probe.json").write_text(json.dumps(rows, indent=2))
PY
echo "[probe] done. fetch: $OUT/gf_flow_probe.json $LOGS"
