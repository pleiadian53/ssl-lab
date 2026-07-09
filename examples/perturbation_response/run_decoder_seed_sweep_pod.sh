#!/usr/bin/env bash
# Pod entrypoint: SEED SWEEP of the decoder ablation (chapter-8 readout levers B1/B2).
#
# The seeded counterpart to run_decoder_ablation_pod.sh: sweeps each decoder arm across
# training seeds so point estimates become verdicts, on held-out combos. Reuses the
# encoder + transport flow (cond_flow.pt, flow_base=control) across every run -- the flow
# is decoder-independent, so sharing it is fair -- and retrains only Stage C (03) per
# (arm, seed). The EVAL seed is fixed at 0, so the spread reflects decoder-TRAINING
# variance (mirrors run_seed_sweep_pod.sh). Both axes are scored: effect size (06) and
# calibration (10). The collation seed-averages each arm and runs a paired bootstrap on
# the seed-averaged per-combo delta_r vs baseline and vs the NB-VAE reference.
#
#   arms:  baseline | anchmean (B1) | statedisp (B2) | both     (set ARMS to subset)
#   seeds: SEEDS (default "0 1 2")
#
# Usage:
#   DATA_DIR=/cache ENCODER=/cache/encoder.pt FLOW=/cache/cond_flow.pt \
#     bash examples/perturbation_response/run_decoder_seed_sweep_pod.sh [OUTPUT_ROOT]
#   # after a single-seed read, sweep only the promising arms:
#   ARMS="baseline both" SEEDS="0 1 2" bash examples/perturbation_response/run_decoder_seed_sweep_pod.sh
set -uo pipefail

OUT_ROOT="${1:-output}"
DATA_DIR="${DATA_DIR:-data}"
ARTIFACT="${ARTIFACT:-norman2019}"
ENCODER="${ENCODER:-output/norman_flow_control/checkpoints/encoder.pt}"
FLOW="${FLOW:-output/norman_flow_control/checkpoints/cond_flow.pt}"
[ "$OUT_ROOT" != "output" ] && { mkdir -p "$OUT_ROOT"; ln -sfn "$OUT_ROOT" output; }

: "${DEC_EPOCHS:=30}"
: "${ANCHOR_WEIGHT:=0.1}"
: "${EVAL_N:=200}"
: "${TOP_K:=20}"
: "${SEEDS:=0 1 2}"
: "${ARMS:=baseline anchmean statedisp both}"

[ -f "$DATA_DIR/$ARTIFACT/manifest.json" ] || { echo "[dec_sweep] ERROR: cache absent at $DATA_DIR/$ARTIFACT" >&2; exit 2; }
[ -f "$FLOW" ]    || { echo "[dec_sweep] ERROR: transport flow absent at $FLOW" >&2; exit 2; }
[ -f "$ENCODER" ] || { echo "[dec_sweep] ERROR: encoder absent at $ENCODER" >&2; exit 2; }
python -c 'import torch; print("[dec_sweep] cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")'
echo "[dec_sweep] arms='$ARMS' seeds='$SEEDS' dec_epochs=$DEC_EPOCHS anchor=$ANCHOR_WEIGHT eval_n=$EVAL_N"

arm_flags() {  # arm name -> Stage-C (03) flags
  case "$1" in
    baseline)  echo "" ;;
    anchmean)  echo "--anchored-mean" ;;
    statedisp) echo "--state-dispersion --anchor-weight $ANCHOR_WEIGHT" ;;
    both)      echo "--anchored-mean --state-dispersion --anchor-weight $ANCHOR_WEIGHT" ;;
    *)         echo "__UNKNOWN__" ;;
  esac
}

run_one() {  # run_one <arm> <seed>
  local arm="$1" seed="$2" exp="norman_dec_${1}_s${2}" flags
  flags="$(arm_flags "$arm")"
  [ "$flags" = "__UNKNOWN__" ] && { echo "[dec_sweep] !!! unknown arm '$arm'"; return 1; }
  mkdir -p "output/$exp/checkpoints"
  cp -f "$FLOW" "output/$exp/checkpoints/cond_flow.pt"   # shared transport flow, read by 06/10
  echo "================ arm=$arm seed=$seed (exp=$exp) 03-flags: ${flags:-none} ================"
  # shellcheck disable=SC2086  (word-splitting of $flags is intentional)
  python examples/perturbation_response/03_train_count_decoder.py --experiment "$exp" \
    --data-dir "$DATA_DIR" --artifact "$ARTIFACT" --encoder "$ENCODER" \
    --split combo --epochs "$DEC_EPOCHS" --seed "$seed" $flags \
  && python examples/perturbation_response/06_eval_effect_size.py --experiment "$exp" \
    --data-dir "$DATA_DIR" --artifact "$ARTIFACT" --split combo --n "$EVAL_N" --top-k "$TOP_K" --seed 0 \
  && python examples/perturbation_response/10_eval_calibration.py --experiment "$exp" \
    --data-dir "$DATA_DIR" --artifact "$ARTIFACT" --model flow --split combo --n "$EVAL_N" --top-k "$TOP_K" --seed 0 \
  || { echo "[dec_sweep] !!! FAILED arm=$arm seed=$seed"; return 1; }
}

status=0
for arm in $ARMS; do
  for s in $SEEDS; do
    run_one "$arm" "$s" || status=1
  done
done

echo "================ collation (seed-averaged + paired bootstrap) ================"
ARMS="$ARMS" SEEDS="$SEEDS" python - <<'PY'
import json, os
import numpy as np

arms = os.environ["ARMS"].split()
seeds = os.environ["SEEDS"].split()

def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None

def per_combo(es):
    pp = (es or {}).get("per_pert", {})
    return {k: (v if isinstance(v, (int, float)) else v.get("delta_r")) for k, v in pp.items()}

# Seed-average each arm: scalar means (delta_r, coverage, energy) and per-combo delta_r.
arm_combo, summary = {}, []
for a in arms:
    drs, covs, ens, acc = [], [], [], {}
    for s in seeds:
        es = load(f"output/norman_dec_{a}_s{s}/reports/effect_size.json")
        cal = load(f"output/norman_dec_{a}_s{s}/reports/calibration_flow.json")
        if es and es.get("mean_delta_r") is not None:
            drs.append(es["mean_delta_r"])
        if cal:
            covs.append(cal.get("mean_coverage")); ens.append(cal.get("mean_energy"))
        for k, v in per_combo(es).items():
            if v == v:
                acc.setdefault(k, []).append(v)
    arm_combo[a] = {k: float(np.mean(v)) for k, v in acc.items() if v}
    def m(x):
        x = [v for v in x if isinstance(v, (int, float))]
        return float(np.mean(x)) if x else float("nan")
    summary.append((a, m(drs), drs, m(covs), m(ens)))

ves = load("output/norman_combo/reports/effect_size_cvae.json") or {}
vcal = load("output/norman_combo/reports/calibration_vae.json") or {}
vae_combo = per_combo(ves)

def fmt(x, p=3):
    return f"{x:.{p}f}" if isinstance(x, (int, float)) and x == x else "  -  "

print(f"\n{'arm':<12}{'delta_r(avg)':>13}{'per-seed':>20}{'coverage':>10}{'energy':>9}")
print("-" * 64)
for a, dr, drs, cov, en in summary:
    print(f"{a:<12}{fmt(dr):>13}{' '.join(fmt(v,2) for v in drs):>20}{fmt(cov):>10}{fmt(en,4):>9}")
print(f"{'NB-VAE ref':<12}{fmt(ves.get('mean_delta_r')):>13}{'':>20}{fmt(vcal.get('mean_coverage')):>10}{fmt(vcal.get('mean_energy'),4):>9}")

def paired_boot(a, b, n=10000, seed=0):
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return None
    d = np.array([a[k] - b[k] for k in common])
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return d.mean(), lo, hi, float((means > 0).mean()), len(common)

print("\npaired bootstrap on seed-averaged per-combo delta_r (10k resamples):")
pairs = [(a, "baseline", arm_combo.get("baseline", {})) for a in arms if a != "baseline"]
if "both" in arm_combo:
    pairs.append(("both", "NB-VAE", vae_combo))
for a, bname, bdict in pairs:
    r = paired_boot(arm_combo.get(a, {}), bdict)
    if r:
        dm, lo, hi, p, ncmb = r
        sig = "SIG" if (lo > 0 or hi < 0) else "ns"
        print(f"  {a:<10} - {bname:<8} d={dm:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  P(>0)={p:.2f}  {sig}  (n={ncmb})")
print("\nread: coverage nominal 0.80 (1.00 = over-dispersed); VAE energy ~0.032, delta_r ~0.633.")
PY
echo "[dec_sweep] done (status=$status)."
exit "$status"
