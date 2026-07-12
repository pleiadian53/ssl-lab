# Running the pipeline: from raw Perturb-seq to the effect-size scoreboard

*The runnable companion to [Chapter 3](conditional-flow-jepa/03-training-and-evaluation.md). Chapter 3 explains the three training stages as ideas; this page is the exact sequence of commands that produces them, what each one reads and writes, how the outputs add up to the numbers in [Chapter 4](conditional-flow-jepa/04-results.md), and the pattern we use to run a lever experiment. It is meant to be followed top to bottom the first time, then used as a reference and a recovery guide.*

The whole point of writing this down is research velocity. Once the workflow is precise, changing one component and re-measuring becomes a mechanical, low-risk operation rather than a fresh archaeology each time. That is exactly what the lever-A (operator) and lever-B (decoder) experiments need.

## 1. The pipeline at a glance

Five kinds of script produce everything: one data-prep step, the three training stages, and the evaluators. A from-scratch NB-VAE baseline hangs off to the side as a comparator. The dependency graph is small:

```
00_process_norman            data prep  ->  data/norman2019/  (the cache)
        |
        v
01_pretrain_stage_a          Stage A    ->  encoder.pt        (frozen from here on)
        |
        +---> 02_probe_cell_encoder     encoder quality gate  ->  stage_a_probe.json
        |
        +---> 03_train_count_decoder    Stage C  ->  count_decoder.pt   (on frozen latents)
        |
        +---> 04_train_cond_flow        Stage B  ->  cond_flow.pt       (on frozen latents)
                     |
   03 + 04 -------->  +--> 05_sample_perturbed          generate a population (.npz)
                      +--> 06_eval_effect_size          -> effect_size.json      (the headline metric)
                      +--> 10_eval_calibration --model flow  -> calibration_flow.json

NB-VAE baseline (independent of Stages A/B/C; needs only the cache):
00  ->  08_train_cvae_baseline  ->  cvae_baseline.pt
                                     +--> 09_eval_cvae_baseline           -> effect_size_cvae.json
                                     +--> 10_eval_calibration --model vae -> calibration_vae.json
```

Two facts about this graph decide almost everything downstream.

**The encoder is trained once and then frozen.** Stage A produces `encoder.pt`; the decoder (Stage C) and the flow (Stage B) are both fit on the *fixed* latents it emits, and they are independent of each other. The flow is trained on latents alone and never sees the decoder; the decoder maps a latent to counts and never sees the flow. This independence is not an incidental detail, it is the lever that makes experiments cheap: **an experiment retrains only the stage it changes and reuses the rest.** The decoder ablation reuses one encoder and one flow across four decoder variants; the flow comparison reuses one encoder and one decoder across two flow variants.

**Two data splits, chosen deliberately, and it matters which you pass.** The `combo` split holds out twenty two-gene combinations whose individual genes are still seen singly in training, so it measures compositional generalization. The `cells` split holds out random cells of seen perturbations, so it measures in-distribution recovery. The convention across the scripts:

| stage | script | default split | why |
|---|---|---|---|
| pretrain | `01` | `combo` | so no held-out test combination can leak into the representation |
| probe | `02` | `cells` | a multiclass probe is only well-posed when train and test share the vocabulary |
| in-distribution decoder/flow/eval | `03`/`04`/`06` | `cells` | recover effect size when generalization is not being asked |
| generalization + baseline | flow `04`, VAE `08`, evals | `combo` | the headline test the method exists to pass |

## 2. Assumptions and environment

- **Python**: the `ssllab` conda env (has torch and the `ssllab` package). Do not use a bare system `python`.
- **Data**: the Norman cache lives at `data/norman2019/` (a symlink into the shared lake, about 8 GB: `processed.h5ad`, `tokens_meta.npz`, `splits.json`, `de_genes.json`, `manifest.json`). Step 0 builds it; every later step reads it through a light loader that needs no scanpy or pertpy.
- **Compute**: the training stages are GPU work. They run on a RunPod A40 through SkyPilot, driven by the `examples/ops/` helpers. CPU is the default device everywhere and MPS is never used.
- **Outputs**: each run writes under `output/<experiment>/{checkpoints,reports,samples,logs}/`. Reports are small JSON files named by the step (for example `effect_size.json`), which is what the evaluators and the docs read.

The heavy steps run on a pod, but every script is a plain Python entry point you can also run locally on a subset with `--limit` to smoke-test the wiring before spending GPU time.

## 3. The steps

Each step below gives the direct command (what runs on the machine that has the cache and, for the trainers, a GPU). Section 4 shows how the pod wrappers automate the same commands. Inputs are what the step reads; outputs are the checkpoint and the report it writes.

### Step 0 — build the cache (data prep)

```bash
python examples/perturbation_response/00_process_norman.py --artifact norman2019
```

Acquires Norman 2019 (via pertpy, or `--source h5ad --h5ad <path>`), runs QC, selects highly-variable genes, normalizes, computes per-perturbation differential expression, tokenizes, and writes both splits into `data/norman2019/`. This is the one step that needs scanpy/pertpy, it runs once on CPU, and it is the reason the rest of the pipeline stays light. Use `--smoke` for a synthetic no-network check of the wiring.

- **reads**: pertpy download or a local `.h5ad`
- **writes**: `data/norman2019/{processed.h5ad, tokens_meta.npz, splits.json, de_genes.json, manifest.json}` (no report)
- **check**: `manifest.json` exists and `de_genes.json` is non-empty.

### Step 1 — pretrain the encoder (Stage A) and gate it (probe)

```bash
python examples/perturbation_response/01_pretrain_stage_a.py --experiment norman_stage_a --split combo --epochs 50
python examples/perturbation_response/02_probe_cell_encoder.py --experiment norman_stage_a --split cells
```

Stage A trains the intra-cell JEPA encoder self-supervised (mask gene-group tokens, predict their embeddings), with a VICReg collapse guard at `--reg-coef 0.04`. The probe then freezes the encoder and asks whether a linear classifier can read the perturbation label off a single latent, and whether the representation collapsed.

- **reads**: the cache (train side of `combo`)
- **writes**: `output/norman_stage_a/checkpoints/encoder.pt`; reports `stage_a_train.json` (effective rank, feature std) and `stage_a_probe.json` (probe accuracy, chance, effective rank)
- **check**: effective rank well above 1 (about 176 of 256 on full data), probe accuracy many times the chance rate.

### Step 2 — train the count decoder (Stage C)

```bash
python examples/perturbation_response/03_train_count_decoder.py --experiment norman_stage_a --split cells --epochs 30
```

Fits the negative-binomial count decoder that maps a frozen latent to gene counts, by NB likelihood. The encoder stays frozen. This is the readout, and it is where the two chapter-8 levers live: `--anchored-mean` (B1, an identity-anchored mean head) and `--state-dispersion --anchor-weight 0.1` (B2, a per-cell dispersion with a moment-matching anchor). Both default off; with them off you get the standard scVI-style decoder. A one-time control pre-pass estimates the B1 baseline profile and the B2 dispersion target from the control cells.

- **reads**: `encoder.pt` + cache
- **writes**: `output/norman_stage_a/checkpoints/count_decoder.pt`; report `stage_c_decoder.json` (final NB NLL, the flags used)
- **check**: NB NLL decreasing across epochs; the report records `anchored_mean`/`state_dispersion` so a later loader rebuilds the right architecture.

### Step 3 — train the conditional flow (Stage B)

```bash
# in-distribution (cells split, table condition):
python examples/perturbation_response/04_train_cond_flow.py --experiment norman_stage_a --split cells --cond-type table --epochs 60
# generalization (combo split, control-transport, gene-set condition):
python examples/perturbation_response/04_train_cond_flow.py --experiment norman_flow_control --split combo \
    --cond-type geneset --flow-base control --epochs 60
```

Fits the flow-matching velocity field over the frozen latents. Two knobs carry the method's design. `--flow-base` chooses the source distribution: `gaussian` transports noise to the outcome (condition fuses baseline and intervention), `control` transports a real control latent to the outcome (condition is the intervention alone, so the field models the displacement directly). `--cond-type` chooses how the perturbation is embedded: `table` is a learned per-perturbation lookup that works in distribution, `geneset` builds a combination from its single-gene parts and is the only way to embed an unseen combo. For held-out combos you need `geneset`.

- **reads**: `encoder.pt` + cache (precomputes and standardizes all latents once, needs the control subset for the baseline pool)
- **writes**: `output/<exp>/checkpoints/cond_flow.pt`; report `stage_b_flow.json` (final flow-matching loss, `flow_base`, `cond_type`, `coupling`)
- **check**: a non-empty control pool (bump `--limit` if it raises "no control cells"); the report records the flow configuration.

### Step 4 — sample and evaluate

```bash
# generate a population for one perturbation (optional, for inspection):
python examples/perturbation_response/05_sample_perturbed.py --experiment norman_stage_a --pert CEBPE --n 500
# effect size (the headline benchmark):
python examples/perturbation_response/06_eval_effect_size.py --experiment norman_flow_control --split combo --n 200 --top-k 20
# calibration (the distributional axis):
python examples/perturbation_response/10_eval_calibration.py --experiment norman_flow_control --model flow --split combo --n 200
```

Effect size generates a predicted population per perturbation, forms its differential expression relative to control, and correlates it against the true differential on the top-DE genes (the Δ-correlation of [Chapter 4](conditional-flow-jepa/04-results.md)). Calibration compares the generated population against the held-out real cells on spread, coverage, 1-Wasserstein, and joint energy. Both read the flow and the decoder from the experiment's checkpoint directory.

- **reads**: `cond_flow.pt` + `count_decoder.pt` + cache
- **writes**: `effect_size.json` (mean and median Δ-correlation, per-perturbation scores) and `calibration_flow.json`
- **check**: `mean_delta_r` in the report; for calibration, coverage near the nominal 0.80 (it currently pins at 1.00, a decoder property, see [Chapter 3b](conditional-flow-jepa/3b-reading-the-calibration-metrics.md)).

### The baseline branch — the NB-VAE

```bash
python examples/perturbation_response/08_train_cvae_baseline.py --experiment norman_combo --split combo --epochs 60
python examples/perturbation_response/09_eval_cvae_baseline.py --experiment norman_combo --split combo --n 200 --top-k 20
python examples/perturbation_response/10_eval_calibration.py --experiment norman_combo --model vae --split combo --n 200
```

The from-scratch conditional NB-VAE uses no JEPA and no flow. It trains end to end on counts, conditioned on the *same* gene-set embedding as the flow, so the head-to-head isolates the generative machinery rather than the perturbation encoding. Its `effect_size_cvae.json` and `calibration_vae.json` are the reference every method arm is measured against. Because it is independent of Stages A/B/C, it can run at any time from the cache alone.

## 4. Running it on a pod

The stages are GPU work, and the shell wrappers in `examples/perturbation_response/run_*_pod.sh` bundle the right sequence of the commands above into one on-pod entrypoint. Each wrapper reuses whatever it can (a staged encoder, a trained flow) and retrains only what it must.

| wrapper | what it runs | writes to |
|---|---|---|
| `run_stage_a_pod.sh` | 00 (only if cache absent) -> 01 -> 02 | `norman_stage_a` |
| `run_full_pipeline_pod.sh` | 01 -> 02 -> 03 -> 04 -> 06, all on `cells` | `norman_stage_a` (the 0.469 in-distribution number) |
| `run_combo_generalization_pod.sh` | 03 -> 04 -> 06 (combo) + 08 -> 09 (VAE) | `norman_combo` |
| `run_flow_compare_pod.sh` | 04 -> 06 for `--flow-base {gaussian,control}` | `norman_flow_{gaussian,control}` |
| `run_seed_sweep_pod.sh` | 04 -> 06 for 3 flow configs x 3 seeds | `norman_sweep_*` |
| `run_decoder_ablation_pod.sh` | 03 -> 06 -> 10 for 4 decoder arms (lever B) | `norman_dec_*` |
| `run_decoder_seed_sweep_pod.sh` | the above x seeds, with a paired-bootstrap collation | `norman_dec_*_s*` |

The pod workflow itself is four steps, and it exists because the SkyPilot workdir upload excludes `/data` and `/output` (they are large or transient), so the cache and the reused checkpoints must be staged to the pod's persistent volume once, then read from there:

```bash
# 1. provision a pod (kept alive)
python examples/ops/ops_run_pipeline.py --execute --gpu a40 -- true
# 2. stage inputs to the volume, once (persists across pods)
python examples/ops/ops_stage_data.py data/norman2019 output/norman_flow_control/checkpoints
# 3. launch a wrapper (reads inputs from the volume)
python examples/ops/ops_run_pipeline.py --execute --gpu a40 -- \
    bash examples/perturbation_response/run_decoder_ablation_pod.sh
# 4. results persist on the volume and rsync back to output/ (or fetch manually):
#    rsync -Pavz <cluster>:/runpod-volume/ssl-lab/output/ output/
# 5. release the pod when done
sky down <cluster> -y
```

Each wrapper opens with a preflight that checks its inputs on the volume *before any training*, so a launch either runs to completion or exits in seconds naming the exact missing input and how to produce it. See Section 6.

## 5. From the workflow to Chapter 4

Every headline number in [Chapter 4](conditional-flow-jepa/04-results.md) traces to a `(script, split, report)`. This is the provenance map, audited in [Chapter 4a](conditional-flow-jepa/04a-reading-the-head-to-head.md).

| result | value | script | split / config | report (experiment) |
|---|---|---|---|---|
| encoder effective rank | ~176 / 256 | `01` | combo | `stage_a_train.json` (`norman_stage_a`) |
| linear probe (chance 0.42%) | 5.2% | `02` | cells | `stage_a_probe.json` (`norman_stage_a`) |
| in-distribution effect size, table | 0.469 | `06` | cells, `--cond-type table` | `effect_size.json` (`norman_stage_a`) |
| combo, transport flow (single seed) | 0.621 | `06` | combo, geneset, control | `effect_size.json` (`norman_flow_control`) |
| combo, Gaussian flow (single seed) | 0.580 | `06` | combo, geneset, gaussian | `effect_size.json` (`norman_flow_gaussian`) |
| combo, NB-VAE baseline | 0.633 | `09` | combo, geneset | `effect_size_cvae.json` (`norman_combo`) |
| seed-averaged (0.584 / 0.613 / 0.590) | 3 seeds x 3 configs | `04`+`06` via `run_seed_sweep_pod.sh` | combo, geneset | `effect_size.json` in `norman_sweep_*` |
| calibration, flow variants | energy 0.038 / 0.045 | `10 --model flow` | combo | `calibration_flow.json` |
| calibration, NB-VAE | energy 0.032 | `10 --model vae` | combo | `calibration_vae.json` |

Two subtleties to carry, both from [Chapter 4a](conditional-flow-jepa/04a-reading-the-head-to-head.md). The pair 0.621 / 0.580 is the single-seed A/B (`norman_flow_control` / `norman_flow_gaussian`); the pair 0.613 / 0.584 is the seed-averaged version (`norman_sweep_*`). Both are real and differ only in aggregation. And the paired-bootstrap confidence intervals are recomputed after the fact from the per-combination scores saved inside the sweep `effect_size.json` reports; they are a re-analysis, not a separately saved report.

## 6. Failure modes and how to recover

- **Preflight names a missing input.** The pod wrapper prints, per input, exactly how to produce it: build the cache (`00`), or train Stage A (`run_stage_a_pod.sh`) or Stage B (`run_flow_compare_pod.sh`), then stage. An input missing both locally and on the volume means an upstream training step was never run.
- **The local log stream stops but the job did not fail.** SkyPilot streams the pod log over your ssh session; if the laptop sleeps or the network drops, the stream ends with a non-zero local exit while the *job keeps running on the pod*. Confirm the truth with `sky queue <cluster>` (look for `SUCCEEDED`), then fetch results with the Section 4 rsync. Do not assume a stream drop is a failure.
- **`00` breaks the torch build.** Installing pertpy upgrades NumPy/scipy/sklearn and can shadow the pod's CUDA torch. The Stage-A wrapper captures and restores the base stack around `00`; if you run `00` by hand on a pod, isolate it or reinstall torch afterward. Best practice: build the cache once and stage it, so pods never install pertpy.
- **"No control cells."** Stage B raises if the precomputed set has no controls (bump `--limit`); Stage C's B1/B2 anchors silently no-op if controls are absent. If a decoder ablation shows the levers doing nothing, check the control pre-pass logged a non-zero control count.
- **A held-out combo scores near zero with the table condition.** The `table` condition cannot embed a combination it never saw. Use `--cond-type geneset` for any `combo`-split run.
- **A single seed disagrees with an earlier run.** Seed-to-seed spread on the twenty-combo test set is about 0.06 to 0.08, larger than the effects we chase, so a single seed is directional only. Use the seed-sweep wrapper and the paired bootstrap before calling any difference real. See [Chapter 4a §3](conditional-flow-jepa/04a-reading-the-head-to-head.md).

## 7. Running a lever experiment

This is the payoff of the workflow being precise, and it is the same shape for both levers. The discipline has four parts.

**Reuse everything the lever does not change.** A decoder experiment (lever B) reuses the frozen encoder *and* the trained transport flow, and retrains only Stage C. An operator experiment (lever A) will reuse the frozen encoder and, depending on the design, the decoder, and retrain only the transition. This is legitimate precisely because of the independence in Section 1, and it is what keeps an A/B to minutes of GPU rather than a full pipeline.

**A/B against the same reference.** Every arm is measured with the same evaluators (`06`, `10`) against the same NB-VAE reference (`effect_size_cvae.json` 0.633, `calibration_vae.json` energy 0.032). The reference does not need retraining; it is fixed.

**Single seed for direction, then a seed sweep for the verdict.** The decoder ablation (`run_decoder_ablation_pod.sh`) runs four arms at one seed to see which way each lever moves the number; `run_decoder_seed_sweep_pod.sh` then sweeps the promising arm across seeds and runs a paired bootstrap, because a single seed cannot separate a real effect from seed noise.

**Watch the axis the lever targets.** B1 (the mean head) targets effect size; B2 (the dispersion) targets calibration coverage and joint energy. A lever that improves its own axis while leaving the other flat is working as intended; a lever that improves the training loss but not its target metric is a warning, not a win.

The decoder ablation already exercises this whole loop end to end: it is the worked example, and its wrapper is the template to copy when lever A's operator experiment is wired in. When you reach for a design beyond the two planned levers, this same scaffold, reuse-what-you-can plus A/B against the fixed reference plus seeds-before-claims, is what will keep the comparison honest.

---

*Up: [the perturbation-response docs](index.md). Concepts behind these stages: [Chapter 3 — Training and evaluation](conditional-flow-jepa/03-training-and-evaluation.md). The numbers this workflow produces: [Chapter 4 — Results](conditional-flow-jepa/04-results.md).*
