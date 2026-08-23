# 1. Build: producing an artifact something else depends on

*The only workflow that creates state. Everything else in this series consumes what Build produces. It is also where all the money and all the waiting go, which is why the [Diagnose](03-diagnose.md) workflow exists to tell you whether a build is worth starting.*

---

## What makes a step a Build step

It writes a checkpoint that another step loads. That is the whole test, and it has three consequences worth internalizing.

**It is the only workflow with a dependency order.** You cannot train a decoder before an encoder exists. Score, Diagnose, and Decide can run in any order on artifacts that already exist; Build cannot.

**Its cost is real.** Minutes to hours on a GPU, and on a pod that means billing. Every other workflow here runs on a laptop in minutes.

**Its output is a contract.** A checkpoint is not just weights. It carries the configuration that produced it, and downstream steps read that configuration to reconstruct the model correctly. Look at any `load_*` function in [`perturb.py`](../../../../src/ssllab/generative/perturb.py) and you will see it pulling `dim`, `pert_gene`, `mean`, `std`, and the flags the arm was trained with. Break the contract and the failure surfaces at *load* time, which is the good case, or at *score* time as a silently wrong number, which is not.

## The dependency spine

```
00 cache ──┬─→ 01 encoder.pt ──┬─→ 03 count_decoder.pt ──┐
           │      (FROZEN)     │                          ├─→ Score / Diagnose
           │                   ├─→ 04 cond_flow.pt ───────┤
           │                   ├─→ 13 operator.pt ────────┤
           │                   └─→ 16 operator_algebra.pt ┘
           │
           └─────────────────────→ 08 cvae_baseline.pt  (no encoder, trained end to end)
```

Read the fan-out at `encoder.pt`. Three different Stage-B designs and one decoder all hang off **the same frozen encoder**, and that is what makes them comparable. The baseline hangs off nothing, which is the point of a baseline.

## The steps

### `00` — the cache

The one step that needs scanpy and pertpy, runs once, on CPU. It has its own chapter, [Preparing a dataset](01a-preparing-a-dataset.md), covering the transformation chain, the `--n-hvg` / `--n-tokens` / `--source` knobs, and what a new cache invalidates downstream.

```bash
python examples/perturbation_response/00_process_norman.py --artifact norman2019
```

It also owns the **scoring seam**: `de_genes.json`, the per-perturbation gene list every metric is computed on. That file is not a training input, so if only the gene selection changes:

```bash
python examples/perturbation_response/00_process_norman.py --de-only
```

regenerates it while leaving `splits.json` and `tokens_meta.npz` alone. **Every trained checkpoint stays valid**; only the evaluators need re-running. This seam is important enough to have its own chapter ([3e](../conditional-flow-jepa/3e-the-genes-the-metric-scores.md)) and its own acceptance gate, because a bad gene list once made the entire benchmark score models on a matrix of zeros.

### `01` — the encoder, then `02` to gate it

Stage A trains the intra-cell JEPA encoder by masked prediction, with a VICReg collapse guard.

```bash
python examples/perturbation_response/01_pretrain_stage_a.py --experiment norman_stage_a --split combo --epochs 50
python examples/perturbation_response/02_probe_cell_encoder.py --experiment norman_stage_a --split cells
```

Always run the probe. It is a [Diagnose](03-diagnose.md) step, it takes a minute, and it is the only thing standing between you and building three more stages on a collapsed representation.

### `03` — the decoder

Maps a frozen latent to gene counts by NB likelihood.

```bash
python examples/perturbation_response/03_train_count_decoder.py --experiment norman_stage_a --split cells --epochs 30
```

Two opt-in levers live here: `--anchored-mean` (an identity-anchored mean head) and `--state-dispersion --anchor-weight 0.1` (per-cell dispersion with a moment-matching anchor). Both default off, and with both off you get a standard scVI-style decoder.

### `04` — the flow

Two knobs carry the method's entire design.

```bash
python examples/perturbation_response/04_train_cond_flow.py --experiment norman_flow_control --split combo \
    --cond-type geneset --flow-base control --epochs 60
```

`--flow-base` picks the source: `gaussian` transports noise to the outcome, `control` transports a real control latent so the field models the *displacement*. `--cond-type` picks the perturbation embedding: `table` is a per-perturbation lookup that only works in distribution, `geneset` composes a combination from its single-gene parts. **For held-out combos you need `geneset`**, because a lookup table has no row for a combination it never saw.

### `13` and `16` — the Stage-B variants

Same slot as `04`, different design. They reuse the frozen encoder and retrain only Stage B, which is the [Vary](05-vary.md) pattern.

### `08` — the baseline

A from-scratch conditional NB-VAE with no JEPA and no flow, trained end to end on the same compositional condition.

```bash
python examples/perturbation_response/08_train_cvae_baseline.py --experiment norman_combo --split combo --epochs 60
```

It shares the condition encoder deliberately, so the comparison isolates the generative machinery rather than the compositional embedding. Note it is *not* an ablation of the method; it is a control, and [Chapter 4](../../../../docs/experimental-method/04-ablations-and-controls.md) of the methodology series is about why that difference matters.

## The frozen-encoder invariant

`encoder.pt` is trained once and never retrained by a downstream experiment.

This is the convention that makes the [ledger](../conditional-flow-jepa/results-ledger.md) coherent across rounds. Every Stage-B and Stage-C variant sits on the same frozen latents, so a difference between two arms is a difference in the thing that changed. Retrain the encoder inside an arm and you have silently changed two things, and the arm can no longer explain its own result.

Relaxing it is a legitimate and probably necessary move, and it is the ledger's named fork. But it must be **declared**, because at that point the method stops being "a flow prior over frozen JEPA latents" and becomes something else that the standing scoreboard does not describe.

## Before you build

The expensive mistake this project made three times is building a lever before checking whether the stage it improves has any room. The [ceiling analysis](03-diagnose.md) trains nothing, runs in minutes, and answers exactly that. Run it first. If the stage you are about to spend a GPU-hour on has $0.03$ of headroom, you want to know before the hour, not after.

---

*Next: [Score](02-score.md). Up: [the workflow map](index.md). Exact commands in order: [the runbook](../running-the-pipeline.md).*
