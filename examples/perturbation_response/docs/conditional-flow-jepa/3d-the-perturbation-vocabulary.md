# Chapter 3d — The perturbation vocabulary: what the 237 classes are

*A companion to [Chapter 3 §2](03-training-and-evaluation.md), alongside [3a](3a-the-models-in-the-head-to-head.md), [3b](3b-reading-the-calibration-metrics.md), and [3c](3c-the-vicreg-collapse-guard.md). Chapter 3 names a vocabulary of 237 perturbation classes and moves on to splits and tokenization. This note stays with that vocabulary: what each class is, why the non-targeting control is counted as one of them rather than treated as missing data, and where the choice shows up downstream — in the gene-set condition, the control pool, and the perturbation-identity probe.*

> **Where this sits.** The biology of controls and baseline pairing is developed in [Reading Perturb-seq, Part 2 §4](../reading-perturb-seq/02-reading-the-dataset.md). Here the focus is the *labeling contract* the method series assumes: integer IDs, string names, and the boolean `is_control` flag that travel together through every loader batch and checkpoint.

## 1. Two fields on every cell, one meaning

Each cell in the processed cache carries its condition in two aligned fields:

| Field | Type | What it is |
|---|---|---|
| `pert_id` | int | Index into `pert_names` — which perturbation class this cell belongs to |
| `is_control` | bool | `True` when that class is the non-targeting control |

They are redundant for control cells (`is_control=True` always means `pert_names[pert_id] == "control"`), but both are kept because different code paths care about different views. The flow's condition encoder keys off `pert_id` (or the gene set parsed from the name). Differential expression, effect-size benchmarks, and `ctrl_pool` construction key off `is_control` to find baseline cells without re-parsing strings. In Stage B's one-time encoding pass, all three tensors — latent `Z`, `pert_id`, and `is_control` — are stacked row-aligned per cell ([`precompute_latents`](../../../../examples/perturbation_response/04_train_cond_flow.py)); row $i$ of each always refers to the same cell.

## 2. The accounting: 105 singles, 131 pairs, one control

After quality control on the full Norman 2019 run, the vocabulary has **237** entries:

| Kind | Count | Example label |
|---|---|---|
| Single-gene CRISPR activation | 105 | `CEBPE`, `KLF1`, `TBX2` |
| Two-gene combination | 131 | `CEBPE+RUNX1T1`, `AHR+KLF1` |
| Non-targeting control | 1 | `control` |
| **Total** | **237** | |

The 105 singles and 131 pairs are genuine interventions — cells where CRISPRa machinery targeted one or two genes. The final entry, **`control`**, is not an intervention at all in the biological sense: those cells went through the same CRISPR protocol but received a **non-targeting guide** (the activation machinery present, aimed at nothing). They sit at the unperturbed baseline the whole benchmark is measured against.

Combination labels are **structured**, not opaque IDs. The processing script normalizes every raw string to a canonical form — alphabetized genes joined by `+`, so `KLF1+CEBPA` and `CEBPA+KLF1` become the same class — and maps non-targeting synonyms (`nt`, `non-targeting`, empty string, …) to `control`:

```python
_NT_TOKENS = {"control", "ctrl", "nt", "non-targeting", "nontargeting", "neg", "none", ""}

def normalize_label(raw: str, combo_sep: str | None) -> str:
    ...
    genes = [g for g in genes if g.lower() not in _NT_TOKENS]
    if not genes:
        return "control"
    return "+".join(sorted(set(genes)))
```

The sorted unique set of canonical names becomes `pert_names`; `pert_id` is just the row index in that table ([`00_process_norman.py`](../../../../examples/perturbation_response/00_process_norman.py)).

## 3. Why control is the 237th class, not "no label"

A natural first reaction is to treat control cells as unlabeled — baseline data with no condition. This pipeline does the opposite: **control is an explicit perturbation class**, counted alongside the 236 interventions rather than set aside.

That choice is deliberate and shows up in two places:

**Conditioning.** The generative stack conditions on a perturbation embedding $z_p$ the same way for every class. Giving control its own ID means the unperturbed state has an addressable label — the model can be asked "what does the no-intervention condition look like?" on equal footing with `CEBPE` or `AHR+KLF1`. Under the **gene-set condition** from [Chapter 2](02-implementation.md), each class is represented by summing learned per-gene embeddings over the genes it targets. Control targets no genes, so its gene set is empty, the additive sum is the **zero vector**, and that zero is precisely the no-op intervention embedding.

**The perturbation-identity probe.** Stage A never sees labels during pretraining. To test whether the frozen encoder learned perturbation-relevant biology anyway, we train a linear classifier to predict `pert_id` from a single cell's latent $z$ ([Chapter 4 §encoder diagnostics](04-results.md)). That probe is a **237-way** problem: random guessing scores $1/237 \approx 0.42\%$. If control were excluded from the vocabulary, the task would be 236-way with a slightly higher chance floor — and, more importantly, the probe would no longer ask whether the representation distinguishes *unperturbed* cells from perturbed ones at all.

## 4. One cell, one latent — not a population average

A separate misconception worth clearing while we are on labels: a cell's JEPA latent is **not** an average over a population. Each row in the cache is one real cell's transcriptome, tokenized and encoded independently. The encoder's pooled output $z$ is that **single cell's** embedding — one point in latent space among thousands of other points, control and perturbed alike.

When the flow needs a baseline $z_b$, it does **not** use one canonical "average control." It draws a fresh control cell's latent from `ctrl_pool` — the set of standardized latents of real non-targeting cells, saved once during Stage B training. Population-level pairing (baseline drawn from the control *distribution*, not paired to the same cell) is forced by the destructiveness of scRNA-seq, not by averaging latents. See [Reading Perturb-seq, Part 2 §4](../reading-perturb-seq/02-reading-the-dataset.md) for the biology and [Chapter 3 §1](03-training-and-evaluation.md) for where `ctrl_pool` enters generation.

Control cells are diverse *within* the homogeneous K562 line — library depth (mostly corrected before tokenization), cell-cycle phase, transcriptional bursting, capture dropout — and that diversity is exactly what `ctrl_pool` preserves. The 237th class names a **population** of baseline cells, but each latent in the pool still belongs to one measured cell.

## 5. Where the vocabulary meets the method

Three downstream touchpoints, all assuming the same 237-name table:

| Component | How it uses the vocabulary |
|---|---|
| **Table condition** (in-distribution) | One learned embedding row per `pert_id`; fetch row $i$ to condition on perturbation $i$ |
| **Gene-set condition** (combo generalization) | Parse each name into genes; $z_p = \sum e(g)$; control $\Rightarrow$ empty sum $\Rightarrow$ zero |
| **Linear probe** | 237-class logistic regression from frozen $z$; chance $= 1/237$; reported accuracy is $\approx 12\times$ chance at 5.2% ([Chapter 4](04-results.md)) |

On the **`combo` split** ([Chapter 3 §2](03-training-and-evaluation.md)), control and all single-gene classes always sit in training — the baseline and the compositional building blocks must be available. Held-out combinations are drawn only from the 131 two-gene entries whose both singles appear separately in training; the vocabulary size itself does not change, only which IDs appear in which split.

## 6. What to carry forward

The 237 classes are not 237 cell types. They are **237 intervention conditions** in one cell line (K562): 105 single-gene activations, 131 two-gene pairs, and one non-targeting control counted as a first-class label. Every cell carries that condition as `pert_id`; control cells additionally have `is_control=True`. Treating control as class 237 (by name: `"control"`) lets the gene-set encoder represent the no-op as a zero vector, lets the flow condition on baseline the same way it conditions on any perturbation, and sets the probe's chance floor honestly. Each cell's latent is its own embedding, not an average — and when the method needs a baseline, it resamples real control cells from `ctrl_pool`, not a single summary point.

---

*Previous: [Chapter 3 — Training and evaluation](03-training-and-evaluation.md). Up: [the method series](index.md). Related: [Reading Perturb-seq, Part 2](../reading-perturb-seq/02-reading-the-dataset.md) · [Chapter 4 — Results](04-results.md).*
