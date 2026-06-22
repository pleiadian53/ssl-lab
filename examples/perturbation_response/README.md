# Perturbation response — single-cell genetic perturbation

The bio application of the generative-JEPA toolkit: predict the **distribution** of
a cell's response to a genetic perturbation, graded on **effect size** (Pearson
correlation of predicted vs. true differential expression on top-DE genes). This
is the scGen / CPA / GEARS / scPPDM lineage, and the case study the [verdict](../../dev/generative_jepa/QA/verdict-which-route-to-build-first.md)
selected for validating the conditional flow prior before refactoring it back into
genai-lab.

**Dataset:** Norman et al. 2019 — K562 CRISPRa, single + combinatorial genetic
perturbations. The combinatorial structure is the headline: train on single-gene
perturbations (and some combos), then **predict unseen 2-gene combinations**.

> **New here? Read the tutorial first.** [docs/](docs/index.md) is a four-part,
> worked-example series that interprets this dataset end to end — what Perturb-seq
> measures and what the intervention *is* biologically, how a cell becomes tokens,
> and how success is scored (effect size). Every number in it is real, taken from
> the small slice the commands below produce.

## Pipeline stages

| Stage | Script | Compute |
|---|---|---|
| **1a · data pipeline** (this folder) | `00_process_norman.py` | Local, CPU — the long pole |
| 1b · Stage A — cell JEPA encoder | *(next phase)* `01_pretrain_stage_a.py` | **Pod (A40)** |
| 1b · Stage C / baseline / Stage B | *(later)* | Mixed |

This folder currently implements **Phase 1a only** — turning Norman 2019 into an
analysis-ready cache that the rest of the project consumes. No GPU needed.

## 0 · Process Norman 2019 → cache

Heavy deps (`anndata`, `scanpy`, `pertpy`) are needed only for this step:

```bash
pip install -e ".[perturb]"
```

```bash
# tiny synthetic smoke run (no network, seconds) — verifies the whole pipeline
python examples/perturbation_response/00_process_norman.py --smoke

# real run: pertpy auto-fetch, 5000-gene HVG panel (multi-GB download, CPU, no pod)
python examples/perturbation_response/00_process_norman.py --source pertpy --n-hvg 5000

# or from a pinned local .h5ad instead of pertpy
python examples/perturbation_response/00_process_norman.py --source h5ad --h5ad <path>.h5ad
```

If pertpy exposes the perturbation label under a non-standard column, or uses a
different combo separator, inspect `adata.obs` and pass `--pert-col` / `--combo-sep`.

### Lake-staging the artifact

The processed cache is multi-GB. Stage it in the project-neutral data lake (rather
than committing it under `data/`) using the existing `ops` helper:

```bash
export SSLLAB_DATA_ROOT=~/work/data
python -c "from ops.datasets import link_dataset; link_dataset('scrna/perturb_seq/norman2019')"
# -> data/norman2019 -> $SSLLAB_DATA_ROOT/scrna/perturb_seq/norman2019
```

The lake follows a `<modality>/<sub-topic>/<dataset>` layout; the nested lake
path produces a flat local alias (`data/norman2019`) that the loader reads.

## Cache contents (`data/norman2019/`)

| File | What |
|---|---|
| `tokens_meta.npz` | torch-native arrays read at train time (no anndata): `hvg_X`, raw `counts`, `libsize`, `pert_id`, `is_control`, `ctrl_group`, per-cell split codes, `gene_ids`, `pert_names` |
| `processed.h5ad` | canonical biology-native AnnData (for inspection / re-derivation) |
| `splits.json` | perturbation-level `combo` (held-out unseen combos) and `cells` (sanity) splits |
| `de_genes.json` | top-DE HVG indices per perturbation — the effect-size metric seam |
| `manifest.json` | provenance + every knob, incl. the token-partition seed |

## Consuming the cache

```python
from ssllab.data import get_perturbseq_dataloaders

train, val, test = get_perturbseq_dataloaders(data_dir="data", split="combo", seed=0)
batch = next(iter(train))
# batch: tokens (B, n_tokens, token_dim), counts (B, n_hvg), libsize, pert_id,
#        is_control, ctrl_group ; plus train.control_sampler and train.meta
```

The `tokens` field is the universal `(B, n_tokens, token_dim)` contract — it feeds
`build_jepa(token_dim=..., n_tokens=...)` unchanged, exactly like MNIST patches.
`counts` + `libsize` are preserved for the later NB/ZINB decoder; `control_sampler`
draws baseline `z_b` cells from the matched control population for the flow.

## Design notes (seams to later phases)

- **Tokenization** is a fixed random partition of the HVG panel into gene groups —
  a sensible default, *revisitable* at Stage A (pathway / co-expression groups).
- **`extract_latents`** (`src/ssllab/extract.py`) currently hardcodes image
  `patchify`; the generalization to accept pre-tokenized batches is deferred to the
  Stage A extraction driver. Phase 1a deliberately touches no existing logic.
