# Perturbation response: single-cell genetic perturbation

This is the biological application of the generative-JEPA toolkit. A genetic perturbation switches a gene on or off inside a cell, and the cell's gene-expression profile, the activity level of each of its thousands of genes, shifts in response. The shift is not the same in every cell. Genetically identical cells respond differently, so a perturbation produces a population of expression profiles rather than a single answer. The task is to predict that population: the **distribution of gene-expression profiles** across perturbed cells, given the unperturbed control population and the identity of the perturbation. The case that matters most is a perturbation the model has never seen, here a combination of two genes whose single-gene effects it has only seen separately. Success is graded on **effect size**: the Pearson correlation between the predicted and the true change in expression, computed over each perturbation's most strongly responding genes. This places the work in the scGen, CPA, GEARS and scPPDM lineage. It is the case study the [generative-JEPA design-space series](../../docs/generative_jepa/13-choosing-a-route.md) chose for validating the conditional flow prior, the route that series recommends building first.

**The method** has three stages, each trained separately so that its contribution can be measured on its own:

1. **Stage A, the encoder.** A JEPA encoder is pretrained on cells with no labels at all. It masks some groups of genes in a cell and learns to predict their embeddings from the rest. After pretraining it is frozen and maps every cell to a latent vector.
2. **Stage B, the transition.** A conditional generative model in that latent space turns a perturbation into a cloud of outcome latents. The main version is a conditional flow that transports control-cell latents toward the perturbed population. Variants replace the flow with a learned linear operator.
3. **Stage C, the readout.** A negative-binomial count decoder turns each latent back into gene counts, because effect size is measured on genes, not on latents.

The stack is compared against a from-scratch conditional NB-VAE, which has no JEPA and no flow and is conditioned on the same perturbation embedding. That baseline is the bar the method has to clear.

**Dataset:** Norman et al. 2019, K562 cells under CRISPR activation, with both single-gene and two-gene perturbations. The combinatorial structure is the headline test. Models train on single-gene perturbations and some pairs, then must **predict twenty held-out two-gene combinations** they have never seen.

## Where it stands

The full scoreboard and every round's details are in the [results ledger](docs/conditional-flow-jepa/results-ledger.md). The short version follows, with Δr meaning the effect-size correlation on the twenty held-out combinations.

| | Δr | what it says |
|---|---|---|
| Conditional NB-VAE, the bar | **0.766** | stable across seeds, spread 0.006 |
| Transport flow, control to perturbed | 0.648 | the best generative configuration built so far |
| Action operator | 0.645 | a dead tie with the flow |
| Oracle Stage B, handed the real held-out latents | *0.679* | a ceiling, not a model |
| Linear readout of the frozen latents | *0.852* | a diagnostic, not a model |

Five findings carry the story so far.

- **The method loses to its baseline.** Every generative configuration lands between 0.61 and 0.65, about 0.12 below the NB-VAE, and the gap survives simultaneous confidence intervals over the whole family of comparisons.
- **The transition was never the bottleneck.** A Stage B that is perfect by construction reaches only 0.679 through the current decoder, so improving the flow could gain at most 0.03. Three rounds spent on the transition (the flow variants, the action operator, the operator algebra) were bounded by that ceiling from the start.
- **The decoder is where the effect is lost.** A plain linear readout of the frozen latents scores 0.852, above the baseline. The representation holds the signal, and the trained count decoder throws away 0.173 of it.
- **The operator-algebra hypothesis is refuted.** The idea that how far two single-gene operators fail to commute predicts the pair's genetic interaction failed every test it was given.
- **Two flow design choices are settled.** Transporting from a real control latent beats generating from noise (+0.036, significant). Optimal-transport coupling of the pairs makes things worse (−0.021, significant), even though it lowers the training loss.

**What comes next** is the [co-adaptation](co-adaptation/README.md) program, which unfreezes the encoder one stage at a time. Its Phase 0 diagnostic has run. On the frozen latents, a nonlinear map from control to perturbed explains only about 1.5% more of the response than an affine map does, across the principal directions that can be measured reliably. The latent response is therefore close to affine, which lowers the priority of reshaping the encoder's geometry for the operator. Phase 1 trains the encoder jointly with the decoder, which targets the 0.173 directly while keeping the transition frozen so the result remains attributable.

## Where to read

All the written material lives under [docs/](docs/index.md). Pick by the question you have.

| if you want to know... | read |
|---|---|
| what a Perturb-seq experiment measures, how a cell becomes tokens, and how success is scored | [Reading Perturb-seq](docs/reading-perturb-seq/index.md), a four-part primer on the data with real numbers throughout |
| what the method is, how it was built, what it scored, and why it fell short | [The conditional-flow + JEPA method](docs/conditional-flow-jepa/index.md), plus its [results ledger](docs/conditional-flow-jepa/results-ledger.md) |
| what an action operator is and why it was a natural candidate for the transition | [Action operators: a development](docs/action-operator/index.md), written after the refutation and honest about which premises failed |
| how to reproduce the headline result, command by command | [Running the pipeline](docs/running-the-pipeline.md), the runbook |
| which script answers which question | [Workflows](docs/workflows/index.md), the map of all the scripts by the kind of work they do |
| how the comparisons were made trustworthy | [Running an experiment you can trust](../../docs/experimental-method/index.md), the general methodology, with this project as its worked example |
| why this method was chosen in the first place | the [generative-JEPA design-space survey](../../docs/generative_jepa/index.md) |

A first-time reader is best served by the data primer, then the method series up to its results chapter, then the ledger.

## The scripts

The scripts are numbered `00` to `17` in the order they were written, which is not the order anyone runs them in. They fall into five kinds of workflow, and the [Workflows](docs/workflows/index.md) map explains each kind in depth. There is no `07`; the gap is kept so that existing references to script numbers stay valid.

**Build** an artifact that later steps depend on. These are the only scripts that train anything, and apart from `00` they belong on the GPU pod.

| script | produces |
|---|---|
| `00_process_norman.py` | the data cache: gene panel, tokens, splits, and the genes each perturbation is scored on |
| `01_pretrain_stage_a.py` | Stage A, the JEPA cell encoder, frozen from then on |
| `03_train_count_decoder.py` | Stage C, the negative-binomial count decoder on frozen latents |
| `04_train_cond_flow.py` | Stage B, the conditional flow, either from noise (`--flow-base gaussian`) or from control latents (`--flow-base control`) |
| `08_train_cvae_baseline.py` | the from-scratch conditional NB-VAE baseline |
| `13_train_operator.py` | Stage B as an action operator, `A_p = exp(M(e(p)))`, carrying control latents to perturbed ones |
| `16_train_operator_algebra.py` | Stage B with one generator per gene, so that pairs compose inside the operator group |

**Score** a trained model on the standing benchmark. These run on CPU in minutes.

| script | produces |
|---|---|
| `05_sample_perturbed.py` | a generated population for one perturbation, for inspection |
| `06_eval_effect_size.py` | the primary benchmark, Δr on the held-out combinations; `--stage-b` picks which Stage B to grade |
| `09_eval_cvae_baseline.py` | the same metric through the same harness, for the baseline |
| `10_eval_calibration.py` | the distributional axis: spread correlation, interval coverage, Wasserstein distance |

**Diagnose** where the loss lives. These train nothing, run in minutes, and several of them changed the direction of the project.

| script | answers |
|---|---|
| `00a_probe_hvg_coverage.py` | does the gene panel already contain the genes that respond? |
| `02_probe_cell_encoder.py` | is the frozen encoder any good, judged by a linear probe? |
| `11_diagnose_variance.py` | how much of the predicted spread comes from the latent distribution, and how much from the decoder? |
| `14_ceiling_analysis.py` | what is the best any Stage B could do with this encoder and decoder? |
| `15_empirical_epistasis.py` | how far does each pair's response depart from the sum of its two singles? |

**Decide** whether a difference is real. These take seconds.

| script | produces |
|---|---|
| `12_compare_arms.py` | a verdict on a family of comparisons, from a joint bootstrap with simultaneous intervals |
| `17_eval_bracket_epistasis.py` | the pre-registered test of whether operator non-commutativity predicts epistasis, against a permutation null |

**Vary** one lever at a time on the GPU pod. Each `run_*_pod.sh` wrapper reuses every artifact it can and retrains only the stage under test. The decoder and operator wrappers also open with a preflight that checks their staged inputs and stops within seconds, naming whatever is missing, before any training starts.

| wrapper | runs |
|---|---|
| `run_stage_a_pod.sh` | process the data if the cache is absent, then pretrain and probe the encoder |
| `run_full_pipeline_pod.sh` | the whole chain, encoder through effect-size evaluation, on the in-distribution split |
| `run_combo_generalization_pod.sh` | decoder, flow and evaluation on held-out combinations, plus the NB-VAE baseline |
| `run_flow_compare_pod.sh` | the noise-source flow against the control-source flow |
| `run_seed_sweep_pod.sh` | three flow configurations across three seeds, for confidence intervals |
| `run_decoder_ablation_pod.sh`, `run_decoder_seed_sweep_pod.sh` | the decoder levers, once and across seeds |
| `run_operator_pod.sh` | the action operator, deterministic and stochastic |
| `run_operator_algebra_pod.sh` | the per-gene operator algebra |
| `sweep_action_weight_pod.sh` | the strength of the penalty that keeps operators near the identity |

## 0 · Process Norman 2019 into a cache

Processing needs a few heavy single-cell libraries (`anndata`, `scanpy`, `pertpy`). Only this step uses them; the loader and every later script read the cache without them.

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

If pertpy stores the perturbation label under a non-standard column, or joins combinations with a different separator, inspect `adata.obs` and pass `--pert-col` or `--combo-sep`. To rebuild only the list of scored genes without touching the splits, pass `--de-only`. Before changing the gene panel, the token layout, or the dataset, read [Preparing a dataset](docs/workflows/01a-preparing-a-dataset.md): a new cache invalidates the encoder and everything trained on it.

### Staging the cache in the data lake

The processed cache is several gigabytes, so it lives in the project-neutral data lake rather than under `data/` in the repository. The `ops` helper links it in:

```bash
export SSLLAB_DATA_ROOT=~/work/data
python -c "from ops.datasets import link_dataset; link_dataset('scrna/perturb_seq/norman2019')"
# -> data/norman2019 -> $SSLLAB_DATA_ROOT/scrna/perturb_seq/norman2019
```

The lake is organized as `<modality>/<sub-topic>/<dataset>`, and the nested lake path becomes a flat local alias, `data/norman2019`, which is what the loader reads.

### Cache contents (`data/norman2019/`)

| File | What |
|---|---|
| `tokens_meta.npz` | the arrays read at train time, with no anndata needed: `hvg_X`, raw `counts`, `libsize`, `pert_id`, `is_control`, `ctrl_group`, per-cell split codes, `gene_ids`, `pert_names` |
| `processed.h5ad` | the canonical AnnData, for inspection and re-derivation |
| `splits.json` | perturbation-level splits: `combo` holds out unseen combinations, `cells` is an in-distribution sanity split |
| `de_genes.json` | the top differentially expressed genes per perturbation, which is the set every metric scores; [chapter 3e](docs/conditional-flow-jepa/3e-the-genes-the-metric-scores.md) explains how they are chosen and why the choice matters |
| `manifest.json` | provenance and every processing setting, including the seed of the token partition |

### Consuming the cache

```python
from ssllab.data import get_perturbseq_dataloaders

train, val, test = get_perturbseq_dataloaders(data_dir="data", split="combo", seed=0)
batch = next(iter(train))
# batch: tokens (B, n_tokens, token_dim), counts (B, n_hvg), libsize, pert_id,
#        is_control, ctrl_group ; plus train.control_sampler and train.meta
```

The `tokens` field follows the toolkit's universal `(B, n_tokens, token_dim)` contract, so it feeds `build_jepa(token_dim=..., n_tokens=...)` unchanged, exactly as MNIST patches do. `counts` and `libsize` are kept for the count decoder, and `control_sampler` draws baseline cells from the matched control population.

## Running the pipeline

The order of operations is cache, then encoder, then decoder and flow on the frozen latents, then evaluation; the baseline needs only the cache. [Running the pipeline](docs/running-the-pipeline.md) gives the exact commands, what each step reads and writes, and how every number in the results traces back to a script and a report.

Data preparation, tests and smoke runs happen locally on CPU. Training happens on a RunPod GPU pod, driven from `examples/ops/`: provision a pod, stage the cache and any reused checkpoints to the persistent volume once, launch a `run_*_pod.sh` wrapper, then bring the outputs back. **Evaluate locally**, against the local `de_genes.json`. A scored-gene list on a pod volume can be out of date, and grading against it silently produces wrong numbers. Section 4 of the runbook walks through the pod workflow step by step.

## Co-adaptation

[`co-adaptation/`](co-adaptation/README.md) is the next round. Every earlier round froze the encoder, which is what made a difference between two arms attributable to the one thing that changed. Co-adaptation relaxes that freeze one stage at a time, and its results join the same ledger as Round 5 onward, on the same twenty held-out combinations and against the same NB-VAE bar.

| script | phase | trained jointly with the encoder |
|---|---|---|
| `01_koopman_linearity_probe.py` | Phase 0, diagnostic, done | nothing; it measures whether the frozen latents respond affinely |
| `02_coadapt_decoder.py` | Phase 1, planned | the count decoder, with the transition frozen |
| `03_coadapt_operator.py` | Phase 2, planned, now lower priority | the operator, under a linearizing objective |

Its narrative companion is [co-adaptation/docs](co-adaptation/docs/index.md).

## Design notes

- **Tokenization** splits the 5000-gene panel into a fixed random partition of gene groups. It is a sensible default and can be revisited, for example with pathway or co-expression groups. [The tokenization design space](docs/conditional-flow-jepa/06a-the-tokenization-design-space.md) lays out the alternatives.
- **Latent extraction** is shared with the image examples. `extract_latents` in `src/ssllab/extract.py` takes an optional `prepare` callable that turns a Perturb-seq batch into tokens, and falls back to MNIST patchify when none is given. The diagnostics encode cells through `encode_rows` in `src/ssllab/latents.py`.
- **One frozen encoder, one grading harness.** Every Stage B and Stage C variant sits on the same frozen latents, and every model, including the baseline and the oracle arms of the ceiling, is scored through the same `run_effect_size_eval` function. That is what lets arms from different rounds be compared at all. Co-adaptation is the declared exception to the first rule, and it re-measures its own ceiling for that reason.
