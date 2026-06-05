# ssl-lab

[![Documentation](https://img.shields.io/badge/docs-pleiadian53.github.io%2Fssl--lab-indigo)](https://pleiadian53.github.io/ssl-lab/)

📖 **Documentation site (full math rendering):** <https://pleiadian53.github.io/ssl-lab/>

A research lab for **self-supervised learning (SSL)**. The aim is to track the state of the art across the SSL families and selectively go deep on the methods most worth mastering — staying at or ahead of the frontier rather than covering everything shallowly.

**Current focus — JEPA.** The first research line studies JEPA (joint-embedding predictive architectures) and extends it into a *sampleable generative model*. JEPA learns a representation by predicting the *embeddings* of masked/target regions from context embeddings — no pixel reconstruction, no likelihood. It is a representation learner, not a generative model. To *sample* data you bolt on two pieces: a **prior** over the latent and a **decoder** back to data space. ssl-lab builds that full vertical slice as a walking skeleton on MNIST (a POC; the core is modality-agnostic so a practically meaningful modality drops in as a data adapter later).

```mermaid
flowchart LR
    NOISE(["noise ε"]) --> PRIOR["flow-matching<br/>prior p(z)"]
    PRIOR -- "sample z ~ p(z)" --> DEC["decoder<br/>z → x"]
    DEC --> GEN(["generated sample"])
    ENC["JEPA encoder<br/>(frozen)"] -. "defines the latent z<br/>the prior is fit to" .-> PRIOR

    classDef accent fill:#eef2ff,stroke:#6366f1,color:#1e1b4b;
    classDef io fill:#f8fafc,stroke:#94a3b8,color:#0f172a;
    class ENC,PRIOR,DEC accent;
    class NOISE,GEN io;
```

This *generative* slice is one of **two complementary extensions of JEPA** the lab pursues. The second turns JEPA from a passive predictor into an *active* world-model — see [Research directions](#research-directions) below.

## Why ssl-lab exists

ssl-lab is a spin-off of its sibling project [genai-lab](https://github.com/pleiadian53/genai-lab) (generative AI for computational biology). It's a focused R&D sandbox for state-of-the-art self-supervised learning — JEPA and new ideas built around it — kept deliberately **modality-agnostic** (MNIST is only a fast proof-of-concept). The aim is to mature these methods here and bring them back to genai-lab to develop more meaningful **genomic generative models**.

## Research directions

JEPA is a *representation learner* — it learns by predicting the embeddings of masked regions from context. ssl-lab pushes it past that in two complementary ways:

**1. Generative JEPA — make the representation *sampleable*.** *(built)* Add a prior over the latent and a decoder back to data space — the vertical slice above. Trained end to end on MNIST: a rich, non-collapsed latent (≈95% linear-probe accuracy with no labels in pretraining) and samples that are recognizable, varied, and *genuinely novel* rather than memorized.

**2. Action operators on JEPA — make prediction *active*.** *(emerging research line)* Go from passively in-filling hidden regions to *exploring*. JEPA's fixed "predict region $p$" mask is really a **frozen action**; promote it to a **learned operator the model chooses** — *sensing* ("where should I look?") and *perturbing* ("what happens if I edit this?") — so the system can form and test hypotheses, not just reconstruct what's masked. This reframes JEPA's predictor as the special case of an **action-operator world-model**, and builds on the action-operator formalism from the sibling project [GRL](https://github.com/pleiadian53/GRL).

> 📄 Write-up: ***JEPA as an Action-Operator World Model*** — read it with full math on the [documentation site](https://pleiadian53.github.io/ssl-lab/action_operator/01-jepa-action-operators/), or as source under [docs/action_operator/](docs/action_operator/01-jepa-action-operators.md) (with a standalone [notation reference](docs/action_operator/notation.md)).

For a concrete motivation: given an RNA sequence, *sense* to localize candidate splice sites, then *perturb* (in-silico edits) to learn what they mean — the kind of active, hypothesis-driven understanding passive prediction alone can't reach.

## Related projects

ssl-lab sits in a small constellation of related work:

- **[genai-lab](https://github.com/pleiadian53/genai-lab)** — the parent project: generative AI for computational biology. ssl-lab matures methods here and feeds them back to genai-lab for genomic generative models.

- **[NMDiff](https://github.com/pleiadian53/nmdiff)** — a **distinct branch of SSL** that sits outside the three classical families. Contrastive (SimCLR/CLIP), predictive (BERT/GPT/MAE), and self-distillation (BYOL/DINO) methods all share one structural property: the labeling function `T` is *fixed*, and only the encoder is learned. NMDiff promotes `T` itself to a domain-parameterized hypothesis `L(·; θ)` and jointly optimizes it with the downstream classifier via an outer-loop cross-view AUC criterion — the labeling function isn't a hyperparameter, it *is* the scientific hypothesis. Call it **hypothesis-driven SSL**: applied to transcript NMD efficiency (where ground-truth labels don't exist), but the pattern generalizes to any label-scarce setting where domain knowledge can parameterize a candidate labeling. Within the 2026 macro-trend of *learning the supervision signal itself* — alongside self-improving SSL, learned-augmentation SSL, and programmatic weak supervision — NMDiff is the interpretable, low-dimensional, domain-knowledge instance.

- **[GRL](https://github.com/pleiadian53/GRL)** — origin of the action-operator formalism that grounds the JEPA-as-action-operator research direction above.

## Layout

This project follows a use-case-driven R&D convention: reusable primitives in `src/`, thin driver scripts in `examples/`, intuition in `notebooks/`.

```
src/ssllab/
  data/        MNIST adapter -> modality-agnostic token tensors (B, N, token_dim)
  models/      TinyViT backbone, JEPA encoder/predictor, latent decoder
  jepa/        block masking, EMA target, the assembled JEPA module
  objectives/  prediction loss + VICReg collapse regularizer
  generative/  flow-matching prior over the latent (rectified flow)
  eval/        collapse diagnostics, linear probe, image-grid viz
  utils/       seeding, device selection
examples/
  jepa_basics/      01 train JEPA · 02 linear probe
  generative_jepa/  03 train decoder · 04 train flow prior · 05 sample & decode
```

## Setup

```bash
mamba env create -f environment.yml    # or: conda env create -f environment.yml
conda activate ssllab
pip install -e .
```

## Run the vertical slice

```bash
# 1. learn the representation (prints collapse diagnostics each epoch)
python examples/jepa_basics/01_train_jepa_mnist.py --epochs 5

# 2. confirm the frozen latent is semantically useful
python examples/jepa_basics/02_linear_probe.py

# 3. learn to decode the frozen latent back to pixels
python examples/generative_jepa/03_train_decoder.py --epochs 5

# 4. learn a sampleable prior over the frozen latent
python examples/generative_jepa/04_train_flow_prior.py --epochs 20

# 5. sample z ~ prior, decode -> output/jepa_mnist/samples/samples.png
python examples/generative_jepa/05_sample_and_decode.py
```

Artifacts are organized per experiment under `output/<experiment>/`:

```
output/jepa_mnist/
  checkpoints/  encoder.pt  decoder.pt  prior.pt
  samples/      samples.png  recon.png
  reports/      jepa_train.json  probe.json
  logs/         train.log
```

On a GPU pod this same tree is written under the network volume (`/runpod-volume/ssl-lab/output/<experiment>/`) and rsynced back to the local `output/` — same structure, env-dependent prefix swapped.

All scripts share an `--experiment` name (default `jepa_mnist`); pass a new name to keep separate runs side by side. On a GPU pod, the whole chain runs via `examples/generative_jepa/run_pod_pipeline.sh`.

## Remote training

Models that don't fit in your local environment train on remote GPU pods. That infra lives in a separate, decoupled package [`ops/`](ops/) (SkyPilot + RunPod) — the SSL library never imports it. Quick path:

```bash
pip install -e ops/                          # compute-check + dry-run
python examples/ops/ops_compute_check.py     # local vs pod?
python examples/ops/ops_run_pipeline.py --gpu a40 -- \
    python examples/jepa_basics/01_train_jepa_mnist.py --epochs 50   # dry-run
```

See [ops/README.md](ops/README.md) for the full provision → run → fetch workflow.
