# Evaluating generated samples: beyond eyeballing

For images you can eyeball a sample grid. For DNA/RNA, designed proteins, gene
expression vectors, or any abstract object, you cannot — so you need
*quantitative* evaluation. This doc lays out how to evaluate a generative model's
samples, with an eye toward **modality-agnostic** metrics that transfer from
MNIST to sequences and molecules.

> Scope note: this is about the **generated samples**. Whether the *encoder's*
> latent is healthy (effective rank, feature std) is a different, upstream
> question — see
> [../../jepa_basics/docs/representation-diagnostics.md](../../jepa_basics/docs/representation-diagnostics.md).

---

## Two axes: extrinsic vs intrinsic

**Extrinsic** — does the generated data *help a downstream task*?
- **TSTR (Train on Synthetic, Test on Real):** train a supervised model purely on
  generated samples, evaluate on real held-out data. High TSTR ⇒ the samples
  carry the real signal. The mirror, TRTS, is also informative.
- **Augmentation gain:** add generated samples to a real training set; measure
  downstream lift. This is the "are they *useful*" test.

Extrinsic eval is decisive but expensive and task-specific. The rest of this doc
is about **intrinsic** evaluation: judging the samples by comparing the
*generated distribution* to the *real data distribution* (or to properties real
data must satisfy), with no downstream task.

---

## The core idea of intrinsic evaluation

You almost never have the true data density, so you can't just compute a
likelihood of "realness." Instead, intrinsic metrics compare **a set of generated
samples** against **a set of real samples** — either as whole distributions, or
via summary statistics that valid objects must obey. Everything below is a
variation on that theme.

### 1. Distributional distance in a feature space (FID / KID family)

Embed real and generated samples with a fixed feature extractor $\phi$, then
measure the distance between the two embedding distributions.

- **FID (Fréchet Inception Distance):** fit a Gaussian to each set of embeddings
  and take the Fréchet (2-Wasserstein) distance:
  $$\text{FID} = \lVert \mu_r - \mu_g\rVert^2 + \mathrm{Tr}\big(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2}\big).$$
- **KID (Kernel Inception Distance):** the squared **MMD** between embedding sets
  with a polynomial kernel — unbiased and more reliable at small sample sizes.

**The generalization that matters:** "Inception" is just the image-domain choice
of $\phi$. FID/KID are really *"distance between real and generated in a semantic
embedding space."* Swap $\phi$ for a domain encoder and the metric transfers:
- DNA/RNA → an embedding from a genomic foundation model (e.g. Evo2, a
  nucleotide-transformer), giving a "FxD" (Fréchet x Distance).
- proteins → ESM/ProtT5 embeddings (Fréchet ProtT5 Distance is used in practice).
- ssl-lab → in principle the JEPA encoder itself. **Caveat:** scoring your own
  generator with your own encoder is partly circular (the decoder was trained to
  satisfy that encoder); prefer an *independent* extractor for an honest number.

### 2. Precision & Recall (fidelity vs diversity)

A single distance can't tell *why* two distributions differ. Precision/Recall for
generative models (Sajjadi 2018; Kynkäänniemi 2019) disentangle it by building
$k$-NN manifolds in embedding space:

- **Precision** = fraction of *generated* samples lying inside the *real*
  manifold → **fidelity** (are samples realistic?).
- **Recall** = fraction of the *real* manifold covered by *generated* samples →
  **diversity / coverage** (did we capture all the modes, or collapse to a few?).

**Density & Coverage** (Naeem 2020) are more robust, outlier-resistant variants of
the same idea. These are the key metrics for catching **mode collapse** — high
precision + low recall = "pretty but repetitive."

### 3. Two-sample tests (MMD)

**Maximum Mean Discrepancy** between the real and generated sets in a kernel
feature space is a proper statistical test of "are these drawn from the same
distribution?" Fully domain-agnostic given a kernel; underlies KID.

### 4. Likelihood / density (only some model families)

If the model assigns probabilities — normalizing flows, autoregressive models,
VAEs (ELBO) — report held-out **negative log-likelihood**, **perplexity**, or
**bits-per-dim**. ssl-lab's flow prior is a continuous normalizing flow, so it can
in principle score latents via change-of-variables (the starter only *samples*).
For DNA, an Evo2-style log-likelihood of generated sequences is a natural intrinsic
signal. Note: likelihood measures density fit, *not* perceptual quality — high
likelihood ≠ good samples and vice versa, so pair it with the metrics above.

### 5. Novelty & memorization

Crucial for the stated goal — *sampling meaningful **new** points*. For each
generated sample, compute the distance to its nearest neighbor in the **training
set** (in data or embedding space):

- too **close** ⇒ the model **memorized** / copied training data (not generating);
- too **far** ⇒ off-distribution garbage;
- a healthy generator sits in between, and you report the *distribution* of NN
  distances vs the real-data train↔test NN-distance baseline.

A generator can score great on FID by regurgitating the training set — novelty
checks are what stop that from fooling you.

### 6. Domain-specific validity oracles (the key for non-visual data)

This is where you replace the human eye for sequences/molecules: define
**measurable properties valid objects satisfy**, and check the generated set
matches the *real distribution* of those properties.

- **DNA/RNA:** $k$-mer frequency spectra (χ²/KL between generated and real $k$-mer
  distributions), GC content distribution, presence/position of motifs (splice-site
  consensus, TF binding sites), ORF integrity. For designed regulatory elements,
  score them through a **predictive oracle** (e.g. a splice-site predictor or an
  expression model) and check the predicted-activity distribution.
- **Proteins:** amino-acid composition, secondary-structure propensity,
  **foldability** (does ESMFold/AlphaFold give high pLDDT?), absence of steric
  clashes, and **novelty** as sequence identity to the nearest natural protein.
- **General principle:** an **oracle model** (a trusted predictor/classifier or a
  physics check) stands in for "looking at it." If real valid objects have
  property distribution $P$, good generated objects should reproduce $P$.

---

## Mapping back to MNIST (your intuition, made precise)

Your instinct — "compare generated digits to known labeled digits" — is exactly
the **classifier-oracle** instantiation of the principles above, and it
generalizes:

1. **Classifier oracle.** Run a pretrained MNIST classifier on the generated grid.
   - mean **confidence / low entropy** per sample ⇒ samples look like *some* real
     digit (a fidelity proxy, akin to the old Inception Score idea);
   - **class balance** of predicted labels ⇒ coverage/diversity — if only 3 of 10
     digits appear, that's mode collapse (a recall failure). Label-free at sample
     time, since the classifier supplies the labels.
2. **FID/KID** using a small MNIST CNN as $\phi$ (instead of Inception).
3. **Precision/Recall (or Density/Coverage)** in that CNN's feature space.
4. **Novelty:** nearest-neighbor distance from each sample to the training images.

---

## A practical battery for ssl-lab generative-JEPA

Recommended order to add to the repo (cheap → thorough), all label-free at sample
time and each transfers to a bio modality by swapping the oracle/encoder:

1. **Classifier-oracle confidence + class-coverage entropy** — cheapest, catches
   mode collapse and obvious garbage.
2. **Precision/Recall or Density/Coverage** in an independent encoder's space.
3. **Novelty / memorization** NN-distance check.
4. **FID/KID** with an independent embedder, once 1–3 look good.

**Implemented** in this starter:
- modality-agnostic metrics — [`src/ssllab/eval/generative.py`](../../../src/ssllab/eval/generative.py)
  (`fid`, `kid`, `precision_recall`, `density_coverage`, `nn_distance_stats`,
  `classifier_metrics`);
- the MNIST oracle — [`src/ssllab/eval/oracle_mnist.py`](../../../src/ssllab/eval/oracle_mnist.py)
  (an independent CNN; swap it for a domain model on a bio modality);
- the driver — [`06_eval_samples.py`](../06_eval_samples.py), writing
  `reports/sample_eval.json`.

See the [results doc](../results/README.md) for the numbers on the reference run.
Two practical notes that the implementation bakes in: (1) features are
**standardized by the real set** before the distance metrics, so an arbitrary
(non-Inception) feature space gives stable FID/PR/coverage; (2) the oracle is
**cached** so FID/KID stay comparable across runs (they are not comparable under a
different oracle or sample size).

---

## References (entry points, not exhaustive)

- Heusel et al. 2017 — FID. Bińkowski et al. 2018 — KID/MMD.
- Sajjadi et al. 2018; Kynkäänniemi et al. 2019 — Precision/Recall for generative models.
- Naeem et al. 2020 — Density & Coverage.
- Roy & Vetterli 2007 — effective rank (see the representation-diagnostics doc).
