# Representation diagnostics: effective rank & feature std

**What these measure:** the health of the JEPA **encoder's latent space** — *not*
the quality of generated samples. They answer "did the self-supervised encoder
learn a rich, non-degenerate representation, or did it collapse?" Sample quality
is a separate question, covered in
[../../generative_jepa/docs/evaluating-generated-samples.md](../../generative_jepa/docs/evaluating-generated-samples.md).

Both are computed in [`src/ssllab/eval/collapse.py`](../../../src/ssllab/eval/collapse.py)
and logged each epoch by [`01_train_jepa_mnist.py`](../01_train_jepa_mnist.py).

---

## Why we need them: the collapse failure mode

JEPA trains a predictor to map context embeddings to target embeddings *in latent
space*. There is a trivial cheat: if the encoder maps **every** input to the
**same** vector, the predictor's job becomes trivial and the loss goes to zero —
while the representation has learned *nothing*. This is **representation
collapse**. The EMA target encoder and the optional VICReg term exist to prevent
it; these two diagnostics let us *verify* it didn't happen.

There are two flavors of collapse, and we need one metric for each:

| Failure | Symptom | Caught by |
|---|---|---|
| **Complete collapse** | all embeddings ≈ one point | feature std → 0 |
| **Dimensional collapse** | variance crammed into a few axes; the rest dead | effective rank → small |

`feature_std` alone can miss dimensional collapse (a few high-variance dims keep
the mean std off the floor), so we report both.

---

## Feature standard deviation

For a batch of embeddings $Z \in \mathbb{R}^{N\times D}$ ($N$ samples, $D$ dims):

$$
\text{feature\_std}(Z) \;=\; \frac{1}{D}\sum_{d=1}^{D}\operatorname{std}_n\!\big(Z_{n,d}\big)
$$

i.e. compute each dimension's standard deviation across the batch, then average
over dimensions. Code: `z.std(dim=0).mean()`.

- **→ 0**: every input produces nearly the same embedding ⇒ complete collapse.
- **healthy**: clearly above 0 and **stable** across training.

**How to read the absolute value.** ssl-lab does not L2-normalize embeddings, so
the scale is arbitrary — `0.7` is not a magic number. What matters is the
**trend**: it should not drift toward 0. In our reference run it sits at
~0.70–0.77 with mild batch-to-batch fluctuation, which is exactly the healthy,
non-collapsed signature you observed in `train.log`. (If you *did* normalize
embeddings to the unit sphere, a per-dim std near $1/\sqrt{D}$ would be the
isotropic reference.)

---

## Effective rank

Feature std says "are the dimensions alive?" Effective rank says "**how many
independent directions** does the representation actually use?" — a soft,
continuous count of active latent dimensions.

Center the batch ($\tilde Z = Z - \bar Z$), take its singular values
$\sigma_1 \ge \dots \ge \sigma_D \ge 0$, normalize them into a distribution, and
take the exponential of its Shannon entropy (Roy & Vetterli, 2007):

$$
p_k = \frac{\sigma_k}{\sum_{j}\sigma_j},\qquad
H(p) = -\sum_{k} p_k \ln p_k,\qquad
\text{erank}(Z) = \exp\big(H(p)\big).
$$

Code: `svdvals(centered Z)` → normalize → entropy → `exp`.

- **Range:** $[1, D]$.
- **erank ≈ 1**: all variance in a single direction ⇒ (dimensional) collapse.
- **erank ≈ $D$**: variance spread evenly over all $D$ axes ⇒ maximally isotropic.
- **rising over training**: the encoder is progressively using more directions —
  learning richer structure.

Intuition: if the normalized spectrum is a flat distribution over $r$ directions
and zero elsewhere, $H = \ln r$ and $\text{erank} = r$ exactly. For a non-flat
spectrum it interpolates smoothly — hence "effective" rank rather than the hard
matrix rank (which would just be $\min(N,D)$ and tells you nothing about *how the
energy is distributed*).

In our reference run, effective rank climbed **74 → 103.7 / 128** over 100 epochs:
the 128-dim latent ends up genuinely exercising ~104 directions. That is a strong,
information-rich, non-collapsed representation.

---

## What they do *not* tell you

- **Not sample quality.** A high-rank, non-collapsed encoder is a *prerequisite*
  for good downstream use and decoding, but says nothing about whether sampled
  digits look right. Use the [sample-evaluation
  metrics](../../generative_jepa/docs/evaluating-generated-samples.md) for that.
- **Not class separability.** That's what the linear probe measures (98%/95.3%
  train/test in our run) — a complementary, label-using check that the latent is
  *semantically* organized, not just high-variance.
- **Not absolute across configs.** Because embeddings are unnormalized and rank
  depends on $D$, compare these numbers *within* a model family / across training,
  not across architectures with different $D$ or normalization.

## Code

- `feature_std`, `effective_rank`, `collapse_report` — [`src/ssllab/eval/collapse.py`](../../../src/ssllab/eval/collapse.py)
- Linear probe (complementary representation check) — [`src/ssllab/eval/probe.py`](../../../src/ssllab/eval/probe.py)
