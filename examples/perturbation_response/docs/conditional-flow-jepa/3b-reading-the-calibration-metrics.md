# Chapter 3b — Reading the calibration metrics: spread, coverage, Wasserstein, and energy

*A companion to [Chapter 3](03-training-and-evaluation.md), alongside [Chapter 3a](3a-the-models-in-the-head-to-head.md). Chapter 3a named the three models and the way they are compared; this note names the four numbers that grade the *distribution* they produce. [Chapter 4](04-results.md) reports those numbers and [Chapter 4a](04a-reading-the-head-to-head.md) audits them; the point here is only to make them legible in advance — what each metric asks, how it is computed, and, above all, how to read the one that trips everyone: coverage.*

> **Where this sits.** Read [Chapter 3](03-training-and-evaluation.md) for how effect size and calibration are scored in the harness. This note zooms in on the calibration half so that the calibration table in [Chapter 4](04-results.md) reads as a set of clean claims. Everything below is implemented in [`src/ssllab/eval/calibration.py`](../../../../src/ssllab/eval/calibration.py); the statistics for deciding whether two calibration numbers *differ* live in [Chapter 3a §5](3a-the-models-in-the-head-to-head.md) and are applied in [Chapter 4a](04a-reading-the-head-to-head.md).

## 1. Why grade the distribution at all

Effect size, the headline metric, grades exactly one thing: the **mean** response. For a perturbation you generate a population of cells, average their expression gene by gene, subtract the control mean, and check that the resulting shift $\Delta$ points the right way. That is a statement about the *center* of the predicted response and nothing else.

But a perturbation does not produce one outcome. Identical cells given the identical intervention respond differently, and the responding population has a real *spread* — sometimes it even splits into two fates. A good generative model should get that spread right, not only the center. And this is where the whole argument for a **flow** lives: a flow can bend noise into an arbitrarily shaped, multimodal population, structure that a point predictor or a plain Gaussian generator cannot represent. If that capability is real, it shows up in the *distribution*, not in the mean. So once the means tie (as [Chapter 4](04-results.md) finds they do) calibration is the axis that could still separate the models. It is where the flow's machinery would earn its keep, if it earns it anywhere.

## 2. The shared setup, and the one rule you cannot skip

All four metrics do the same thing at the top level. For one perturbation:

1. **Generate** a predicted population — draw many latents from the model, decode each to gene counts, normalize. Call it `pred_pop`, shape `(n_generated, G)`.
2. **Take the truth** — the held-out *real* cells of that perturbation, `true_pop`.
3. **Restrict to the top-DE genes** — the ~20 genes the perturbation actually moved. Grading genes the intervention never touched would just be measuring noise.
4. **Compare** `pred_pop` against `true_pop` on those genes, four ways.

Then average each metric over all evaluable perturbations.

> **The measurement rule.** The predicted population must be built by **sampling counts** from the decoder's negative binomial, *not* by reading its expected rate. A population of expected rates has almost no spread — every latent decodes to nearly the same rate vector — so its predicted interval is razor-thin and every calibration number collapses (coverage near $0$). Real single-cell variation is dominated by count-level noise that appears only once you actually draw counts. This is the single easiest way to get calibration wrong, and [Chapter 4a](04a-reading-the-head-to-head.md) states explicitly that its numbers use sampled counts.

## 3. The four metrics

### Spread correlation — *which* genes vary

**Question:** does the model know which genes vary a lot in the response, and which vary little?

**Computed** (`spread_correlation`): for each top-DE gene, take the standard deviation of the predicted population and of the true population. That gives two vectors of one std per gene; the metric is the **Pearson correlation** between them,

$$\text{spread\_r} = \mathrm{corr}\big(\mathrm{std}_g[\text{pred}], \mathrm{std}_g[\text{true}]\big)_{g \in \text{top-DE}}.$$

**Reading it.** A correlation in $[-1, 1]$, about *ranking* rather than magnitude. High positive means the genes the model calls high-variance are the ones that really are. Zero means no relationship. **Negative means the ranking is backwards** — the model predicts the largest spread on the genes that are actually tightest. What it does *not* see: whether the overall spread is too big or too small. That is coverage's job.

### Central-interval coverage — is the spread the *right size*

This is the metric worth slowing down on.

**Question:** is the predicted spread neither too tight nor too wide?

**Computed** (`interval_coverage`, with `lo=0.1`, `hi=0.9`): for each top-DE gene, look at the **predicted** population and find its central 80% band — the 10th and 90th percentiles of the predicted expression, $[q_{10}^{\text{pred}}, q_{90}^{\text{pred}}]$. Now take the **real** cells for that gene and count what fraction fall inside that predicted band. Average that fraction over the genes,

$$\text{coverage} = \operatorname{mean}_{g}\Big(\text{fraction of true cells with } q_{10}^{\text{pred}} \le x_g^{\text{true}} \le q_{90}^{\text{pred}}\Big).$$

**The key idea:** you built an interval meant to contain 80% of outcomes — that is what "10th-to-90th percentile" means — and then you check what fraction of *real* outcomes it actually contains. So:

- **coverage ≈ 0.80** (the nominal target): calibrated. An 80% predicted interval really does contain about 80% of real cells.
- **coverage < 0.80**: the band is **too narrow** → the model is **over-confident** (under-dispersed). Reality is wider than the model thinks, so its interval catches less than it promised.
- **coverage > 0.80**: the band is **too wide** → the model is **under-confident / over-dispersed**. Its interval swallows more than 80% of real cells because the predicted population is more spread out than reality.

The signed gap `coverage − 0.80` says which way, and by how much.

**A worked example, one gene.** Say your 200 generated cells for gene *G* have 10th percentile $0.5$ and 90th percentile $3.0$ on the log1p-CP10K scale, so the predicted central band is $[0.5, 3.0]$. Now look at the real held-out cells for *G*:

- **80 of 100** fall in $[0.5, 3.0]$ → coverage $0.80$. Perfect.
- **100 of 100** fall in → coverage $1.00$. The band was so wide it caught *everyone*; the model thinks the response varies more than it does. **Over-dispersed.**
- **50 of 100** fall in → coverage $0.50$. The band was too tight; real cells spill outside it. **Over-confident.**

Average the per-gene fraction over the ~20 top-DE genes to get the reported number.

**Why Chapter 4 shows 1.00, and why that is a bad sign.** Every model reports coverage $1.00$ against nominal $0.80$. By the rule above, the predicted 80% bands are so wide they contain essentially *all* the real cells — the predicted populations are **over-dispersed** on the top-DE genes. The cause is the shared negative-binomial decoder: on these high-signal genes its sampled counts carry more variance than the real cells do. So $1.00$ is not "great, we cover everything"; it is "our uncertainty is too large, and saturated so hard we cannot even see the gap." And because *all* models share that decoder, they all pin at $1.00$, which is why coverage cannot tell them apart — the tell that this metric is reporting the decoder, not the flow.

### Mean 1-Wasserstein — the holistic per-gene distance

**Question:** overall, how far is each predicted per-gene distribution from the true one?

**Computed** (`mean_wasserstein`): for each top-DE gene, treat predicted and true values as two 1-D distributions and compute the **1-Wasserstein** (earth-mover) distance — the minimum work to reshape one pile of probability mass into the other. Average over genes. Lower is better; $0$ means an exact per-gene match.

**Reading it.** Wasserstein is holistic: unlike coverage (spread only) or a mean check (center only), it folds **location, spread, and shape** into one number. Shift the predicted distribution, over-widen it, or get its shape wrong, and Wasserstein grows. A good single summary of per-gene fit — but still per-gene, so blind to how genes move together.

### Energy distance — the *joint* metric

**Question:** does the predicted population match the true one in its *joint* structure — the correlations between genes, and any multimodality — not just gene by gene?

**Computed** (`energy_distance`): treat each population as a cloud of points in the ~20-dimensional top-DE gene space and compute the two-sample **energy distance**

$$\mathcal{E} = 2 \mathbb{E}\lVert P - Q\rVert - \mathbb{E}\lVert P - P'\rVert - \mathbb{E}\lVert Q - Q'\rVert,$$

where $P$ is a predicted cell, $Q$ a true cell, and primes are independent copies: twice the average distance between predicted and true cells, minus the average spread within each population. It is $0$ exactly when the two clouds are identically distributed, positive otherwise. Lower is better.

**Why it is the important one.** The first three metrics are all **marginal** — one gene at a time, never asking whether genes move *together*. Energy distance looks at the **joint** cloud, so it is sensitive to gene-gene correlation and to multimodality: precisely the structure a rich latent flow can represent and a per-gene view cannot. This is the metric where the flow's theoretical advantage would appear if it exists — which is what makes [Chapter 4](04-results.md)'s finding that it still favors the VAE meaningful rather than incidental.

## 4. The one distinction that ties it together: marginal vs joint

If you keep one thing from this note, make it this. Three of the four metrics are **marginal** (per-gene) — spread correlation, coverage, Wasserstein each look at one gene's distribution at a time. Only **energy distance** is **joint** (multivariate). That split explains the whole shape of the calibration result in Chapter 4.

| metric | what it sees | consequence for the comparison |
|---|---|---|
| spread correlation | per-gene: which genes vary | marginal → dominated by the shared decoder |
| interval coverage | per-gene: is the spread the right size | marginal → saturates at 1.00 for all (over-dispersed decoder) |
| mean 1-Wasserstein | per-gene: full 1-D fit | marginal → near-identical across models |
| energy distance | **joint**: gene-gene correlation, multimodality | the one read that *could* separate the flow |

The three generative models share the negative-binomial decoder, so on every marginal metric they are largely reporting that decoder rather than their latent distributions — which is exactly why they look nearly identical there. The joint metric is the only place the latent distribution itself can show through. When [Chapter 4a](04a-reading-the-head-to-head.md) reports that even the energy distance ranks the VAE first, and that the transport-versus-VAE gap is not significant, the marginal-versus-joint frame is what tells you that result is not an artifact of the decoder: the one metric built to see past the decoder agrees with the mean axis.

## 5. What to carry into the results

Four numbers, all comparing a **sampled** predicted population to the real held-out cells on the top-DE genes. **Spread correlation** asks whether the model ranks which genes vary correctly. **Coverage** asks whether the predicted spread is the right size — $0.80$ is calibrated, below is over-confident, above is over-dispersed — and a value pinned at $1.00$ means over-dispersed and saturated, driven by the shared decoder. **Mean 1-Wasserstein** is a holistic per-gene distance. **Energy distance** is the only *joint* metric, and thus the flow's best chance to show an edge. Three marginal reads plus one joint read: with that in hand, the calibration table in [Chapter 4](04-results.md) reads as one coherent story — the marginal metrics report the shared decoder and cannot separate the models, and the one metric that could still ranks the VAE first, so the distribution axis agrees with the mean axis rather than rescuing the flow.

---

*Previous: [Chapter 3a — The models in the head-to-head](3a-the-models-in-the-head-to-head.md). Up: [the method series](index.md). Next: [Chapter 3c — The VICReg collapse guard](3c-the-vicreg-collapse-guard.md).*
