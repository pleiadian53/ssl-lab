# Chapter 3b — Reading the calibration metrics: spread, coverage, Wasserstein, and energy

*A companion to [Chapter 3](03-training-and-evaluation.md), alongside [Chapter 3a](3a-the-models-in-the-head-to-head.md). Chapter 3a named the three models and the way they are compared; this note names the four numbers that grade the *distribution* they produce. [Chapter 4](04-results.md) reports those numbers and [Chapter 4a](04a-reading-the-head-to-head.md) audits them; the point here is only to make them legible in advance: what each metric asks, how it is computed, and, above all, how to read the one that trips everyone, which is coverage.*

> **Where this sits.** Read [Chapter 3](03-training-and-evaluation.md) for how effect size and calibration are scored in the harness. This note zooms in on the calibration half so that the calibration table in [Chapter 4](04-results.md) reads as a set of clean claims. Everything below is implemented in [`src/ssllab/eval/calibration.py`](../../../../src/ssllab/eval/calibration.py); the statistics for deciding whether two calibration numbers *differ* live in [Chapter 3a §5](3a-the-models-in-the-head-to-head.md) and are applied in [Chapter 4a](04a-reading-the-head-to-head.md).

> **All four of these are secondary endpoints.** The study declares exactly one primary endpoint in advance, the effect-size $\Delta$-correlation, and that is the only metric on which any significance claim is made ([Chapter 4](04-results.md)). Everything on this page is **secondary and exploratory**. Its numbers are reported with intervals and read for direction, never for a verdict. A difference that shows up on a secondary metric after the fact is a hypothesis to test on a larger held-out set, not a finding. Carry that caveat through every section below.

## 1. Why grade the distribution at all

Effect size, the headline metric, grades exactly one thing: the **mean** response. For a perturbation you generate a population of cells, average their expression gene by gene, subtract the control mean, and check that the resulting shift $\Delta$ points the right way. That is a statement about the *center* of the predicted response and nothing else.

But a perturbation does not produce one outcome. Identical cells given the identical intervention respond differently, and the responding population has a real *spread*, sometimes even splitting into two fates. A good generative model should get that spread right, not only the center. And this is where the whole argument for a **flow** lives: a flow can bend noise into an arbitrarily shaped, multimodal population, structure that a point predictor or a plain Gaussian generator cannot represent. If that capability is real, it shows up in the *distribution*, not in the mean. Calibration is therefore a second axis, independent of the first. [Chapter 4](04-results.md) finds that the mean axis already favors the conditional NB-VAE baseline, so the distribution is the place where the flow's machinery could still earn its keep, if it earns it anywhere.

## 2. The shared setup, and the one rule you cannot skip

All four metrics do the same thing at the top level. For one perturbation:

1. **Generate** a predicted population by drawing many latents from the model, decoding each to gene counts, and normalizing. Call it `pred_pop`, shape `(n_generated, G)`.
2. **Take the truth**, the held-out *real* cells of that perturbation, `true_pop`.
3. **Restrict to the top-DE genes**, the ~20 genes the perturbation actually moved. Grading genes the intervention never touched would just be measuring noise.
4. **Compare** `pred_pop` against `true_pop` on those genes, four ways.

Then average each metric over all evaluable perturbations.

> **The measurement rule.** The predicted population must be built by **sampling counts** from the decoder's negative binomial, *not* by reading its expected rate. A population of expected rates has almost no spread, because every latent decodes to nearly the same rate vector, so its predicted interval is razor-thin and every calibration number collapses (coverage near $0$). Real single-cell variation is dominated by count-level noise that appears only once you actually draw counts. This is the single easiest way to get calibration wrong, and [Chapter 4a](04a-reading-the-head-to-head.md) states explicitly that its numbers use sampled counts.

## 3. The four metrics

### Spread correlation — *which* genes vary

**Question:** does the model know which genes vary a lot in the response, and which vary little?

**Computed** (`spread_correlation`): for each top-DE gene, take the standard deviation of the predicted population and of the true population. That gives two vectors of one std per gene; the metric is the **Pearson correlation** between them,

$$\text{spread\_r} = \mathrm{corr}\big(\mathrm{std}_g[\text{pred}], \mathrm{std}_g[\text{true}]\big)_{g \in \text{top-DE}},$$

where $\mathrm{std}_g[\cdot]$ is the standard deviation of gene $g$'s expression across the cells of a population, and the correlation runs over the genes in the top-DE list.

**Reading it.** A correlation in $[-1, 1]$, about *ranking* rather than magnitude. High positive means the genes the model calls high-variance are the ones that really are. Zero means no relationship. Negative means the ranking is backwards, i.e. the model predicts the largest spread on the genes that are actually tightest. What it does *not* see: whether the overall spread is too big or too small. That is coverage's job.

**What Chapter 4 shows.** Every model lands positive, from $0.205$ for the Gaussian flow to $0.522$ for the NB-VAE, with the transport flow at $0.234$. So all of them do track which genes vary, and the VAE tracks it substantially better than any flow variant. Since a positive spread correlation says nothing about the *size* of the spread, this is entirely compatible with the coverage reading below, where every model turns out to be too narrow. Knowing which genes vary most and knowing how much they vary are separate skills, and the metrics are built to grade them separately.

### Central-interval coverage — is the spread the *right size*

This is the metric worth slowing down on.

**Question:** is the predicted spread neither too tight nor too wide?

**Computed** (`interval_coverage`, with `lo=0.1`, `hi=0.9`): for each top-DE gene, look at the **predicted** population and find its central 80% band, the 10th and 90th percentiles of the predicted expression, $[q_{10}^{\text{pred}}, q_{90}^{\text{pred}}]$. Now take the **real** cells for that gene and count what fraction fall inside that predicted band. Average that fraction over the genes,

$$\text{coverage} = \operatorname{mean}_{g}\Big(\text{fraction of true cells with } q_{10}^{\text{pred}} \le x_g^{\text{true}} \le q_{90}^{\text{pred}}\Big),$$

where $x_g^{\text{true}}$ is the observed expression of gene $g$ in one real held-out cell.

**The key idea:** you built an interval meant to contain 80% of outcomes, which is what "10th-to-90th percentile" means, and then you check what fraction of *real* outcomes it actually contains. So:

- **coverage ≈ 0.80** (the nominal target): calibrated. An 80% predicted interval really does contain about 80% of real cells.
- **coverage < 0.80**: the band is **too narrow**, so the model is **over-confident** (under-dispersed). Reality is wider than the model thinks, and its interval catches less than it promised.
- **coverage > 0.80**: the band is **too wide**, so the model is **under-confident** (over-dispersed). Its interval swallows more than 80% of real cells because the predicted population is more spread out than reality.

The signed gap `coverage − 0.80` says which way, and by how much.

**A worked example, one gene.** Say your 200 generated cells for gene *G* have 10th percentile $0.5$ and 90th percentile $3.0$ on the log1p-CP10K scale, so the predicted central band is $[0.5, 3.0]$. Now look at the real held-out cells for *G*:

- **80 of 100** fall in $[0.5, 3.0]$, so coverage is $0.80$. Perfect.
- **50 of 100** fall in, so coverage is $0.50$. The band was too tight and real cells spill outside it. **Over-confident.**
- **100 of 100** fall in, so coverage is $1.00$. The band caught *everyone*, and on this one gene that reads as **over-dispersed**. Hold that reading loosely: a coverage pinned at exactly $1.00$ across the *whole* gene list has a second and far more likely explanation, which the warning below spells out.

Average the per-gene fraction over the ~20 top-DE genes to get the reported number.

**What Chapter 4 shows: every model is too narrow.** Coverage lands between $0.33$ and $0.38$ against a nominal $0.80$ (Gaussian flow $0.357$, transport flow $0.375$, transport with OT coupling $0.365$, NB-VAE $0.328$). Each model's 80% predicted interval therefore captures only about a third of the real cells, so the predicted populations are far **too narrow**: these models are **over-confident**, i.e. under-dispersed. This is the most robust fact on the whole calibration axis, and it holds for the flow and the VAE alike, which points at the count decoders they both read out through rather than at anything the latent distribution does.

**Where the missing spread went.** Under-dispersion is worth one more measurement, because it says *which component* is short. By the law of total variance, the per-gene variance of a generated population splits exactly in two:

$$\underbrace{\mathrm{Var}[x_g]}_{\text{predicted total}} = \underbrace{\mathbb{E}_{z}\big[\mathrm{Var}[x_g \mid z]\big]}_{\sigma^2_{\text{dec}}} + \underbrace{\mathrm{Var}_{z}\big[\mathbb{E}[x_g \mid z]\big]}_{\sigma^2_{\text{bio}}},$$

where $x_g$ is gene $g$'s expression in a generated cell, $z$ is the latent that cell was decoded from, $\sigma^2_{\text{dec}}$ is the count noise the decoder adds around each cell's own mean, and $\sigma^2_{\text{bio}}$ is the spread of the decoded mean *across* the latent cloud. That second term is the latent distribution's entire contribution to the predicted spread, and it is the only part the flow and the VAE do differently. Measured on the held-out combinations (script `11_diagnose_variance.py`):

| | real variance | predicted total | $\sigma^2_{\text{dec}}$ (decoder) | $\sigma^2_{\text{bio}}$ (latent) | latent's share |
|---|---|---|---|---|---|
| transport flow | 0.824 | 0.678 (0.84×) | 0.538 | 0.140 | 22% |
| NB-VAE | 0.824 | 0.355 (0.46×) | 0.226 | 0.128 | 38% |

Both models under-produce total spread, the VAE the more severely of the two, and the two latent contributions are close ($0.140$ against $0.128$). The shortfall is therefore mostly in the decoder each model learned, not in what its latent distribution does, which tells the decoder work in [Chapter 8](08-modeling-the-readout-count-decoder.md) exactly which way to push. The fix is **more** dispersion, not less.

> **A coverage of exactly 1.00 is not a decoder reading. It is a gene-list reading.** This is the trap worth internalizing, because it survives this dataset. Suppose the genes being scored are effectively silent, so that every real held-out cell is exactly zero on them. Then *any* predicted interval that contains zero contains 100% of the real cells, and coverage is forced to $1.00$ no matter what the decoder does. The number is stable, it sits in $[0, 1]$, and it is meaningless. Worse, it *mimics* an over-dispersed decoder perfectly, so the natural response is to go tighten a dispersion that was never the cause, and coverage will not move. The rule to keep: **if you ever see coverage pinned at exactly $1.00$, audit the gene selection before you conclude anything about dispersion.** The companion [Chapter 3e](3e-the-genes-the-metric-scores.md) works through why the top-DE list must be ranked by the Wilcoxon $z$ statistic, which cannot blow up on a silent gene, rather than by fold change, whose ratio explodes precisely on genes that are barely expressed. A real dispersion signal, of the kind reported just above, looks nothing like saturation: it is a graded number strictly between $0$ and $1$ that responds when you change the decoder.

### Mean 1-Wasserstein — the holistic per-gene distance

**Question:** overall, how far is each predicted per-gene distribution from the true one?

**Computed** (`mean_wasserstein`): for each top-DE gene, treat predicted and true values as two 1-D distributions and compute the **1-Wasserstein** (earth-mover) distance, the minimum work needed to reshape one pile of probability mass into the other. Average over genes. Lower is better, and $0$ means an exact per-gene match.

**Reading it.** Wasserstein is holistic: unlike coverage (spread only) or a mean check (center only), it folds **location, spread, and shape** into one number. Shift the predicted distribution, over-widen it, or get its shape wrong, and Wasserstein grows. It is a good single summary of per-gene fit, but still per-gene, so blind to how genes move together.

**What Chapter 4 shows.** The four models fall in a narrow band, from $0.956$ for the NB-VAE to $1.013$ for the Gaussian flow, with the transport flow at $0.982$. The ordering matches the spread correlation, and the margins are small enough that this metric mostly confirms the others rather than adding a separate verdict.

### Energy distance — the *joint* metric

**Question:** does the predicted population match the true one in its *joint* structure, meaning the correlations between genes and any multimodality, and not merely gene by gene?

**Computed** (`energy_distance`): treat each population as a cloud of points in the ~20-dimensional top-DE gene space and compute the two-sample **energy distance**

$$\mathcal{E} = 2 \mathbb{E}\lVert P - Q\rVert - \mathbb{E}\lVert P - P'\rVert - \mathbb{E}\lVert Q - Q'\rVert,$$

where $P$ is a predicted cell, $Q$ a true cell, primes denote independent copies drawn from the same population, and $\lVert \cdot \rVert$ is the Euclidean norm in gene space. In words: twice the average distance between predicted and true cells, minus the average spread within each population. It is $0$ exactly when the two clouds are identically distributed, and positive otherwise. Lower is better.

**Why it is the interesting one.** The first three metrics are all **marginal**, looking at one gene at a time and never asking whether genes move *together*. Energy distance looks at the **joint** cloud, so it is sensitive to gene-gene correlation and to multimodality, which is precisely the structure a rich latent flow can represent and a per-gene view cannot. This is the metric where the flow's theoretical advantage would appear if it exists anywhere.

**What Chapter 4 shows, and how far it can be pushed.** The transport flow posts the best joint energy distance, $3.578$ against the NB-VAE's $3.962$, with the Gaussian flow at $3.794$ and the OT variant at $3.746$. It is tempting to read that as the flow finally earning its keep on the axis built to reward it. Resist the temptation. The transport-minus-VAE contrast is $-0.383$ with a 95% interval of $[-0.960, +0.172]$, and that interval **crosses zero**: on twenty held-out perturbations the difference simply does not resolve. It is also a secondary endpoint, so no significance is claimed on it in any case. The honest statement is that this is a **hypothesis** worth testing on a larger held-out set, and today it is nothing more than that.

## 4. The one distinction that ties it together: marginal vs joint

If you keep one thing from this note, make it this. Three of the four metrics are **marginal** (per-gene), because spread correlation, coverage, and Wasserstein each look at one gene's distribution at a time. Only **energy distance** is **joint** (multivariate). That split explains the shape of the calibration result in Chapter 4.

| metric | what it sees | what it says in Chapter 4 |
|---|---|---|
| spread correlation | per-gene: which genes vary | positive for every model ($0.205$ to $0.522$); the VAE ranks variability much better |
| interval coverage | per-gene: is the spread the right size | $0.33$ to $0.38$ against nominal $0.80$; every model is under-dispersed |
| mean 1-Wasserstein | per-gene: full 1-D fit | a narrow band, $0.956$ to $1.013$; largely confirms the ordering above |
| energy distance | **joint**: gene-gene correlation, multimodality | the transport flow leads, but the interval crosses zero and it does not resolve |

The marginal reads all point the same way, and the variance decomposition of §3 explains why: what separates these models on a per-gene view is mostly the decoder each one learned, since the latent's share of the predicted variance is a minority in both ($22\%$ for the transport flow, $38\%$ for the VAE) and the two latent contributions are nearly equal in absolute terms. The three flow variants, which share a decoder, land within a whisker of each other on every marginal metric, exactly as that account predicts. The joint metric is the only place where the latent distribution itself has room to show through, and it is the only calibration number that ranks the transport flow first. But it ranks it first by a margin the data cannot certify, on a secondary endpoint, so the marginal-versus-joint frame does not deliver a verdict here. What it delivers is a well-posed question: does the flow's joint advantage survive a held-out set large enough to measure it?

## 5. What to carry into the results

Four numbers, all comparing a **sampled** predicted population to the real held-out cells on the top-DE genes, and all of them **secondary endpoints** on which no significance is claimed. **Spread correlation** asks whether the model ranks which genes vary correctly, and every model does, the VAE best. **Coverage** asks whether the predicted spread is the right size, where $0.80$ is calibrated, above is over-dispersed, and below is over-confident; every model sits near $0.35$, so every model is too narrow, and a variance decomposition puts most of that shortfall in the decoder rather than the latent. If you ever see coverage pinned at exactly $1.00$, that is a gene-list alarm, not a dispersion reading, and [Chapter 3e](3e-the-genes-the-metric-scores.md) is the place to go. **Mean 1-Wasserstein** is a holistic per-gene distance and moves with the other marginals. **Energy distance** is the only *joint* metric, and thus the flow's best chance to show an edge; the transport flow does lead on it, by a margin whose interval crosses zero. Three marginal reads plus one joint read: with that in hand, the calibration table in [Chapter 4](04-results.md) resolves into one coherent story, in which the models are uniformly over-confident for reasons that live in the decoder, and the one metric that could still favor the flow does so too faintly to be called a result.

---

*Previous: [Chapter 3a — The models in the head-to-head](3a-the-models-in-the-head-to-head.md). Up: [the method series](index.md). Next: [Chapter 3c — The VICReg collapse guard](3c-the-vicreg-collapse-guard.md).*
