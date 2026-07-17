# Chapter 4 — Results

*The numbers, with simultaneous confidence intervals. The encoder learns real structure and the method generalizes to unseen gene combinations. But measured against a from-scratch baseline, it loses, and it loses by a margin that survives every correction we can apply. This chapter reports that honestly, because a clean negative result is worth more than an ambiguous positive one.*

Every result here is on Norman 2019. The headline metric is the **effect size**, defined in [Chapter 3](03-training-and-evaluation.md): for a perturbation, generate a predicted response population, form its differential expression $\Delta = \mathrm{mean}(\text{predicted}) - \mathrm{mean}(\text{control})$, and correlate it against the true $\Delta$ on the perturbation's top differentially-expressed genes. The score is the Pearson correlation $r$, which we call the $\Delta$-correlation. Higher is better.

Which genes enter that correlation is a design decision rather than a detail, and [Chapter 3e](3e-the-genes-the-metric-scores.md) is the companion that explains it. The genes are the top 20 by $|z|$, the Wilcoxon rank-sum statistic against control, and they are detected in about 83% of the perturbation's cells. Ranking instead by fold change would select genes that are barely expressed at all, which produces metrics that are well-formed and meaningless. The numbers below are all on the $|z|$-selected genes.

[Chapter 4a](04a-reading-the-head-to-head.md) is the audit behind this chapter: the full scoreboard, the statistical procedure, and the map from each number to the run that produced it.

## The encoder learns real structure

Before any generation, Stage A has to produce a representation worth generating in, and two diagnostics say it does. Neither depends on the gene selection above, because neither involves the effect-size metric.

The first is a collapse check. A self-supervised objective that predicts embeddings from embeddings has a trivial cheat available to it: map every cell to the same latent, and the prediction loss vanishes while the representation carries nothing. The guard is *effective rank*, a soft count of how many latent directions the representation actually spreads across, ranging from $1$ (collapsed onto a single direction) to $256$ (every direction used equally). The pretrained encoder reaches an effective rank of about $176$ out of $256$, so it is far from collapse and uses most of the space available to it.

The second asks whether that space encodes anything *biological*. Stage A never sees a perturbation label. We freeze the encoder, encode each cell to its pooled latent $z$, and train a plain logistic regression to predict which of the $237$ perturbations a cell received from $z$ alone. The probe reaches $5.2\%$ on held-out cells. Against $100\%$ that looks poor, but $100\%$ is the wrong yardstick: with $237$ classes, chance is $0.42\%$, so $5.2\%$ is roughly twelve times chance. The absolute number is low for a biological reason rather than a modeling failure, because a single cell's transcriptome is dominated by its baseline state and a perturbation is a small shift riding on top of it. The perturbation signal is a population-level property. The honest reading is modest but real: the frozen latent carries perturbation-relevant information for the flow to use.

## In-distribution effect size

The easier test is held-out *cells* of *seen* perturbations, the `cells` split, with the simplest of the two condition encoders from [Chapter 2](02-implementation.md), the *table* condition: a learned lookup with one row per perturbation ID. This suffices in distribution, because every perturbation seen at test time was also seen in training.

The conditional flow scores a mean $\Delta$-correlation of $\mathbf{0.612}$ across $216$ perturbations, with a median of $0.626$. Two-gene combinations score higher than singles on average, because a combination usually moves more genes by larger amounts, which gives the correlation more signal to lock onto. The weak singles are the biologically weak perturbations, where the true $\Delta$ is small and dominated by noise, so a near-zero correlation there is expected rather than a model failure.

## The harder test: generalizing to unseen combinations

The test the method exists to pass is the `combo` split: hold out twenty two-gene combinations entirely, train on everything else, and predict a combination never seen.

It is worth being precise about *which* representation makes this possible, because it is not the one this chapter has been discussing. Two separate encoders are in play. The JEPA encoder maps a cell's expression to a latent and represents *cell state*; it never sees a perturbation label. The condition encoder is a distinct module that maps a perturbation *label* to an intervention embedding, and it is the one that must handle unseen combinations. The default table version assigns each perturbation its own row, so an unseen combination has no row and cannot be predicted at all. The gene-set version builds a combination's embedding from its single-gene parts, $z_p(A{+}B) = e(A) + e(B)$, and both parts were trained whenever their genes appeared in any perturbation. So the capability tested here is a property of how the *condition* is represented, not of the JEPA cell latents and not of the flow. The baseline below makes that attribution concrete.

With the compositional condition in place, the method generalizes: on the twenty held-out combinations the transport flow reaches a seed-averaged $\Delta$-correlation of $\mathbf{0.648}$.

## The baseline beats it

A number in isolation means little, so we built the obvious control: a from-scratch conditional negative-binomial VAE, with no JEPA pretraining and no flow, conditioned on the *same* gene-set embedding, so the comparison isolates the generative machinery rather than the perturbation encoding. On the same twenty held-out combinations it scores $\mathbf{0.766}$.

That is not a tie. The full stack, a self-supervised representation plus a learned conditional flow, is beaten by a plain conditional VAE that has neither, by $0.118$ in $\Delta$-correlation. The gap is larger than every effect we have chased inside the flow family, and the statistics below say it is real.

## The statistics, and why they are stricter than they look

Two facts about the test set drive the whole procedure. The unit of analysis is the **perturbation**, not the cell and not the gene, because the metric is defined per perturbation. And there are only **twenty** of them. That is a small sample, so the analysis has to work hard to be honest.

Three choices do that work, and [Chapter 4a](04a-reading-the-head-to-head.md) develops each.

**Seeds are averaged before testing.** Every configuration is retrained at three seeds and its per-perturbation scores averaged, so a difference is not an artifact of one lucky initialization.

**One primary endpoint, declared in advance.** The $\Delta$-correlation is the only metric on which we make a significance claim. Everything else is secondary and reported as an interval without a verdict, because a difference discovered on a secondary metric after the fact is a hypothesis rather than a result.

**A joint bootstrap with simultaneous intervals.** The contrasts below share arms and are computed on the same twenty perturbations, so they are not independent. We resample the perturbations once per bootstrap iteration, evaluate every contrast on that shared resample, and take a **max-$t$** critical value across the family. The result is intervals that hold *simultaneously* at 95% over all three contrasts. The critical value comes out at $2.95$, against $1.96$ for a single unadjusted test, so this is a materially higher bar than testing each comparison on its own.

| configuration | mean $\Delta$-correlation (3 seeds) |
|---|---|
| Gaussian flow (noise to outcome) | 0.612 |
| transport flow with OT coupling | 0.627 |
| transport flow (control to outcome) | 0.648 |
| **conditional NB-VAE (baseline)** | **0.766** |

| contrast | difference | simultaneous 95% CI | reading |
|---|---|---|---|
| transport − Gaussian | $+0.036$ | $[+0.019, +0.052]$ | **significant**: the transport reformulation is a real improvement |
| OT − transport | $-0.021$ | $[-0.039, -0.002]$ | **significant**: optimal-transport coupling *hurts* |
| **transport − NB-VAE** | $\mathbf{-0.118}$ | $\mathbf{[-0.228, -0.008]}$ | **significant**: the baseline wins |

All three survive the simultaneous correction. Two of them are findings we keep and build on. The third is the result of the chapter.

Note also what the seeds say. The VAE's three seeds score $0.762$, $0.767$, $0.768$, a spread of $0.006$. This is not a lucky draw, and no amount of reseeding the flow will close a gap of $0.118$ against a baseline that stable.

## Calibration: a second axis, and it does not rescue the flow

Effect size grades only the mean of the response. A generative model's real promise is the whole predictive *distribution*, and a flow can in principle bend noise into a multimodal, correlated population that a cruder generator cannot. So we measured calibration directly, comparing each model's generated population against the held-out real cells on the top-DE genes, with the four metrics of [Chapter 3b](3b-reading-the-calibration-metrics.md).

These are **secondary endpoints**. The intervals are reported; no significance is claimed on them.

| model | spread correlation ↑ | coverage (nominal 0.80) | 1-Wasserstein ↓ | joint energy ↓ |
|---|---|---|---|---|
| Gaussian flow | 0.205 | 0.357 | 1.013 | 3.794 |
| transport flow | 0.234 | 0.375 | 0.982 | **3.578** |
| transport + OT | 0.214 | 0.365 | 1.010 | 3.746 |
| NB-VAE | **0.522** | 0.328 | **0.956** | 3.962 |

Three things to read here, in descending order of confidence.

**Every model is badly under-dispersed.** Coverage sits between $0.33$ and $0.38$ against a nominal $0.80$, meaning each model's predicted $80\%$ interval captures only about a third of the real cells. The predicted populations are far *too narrow*. This is the single most robust fact on the calibration axis and it applies to the flow and the VAE alike, so it is a property of the count decoders that both read out through rather than of the flow.

**The VAE tracks per-gene variability much better.** Its spread correlation of $0.522$ against the transport flow's $0.234$ says it is substantially better at knowing *which* genes vary in a response. The contrast is $-0.288$ with an interval of $[-0.429, -0.136]$, which does not go near zero.

**The flow's apparent advantage on joint structure does not resolve.** The transport flow posts the best joint energy distance, $3.578$ against the VAE's $3.962$, and the energy distance is precisely the metric built to see the gene-gene structure a flow should capture and a marginal metric cannot. It is tempting to call this the flow earning its keep. But the contrast is $-0.383$ with an interval of $[-0.960, +0.172]$, which **crosses zero**. On twenty perturbations this difference is not resolvable, and it is a secondary endpoint besides. It is a hypothesis worth testing on a larger held-out set, and nothing more than that today.

## Where the predicted spread actually goes

The under-dispersion is worth one more measurement, because it says which component is responsible. By the law of total variance, a generated population's per-gene variance splits exactly into two parts: the count noise the decoder adds around each cell's own mean, and the spread of the decoded mean across the latent cloud. The second is the latent distribution's entire contribution, and it is the only part the flow and the VAE do differently.

| | real variance | predicted total | $\sigma^2_{\text{dec}}$ (decoder) | $\sigma^2_{\text{bio}}$ (latent) | latent's share |
|---|---|---|---|---|---|
| transport flow | 0.824 | 0.678 (0.84×) | 0.538 | 0.140 | 22% |
| NB-VAE | 0.824 | 0.355 (0.46×) | 0.226 | 0.128 | 38% |

Both models under-produce total spread, the VAE more severely than the flow. The latent distribution contributes a real but minority share in both, and the two models' latent contributions are close ($0.140$ against $0.128$). The gap between the models is therefore not a gap in what their latent distributions do; it is mostly in the decoder each learned. This is the measurement that tells the decoder work in [Chapter 8](08-modeling-the-readout-count-decoder.md) which direction to push, and the answer is *more* dispersion, not less.

## The honest scoreboard

On Norman held-out combinations, **the JEPA-plus-conditional-flow stack does not beat a from-scratch conditional NB-VAE. It loses to it, by $0.118$ in $\Delta$-correlation, significantly, and against a baseline whose seed-to-seed spread is $0.006$.** The compositional gene-set embedding, shared by both models, is what drives combination generalization, and the baseline's strength makes that attribution concrete: almost everything the method achieves on unseen combinations is achievable without JEPA and without a flow.

Within the flow family the picture is coherent and useful. Transporting from a real control latent beats transporting from noise, significantly. Optimal-transport coupling makes things worse, significantly, even though it lowers the training loss. Both of those findings survive the simultaneous correction and both are worth keeping.

This is a negative result, and it is a clean one. The method was built carefully, measured carefully, and it lost to a simpler thing. That is more useful than an ambiguous tie, because it forces the question of *why*, and the answer is not "we needed a slightly better decoder." [Chapter 5](05-challenges-and-limitations.md) dissects the causes, and [Chapter 6](06-beyond-the-current-limit.md) argues that the remaining hope is structural: the encoder is frozen and condition-blind, and the one axis on which the self-supervised premise makes a differential prediction, data efficiency, is the one we have not yet measured.

Two caveats bound everything above. The evaluation rests on a single dataset and twenty held-out combinations, and those combinations share genes, so they are not strictly independent draws. And the intervals capture uncertainty from the finite test set but not from training-seed randomness, which is averaged out before the bootstrap rather than propagated through it. Both push in the direction of the intervals being, if anything, too narrow.

---

*Previous: [Chapter 3 — Training and evaluation](03-training-and-evaluation.md). Up: [the method series](index.md). Next: [Chapter 5 — Challenges and limitations](05-challenges-and-limitations.md). The audit: [Chapter 4a](04a-reading-the-head-to-head.md). The gene selection: [Chapter 3e](3e-the-genes-the-metric-scores.md). Current state of play across all rounds: [the results ledger](results-ledger.md).*
