# Chapter 4 — Results

*The numbers, with confidence intervals. The encoder learns real structure; the method generalizes to unseen gene combinations; and measured against a from-scratch baseline, it does not yet win. This chapter reports the head-to-head honestly, which is what will point to the next move.*

Every result here is on Norman 2019, evaluated by the effect-size metric of [Chapter 3](03-training-and-evaluation.md): for a perturbation, generate a predicted response population, form its differential expression $\Delta = \mathrm{mean}(\text{predicted}) - \mathrm{mean}(\text{control})$, and correlate it against the true $\Delta$ on the perturbation's top differentially-expressed genes. The score is the Pearson correlation $r$, which we call the $\Delta$-correlation. Higher is better; $1.0$ is a perfect match of the response's shape across those genes.

This chapter is the narrative. A companion note, [Chapter 4a](04a-reading-the-head-to-head.md), is the audit behind it: one consolidated scoreboard at full precision, the paired-bootstrap statistics that make the gap a tie, the seed-noise evidence, and a map from each number to the experiment that produced it.

## The encoder learns real structure

Before any generation, Stage A has to produce a representation worth generating in, and two diagnostics say it does.

The first is a collapse check. A self-supervised objective that predicts embeddings from embeddings has a trivial cheat available to it: map every cell to the same latent, and the prediction loss vanishes while the representation carries nothing. The guard against this is *effective rank*, a soft count of how many latent directions the representation actually spreads across, ranging from $1$ (fully collapsed onto a single direction) to $256$ (every direction used equally). On the full data the pretrained encoder reaches an effective rank of about $176$ out of $256$, so it is far from the collapsed, low-rank failure a self-supervised objective can fall into and uses most of the space available to it.

The second diagnostic asks whether that space encodes anything *biological*, and the result only makes sense against its setup. Stage A never sees a perturbation label. We freeze the encoder, encode each cell to its single pooled latent $z$, and then train a plain logistic-regression classifier to predict which of the $237$ perturbations a cell received from $z$ alone. If a linear rule can read perturbation identity off the frozen latent, the encoder learned perturbation-relevant biology without ever being told about it. That is the headline self-supervised claim, and this probe is how we test it.

The probe reaches $5.2\%$ accuracy on held-out cells. Measured against $100\%$ that looks poor, but $100\%$ is the wrong yardstick: with $237$ classes, random guessing scores only $0.42\%$, so $5.2\%$ is roughly twelve times chance, a lift far beyond what noise across thousands of test cells could produce. The absolute number is low for a biological reason rather than a modeling failure. A single cell's transcriptome is dominated by its baseline state, and a perturbation is a small shift riding on top of that baseline, so one cell rarely reveals which intervention it received. The perturbation signal is a population-level property, which is exactly what the flow will condition on. The honest reading is therefore modest but real: the frozen latent carries perturbation-relevant information for the flow to use, even when no single cell pins its own label down.

## In-distribution effect size

The first question is whether the full stack recovers effect size at all, on the easier test: held-out *cells* of *seen* perturbations, the `cells` split. The condition fed to the flow here is the simplest of the two encoders from [Chapter 2](02-implementation.md), the *table* condition. A perturbation is a label naming which gene or genes were activated, a single gene such as `CEBPE` or a two-gene combination such as `CEBPE+RUNX1T1`, and each of the $237$ perturbations carries its own integer ID. The table condition is a learned lookup table with one row per ID: to condition on a perturbation you fetch its row, and that row is the perturbation embedding. This is all we need in distribution, because every perturbation seen at test time was also seen in training, so its row was trained. The harder combination test in the next section is exactly where a per-perturbation table breaks, and a compositional encoder has to take over. With the table condition, the conditional flow scores a mean $\Delta$-correlation of $0.469$ across $216$ perturbations, with a median of $0.470$. The spread is wide and informative. About $45\%$ of perturbations exceed $0.5$ and $22\%$ exceed $0.7$, while only $5\%$ go negative. Two-gene combinations score higher on average than singles, $0.548$ against $0.384$, because a combination usually moves more genes by larger amounts, which gives the correlation more signal to lock onto. The weak singles are the biologically weak perturbations, where the true $\Delta$ is small and dominated by noise, so a near-zero correlation there is expected rather than a model failure.

## The harder test: generalizing to unseen combinations

The test that actually matters for this method is the `combo` split: hold out twenty two-gene combinations entirely, train on everything else, and ask the model to predict a combination it has never seen. This is only possible because of the compositional gene-set condition of [Chapter 2](02-implementation.md).

It is worth being precise about *which* representation makes this work, because it is not the one the rest of the chapter has been discussing. Two separate encoders are in play. The JEPA encoder maps a cell's expression to a latent and represents *cell state*; it never sees a perturbation label. The condition encoder is a distinct module that maps a perturbation *label* to the intervention embedding $z_p$, and it is the one that must handle unseen combinations. The default table version of it assigns each perturbation its own learned embedding row, so an unseen combination has no row to use and cannot be predicted at all. The gene-set version instead builds the combination's embedding additively from its two single-gene parts, $z_p(A{+}B) = e(A) + e(B)$, and both parts were trained whenever their genes appeared in any perturbation. So the capability being tested here is a property of how the *condition* is represented, not of the JEPA cell latents and not of the flow. The result does not say the latent space composes gene combinations; it says the gene-set condition does. The baseline below makes that attribution concrete.

With that compositional condition in place, the method does generalize. On the twenty held-out combinations the flow reaches a mean $\Delta$-correlation of $0.61$. That is higher than the in-distribution cells number above, which is not a contradiction: these held-out combinations happen to have large, clean effects, exactly the regime where the metric is easiest. Generalization here is about the *kind* of held-out example, not about the raw difficulty of the number.

## The baseline that reframes everything

A number in isolation means little. To calibrate it we built the obvious control: a from-scratch conditional negative-binomial VAE, described in [Chapter 5](05-challenges-and-limitations.md), with no JEPA pretraining and no flow, conditioned on the same gene-set embedding, so the comparison isolates the generative machinery rather than being confounded by the perturbation encoding. On the same twenty held-out combinations the baseline scores a mean $\Delta$-correlation of $0.633$.

That result reframes the project. Line the numbers up: the baseline's $0.633$ is level with the flow's $0.62$, and on the raw average it sits a shade above. The elaborate part of the method, a JEPA representation plus a learned conditional flow, was meant to be the contribution here, and on this metric it does not beat a plain conditional VAE that has neither piece. Whether the small gap in the baseline's favor is even real is what the seed analysis below settles; the point for now is only that the fancy stack has not pulled ahead of the simple generator.

## The transport reformulation helps the flow

Investigating why led to a real improvement. In the original formulation the flow transports Gaussian noise to the outcome latent, and the baseline state $z_b$ enters only as part of the condition, drawn as a random, unpaired control cell. Because that $z_b$ is statistically independent of the specific target, it cannot shift the predicted mean, so the condition collapses to "generate the perturbed population," which is the same object the VAE fits. The reformulation, developed in [Chapter 5](05-challenges-and-limitations.md), makes the flow transport a real control latent to the outcome, so it models the displacement, the effect itself, with $z_b$ anchoring each sample.

Run head to head with the same encoder, decoder, and seed, the transport formulation lifts the mean $\Delta$-correlation from $0.580$ to $0.621$, and improves on fourteen of the twenty combinations. The largest gains land on the hard, low-signal combinations, and the worst case moves from slightly negative to positive. So the reformulation is a genuine improvement to the flow. It is not, however, enough to clear the baseline: $0.621$ still sits below the VAE's $0.633$.

## Seeds change the ranking, so we measured the noise

A single training seed turned out to be an unreliable narrator. Retraining each configuration at three seeds shows the per-seed $\Delta$-correlation swinging by more than the effects we care about. The Gaussian flow alone ranges across $0.623$, $0.562$, and $0.569$; the transport flow across $0.575$, $0.653$, and $0.611$. At one seed the Gaussian flow wins, at another the transport flow wins by a wide margin. Only by averaging over seeds and comparing per-combination pairs does a stable picture emerge.

The table below reports the seed-averaged mean $\Delta$-correlation for each configuration, and the paired bootstrap difference between configurations, resampling the twenty combinations ten thousand times. A confidence interval that excludes zero is a difference we can trust; one that spans zero is within the noise of a twenty-combination test set.

| configuration | mean $\Delta$-correlation |
|---|---|
| Gaussian flow (noise to outcome) | 0.584 |
| transport flow (control to outcome) | 0.613 |
| transport flow with OT coupling | 0.590 |
| conditional NB-VAE (baseline) | 0.633 |

| comparison | difference | 95% CI | reading |
|---|---|---|---|
| transport − Gaussian | +0.028 | [−0.006, +0.064] | borderline; transport's edge is likely real |
| OT − transport | −0.023 | [−0.041, −0.007] | significant; OT coupling **hurts** |
| transport − VAE | −0.020 | [−0.077, +0.044] | not significant; a tie |
| VAE − Gaussian | +0.049 | [−0.021, +0.112] | not significant |

Two things stand out. First, the transport reformulation's edge over the Gaussian flow survives seed averaging, at borderline significance. Second, and this is the one clearly significant effect in the whole sweep, optimal-transport coupling *hurts*: it lowers the flow-matching training loss by straightening the paths, yet it lowers the $\Delta$-correlation. A clean reminder that a better training objective is not the same as a better downstream metric. The reasons are discussed in [Chapter 5](05-challenges-and-limitations.md).

Notably, the last row also corrects an earlier, hastier reading. The single-seed run had the VAE ahead of the Gaussian flow by $0.053$, which looked like the baseline decisively winning. Seed-averaged the gap narrows to $0.049$, and with three seeds and a paired test it is not statistically distinguishable from zero. The most defensible statement is that on this test set the three generative configurations are within noise of each other, with the transport flow the best of the flow variants and the VAE at or slightly above them all.

## Calibration: the same verdict from a different angle

Effect size grades only the mean of the response. A generative model's real promise is the whole predictive *distribution*, and a flow can in principle bend noise into a multimodal, correlated population that a cruder generator cannot. So we measured calibration directly, comparing each model's generated population against the held-out real cells on the top differentially-expressed genes. The metrics are per-gene spread correlation, central-interval coverage, mean 1-Wasserstein distance, and a multivariate two-sample energy distance that sees the joint structure across genes. [Chapter 3b](3b-reading-the-calibration-metrics.md) is a primer on all four — in particular what coverage means and why $1.00$ is a bad sign — and the implementation is in [`calibration.py`](../../../../src/ssllab/eval/calibration.py).

| model | joint energy (lower better) | 1-Wasserstein (lower better) | coverage (nominal 0.80) |
|---|---|---|---|
| transport flow | 0.038 | 0.009 | 1.00 |
| Gaussian flow | 0.045 | 0.010 | 1.00 |
| NB-VAE | 0.032 | 0.008 | 1.00 |

The marginal picture is unflattering for everyone. Coverage sits at $1.00$ against a nominal $0.80$, meaning the sampled populations are over-dispersed on the top-DE genes, a property of the negative-binomial decoders rather than of the flow. The two flow variants look nearly identical on the marginal metrics precisely because they share a decoder, which tells us these metrics are reading the decoder, not the latent distribution. The one metric that can see the flow's would-be advantage, the joint energy distance, again ranks the VAE best at $0.038$ for the transport flow against $0.032$ for the VAE, with the transport-minus-VAE difference not significant. The transport flow beats the Gaussian flow here too, consistent with the effect-size axis.

So both axes we can measure agree. The transport formulation is the right way to build the flow, optimal-transport coupling is the wrong way, and the from-scratch NB-VAE is equal or slightly better than the flow across the board.

## The honest scoreboard

On Norman held-out combinations, the JEPA-plus-conditional-flow stack does not beat a from-scratch conditional NB-VAE on either effect size or calibration. It ties or slightly trails on both. The compositional gene-set embedding, shared by both models, is what drives combination generalization. Within the flow family, transporting from a real control latent beats transporting from noise, and optimal-transport coupling makes things worse.

This is a negative result in the narrow sense that the flagship method did not clear its baseline. It is a productive one in the broader sense that it was measured carefully enough to be trusted, it isolated which component carries the generalization, and it surfaced a formulation fix worth keeping. Two caveats bound it. The evaluation rests on a single dataset and a twenty-combination held-out set, which limits statistical power to effects of roughly $0.05$. And there is one axis we have not yet measured, data efficiency, where the flow's inductive bias could still pay off. [Chapter 5](05-challenges-and-limitations.md) dissects the challenges behind these numbers, and [Chapter 6](06-beyond-the-current-limit.md) lays out where a breakthrough would most plausibly come from.

---

*Previous: [Chapter 3 — Training and evaluation](03-training-and-evaluation.md). Up: [the method series](index.md). Next: [Chapter 5 — Challenges and limitations](05-challenges-and-limitations.md).*
