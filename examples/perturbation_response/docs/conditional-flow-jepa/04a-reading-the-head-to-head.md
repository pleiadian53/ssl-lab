# Chapter 4a — Reading the head-to-head: the statistics behind the scoreboard

*A companion to [Chapter 4](04-results.md). That chapter tells the story of the head-to-head and states the verdict. This one is the audit behind it: what each contrast is actually asking, why the unit of analysis is the perturbation and not the cell, how a joint bootstrap with simultaneous intervals turns four columns of numbers into three defensible claims, what the procedure assumes that is not true, and a map from every number back to the run that produced it.*

> **Where this sits.** Read [Chapter 4](04-results.md) first for the narrative and the conclusion. This note is reference material for anyone who would rather check that conclusion than take it on trust. [Chapter 3a](3a-the-models-in-the-head-to-head.md) introduces the models and the statistical background in the abstract; [Chapter 3b](3b-reading-the-calibration-metrics.md) explains the four calibration metrics; [Chapter 3e](3e-the-genes-the-metric-scores.md) explains which genes enter the score. Everything below is produced by [`12_compare_arms.py`](../../12_compare_arms.py) from reports the pipeline already wrote, so it is re-runnable end to end.

The metric throughout is the $\Delta$-correlation defined in [Chapter 3](03-training-and-evaluation.md). For one perturbation, generate a predicted response population, form its differential expression $\Delta = \mathrm{mean}(\text{predicted}) - \mathrm{mean}(\text{control})$, and take the Pearson correlation $r$ between the predicted $\Delta$ and the true $\Delta$ over that perturbation's top 20 differentially-expressed genes. Higher is better, and the theoretical maximum is $1$. Every number below is computed on the same twenty held-out two-gene combinations of the `combo` split, with 200 generated cells per perturbation, guidance $1.0$, and the compositional gene-set condition encoder, which is the only condition encoder that can represent a combination it has never seen.

## 1. Three contrasts, and only two of them are ablations

Four arms enter the comparison. Three of them are flow variants that differ from each other by exactly one lever. The fourth is a different model entirely.

The three flow arms share what we will call the **frozen-encoder invariant**: the same frozen JEPA cell encoder, the same gene-set condition encoder, the same negative-binomial count decoder, the same data, the same split, the same three training seeds, and the same sampling settings at evaluation time. Nothing moves between them except the one thing under test. The **Gaussian flow** transports samples from standard Gaussian noise to the perturbed outcome latent. The **transport flow** transports from a real control-cell latent to the perturbed outcome latent, which is the same machinery with a different base distribution. The **OT-coupled** arm is the transport flow with one further change, an optimal-transport coupling used to pair control latents with outcome latents when building training pairs, rather than pairing them at random.

The fourth arm, the **conditional NB-VAE**, shares only the condition encoder and the data. It has no JEPA encoder (it learns its own representation from scratch), no flow (it uses a Gaussian latent prior and an amortized encoder), and a different training objective. It is a from-scratch generator built for the same job.

That difference in kind is what decides how each contrast may be read.

| contrast | lever varied | everything else | type | the question it answers |
|---|---|---|---|---|
| transport $-$ Gaussian | the flow's base distribution (noise or a real control latent) | held fixed by the frozen-encoder invariant | **ablation** | what does this one lever contribute? |
| OT $-$ transport | the coupling that builds training pairs (optimal transport or random) | held fixed by the frozen-encoder invariant | **ablation** | what does this one lever contribute? |
| transport $-$ NB-VAE | encoder, prior, objective, and generative machinery, all at once | only the condition encoder and the data are shared | **control** | does the machinery beat a simple generator? |

An ablation isolates a component, so its result attaches to that component. A control does not isolate anything, so its result attaches to the method as a whole. The $-0.118$ that Chapter 4 reports is a statement about the *stack*: a frozen self-supervised encoder plus a conditional flow plus a count decoder, taken together, scores below a from-scratch conditional VAE. It is not a statement that "the flow is the problem" or that "JEPA is the problem," because the contrast moved both at once along with everything else. Reading a control as though it were an ablation is the most common way to turn an honest whole-method result into a false claim about one component. The lever-level claims in this chapter come from the ablations, and only from them.

## 2. The scoreboard, at full precision

The primary endpoint first. Each arm's score is the mean over the twenty held-out combinations of its seed-averaged per-combination $\Delta$-correlation.

| arm | JEPA? | flow? | mean $\Delta$-correlation (3 seeds) |
|---|---|---|---|
| Gaussian flow (noise to outcome) | yes | yes | 0.612 |
| transport flow with OT coupling | yes | yes | 0.627 |
| transport flow (control to outcome) | yes | yes | 0.648 |
| **conditional NB-VAE (baseline)** | no | no | **0.766** |

The four calibration metrics second, on the same twenty combinations and the same top-DE genes, with populations sampled as counts from the negative binomial rather than read off as expected rates. These are **secondary endpoints**, for reasons given in §4, and they carry no significance verdict anywhere in this chapter.

| arm | spread correlation $\uparrow$ | coverage (nominal 0.80) | 1-Wasserstein $\downarrow$ | joint energy $\downarrow$ |
|---|---|---|---|---|
| Gaussian flow | 0.205 | 0.357 | 1.013 | 3.794 |
| transport flow | 0.234 | 0.375 | 0.982 | **3.578** |
| transport + OT | 0.214 | 0.365 | 1.010 | 3.746 |
| NB-VAE | **0.522** | 0.328 | **0.956** | 3.962 |

Two orderings are visible before any statistics are run. Inside the flow family, transport is ahead of OT-coupled transport, which is ahead of the Gaussian flow, and that ordering is the same on the primary metric and on all four calibration metrics. Across the family boundary, the NB-VAE leads on the primary metric by a wide margin and on two of the four calibration metrics, while posting the worst joint energy distance of the four. Whether any of those gaps survives resampling is what the rest of this chapter is for.

For orientation only, the in-distribution reference number is a mean $\Delta$-correlation of $0.612$ across $216$ perturbations, measured on held-out *cells* of *seen* perturbations with the table condition encoder. It is not comparable to the numbers above and no contrast is formed against it, because it uses a different split, a different condition encoder, and a different set of perturbations.

## 3. The unit of analysis is the perturbation, and there are twenty of them

Every statistical choice below follows from one fact, so it is worth stating slowly.

The metric is defined **per perturbation**. To score one held-out combination, we generate a population of 200 cells, average them, subtract the control mean, and correlate the resulting 20-gene vector against the truth. Everything inside that computation is an *input* to a single number. The 200 cells are not 200 samples; they are the ingredients of one score, and generating 2000 instead would sharpen that score slightly without adding a single observation. The 20 genes are not 20 samples; they are the coordinates of one correlation. A training seed is not a sample either; it is a re-draw of the same model on the same data, not a new item to test on.

What remains is the perturbation. An arm yields a vector of $n = 20$ numbers, one per held-out combination, and that vector is the entire sample. The twenty are:

`AHR+KLF1`, `BCL2L11+TGFBR2`, `CBFA2T3+POU3F2`, `CBL+CNN1`, `CBL+TGFBR2`, `CEBPE+FOSB`, `CEBPE+RUNX1T1`, `DLX2+ZBTB10`, `FOSB+UBASH3B`, `FOXA1+FOXF1`, `FOXL2+HOXB9`, `IGDCC3+ZBTB25`, `KLF1+MAP2K6`, `OSR2+UBASH3B`, `PLK4+STIL`, `PTPN12+ZBTB10`, `SAMD1+ZBTB1`, `SNAI1+ZBTB10`, `TBX2+TBX3`, `UBASH3A+UBASH3B`.

Twenty is a small sample, and no amount of statistical machinery will make it a large one. The machinery's job is to keep us honest about that, not to hide it.

Seeds are handled by **averaging before the test**, not by pooling into it. Write $x_{a,s,i}$ for the score of arm $a$, trained with seed $s$, on perturbation $i$. Each arm is retrained at $S = 3$ seeds, and the vector that enters every test below is the per-perturbation seed average

$$\bar{x}_{a,i} = \frac{1}{S} \sum_{s=1}^{S} x_{a,s,i}, \qquad i = 1, \dots, n.$$

This is the right thing to do and it has a real cost, which we state rather than bury. Averaging first means a reported difference cannot be an artifact of one lucky initialization. It also means the intervals below describe **test-set uncertainty only**. They answer "if we had held out a different twenty combinations, would the conclusion hold?" They do not answer "if we had trained with different seeds, would the conclusion hold?" With three seeds there is not enough information to estimate a seed-variance component well enough to propagate it through the bootstrap, so it is not propagated, and every interval in this chapter is conditional on the seed-averaged model. §9 gives the direct evidence on how much seed noise there actually is, which is the honest substitute for propagating it.

## 4. One primary endpoint, declared before the numbers

Four arms and five metrics is a lot of surface area to go looking for a result on. With four contrasts of interest across five metrics, roughly twenty tests are available, and testing each at $\alpha = 0.05$ means that under the global null, where nothing differs from anything, we should *expect* about one significant finding purely by chance. A procedure that lets us pick our favorite from twenty is not a test. It is a search.

So the family is fixed in advance. The $\Delta$-correlation is the **single primary endpoint**, and its three contrasts (the two ablations and the one control of §1) are the entire family on which a significance claim is made. Everything else, meaning all four calibration metrics on all three contrasts, is **secondary and exploratory**. Secondary endpoints are reported with intervals and never with a verdict, even when an interval comfortably excludes zero, because a difference noticed on a secondary endpoint after seeing the data is a hypothesis and not a result. It earns the name "result" by being confirmed on data that did not suggest it, or not at all.

This is a commitment that costs something. The transport flow posts the best joint energy distance of any arm, which is exactly the metric a flow ought to win, and §10 declines to call it a win. That is the point of declaring the endpoint before looking.

## 5. The bootstrap is paired, because combinations differ in difficulty

For two arms $A$ and $B$, let $A_i$ and $B_i$ be their seed-averaged scores on perturbation $i$. The statistic is the mean of the **per-combination difference**

$$d_i = A_i - B_i, \qquad \bar{d} = \frac{1}{n} \sum_{i=1}^{n} d_i .$$

To get its sampling distribution we resample the *perturbations*, not the cells and not the genes. Draw an index vector $I^{(b)} = (i_1^{(b)}, \dots, i_n^{(b)})$ uniformly with replacement from $\{1, \dots, n\}$, and for each of $B = 10{,}000$ draws recompute the mean difference on the resampled combinations. The spread of those ten thousand means is the uncertainty in $\bar{d}$.

The pairing is not a convenience, it is the whole reason the test has any power at $n = 20$. Combinations differ enormously in intrinsic difficulty. Some move many genes by large amounts and score high for *every* arm; others are biologically weak, so the true $\Delta$ is small and noisy and every arm scores low. That between-combination variance is far larger than the differences between arms, and in an unpaired comparison of two column means it would swamp the effect completely. Taking the difference combination by combination cancels it exactly, because the shared difficulty appears in $A_i$ and $B_i$ alike and subtracts out. The evidence that it works is the standard error: the transport-minus-Gaussian difference is $+0.036$ with a paired standard error of $0.0055$, so the effect is more than six standard errors from zero even though both arms range widely across the twenty combinations.

## 6. The bootstrap is joint, because the contrasts share arms

The three primary contrasts are not independent tests. The transport arm appears in all three of them, and every contrast is computed on the *same* twenty perturbations. If a resample happens to draw an unusually easy set of combinations, that lands in all three contrasts at once. They move together.

Bootstrapping each contrast in its own separate run would throw that dependence away, and so would correcting for multiplicity with Bonferroni, which is derived without any use of the dependence structure and is therefore needlessly conservative when the contrasts are strongly positively dependent. The alternative is simpler than either and more faithful than both: resample the perturbation indices **once per bootstrap iteration**, and evaluate *every* contrast on that one shared resample. In code this is a single index array of shape $(B, n)$, reused for every contrast and every metric. The dependence structure is preserved by construction rather than modeled, which means we never have to assume anything about it.

## 7. The max-$t$ critical value, and what the correction costs

Preserving the dependence is what makes a **simultaneous** interval possible. On each bootstrap iteration $b$, and for each contrast $c$ in the primary family $\mathcal{C}$, compute the resampled mean difference $\bar{d}_c^{(b)}$, its resampled standard error $s_c^{(b)}$, and the studentized statistic centered on the observed value

$$t_c^{(b)} = \frac{\bar{d}_c^{(b)} - \bar{d}_c}{s_c^{(b)}} .$$

Studentizing puts contrasts with very different scales onto a common footing, which matters here because the two ablations have standard errors near $0.006$ while the control's is $0.037$. Now take the **largest** absolute studentized statistic across the family on that iteration,

$$M^{(b)} = \max_{c \in \mathcal{C}} \left| t_c^{(b)} \right| ,$$

and let $q_{0.95}$ be the 95th percentile of $M^{(1)}, \dots, M^{(B)}$. That single number is the critical value for the whole family. The simultaneous interval for contrast $c$ is

$$\bar{d}_c \pm q_{0.95} \cdot \hat{s}_c ,$$

where $\hat{s}_c$ is the observed standard error of $\bar{d}_c$. Because $q_{0.95}$ controls the *maximum* deviation across the family, the resulting intervals hold **simultaneously** at 95%: the probability that even one of the three misses its target is at most 5%, not 5% each.

On this data $q_{0.95} = 2.951$, against $1.96$ for a single unadjusted test. The bar is roughly 50% higher, and every interval is 50% wider than it would be if we tested each contrast on its own and pretended the other two did not exist. Nothing in this chapter is claimed at the unadjusted bar.

## 8. The primary result

Three contrasts, one family, one critical value. The unadjusted interval is shown alongside the simultaneous one so that the cost of the correction is visible rather than assumed.

| contrast | difference | std. error | unadjusted 95% CI | **simultaneous 95% CI** | verdict |
|---|---|---|---|---|---|
| transport $-$ Gaussian | $+0.036$ | 0.0055 | $[+0.026, +0.047]$ | $\mathbf{[+0.019, +0.052]}$ | **significant** |
| OT $-$ transport | $-0.021$ | 0.0063 | $[-0.034, -0.009]$ | $\mathbf{[-0.039, -0.002]}$ | **significant** |
| **transport $-$ NB-VAE** | $\mathbf{-0.118}$ | 0.0372 | $[-0.189, -0.049]$ | $\mathbf{[-0.228, -0.008]}$ | **significant** |

All three exclude zero at the simultaneous bar, so all three are claims we are willing to make, and they are the only claims this chapter makes.

The two ablations attach to their levers. Transporting from a real control latent beats transporting from Gaussian noise by $+0.036$, so the transport reformulation is a real improvement and it stays. Optimal-transport coupling costs $-0.021$, so it *hurts*, and it goes, even though it lowers the training loss. Both are lever-level findings because both come from contrasts in which exactly one lever moved.

The control attaches to the whole method. The from-scratch NB-VAE beats the best flow arm by $0.118$, and the simultaneous interval $[-0.228, -0.008]$ stays clear of zero. Two things are worth saying about that interval rather than only about its verdict. Its width, roughly $\pm 0.110$, is more than three times the width of either ablation's, because the two models disagree in very different ways on different combinations and the pairing cancels less of the variance. And its upper edge, $-0.008$, sits close to zero. The finding survives the strictest correction we know how to apply, and it survives without much room to spare. That is what twenty perturbations buys, and it is exactly why the seed evidence in the next section matters.

## 9. What averaging the seeds hides, and what it does not

The intervals above are conditional on the seed-averaged model, so they say nothing about how much a different set of training seeds would have moved things. The direct measurement does.

The NB-VAE's three seeds score $0.762$, $0.767$, and $0.768$, a spread of $0.006$. The Gaussian flow's three seeds span $0.606$ to $0.618$. Configurations do not trade places from one initialization to the next, and no arm's seed-to-seed movement is remotely the size of the gap to the baseline: the $0.118$ that separates the transport flow from the VAE is around twenty times the VAE's entire seed spread. Reseeding is not going to close it.

This does not repair the limitation, and we are not claiming that it does. Seed uncertainty is genuinely absent from every interval in §8, and a properly propagated analysis with many more seeds would produce wider intervals than the ones reported. What the seed measurements establish is the *direction* of that missing uncertainty relative to the finding: the arms are stable, so the omitted component is small, and the omitted component is nowhere near large enough to turn a $0.118$ deficit against a baseline with a $0.006$ spread into a tie.

## 10. The secondary endpoints, reported without verdicts

Here are all three contrasts on all four calibration metrics, with plain bootstrap intervals from the same shared resample, carrying no multiplicity adjustment. No cell of this table is a result. Every cell is a description.

| metric | transport $-$ Gaussian | OT $-$ transport | transport $-$ NB-VAE |
|---|---|---|---|
| spread correlation $\uparrow$ | $+0.029$ $[+0.007, +0.054]$ | $-0.021$ $[-0.040, -0.002]$ | $-0.288$ $[-0.429, -0.136]$ |
| coverage (nominal 0.80) | $+0.018$ $[+0.006, +0.033]$ | $-0.010$ $[-0.020, -0.001]$ | $+0.046$ $[+0.016, +0.082]$ |
| 1-Wasserstein $\downarrow$ | $-0.031$ $[-0.061, -0.009]$ | $+0.027$ $[+0.008, +0.049]$ | $+0.026$ $[-0.054, +0.094]$ |
| joint energy $\downarrow$ | $-0.215$ $[-0.378, -0.082]$ | $+0.168$ $[+0.049, +0.298]$ | $-0.383$ $[-0.960, +0.172]$ |

Three things are worth reading off it, in descending order of how much weight they can bear.

**Every arm is badly under-dispersed, and that is the most robust fact on this axis.** Coverage sits between $0.33$ and $0.38$ against a nominal $0.80$, meaning each model's predicted 80% interval captures barely a third of the real cells. This holds for the flows and the VAE alike, it is far too large to be a resampling artifact, and it is shared across models that differ in almost everything except that they both read out through a count decoder. It is therefore a property of the readout, not of the latent distribution, which is the finding that sets the direction of the decoder work in [Chapter 8](08-modeling-the-readout-count-decoder.md): the decoder needs *more* dispersion, not less.

**The VAE tracks per-gene variability much better than any flow arm.** Its spread correlation of $0.522$ against the transport flow's $0.234$ is a large difference, and the interval $[-0.429, -0.136]$ is nowhere near zero. It says the VAE is substantially better at knowing *which* genes vary within a response. This is a description we take seriously and still do not call a result, because it is a secondary endpoint and it carries no multiplicity correction.

**The flow's apparent advantage on joint structure does not resolve.** The transport flow posts the best joint energy distance of the four arms, $3.578$ against the VAE's $3.962$, and the energy distance is precisely the multivariate metric built to see the gene-gene correlation structure that a rich latent distribution should carry and a marginal metric cannot. This is the one place the flow looks like it is earning its keep. But the contrast is $-0.383$ with an interval of $[-0.960, +0.172]$ that **crosses zero**: on twenty perturbations, this difference is not resolvable. It is a secondary endpoint besides, so even a clean interval would not have made it a claim. It is a hypothesis worth testing on a larger held-out set, and today it is nothing more than that.

## 11. What the bootstrap assumes, and what it cannot fix

The procedure buys real protection and it is not magic. Three things about it should be understood by anyone quoting the intervals.

**It does not assume normality, and that matters here.** The bootstrap builds its sampling distribution from the data by resampling, so it never requires the per-combination differences to be Gaussian. This is worth having, because a $\Delta$-correlation is a Pearson $r$, bounded in $[-1, 1]$ and skewed near its upper end, and a difference of two such quantities across twenty heterogeneous combinations has no reason to be normal. A $t$-test would have imposed a shape the data do not have. The bootstrap does not.

**It does assume the twenty perturbations are independent draws, and they are not.** Resampling combinations with replacement treats them as i.i.d. samples from a population of perturbations. The twenty share genes. `CBL+CNN1` and `CBL+TGFBR2` both contain CBL. `ZBTB10` appears in three of them (`DLX2+ZBTB10`, `PTPN12+ZBTB10`, `SNAI1+ZBTB10`), and `UBASH3B` appears in three more (`FOSB+UBASH3B`, `OSR2+UBASH3B`, `UBASH3A+UBASH3B`). Combinations that share a gene share biology, so their scores are positively correlated, and positively correlated items carry less information than independent ones. The effective sample size is therefore smaller than twenty, and the intervals reported here are, if anything, **too narrow**. This pushes in the same direction as the omitted seed variance of §9, and we would rather state both than have a reader discover them.

**It cannot fix $n = 20$.** A bootstrap re-uses the sample it is given; it does not create information. Ten thousand resamples of twenty numbers is still twenty numbers, and the widest interval in §8 (roughly $\pm 0.110$ on the control contrast) is an honest reading of what twenty heterogeneous combinations can resolve, not a defect of the method. Everything here also rests on a single dataset and a single split. The way to tighten any of this is more held-out perturbations, more datasets, and more seeds, in that order. It is not a cleverer test.

## 12. Provenance: every number, and how to reproduce it

Twelve training runs feed this chapter, four arms at three seeds each. Every evaluation writes a JSON report under `output/<experiment>/reports/`, and the comparison script reads those reports rather than recomputing anything, so the audit trail is a file path.

| number | experiments | script | report |
|---|---|---|---|
| $\Delta$-correlation, flow arms (§2) | `norman_sweep_{gaussian,control,control_ot}_s{0,1,2}` | `06_eval_effect_size.py` | `reports/effect_size.json` |
| calibration, flow arms (§2) | `norman_sweep_{gaussian,control,control_ot}_s{0,1,2}` | `10_eval_calibration.py --model flow` | `reports/calibration_flow.json` |
| $\Delta$-correlation, NB-VAE (§2) | `norman_vae_s{0,1,2}` | `09_eval_cvae_baseline.py` | `reports/effect_size_cvae.json` |
| calibration, NB-VAE (§2) | `norman_vae_s{0,1,2}` | `10_eval_calibration.py --model vae` | `reports/calibration_vae.json` |
| all contrasts, both CIs, and the max-$t$ critical value (§8, §10) | all twelve of the above | `12_compare_arms.py` | `reports/arm_comparison.json` |
| in-distribution reference, $0.612$ over $216$ perturbations (§2) | `norman_stage_a` (`cells` split, table condition) | `06_eval_effect_size.py` | `reports/effect_size.json` |

The comparison itself is one command, and it is the command that produced every interval above:

```bash
python examples/perturbation_response/12_compare_arms.py \
    --arm "gaussian=norman_sweep_gaussian_s{s}:flow:0,1,2" \
    --arm "transport=norman_sweep_control_s{s}:flow:0,1,2" \
    --arm "OT=norman_sweep_control_ot_s{s}:flow:0,1,2" \
    --arm "NBVAE=norman_vae_s{s}:vae:0,1,2" \
    --contrast transport-gaussian \
    --contrast OT-transport \
    --contrast transport-NBVAE \
    --primary delta_r --n-boot 10000
```

Two details of that script are load-bearing and easy to miss. A perturbation is admitted to the analysis only if *every* requested seed produced a score for it, and the tested set is the intersection across all arms and all reported metrics, which is how the same twenty combinations end up backing every number in the chapter. And the resampled index array is drawn once and reused across every contrast and every metric, which is what makes the intervals in §8 simultaneous and the ones in §10 mutually comparable.

## 13. The verdict in one line

On the one pre-committed endpoint, tested at a bar that holds simultaneously across the whole family of contrasts, the two ablations say that transporting from a real control latent helps and that optimal-transport coupling hurts, and the control says that the JEPA-plus-conditional-flow stack loses to a from-scratch conditional NB-VAE by $0.118$ in $\Delta$-correlation. The calibration axis is reported without verdicts and does not rescue the flow, and its one robust reading, that every model is far too narrow, is a statement about the count decoder rather than about the flow. The intervals omit seed uncertainty and treat gene-sharing combinations as independent, so they are if anything too narrow, and both omissions push against the flow rather than for it. [Chapter 5](05-challenges-and-limitations.md) asks why the method lost.

---

*Previous: [Chapter 4 — Results](04-results.md). Up: [the method series](index.md). Next: [Chapter 5 — Challenges and limitations](05-challenges-and-limitations.md). Background on the models and the small-sample reasoning: [Chapter 3a](3a-the-models-in-the-head-to-head.md). The calibration metrics: [Chapter 3b](3b-reading-the-calibration-metrics.md). The gene selection: [Chapter 3e](3e-the-genes-the-metric-scores.md). Current state of play across all rounds: [the results ledger](results-ledger.md).*
