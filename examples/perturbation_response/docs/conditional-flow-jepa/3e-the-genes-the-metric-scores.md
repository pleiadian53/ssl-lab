# 3e — The genes the metric scores

*Companion to [Chapter 3](03-training-and-evaluation.md). Every number in this series is computed on a short list of "top differentially-expressed genes" per perturbation. That list is not a detail of the plumbing. It is a design decision that decides what the benchmark can and cannot see, and the criterion used to build it can quietly make the whole evaluation unreadable. This note explains how the list is chosen, why the obvious criterion is the wrong one, and what guard keeps the choice honest.*

---

## 1. Why the metric scores a short list at all

A Perturb-seq readout has thousands of genes, and a perturbation moves only a handful of them. If the effect-size metric correlated predicted against true $\Delta$ across all $5000$ highly-variable genes, the correlation would be dominated by the thousands of genes the intervention never touched, where both the predicted and the true change are noise around zero. The score would mostly measure how well a model reproduces the *baseline*, which is the one thing every model already does well ([Chapter 5](05-challenges-and-limitations.md) calls this baseline dominance).

So both axes restrict to the perturbation's **top-$k$ differentially expressed genes**, $k = 20$ by default. Effect size correlates $\Delta$ over those genes; calibration compares the predicted and real populations on those genes. The list is per-perturbation, computed once in Stage 0 and cached in `de_genes.json`, and it is the *scoring seam* of the whole project.

That makes the selection criterion load-bearing. Choose it badly and every downstream number is measuring the wrong thing, while still looking perfectly well-formed.

## 2. The criterion that looks obvious, and why it fails

The intuitive choice is **fold change**: rank the genes by how much the perturbation multiplied their expression, and take the top $20$ by $|\log_2 \mathrm{FC}|$. It is the number most people mean by "differentially expressed," and scanpy hands it to you in `rank_genes_groups`.

It is a trap, and the reason is arithmetic. A fold change is a *ratio*, and a ratio is unstable when its denominator is near zero. Scanpy computes

$$\log_2 \mathrm{FC}_g = \log_2 \frac{\overline{x}_g^{\text{pert}} + \varepsilon}{\overline{x}_g^{\text{ctrl}} + \varepsilon}, \qquad \varepsilon = 10^{-9},$$

where $\overline{x}_g$ is the mean expression of gene $g$ in each group and $\varepsilon$ is a pseudocount that only exists to avoid dividing by zero. Now take a gene that is essentially silent in this cell type, expressed in almost no cells at all. Its control mean might be $10^{-7}$ and its perturbed mean $10^{-5}$, purely because a couple of stray transcripts landed in a couple of cells. That is a hundred-fold change, and it will outrank a real responder that genuinely doubled from $2.0$ to $4.0$.

The result is that ranking by $|\log_2 \mathrm{FC}|$ does not select the genes the perturbation moved. It selects **the genes that are barely expressed at all**, because those are the ones whose ratios can explode. On Norman 2019 the effect is not subtle. The twenty genes it picks per held-out combination have:

| ranked by | mean expression | $\lvert\Delta\rvert$ | detected in | significant ($p_{\text{adj}} < 0.05$) |
|---|---|---|---|---|
| $\lvert\log_2 \mathrm{FC}\rvert$ | $0.0006$ | $0.02$ | $0\%$ of cells | $0\%$ |
| $\lvert z\rvert$ (Wilcoxon) | $2.40$ | $0.75$ | $83\%$ of cells | $100\%$ |

Read the top row carefully, because it is worse than "suboptimal." The selected genes are detected in essentially none of the cells, not one of them is statistically significant, and in the held-out population **every single cell is exactly zero on every one of them**.

## 3. What a silent gene list does to the scoreboard

A degenerate gene list does not announce itself. Every script still runs, every report still writes, and every metric still returns a plausible-looking number in $[0, 1]$. What it does instead is quietly destroy the meaning of both axes.

**Coverage is forced to $1.00$.** Interval coverage asks what fraction of real cells fall inside the model's central $80\%$ predicted interval ([Chapter 3b](3b-reading-the-calibration-metrics.md)). If every real cell is exactly zero on a gene, then *any* predicted interval that contains zero contains all of them. Coverage comes back at $1.00$ for every model, at every setting, no matter what the decoder does. It looks exactly like a decoder that is over-dispersed, and it will not budge when you fix the dispersion, because the dispersion was never what produced it.

**Effect size measures the wrong thing.** With the perturbed mean pinned at zero, the true effect reduces to $\Delta_g = 0 - \overline{x}_g^{\text{ctrl}} = -\overline{x}_g^{\text{ctrl}}$, which is the same fixed vector for every perturbation. Any model that learns to emit near-zero on silent genes reproduces it. The $\Delta$-correlation stays respectable and stops discriminating between models.

**Spread correlation goes negative.** Asked which genes vary most, the model ranks by its own decoded rates, while the truth is a set of all-zero columns whose ordering is pure noise. A negative correlation is the natural result, and it reads as "the model has the variance ranking backwards" when the real answer is that there is no ranking to get right.

The general lesson is worth stating on its own, because it outlives this dataset. **A metric computed on the wrong support does not fail loudly. It returns a number, and the number is stable, and it is meaningless.** Every stage of a pipeline should assert something about its own output, or a defect like this travels all the way to the results chapter wearing a suit.

## 4. The criterion that works: the Wilcoxon $z$

The fix is to rank by a **test statistic** rather than by an effect ratio. Scanpy's `rank_genes_groups(method="wilcoxon")` already computes one, the `scores` field, which is the standardized Mann-Whitney (Wilcoxon rank-sum) $z$-statistic for the perturbed group against control:

$$z_g = \frac{U_g - \mathbb{E}[U_g]}{\sqrt{\mathrm{Var}[U_g]}},$$

where $U_g$ is the rank-sum of gene $g$'s expression in the perturbed cells relative to the control cells, and the expectation and variance are those of the null (no difference). Ranking by $|z_g|$ is in fact scanpy's *default* ordering, and it has exactly the property fold change lacks.

**A rank statistic cannot blow up on a silent gene.** $U_g$ is built from the ordering of cells, not from the magnitude of a ratio. If a gene is zero in almost every cell of both groups, then almost every cell is tied, the rank-sum sits at its null expectation, and $z_g \approx 0$. There is no pseudocount to divide by and no way for two stray transcripts to manufacture a large statistic. The criterion is **self-guarding**: the pathology of section 2 is not merely reduced, it is structurally unreachable.

It also selects on the right thing. A large $|z|$ requires the gene to be *consistently* shifted across many cells, which is what "this perturbation moved this gene" actually means, and which is what a model can be fairly asked to predict.

## 5. The guards, and why they stay even though they do nothing

Stage 0 applies two filters on top of the $|z|$ ranking. A gene is eligible only if it is

- **detected** in at least $10\%$ of the perturbation's cells (a nonzero raw count), and
- **significant** at $p_{\text{adj}} < 0.05$.

On Norman these are very nearly no-ops. Ranking by $|z|$ already produces genes that are $100\%$ significant and detected in $83\%$ of cells, so the guards reject almost nothing. Keeping them is deliberate anyway, for two reasons.

The first is that they turn an implicit property into an **enforced invariant**. The ranking criterion is the kind of thing a future change touches casually, and the guards mean such a change cannot silently reintroduce silent genes: the genes would fail the filters and the selection would come back short rather than come back wrong.

The second is that Stage 0 asserts on the result and **refuses to write a cache that violates it**. If the median detection rate across all selected genes falls below $5\%$, or fewer than half of them are significant, the pipeline raises instead of caching. This is the acceptance gate described in [Running the pipeline §3](../running-the-pipeline.md), and it is the cheapest insurance in the project, because a scoring seam that goes wrong is invisible from every point downstream of it.

One consequence to expect and accept: biologically weak perturbations genuinely have fewer than $20$ genes that pass. About $11$ of $236$ perturbations in Norman do, some with as few as three. The honest response is to score them on the genes they have rather than pad the list back to twenty with genes that failed the guard, which is precisely how the padding would smuggle silent genes back in.

## 6. What this does not fix

Choosing the genes well makes the metric *readable*. It does not make it *complete*, and two limits carry forward.

The list is computed from the perturbation's own cells, including the held-out ones. This is the field convention (the DE genes define *where* to score, not what the answer is) and it is unavoidable for a held-out combination, which by construction has no training cells of its own. But it does mean the benchmark tells a model which genes it will be graded on, and a method that exploited that would be flattered.

And a top-$20$ list is still a marginal, per-gene view of a response that is jointly distributed across genes. It is the right support for the mean, and a reasonable one for per-gene spread, but the joint structure a flow is built to capture lives across genes rather than within any one of them. That is why the joint energy distance of [Chapter 3b](3b-reading-the-calibration-metrics.md) exists alongside the per-gene reads.

---

*Up: [the method series](index.md). Parent: [Chapter 3 — Training and evaluation](03-training-and-evaluation.md). Related: [Chapter 3b — Reading the calibration metrics](3b-reading-the-calibration-metrics.md), [Running the pipeline](../running-the-pipeline.md).*
