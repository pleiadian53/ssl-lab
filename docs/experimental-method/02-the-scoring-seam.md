# Chapter 2 — The scoring seam

*This is the most important chapter in the series. It is about a piece of code that almost nobody reads, that sits upstream of every metric you compute, that looks like data preparation, and that can silently decide what your entire benchmark is capable of seeing.*

---

## 1. The seam

Every metric is computed on a **support**: the subset of items, features, timesteps, or classes that the score is actually evaluated over. Almost no metric is computed on everything. There is always a selection, and the selection is made somewhere, by something.

Call that selection the **scoring seam**.

The seam is where your data ends and your measurement begins, and it has three properties that together make it the most dangerous surface in an empirical pipeline.

It is **upstream of the metric**, so every check you run downstream (the loss curves, the unit tests on the metric function, the sanity plots, the ablations, the significance tests) runs *after* the damage and cannot see it. Those checks all verify that the metric was computed correctly. None of them verify that it was computed on the right thing.

It **looks like plumbing**. The seam is usually implemented in a preprocessing step, a caching stage, a filter, a call to a library helper with sensible-looking defaults. It carries no modeling ideas, so it attracts no scrutiny in review. When a project audits itself, it audits the model and the metric. The seam is in neither.

And it is **chosen by a criterion**, which means somebody wrote down a rule for which items are in and which are out. That rule is a modeling assumption wearing the clothes of a utility function.

## 2. The general law

Here is the property that makes this worth an entire chapter, stated as generally as it can be stated:

> **A metric computed on the wrong support does not fail loudly. It returns a well-formed number. The number is in the right range, it is stable across runs and across seeds, it responds plausibly to changes in the model, and it is meaningless.**

Compare this to how other bugs behave. A shape mismatch throws. A NaN propagates and is visible in the first plot. A data leak usually produces a score that is *too good*, which trips somebody's intuition. A wrong-support bug does none of that. It produces exactly the kind of number you were expecting, which is why it can travel through an entire project, into a results table, past every reviewer, and into a conclusion.

Everything else in this series is about extracting a trustworthy signal from noisy measurements. This chapter is about the case where there is no signal in the measurement at all, and the measurement declines to mention it.

What follows is a real instance, told from the evidence forward.

## 3. The clue: a number that would not move

The generative model in our project reads out through a **decoder**, the component that turns a latent vector into a predicted population of cells with realistic counts. A decoder is judged largely on whether the *spread* of what it emits matches the spread of reality, and the standard instrument for that is **interval coverage**: form the model's central 80 percent predicted interval for each gene, and ask what fraction of the real cells fall inside it. If the model is well calibrated, coverage lands near $0.80$. If the model is too confident, coverage lands well below. If the model is too diffuse, coverage lands above.

Coverage came back at exactly $1.00$.

Not $0.97$. Not $0.99$ with a little wobble across arms. Exactly $1.00$, for every model, at every hyperparameter setting, on every seed.

The reading of that is unambiguous, and it is the reading we took: the decoder is wildly over-dispersed. Its predicted intervals are so wide that they swallow all of reality. So the decoder was rebuilt. The dispersion parameterization was changed. The count distribution was changed. Regularization was added, then removed. Each time, the pipeline ran, the metrics wrote, and coverage came back at exactly $1.00$.

That is the clue, and it is a good one, because it is the wrong *kind* of stability. Real metrics move. A quantity that is genuinely responding to a decoder should respond to a change in the decoder, if only by a little, if only in the third decimal place. A number that is bit-for-bit identical across models that share almost nothing is not measuring those models. It is measuring something they have in common, and the only thing they had in common was the data they were being scored against.

So the question stops being "what is wrong with the decoder" and becomes "what is wrong with the genes."

## 4. The seam, and the criterion that built it

The metric does not score all five thousand genes. A perturbation moves only a handful of them, and scoring all five thousand would drown the signal in the thousands of genes the intervention never touched. So both axes of the evaluation restrict to a per-perturbation list: **the top twenty differentially expressed genes** for that perturbation, computed once in a preprocessing stage and cached to a JSON file.

That JSON file is the scoring seam of the entire project. Every number in every results table is computed on it.

The genes in it were selected by ranking on **fold change**, which is the amount by which the perturbation multiplied a gene's expression. It is the number most people mean by "differentially expressed." It is intuitive, it is what the field reports, and it is what the standard tooling hands you: a call to scanpy's `rank_genes_groups` returns a `logfoldchanges` field, and taking the top twenty by absolute value is a two-line operation that looks like the obvious thing to do.

It is a trap, and the reason is arithmetic.

## 5. The arithmetic

A fold change is a **ratio**, and a ratio is unstable when its denominator approaches zero. The library computes, for each gene $g$,

$$\log_2 \mathrm{FC}_g = \log_2 \frac{\overline{x}_g^{\text{pert}} + \varepsilon}{\overline{x}_g^{\text{ctrl}} + \varepsilon}, \qquad \varepsilon = 10^{-9},$$

where $\overline{x}_g^{\text{pert}}$ is the mean expression of gene $g$ across the perturbed cells, $\overline{x}_g^{\text{ctrl}}$ is its mean across the control cells, and $\varepsilon$ is a pseudocount whose only purpose is to prevent a division by zero.

Now consider a gene that is essentially silent in this cell type, detected in almost no cells at all. Its control mean might be $10^{-7}$ and its perturbed mean $10^{-5}$, not because anything happened to it, but because two stray transcripts happened to land in two cells. The ratio of those two numbers is one hundred. On a $\log_2$ scale that is a change of about $6.6$, and it will comfortably outrank a real responder that genuinely doubled from $2.0$ to $4.0$, whose $\log_2$ fold change is $1.0$.

The pseudocount does not protect against this. It only guarantees that the ratio is finite. A finite ratio of two pieces of noise is still a ratio of two pieces of noise, and it is *large*, because noise divided by noise is unbounded in exactly the regime where both are small.

So the criterion does not select the genes the perturbation moved. It selects, with high reliability, **the genes that are barely expressed at all**, because those are the ones whose ratios can explode. The criterion is not noisy. It is not approximately right. It is a near-perfect detector of the one class of gene that carries no information.

## 6. What was actually in the file

The consequence is not subtle, and it is measurable directly, without any model in the loop. Here is what the twenty selected genes per held-out perturbation look like on real data, under each of two ranking criteria (the second is introduced in §8):

| ranked by | mean expression | $\lvert \Delta \rvert$ | detected in | significant at $p_{\text{adj}} < 0.05$ |
|---|---|---|---|---|
| $\lvert \log_2 \mathrm{FC} \rvert$ | $0.0006$ | $0.02$ | $0\%$ of cells | $0\%$ |
| $\lvert z \rvert$ (Wilcoxon) | $2.40$ | $0.75$ | $83\%$ of cells | $100\%$ |

Read the top row slowly. A typical gene in this dataset has a mean expression around $0.39$. The selected genes have a mean expression of $0.0006$, roughly six hundred times smaller. They are detected in zero percent of cells. Their median adjusted p-value is $1.000$, and not one of them is statistically significant.

And the fact that ends the investigation: in the held-out population, **every single cell is exactly zero on every one of the selected genes.**

The benchmark was scoring models on their ability to predict a matrix of zeros.

## 7. What that did to the scoreboard

Now go back to the clue and watch it resolve.

**Coverage was forced to exactly $1.00$, and the decoder was never involved.** Coverage asks what fraction of real cells fall inside the model's predicted interval. If every real cell is exactly zero on a gene, then *any* predicted interval that contains zero contains one hundred percent of them. The model's dispersion is irrelevant. Its mean is irrelevant. Its architecture is irrelevant. The number was pinned by the data, and it looked exactly like an over-dispersed decoder, and it would not budge no matter how the decoder was changed, because the decoder was never what produced it. Months of decoder engineering were aimed at a phantom, and the pipeline reported a clean, plausible, perfectly stable $1.00$ at every step of that work.

**The per-gene spread correlation went negative.** This metric asks whether the model knows *which* genes vary most within a response, by correlating the model's per-gene predicted standard deviations against the true ones. But the true columns were all zeros, so their "ordering" was pure floating-point noise. Correlating anything against a noise ordering gives you a number near zero that is as likely to be negative as positive. It came back negative, and it reads, on a results table, as "the model has the variance ranking backwards," which is a specific and damning-sounding failure. There was no ranking to get right.

**The effect-size correlation degenerated into a constant.** With the perturbed mean pinned at zero for every selected gene, the true differential expression reduces to

$$\Delta_g = 0 - \overline{x}_g^{\text{ctrl}} = -\overline{x}_g^{\text{ctrl}},$$

which is *the same fixed vector for every perturbation*, and it does not depend on the perturbation at all. Any model that has learned to emit near-zero on silent genes reproduces it. The correlation stayed respectably high, which is why nothing looked wrong, and it stopped discriminating between models, which is why everything was close.

That last one is the expensive part. The headline conclusion of the entire project at that stage was that the elaborate method **tied** its simple baseline. Every arm was scoring in the same narrow band, and a tie is what it looks like when a metric has stopped being able to tell things apart. Recomputed on a correct support, the arms separate cleanly and the baseline **wins decisively**.

The project's central result was an artifact of a two-line gene selection.

Notice what this bug never did. It never crashed. It never produced a NaN, an infinity, a negative variance, or a correlation outside $[-1, 1]$. It never produced a score that was suspiciously good. Every unit test on the metric functions passed, because the metric functions were correct. Every seed reproduced. Every ablation ran. The pipeline was, in the only sense that automated checking can establish, working.

## 8. The fix, stated generally

The general principle first, because it is the transferable part:

> **Rank by a test statistic, not by an effect ratio.**

An effect ratio answers "how much did this change, in relative terms," and its magnitude is a function of both the change and the baseline. When the baseline is small, the ratio is large, and the ratio has no way of telling you that it is large for the boring reason. It needs an external guard (a minimum-expression filter, a detection threshold, a significance cut) to be usable, and that guard is something a human has to remember to apply.

A test statistic answers a different question: "how surprising would this difference be if nothing were happening?" It has the null hypothesis built into its denominator, so it is *already* normalized by how much noise a quantity of this kind carries. A quantity that is small and noisy cannot post a large test statistic, because the noise is in the denominator.

Concretely: rank the genes by the **Wilcoxon rank-sum $z$-statistic**, which the same library already computes and returns in its `scores` field, and which is in fact its own default ordering. The project's code had explicitly overridden that default in favor of fold change.

$$z_g = \frac{U_g - \mathbb{E}[U_g]}{\sqrt{\mathrm{Var}[U_g]}},$$

where $U_g$ is the rank-sum of gene $g$'s expression in the perturbed cells relative to the control cells, and $\mathbb{E}[U_g]$ and $\mathrm{Var}[U_g]$ are the mean and variance of that rank-sum under the null hypothesis of no difference between the groups.

## 9. Why it works structurally, and what "self-guarding" means

The argument for $z$ is not that it is a better heuristic. It is that the failure mode of §5 is **structurally unreachable** under it, and that is a much stronger property than being reduced.

$U_g$ is built from the **ordering** of samples, not from the magnitude of a ratio. To compute it you rank all the cells by their expression of gene $g$ and sum the ranks belonging to the perturbed group. Now take the pathological gene: it is zero in almost every cell of both groups. Almost every cell is therefore **tied** with almost every other cell, ties are assigned their average rank, the rank-sum lands essentially at its null expectation $\mathbb{E}[U_g]$, and the numerator of $z_g$ goes to zero. The statistic is near zero, so the gene ranks near the bottom, so it is never selected.

There is no pseudocount to divide by. There is no regime in which two stray transcripts can manufacture a large statistic, because two stray transcripts move two cells by one rank each, out of thousands. The quantity is bounded by construction in exactly the place where the fold change was unbounded.

This is the property worth naming and worth looking for everywhere: the criterion is **self-guarding**. The pathology is not filtered out after the fact by a guard someone has to remember to write. It cannot occur.

The general design principle, which is the real takeaway of this chapter:

> **Prefer a criterion whose failure mode is structurally impossible over a criterion whose failure mode you have to remember to filter.**

The second kind works, right up until the moment someone refactors the pipeline, or reuses the selection code on a new dataset, or copies the two lines into a different script and leaves the filter behind. The first kind keeps working, because there is nothing to leave behind.

And $z$ selects on the right thing, which is the other half of the argument. A large $\lvert z \rvert$ requires the gene to be *consistently* shifted across many cells. That is what "this perturbation moved this gene" actually means, and it is what a model can fairly be asked to predict.

## 10. What the fix bought

Same data, same pipeline, same models, one changed line in the selection stage. The selected genes now have a mean expression of $2.40$ rather than $0.0006$. They are detected in $83$ percent of the perturbation's cells rather than zero percent. They are $100$ percent significant rather than zero percent significant. And the true effect they are supposed to be measuring is about **thirty-six times larger** in magnitude.

The benchmark went from grading models on a matrix of zeros to grading them on the genes the perturbation actually moved, and the arms stopped tying.

## 11. The acceptance gate, and the guard that never fires

Fixing the criterion is not the end of the work, because a criterion is a line of code and lines of code get changed. The seam needs a **gate**: an explicit, machine-checked statement of what a valid support looks like, enforced at the point of production.

Two filters now sit on top of the $\lvert z \rvert$ ranking. A gene is eligible only if it is **detected** in at least ten percent of the perturbation's cells, and only if it is **significant** at $p_{\text{adj}} < 0.05$. And the selection stage **asserts on its own output**: if the median detection rate across the selected genes falls below five percent, or if fewer than half of them are significant, the stage raises an exception and **refuses to write the cache**.

Under the correct ranking, these guards reject almost nothing. Ranking by $\lvert z \rvert$ already yields genes that are one hundred percent significant and detected in eighty-three percent of cells, so the filters are very nearly no-ops and the assertion never fires.

**That is precisely the point, and it is worth being stubborn about**, because "this check never fails, delete it" is one of the most common and most expensive pieces of code-review advice in existence.

A guard that never fires is not dead code. It is a **tripwire**. Its purpose is not to catch a problem that exists today; today's criterion is sound and the guard has nothing to do. Its purpose is to convert a future silent failure into a loud one. The ranking criterion is exactly the kind of thing a later change touches casually: a refactor, a new dataset, a well-meaning switch back to fold change because someone found it more interpretable. With the gate in place, such a change cannot quietly reintroduce silent genes into the support. The genes would fail the filters, the assertion would fire, the cache would not be written, and the pipeline would stop with an error naming the invariant it violated.

The invariant is the deliverable. The guard is just how the invariant is spelled in a language the machine can check.

This generalizes past this one seam and is the organizing habit of the whole series: **make every stage assert something about its own output before the next stage consumes it.** Not because assertions catch bugs, though they do. Because the alternative is discovering the bug at the metric, and the metric is the worst place in the pipeline to discover anything, since it is precisely the thing that will hand you a confident number no matter how badly you have fed it.

One consequence to accept rather than engineer around: biologically weak perturbations genuinely have fewer than twenty genes that pass the gate. About eleven of the two hundred thirty-six perturbations in our dataset are in that position, some with as few as three. The honest response is to score them on the genes they have, not to pad the list back up to twenty with the next-ranked genes, because the next-ranked genes are exactly the ones that failed the guard. Padding is how the pathology gets back in through the front door, wearing a uniform.

## 12. Your seam is somewhere. Go find it.

Nothing in the last nine sections is about biology. The gene list is an instance. The structure is universal, and it looks like this: a criterion, written once, in a preprocessing stage, using a library default, selecting the support that every downstream number is computed on.

Here is where the same seam lives in other fields, so that you can go and read the code that implements it.

**In an LLM evaluation**, the seam is the benchmark subset. Which prompts are in it? Who chose them, and on what criterion? If your suite advertises a "hard" split, what defined hard? A very common construction is to define "hard" as *the items a reference model got wrong*, which is a criterion with exactly the structure of a fold change: it selects on an outcome rather than on a property, and it will happily fill your hard split with items that are mislabeled, ambiguous, or unanswerable, because those are the items every model gets wrong. Your leaderboard then measures agreement with annotation noise, stably and reproducibly, forever. Also in this family: which of $k$ sampled completions are scored, how ties are broken, and which items were silently dropped for being too long to fit the context window (a filter that quietly removes exactly the hardest items).

**In reinforcement learning**, the seam is the success criterion and the episode filter. Which episodes count as successes? Are timed-out episodes excluded from the return average, and if so, does your "mean return" silently exclude the failures? Which environment seeds are in the evaluation set, and were they chosen before or after somebody looked at which ones the agent could solve?

**In time-series forecasting**, the seam is the windowing. Which windows are held out, and how were they chosen? A split that drops windows containing missing values will preferentially drop the anomalous periods, which are the only periods anyone cares about forecasting. Your model will post an excellent score on the calm weather.

**In information retrieval**, the seam is the gold set. Which documents were labelled relevant, by whom, and using what pooling procedure? If the pool was constructed from the top results of an earlier generation of systems, then a genuinely better system that surfaces a genuinely relevant document nobody pooled gets *penalized* for it, and the benchmark is measuring similarity to the systems of 2009.

**In classification generally**, the seam is class filtering and thresholding. Which classes were kept? Were the rare ones dropped for having too few examples? Rare classes are usually the ones that matter, and dropping them is a two-line preprocessing step that reviewers never see.

In every one of these, the seam is chosen by code that nobody re-reads, on a criterion that looked obvious at the time, using a library default that was designed for a different purpose. And in every one of these, a bad seam returns confident, stable, plausible, meaningless numbers.

So the practice is simple, and it is a practice rather than a check, because it has to be done once per project and it takes an afternoon.

**Find the line of code that chooses your support.** Not the metric. The selection upstream of it. If you cannot find it in ten minutes, that is itself the finding, and it means a library chose it for you.

**Read the criterion and ask what it maximizes.** Not what it is *for*, what it *maximizes*. Those come apart exactly when a criterion is a ratio, a difference of noisy quantities, or a function of an outcome rather than a property.

**Look at the selected items directly.** Not the score computed on them. The items. Print twenty of them and inspect them by hand. Every fact in §6 was available from a five-line script and none of it required a model, and any one of those facts would have ended the investigation on day one.

**Then gate it**, so that the next person to touch that criterion, who will be you, in six months, having forgotten all of this, gets an exception instead of a plausible number.

---

*Previous: [What are you actually measuring?](01-what-are-you-measuring.md). Up: [Running an experiment you can trust](index.md). Next: [Seeds, and the noise you cannot see](03-seeds-and-noise.md).*
