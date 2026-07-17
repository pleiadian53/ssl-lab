# Chapter 1 — What are you actually measuring?

*The metric is the first thing most projects choose and the last thing most projects examine. This chapter is about the property of a metric that decides whether any statistics are possible at all, and it is not the formula. It is the shape.*

---

## 1. A metric has a shape, not just a value

Ask a researcher what they are measuring and you will usually get a name and a number. "Accuracy, 84 percent." "BLEU, 31.2." "Mean return, 412." "Correlation, 0.648." Every one of those answers is incomplete in the same way, and the missing part is the part that matters.

The question that is actually load-bearing is this: **what is one observation?**

Not "what is the score," but "what is the thing the score is computed on, one instance at a time, such that a different draw of those things would have given me a different score?" That thing is the **unit of analysis**. It is the atom of your experiment. Everything you will ever be able to say about uncertainty, significance, or replication is a statement about how your score varies across units, and if you have not decided what a unit is, you have not decided anything, no matter how carefully you have specified the loss.

Choosing the unit is not a statistical formality that gets attended to at write-up time. It is a design decision that has to be made before the evaluation is written, because it determines what the evaluation code has to emit. A metric that emits one number cannot be repaired later by a cleverer test. There is nothing there to test.

## 2. An aggregate score is not a sample

Here is the failure in its purest form. Suppose your evaluation ends with a line like this:

```python
score = pearson_correlation(all_predictions, all_truths)   # -> 0.648
print(f"score: {score:.3f}")
```

You now have exactly one number. In statistical terms you have a sample of size $n = 1$, where $n$ denotes the number of independent observations available to the analysis. You cannot compute a standard error from it. You cannot bootstrap it, because a bootstrap resamples observations and you have one. You cannot compare it against a baseline's single number in any way that distinguishes a real difference from a coin flip, because you have no estimate of how much either number would have moved had the test set been drawn slightly differently.

People feel this and reach for the wrong repair. They add seeds, or they add test examples, or they run the whole pipeline again on a Tuesday and note that the number came back the same. None of that helps, because stability is not the same as significance. A broken thermometer is extremely stable.

The repair is structural: **define the metric per item, and let it return a vector.** Instead of one score over the whole test set, compute a score for each unit, and keep them all. The aggregate is then a summary of that vector rather than a substitute for it, and the vector is what every subsequent chapter of this series will operate on. The headline number does not change. What changes is that it acquires a distribution behind it, and a distribution is a thing you can reason about.

Concretely, the evaluation should end closer to this:

```python
scores = {item_id: score_one(item_id) for item_id in test_items}   # -> {..., 20 entries}
report = {"per_item": scores, "mean": float(np.mean(list(scores.values())))}
```

The mean is still there for the results table. But the per-item dictionary is the actual output of the experiment, and it is what gets cached, versioned, and handed to the comparison script. If your evaluation writes only the mean, then every statistical claim you make later will have been reconstructed from something you threw away.

## 3. The worked example: one perturbation, one number

Our running example is a method that tries to predict what a cell does when you activate a gene. A **perturbation** is the intervention (in our case a pair of genes activated together), and the test set is twenty two-gene combinations the model has never seen. For each one, the model generates a population of predicted cells.

The primary metric is an **effect-size** score, and it is defined per perturbation. For a held-out perturbation $p$:

$$r_p = \mathrm{corr}\left( \widehat{\Delta}_p, \Delta_p \right),$$

where $\Delta_p$ is the *true* differential expression of perturbation $p$ (the mean expression of the perturbed cells minus the mean expression of the control cells, computed gene by gene), $\widehat{\Delta}_p$ is the same quantity computed from the model's *predicted* cell population, $\mathrm{corr}$ is the Pearson correlation, and both vectors are restricted to the twenty genes that perturbation $p$ moves most (which genes those are, and how badly that choice can go wrong, is the entire subject of [Chapter 2](02-the-scoring-seam.md)).

So one perturbation yields one number. An arm of the experiment, meaning one trained model configuration evaluated end to end, yields **a vector of twenty numbers**, one per held-out combination. That vector is the arm's result. The headline figure that appears in the results table, $0.648$ for our best flow variant, is nothing more exotic than

$$\bar{r} = \frac{1}{n} \sum_{p=1}^{n} r_p, \qquad n = 20 .$$

That is the whole trick, and it is worth being unimpressed by it, because being unimpressed is the point. Nobody who saw the twenty-element vector would have been tempted to do anything unsound with it. The danger was never in the analysis. It was in an evaluation script that could have returned the mean directly, in which case the analysis would have been impossible rather than unsound, and impossible failures are the ones that get papered over.

## 4. What the unit is not, and why it is so tempting to get wrong

The metric $r_p$ is computed from thousands of cells and twenty genes, at three training seeds. None of those are the unit.

**Not the cell.** Each perturbation's score is computed from a population of a couple of hundred generated cells (and thousands of real ones). Those cells are *ingredients* of $r_p$, not replicates of it. They are averaged inside the metric, before the metric exists. Generating ten times as many cells would make $r_p$ slightly less noisy; it would not give you a single additional observation of the thing you are comparing, which is *how well the method handles a perturbation it has not seen*.

**Not the gene.** The twenty genes are the coordinates of one correlation. A correlation over twenty coordinates is one number, not twenty.

**Not the seed.** A retrain with a new random seed is a re-draw of the same model on the same data. It tells you about the stability of your training procedure, which is a genuinely useful and completely different question. It does not give you a new item to be tested on. Seeds get their own chapter, [Chapter 3](03-seeds-and-noise.md), because they are useful for exactly one thing and are routinely conscripted into doing another.

What is left is the perturbation, and there are twenty of them. Twenty is a small sample, and no amount of machinery will make it a large one. The machinery's job is to keep you honest about that, not to hide it.

The temptation to count the wrong thing is enormous, and it is enormous precisely because the wrong thing is more numerous. This is the classic **pseudoreplication** error: inflating $n$ by counting sub-units that are not independent replicates of the effect under study. In our setting you could report $n = 100{,}000$ by counting cells. Every standard error would shrink by a factor of roughly $\sqrt{100000/20} \approx 70$, every interval would tighten, and every p-value you computed would be garbage, because the cells within one perturbation are not independent draws of the quantity being compared. They all share a perturbation, a batch, a biology. The comparison is between methods across *perturbations*, so the perturbation is the exchangeable unit, and it is the only thing you are allowed to resample.

The general rule transfers cleanly, and the specific instances are worth memorizing because they are where the field actually errs:

In an **LLM evaluation**, the unit is almost always the prompt or the task, not the token and not the generated sample. If you score 500 prompts and generate 8 completions each, you have $n = 500$, not $n = 4000$. The eight completions are ingredients of one prompt's score.

In **reinforcement learning**, the unit is the episode, or more often the training seed, and it is never the timestep. A million environment steps inside one episode is one observation of that episode's return, and a run's learning curve is one draw from a training procedure that is notoriously seed-sensitive.

In **time-series forecasting**, the unit is usually the forecast origin or the held-out series, not the individual timestep, because consecutive timesteps within a window are about as independent as consecutive frames of a movie.

In **information retrieval**, the unit is the query, not the retrieved document.

The pattern is identical every time. There is a numerous thing (tokens, cells, timesteps, documents), and there is an exchangeable thing (prompts, perturbations, forecast origins, queries), and the numerous thing lives *inside* the exchangeable thing. Aggregation across the numerous thing happens inside the metric. Statistics happen across the exchangeable thing. Cross that line and your sample size is a work of fiction.

## 5. One primary endpoint, declared before you look

Now the second decision, which is cheaper than the first and skipped more often.

Fix, in advance, a **single primary endpoint**: one metric, on one set of comparisons, that the experiment is going to be judged on. Everything else in the evaluation is **secondary** or **exploratory**, and gets reported with descriptive intervals and no significance verdict.

The argument is arithmetic. Our project had four arms, which gives on the order of four comparisons of interest, and five metrics that could be computed on any of them. That is roughly twenty available tests. Under the **global null**, the hypothetical world in which nothing differs from anything, testing twenty things at a significance threshold of $\alpha = 0.05$ produces an *expected* number of false positives of

$$20 \times 0.05 = 1 .$$

You expect one. Not "you might get one if you are unlucky." Under the assumption that your method does nothing at all, the experimental design as specified will hand you, on average, one publishable-looking finding. And it will not be labelled. It will look exactly like the real ones.

A procedure that lets you pick your favorite result out of twenty is not a test. It is a search, and the thing it searches is noise. The formal version of this problem is called **multiplicity**, and there are corrections for it that [Chapter 5](05-from-difference-to-verdict.md) covers in detail. But the correction is downstream damage control. The upstream fix, the free one, is to declare the endpoint before you have seen the results, because then there is only one thing you could have found, and finding it means something.

In our case the declaration was: the effect-size correlation $r_p$ is the primary endpoint, its comparisons between arms are the entire family on which any significance claim will be made, and the four calibration metrics are secondary and will be reported without verdicts. That commitment cost us a result we liked, which is how you know it was doing work.

## 6. What it costs to skip this: a finding, announced and withdrawn

The four secondary metrics measured **calibration**, that is, whether the *shape* of the predicted cell population matches the real one rather than just its center. One of them, the **joint energy distance**, is a multivariate score that is sensitive to the correlation structure across genes: precisely the kind of structure a flow-based generative model exists to capture, and precisely the kind a per-gene metric is blind to. Lower is better.

The flow posted $3.578$. The from-scratch VAE baseline posted $3.962$. The flow won, and it won on the one metric that its whole architecture was justified by. That is a satisfying story, it was noticed *after* looking at the results table, and it was written down as a finding.

Then the interval was computed. The difference is $-0.383$, and its 95 percent bootstrap interval is

$$[-0.960, +0.172].$$

It crosses zero. On twenty perturbations, this difference is not resolvable. The apparent win is inside the noise, and the finding was withdrawn.

It is worth being precise about how many separate things had to go right for that "finding" to survive, because each one is a rule from this chapter:

It was a **secondary endpoint**, so it carried no pre-commitment and was one of roughly twenty available tests. It was **post-hoc**, noticed by scanning the results table for something favorable. And it was found on a metric that had been *changed* partway through the project, so the comparison it invited was not the comparison anyone had planned to make. That combination has a name. It is the **garden of forking paths**: at each of many small, individually defensible decision points (which metric, which arms, which subset, which of several plausible definitions), the analyst takes the branch that the data suggests, and the resulting claim carries none of the guarantees that the p-value printed next to it appears to offer. No one has to be dishonest for this to happen. The data does the choosing, quietly, through you.

The withdrawal cost nothing except a paragraph, because it happened before publication and because the interval that killed it took four seconds to compute. That is the actual argument for pre-commitment. It is not a moral posture. It is that the alternative is to find out later, in public, and by then the paragraph is a correction.

The honest final statement about the joint energy distance is the one in the record now: it is a hypothesis worth testing on a larger held-out set, and today it is nothing more than that.

## 7. Every metric is blind to something. Name it.

The last habit in this chapter is the shortest to state and the easiest to defer forever.

Write down, next to your primary metric, **the thing it cannot see**.

Our effect-size correlation $r_p$ grades the *mean* predicted response against the *mean* true response. That is all it does. It therefore cannot see whether the predicted population has the right spread, the right shape, the right number of modes, or any correlation structure across genes whatsoever. A model that emits the correct average cell with zero variance, the same cell every time, scores a perfect $r_p = 1$. It is, as a generative model, worthless.

Once that sentence is written down, several things follow immediately and for free. It explains why a second axis of metrics has to exist at all, which is where the calibration metrics came from. It tells you what an adversary would build to game your benchmark, which is usually the fastest way to find the benchmark's real weakness. And it stops you from over-claiming, because "our model predicts the mean response well" and "our model predicts the response well" are different sentences, and only one of them is supported.

The general form of this exercise: for each metric, describe a model that scores perfectly on it and is nonetheless useless. If you cannot think of one, you do not yet understand the metric. If you can think of one easily, you have just learned what your second metric needs to measure.

## 8. The chapter in four questions

Before an experiment is worth running, four answers should be written down, in this order, and none of them should require looking at a result.

**What is one observation?** Name the unit of analysis, and say how many of them you have. If the answer is "the whole test set," stop and redesign the evaluation until the answer is a countable thing.

**Does my evaluation emit a vector?** Check that the code writes the per-unit scores, not just their mean. This is a one-line difference at write time and an unrecoverable one afterwards.

**What is the primary endpoint?** One metric, one family of comparisons, committed before any results are seen. Everything else is described, not adjudicated.

**What is this metric blind to?** Write the sentence. Design the second metric to cover the gap, and expect the gap to be where your method's real problem lives.

Those four answers are what turn a number into a measurement. What decides whether the measurement is of the right thing is one level further upstream, in the code that chose *which* items and features the metric is computed on. That code is almost never audited, and it is where this project's worst bug was hiding.

---

*Up: [Running an experiment you can trust](index.md). Next: [The scoring seam](02-the-scoring-seam.md).*
