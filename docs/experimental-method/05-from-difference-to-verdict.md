# Chapter 5 — From a difference to a verdict

*You have two arms and a per-item metric. Arm A scores higher than arm B. That is a difference, and a difference is not a verdict. This chapter is about the machinery that turns one into the other: the paired bootstrap, why pairing is not optional when your test set is small, why contrasts that share arms have to be resampled together, and how a single critical value can be made to police an entire family of comparisons at once. It ends with a finding we announced and then took back, because the machinery said we should not have believed it.*

By this point in the series you have done the hard part. You have picked a metric, decided what counts as one observation, run enough seeds to know what noise looks like, and built arms that differ by one lever each. You are holding two vectors of numbers and one of them has a bigger mean. The question this chapter answers is the only question left, and it is the one people skip: **is that gap real, or is it the test set you happened to draw?**

## The shape of the problem

Assume the setup that [Chapter 1](01-what-are-you-measuring.md) argued for. Your evaluation has a **unit of analysis**, an item that counts as one independent observation, and there are $n$ of them. Your metric is defined *per item*, so each arm does not produce a score. It produces a **vector** of $n$ scores, one per item.

Write $A_i$ for arm $A$'s score on item $i$ and $B_i$ for arm $B$'s score on the same item $i$, with $i$ running from $1$ to $n$. The thing you want to reason about is the gap between the arm means, $\bar{A} - \bar{B}$, where $\bar{A} = \frac{1}{n}\sum_{i=1}^{n} A_i$ and likewise for $\bar{B}$.

In our worked example the item is a **perturbation**, that is, one held-out two-gene combination, and $n = 20$. The primary metric is a per-perturbation $\Delta$-correlation, an effect-size score in $[-1, 1]$ measuring how well a model's predicted change in expression tracks the real one. Every arm therefore hands us twenty numbers. Seed-averaged over three seeds, the arm means are $0.612$ for the Gaussian flow, $0.627$ for the transport flow with optimal-transport coupling, $0.648$ for the plain transport flow, $0.645$ for the action operator, and $0.766$ for the from-scratch NB-VAE baseline.

Read that list and the story writes itself. Transport beats noise. The baseline crushes everything. Both of those turn out to be true, but *nothing in the list of means establishes either one*, and the same list would look exactly as convincing if all five arms were identical and we had drawn an unlucky twenty items. The means are the input to the analysis, not the output.

If your work is elsewhere, the item is whatever you are actually generalizing over. It is the **task** in an LLM eval, not the token and not the individual generation. It is the **environment seed** or the **level** in RL, not the episode step. It is the **series** in a forecasting benchmark, not the timestamp. The rest of this chapter does not care what the item is. It only cares that you have $n$ of them and that they are the things you want your claim to be about.

## The paired bootstrap

The classical move here is a paired $t$-test. The bootstrap does the same job without asking you to believe anything about the shape of the distribution, and it is about five lines of code.

Form the **per-item difference**:

$$d_i = A_i - B_i, \qquad i = 1, \dots, n .$$

This is now a single vector of $n$ numbers, and the quantity we care about is its mean, $\bar{d} = \frac{1}{n}\sum_{i=1}^{n} d_i$. Note that $\bar{d} = \bar{A} - \bar{B}$ exactly, so we have not changed the estimate at all. We have only changed what we are going to resample.

Now the bootstrap. Let $R$ be the number of bootstrap replicates; we use $R = 10{,}000$. For each replicate $r = 1, \dots, R$, draw $n$ item indices uniformly **with replacement** from $\{1, \dots, n\}$, call them $I^{(r)} = (i_1^{(r)}, \dots, i_n^{(r)})$, and compute the mean difference over that resampled set:

$$\bar{d}^{(r)} = \frac{1}{n} \sum_{k=1}^{n} d_{i_k^{(r)}} .$$

You now have $R$ values of $\bar{d}^{(r)}$, and their spread is an estimate of how much $\bar{d}$ would wobble if you had drawn a different test set of the same size from the same population. Take the $2.5$th and $97.5$th percentiles of $\{\bar{d}^{(r)}\}$ and you have a **95% confidence interval** for the true mean difference. If that interval excludes zero, the difference is real at the 5% level. If it straddles zero, your experiment did not resolve the question, and the honest report is "did not resolve", not "no difference".

That last distinction matters more than it sounds. An interval containing zero is not evidence that the arms are the same. It is evidence that your test set was too small to tell, which is a statement about your experiment and not about the world.

## Why pairing is not optional

You could have skipped the differencing. You could have bootstrapped $\bar{A}$ and $\bar{B}$ separately and asked whether their intervals overlap. On a small test set that is close to useless, and it is worth being precise about why.

Items differ in **intrinsic difficulty**, and on most benchmarks they differ enormously. Some of our twenty held-out combinations produce large, clean, unmistakable transcriptional effects that every model gets roughly right. Others are weak, noisy, and near the detection floor, and every model flounders on them. The variance across items is a property of the *items*, and it is large. The difference between two arms is a property of the *methods*, and it is small: the gaps we are chasing are in the third decimal place of a correlation.

Compare the arms unpaired and you are asking the between-item variance to sit still while you measure a between-arm effect one or two orders of magnitude smaller than it. It will not sit still. It swamps everything, the intervals come out wide enough to drive a truck through, and no comparison you ever run will resolve.

Pairing removes the problem at the root. Because $d_i = A_i - B_i$ is computed **on the same item**, each item's intrinsic difficulty appears in both terms and cancels. An item that is hard for $A$ is hard for $B$ too, so the hardness subtracts out and what survives in $d_i$ is the part that is actually about the methods. The variance that the bootstrap then has to work against is the variance of the *differences*, which is far smaller than the variance of the scores, and that is where the statistical power comes from.

This is exactly the logic of a paired $t$-test, and the only reason to use a bootstrap instead is that the $t$-test would require us to pretend that twenty bounded, skewed correlation coefficients are draws from a normal distribution. They are not, and there is no need to pretend, because the bootstrap does not ask.

The general form of the rule: **if the same items are scored by both arms, difference them before you resample.** If they are not the same items, fix your experiment before you fix your statistics.

## Seeds are averaged before the test, and that costs you something

There is a decision buried in "the arm's score on item $i$" that has to be stated out loud, because it changes what the intervals mean.

Each arm was trained three times, under three seeds. So for item $i$ there are three numbers, not one. We average them: $A_i$ is the **seed-averaged** score of arm $A$ on item $i$, and the averaging happens *before* the differencing and *before* the bootstrap. In the implementation this is `load_arm`, which pools every seed's per-item report and takes a mean, and which drops any item that did not survive all three seeds so that every arm is scored on the same support.

The consequence is unglamorous and needs to be said plainly rather than buried. **The intervals in this chapter capture uncertainty from the finite test set, not uncertainty from training randomness.** They answer "if I drew twenty different held-out combinations, how much would this gap move?" They do not answer "if I retrained with different seeds, how much would this gap move?"

We would like them to answer both. With three seeds we cannot make them. Estimating a seed variance well enough to propagate it into an interval takes considerably more than three draws, and a variance estimated from three numbers is itself so noisy that propagating it would add more error than it removes. So the intervals are **conditional on the seed-averaged model**, and they **understate total uncertainty**. That is a real limitation, it is stated in the script's own docstring, and it is stated here for the same reason: an interval whose meaning you have quietly narrowed is a worse object than a wide interval you have described honestly.

## The multiplicity arithmetic, done explicitly

Here is where most method papers quietly go wrong, and the arithmetic takes ten seconds.

We have roughly four contrasts of interest. We have five metrics: the primary effect size, plus four calibration metrics that are spread correlation, interval coverage, 1-Wasserstein distance, and joint energy distance. Four contrasts times five metrics is **about twenty tests**.

Run twenty tests at $\alpha = 0.05$ and, under the global null in which every arm is truly identical, the expected number of false positives is

$$20 \times 0.05 = 1 .$$

**You expect one.** Not "you might get one if you are unlucky". You expect one, on average, from an experiment in which nothing is happening at all. So an unadjusted "significant" result, pulled from a family of that size and reported on its own, carries almost no information. If you go looking across twenty tests for the one that clears $p < 0.05$, you will find it, and you will find it whether or not your method works. This is not a subtle statistical point. It is counting.

There are two defensible responses and we use both. The first is to **pre-commit one primary endpoint** and make significance claims only there, treating everything else as exploratory. The second is to **correct for multiplicity within the primary family**, which is what the rest of this chapter is about.

## The joint bootstrap

The naive correction is Bonferroni: divide $\alpha$ by the number of tests. It is safe, it is one line, and here it is the wrong tool, for a reason that is structural rather than statistical.

**Our contrasts are not independent**, and they fail independence twice over.

They **share arms**. The contrast `transport - Gaussian` and the contrast `transport - NB-VAE` both contain the transport arm. If the transport arm happened to do well on this particular test set, both contrasts move together, in the same direction, for the same reason. They are not two independent looks at the world. They are two views of a partly shared object.

They are **computed on the same items**. Every contrast is evaluated on the same twenty perturbations. If perturbation 7 happens to be one where all the flow-based models struggle, that fact enters every flow-versus-baseline contrast simultaneously.

Bonferroni *assumes* independence, and when the tests are positively dependent, as ours are, it over-corrects. You pay a penalty calibrated for twenty independent coin flips when what you actually have is twenty heavily overlapping views of the same coin, and you lose real power for nothing.

The fix is elegant enough that there is no excuse for not doing it. **Resample the item indices once per bootstrap iteration, and evaluate every contrast on that same resample.**

That is the whole idea. In the implementation it is a single array,

```python
idx = rng.integers(0, n, size=(args.n_boot, n))   # (R, n): drawn ONCE
```

and every contrast, on every metric, is then evaluated by indexing its own difference vector with that same `idx`. Because the resample is shared, a bootstrap replicate that happens to over-sample the hard perturbations over-samples them *for every contrast at once*, exactly as the real test set does. **The dependence structure is preserved by construction**, not modeled, not assumed, and not corrected for after the fact. It is simply never thrown away.

This costs about ten lines relative to bootstrapping each contrast in its own separate call, and it is strictly better. Separate calls destroy the correlation between contrasts and then force you to buy it back with a conservative correction that assumes the opposite of what is true.

## Max-$t$ simultaneous intervals

With a shared resample in hand, the multiplicity correction becomes almost free, and it takes a form that is much more informative than a $p$-value threshold.

Let $C$ be the family of primary contrasts, and index a member by $c$. For each contrast we have its per-item difference vector $d_c$, its observed mean $\bar{d}_c$, and its standard error

$$\mathrm{se}(d_c) = \frac{s_c}{\sqrt{n}}, \qquad s_c = \text{sample standard deviation of } d_c .$$

Now, on each bootstrap replicate $r$, compute for every contrast the **centered, studentized** statistic

$$t_c^{(r)} = \frac{\bar{d}_c^{(r)} - \bar{d}_c}{\mathrm{se}^{(r)}(d_c)} ,$$

where $\bar{d}_c^{(r)}$ and $\mathrm{se}^{(r)}(d_c)$ are the mean and standard error recomputed **on the resampled items**. Centering by the observed $\bar{d}_c$ is what makes this a sampling distribution for the *error* rather than for the estimate. Dividing by the resample's own standard error is what puts every contrast on a common scale, so that a contrast with a large spread and a contrast with a small spread can be compared at all.

Then take the maximum over the family, on each replicate:

$$M^{(r)} = \max_{c \in C} \left| t_c^{(r)} \right| .$$

The distribution of $M^{(r)}$ over the $R$ replicates is the distribution of the **largest error anywhere in the family**. Take its 95th percentile and call it $q$. That single number is a critical value that covers the whole family at once, and the **simultaneous 95% interval** for each contrast is

$$\bar{d}_c \pm q \cdot \mathrm{se}(d_c) .$$

These intervals hold *simultaneously* at 95% across every contrast in the family. The claim they license is not "each of these intervals covers its true value 95% of the time", which is the weak per-contrast guarantee. It is the strong one: **with probability 0.95, all of the intervals cover their true values at once.** That is the guarantee you actually need when you are going to look at four contrasts and write a paragraph about whichever ones are interesting.

And the price is visible, which is the best thing about the method. In our run,

$$q \approx 2.95 ,$$

against the $1.96$ that an unadjusted single test would use. The bar is materially higher, by about 50% in interval width, and you can see exactly how much of it you are paying. A finding that clears $2.95$ has cleared a family-wide standard, and one that clears $1.96$ but not $2.95$ was never as solid as its $p$-value suggested.

## The verdicts

Here is the primary endpoint, the $\Delta$-correlation, with simultaneous intervals over the four-contrast family at the critical value above.

| contrast | difference | simultaneous 95% CI | verdict |
|---|---|---|---|
| transport − Gaussian | $+0.036$ | $[+0.019, +0.052]$ | **significant**: transporting from a real control latent beats transporting from noise |
| OT − transport | $-0.021$ | $[-0.039, -0.002]$ | **significant**: the optimal-transport coupling *hurts* |
| transport − NB-VAE | $-0.118$ | $[-0.228, -0.008]$ | **significant**: the *baseline wins* |
| operator − transport | $-0.003$ | $[-0.030, +0.024]$ | not significant: a dead tie |

Four contrasts, three surviving claims, one honest null. Read them in order and they say something coherent: the way we build the flow matters and we built it the right way; the clever coupling we added on top of it is actively harmful; the elaborate method loses to the simple baseline by a margin that is not close; and the action operator, which was the deeper of the two structural bets, moves the number by $-0.003$ and cannot be distinguished from the thing it was designed to replace.

The point worth dwelling on is that **all of these survived the stricter bar.** They are not $1.96$ findings dressed up. They cleared $2.95$, family-wide, which is why they are worth stating at all and why they have not moved since. The negative result in particular, that the baseline wins by $0.118$, is the load-bearing claim of the entire project, and it is the one that most deserved a hostile test. It got one and it held.

Notice also what the last row buys you. A null result reported with an interval, $[-0.030, +0.024]$, is a genuinely informative object: it says the operator's effect, if any, is smaller than about $0.03$ in either direction. A null result reported as "no significant difference" says nothing at all. Intervals are strictly more useful than verdicts, even when a verdict is available.

## Secondary endpoints get intervals, not verdicts

Now the story this chapter exists for.

We had a set of calibration metrics alongside the primary one, and among them was the **joint energy distance**, a distributional metric that compares the whole predicted cell population against the whole real one. It is, more than any other metric in the suite, the one built to detect exactly the kind of structure a flow-based generative model *should* capture, and that a variational autoencoder with a diagonal Gaussian posterior *should* miss. If our method had a home-field advantage anywhere, it was here.

And on this metric, the transport flow beat the baseline. The flow scored $3.578$, lower being better; the NB-VAE scored $3.962$. That is a gap of $-0.383$ in our favour, on the one metric that we had the strongest theoretical reason to care about, in a project that had lost on everything else.

It was announced. Internally, in writing, as a real finding.

Then the interval was computed properly, on the same twenty items, and it came out as

$$[-0.960, +0.172] .$$

It **crosses zero.** At $n = 20$ the metric does not resolve, and the point estimate that looked so much like vindication is entirely consistent with the flow being no better than the baseline, or with it being somewhat worse.

It gets worse under scrutiny, and the second problem is more serious than the first. This was a **post-hoc finding on a secondary endpoint**, and it was found *after* the metric had been changed. It was not the pre-committed primary. It was one of roughly twenty tests in the family, which is exactly the family size for which we computed, a few sections ago, an expected false-positive count of one under the global null. We went looking across a wide family for something that made the method look good, and we found something, and the thing we found had a point estimate in the right direction and an interval that could not exclude zero.

So it was **retracted.** It survives in the record as a hypothesis worth testing on a larger held-out set, which is precisely what it is, and it does not survive as a result.

That is the moral of the chapter, and it is why the discipline is worth the trouble. **The machinery is not there to help you argue for your findings. It is there to stop you from believing your own favourite result.** It earns its keep on the day it takes something away from you, and if it has never taken anything away from you, it is not doing anything.

The rule that came out of this, and that is now enforced in the code: **one metric is primary and it is chosen before you look.** Its contrasts get simultaneous intervals and they get verdicts. Every other metric is reported with a plain interval and **no significance verdict at all**, because a difference discovered on a secondary endpoint after the fact is a hypothesis, and it becomes a result only when it is confirmed on data that did not suggest it. The script prints this in the header of the section, in the run log, every time, so that nobody reading the output can mistake an exploratory interval for a claim.

## What the bootstrap assumes, and where ours is wrong

A method that is used to demolish a finding should be held to the same standard as the finding.

Start with what the bootstrap does **not** assume, because it is a lot. It does not assume **normality**, which matters here: correlations are bounded in $[-1, 1]$ and their sampling distribution is skewed, so a $t$-test's normality assumption would be violated in a way that is hard to reason about at $n = 20$. It does not assume **equal variance** between arms. It does not assume any parametric form for the metric at all. All of that is real, and it is why the bootstrap is the right tool.

What it **does** assume is that the $n$ items are **independent and identically distributed draws** from some population. Resampling with replacement is a simulation of drawing a fresh test set, and that simulation is only faithful if the items were drawn independently in the first place.

**In our case that assumption is false, and we know exactly how.** The twenty held-out combinations are two-gene sets, and they **share genes**. Two different held-out combinations may both contain the gene *CBL*. Whatever a model has or has not learned about *CBL* enters both items, in the same direction, so those items are positively dependent. They are not twenty independent draws. They are twenty overlapping draws from a smaller pool of underlying gene-level facts.

The consequence has a sign, and it does not favour us. Positively dependent items carry less information than independent ones, so the **effective sample size is smaller than twenty**. Resampling them as if they were independent therefore **understates the true uncertainty**, which means our intervals are, if anything, **too narrow**. Every interval in the table above should be read as a lower bound on the true width. The three significant findings have margin to spare and survive comfortably. The retracted energy-distance finding, which failed even under the too-narrow interval, fails even harder under an honest one.

And there is one more thing, which no amount of statistics will fix. **The bootstrap does not create information.** It is a way of asking what your existing sample would have said had it come out slightly differently, and if your sample is twenty items, resampling it ten thousand times gives you ten thousand views of twenty items, not two hundred thousand items. $n = 20$ is small. It is small enough that only large effects will ever resolve, which is why the $0.118$ gap to the baseline is significant and the $0.003$ operator gap is not, and it will remain small until somebody holds out more combinations. Everything in this chapter is a way of being honest about a small sample. None of it is a way of escaping one.

---

*Previous: [Chapter 4, Ablations and controls](04-ablations-and-controls.md). Next: [Chapter 6, Failure modes](06-failure-modes.md). Up: [Running an experiment you can trust](index.md).*

*The implementation of everything in this chapter is a single script, [`12_compare_arms.py`](../../examples/perturbation_response/12_compare_arms.py), which loads each arm's per-item reports, seed-averages them, intersects the arms onto a common support, draws one shared resample, and prints the primary family with simultaneous intervals and the secondary family without verdicts.*
