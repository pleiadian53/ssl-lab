# Chapter 3 — Seeds, and the noise you cannot see

*What a random seed actually controls, why a single-seed comparison is a narrator you cannot trust, and the distinction that matters more than either: the noise in your **training** is not the same as the noise in your **measurement**, and confusing them will cost you compute you did not need to spend.*

## The number that moved when nothing moved

Here is the experience that every method developer eventually has, and that almost nobody writes down.

You train a model. You score it. You get $0.623$. You change nothing at all: not the data, not a hyperparameter, not a line of code. You change one integer in a config file, the one labelled `seed`, and you train again. You get $0.562$.

The gap between those two numbers is $0.061$. The improvement you were trying to detect, the whole reason the experiment exists, was worth about $0.02$ to $0.03$. Your noise is twice the size of your signal, and it came from nowhere. Nothing *happened*. You did not do anything.

Except that you did, and this chapter is about understanding exactly what.

## What a seed actually controls

A random seed is not a cosmetic setting. It is the initial state of the pseudo-random number generator that your entire training procedure draws from, and modern training draws from it constantly. At minimum it determines four things.

**Weight initialization.** Every parameter in the network starts at a value sampled from some distribution. A different seed means a different starting point in parameter space, and gradient descent is a local procedure: it walks downhill from where you put it. Two runs that start in different places end in different places. Not slightly different places. In a non-convex landscape they can end in genuinely different basins, with different internal representations and different failure modes.

**Minibatch composition and shuffling order.** The order in which examples are presented determines the sequence of gradient steps. Because the steps are not commutative (each one changes the point at which the next gradient is evaluated), reordering the data changes the trajectory, not just the path length. Early batches matter disproportionately, because they set the direction the rest of training refines.

**Stochastic regularization.** Dropout masks, data augmentation draws, any noise injected into the forward pass. These are sampled fresh from the same generator.

**Sampling at generation time.** This one is easy to forget and it is the one that bites generative work hardest. If your model produces its output by *sampling*, then even a fully trained, completely deterministic set of weights will hand you different outputs on different seeds. In our project the generative model integrates a flow from a source point to a target point, and both the source draw and the integration are stochastic. The seed is still in play long after training has ended.

Put together: **same data, same hyperparameters, same code, different seed, and you have a measurably different trained model.** Not a cosmetically different one. A different model, with a different score, and the difference is real in the sense that it will reproduce if you re-run that seed and will not reproduce if you re-run a different one.

## Why this is fatal on a small test set

Now combine that fact with the thing every method developer is also living with: a test set that is smaller than you would like.

In our worked example the test set is twenty held-out two-gene combinations. The primary metric is a per-perturbation $\Delta$-correlation, where $\Delta$ denotes the *change* induced by a perturbation (the perturbed expression profile minus the matched control profile), and the correlation is taken between the predicted change and the observed change. The unit of analysis is the perturbation, so $n = 20$. Chapter 1 argued for that choice and I will not relitigate it here.

Twenty is a small sample. It means the metric itself has sampling variability, and it means that a training procedure that lands in a slightly different basin will score differently on those particular twenty combinations in ways that have nothing to do with whether it is a better model in general.

So we have two sources of variation stacked on top of each other, and an effect size of $0.02$ to $0.03$ that we are trying to see through both of them. When we retrained the Gaussian-source flow at three seeds, the scores came back as $0.623$, $0.562$, and $0.569$: a spread of $0.061$.

That number is the whole problem in one line. **A single-seed comparison, in that regime, is uninterpretable.** Not "weak evidence". Not "suggestive". Uninterpretable, because a $0.03$ difference between two configurations is entirely consistent with both of them being identical models scored on two different draws from the seed distribution. And this was not a theoretical concern for us: at one seed, the ranking of two configurations *reversed*. Config A beat config B on one seed and lost to it on another. Whichever seed you happened to have run first would have told you a confident story, and one of those stories would have been false.

The uncomfortable corollary is that if you have ever shipped a single-seed result on a small test set, you have shipped a coin flip and called it a finding. Most people have. The fix is not complicated, and it is the next section.

## The seed sweep

A **seed sweep** is the simplest possible remedy and it is not optional: retrain *the same configuration* at $N$ different seeds, score each run independently, and report the average. We use $N = 3$, which is the minimum that buys you anything and less than you would want if compute were free.

The sweep buys two distinct things, and it is worth being precise about them because they are used for different purposes.

The first is a **seed-averaged point estimate**. If $s_1, \dots, s_N$ are the scores of the $N$ runs, the reported score is $\bar{s} = \frac{1}{N} \sum_{i=1}^{N} s_i$. This is what goes on the scoreboard. It is a better estimate of "what this configuration is worth" than any individual run, because it averages out the particular basin that any single initialization happened to fall into.

The second, and the one people skip, is an estimate of the **noise floor**: the spread of $s_1, \dots, s_N$. This is the magnitude of the difference you can produce by *doing nothing*. It is the null hypothesis made concrete, in the units of your actual metric, from your actual pipeline.

And once you have that number, it is a ruler, and the ruler has a rule attached to it: **any claimed improvement smaller than the noise floor is not a claim, it is a coin flip.** You may still believe it. You may have a good mechanistic story for why it should be true. But you have not shown it, and the honest thing to write in the results table is that you have not shown it.

That single discipline, retraining every configuration at three seeds before it is allowed onto the scoreboard, is the cheapest protection against self-deception in the entire methodology. It costs you a multiple of your compute and it saves you from chasing ghosts for a month.

## The central insight: seed noise is not metric noise

Everything above is standard advice, competently followed. Here is the part that is not standard, and it is the reason this chapter exists.

We had a noise floor of $0.061$. The natural reading of that number is: *training is noisy, that is the nature of stochastic optimization on a small dataset, and if we want statistical power we will have to buy it with more seeds.* That reading feels obviously correct. It is also, in our case, mostly wrong.

Chapter 2 tells the story of the scoring seam: the metric was being computed on a **support**, a selected subset of genes, and the code that selected that subset was choosing them by a plausible-looking criterion that in fact selected almost entirely the wrong genes. The gene list the metric scored was largely degenerate. The metric did not crash. It returned numbers. The numbers were plausible and they went on a scoreboard.

When the seam was fixed, we re-ran the seed sweep. Same models. Same architectures. Same seeds. Same training procedure, same data, same everything. The only thing that had changed was *how the output was scored*.

The three seeds of the Gaussian flow, which had previously spanned $0.061$, now spanned $0.012$.

Sit with that for a moment, because it inverts the diagnosis completely. The training was never that noisy. **Most of what we had attributed to seed noise was measurement noise.** A metric computed on a degenerate support is a metric that is largely reading off whatever happens to be in the noise of the model's output on genes that carry no signal, and a noisy readout on a fixed model is indistinguishable, from the outside, from a stable readout on a noisy model. Both produce a scoreboard where the same configuration scores differently on different runs. We had blamed the model. It was the ruler.

The practical consequence is large and it is very much in your favor: **a better-conditioned metric buys you statistical power for free.** It does not cost a single additional GPU-hour. It makes every experiment you have already run more informative, retroactively, because the same underlying differences between models now sit further above the noise.

The concrete case: the comparison between the transport-source flow and the Gaussian-source flow. Under the broken metric this was a difference of $+0.028$, sitting inside a noise floor of $0.061$, with a confidence interval that comfortably crossed zero. It was the definition of borderline, and the honest write-up was "directional hint, insufficient evidence". Under the corrected metric, the same two model families, retrained at the same three seeds, give $0.648$ against $0.612$: a difference of $+0.036$ with an interval of $[+0.019, +0.052]$. That interval does not touch zero. The finding is now a finding.

Nothing about the models changed. We did not make transport better. We made the measurement capable of seeing what transport had been doing all along.

### The general lesson

Generalize it, because this pattern is not about genes and it is not about flows.

**Before you spend compute on more seeds to chase statistical power, check whether your metric is the thing that is noisy.**

The instinct, when a comparison sits inside the noise, is to run more seeds. More seeds is the textbook answer and it does work, but it works *slowly*: the standard error of a mean falls as $1/\sqrt{N}$, so halving your uncertainty costs you four times the compute. Meanwhile, a metric that has been badly conditioned (computed on the wrong support, aggregated at the wrong level, dominated by a degenerate subpopulation, or averaging over items with wildly different scales) can be inflating your apparent run-to-run variance by a factor that no realistic seed budget will out-run.

So the diagnostic question, when your noise floor looks bad, is not "how many more seeds can I afford?" It is "is this variance coming from the model, or from the ruler?" Those have different fixes and only one of them is expensive. Look at the ruler first. It is free, and in our case it was worth more than tripling the seed budget would have been.

## An honest limitation: what our seed sweep does not vary

A methodology chapter that describes a safeguard without describing its holes is doing advertising, so here is the hole.

Our pipeline has three training stages. **Stage A** is a JEPA encoder trained by self-supervision on the expression data, which is then **frozen**. **Stage B** is a generative model over those frozen latents (a conditional flow, or an action operator). **Stage C** is a count decoder that maps a latent back to gene counts. Writing $x$ for a cell's expression profile, $c$ for the perturbation condition, and $z$ for the frozen latent, Stage B models $p(z \mid c)$ and Stage C models $p(x \mid z)$.

Our seed sweep re-seeds Stages B and C. It does **not** re-seed Stage A. The encoder is trained once and reused across every arm of every experiment, by design (the next chapter explains why that design is so valuable), which means the frozen encoder is a *constant* in our variance estimate rather than a random variable.

The consequence is stated plainly: **the seed noise we report excludes encoder-initialization variance, and the true run-to-run variance of the full stack is larger than what we measure.** If someone reran the entire pipeline from a fresh Stage A, they would see a spread wider than $0.012$. We do not know how much wider. That is a real limitation of every number in our flow-family results and it should be read as one.

There is a second asymmetry that follows from the same structure, and it is the more dangerous of the two because it can bias a *comparison* rather than merely a variance estimate.

The from-scratch NB-VAE baseline trains **end to end**. It has no frozen component. When you change its seed, you perturb *everything*: its encoder, its prior, its decoder, all of it. When you change the flow's seed, you perturb only what sits on top of a fixed encoder. These are not the same experiment. The baseline's seed distribution samples a strictly larger space of models than ours does, so its measured spread and our measured spread are not directly comparable quantities, even though they are both reported in the same column of the same table.

This is precisely why the baseline needed its own seed sweep before we would allow it to be declared the winner. A single-seed baseline that beat us would have been open to exactly the objection we would have made against a single-seed *win*: that we got lucky. So we ran it three times. It scored $0.762$, $0.767$, and $0.768$, a spread of $0.006$.

That result is worth reading carefully, because it does two things at once. It confirms the baseline's win (a seed-averaged $0.766$ against the flow's $0.648$ is a gap of roughly $0.118$, which is an order of magnitude above either method's noise floor and is not going to be closed by reseeding). And it also, quietly, undercuts the excuse that was available to us. The end-to-end model, whose seed perturbs strictly more of the pipeline than ours does, turned out to be *more* stable than ours, not less. We could not attribute the gap to it having gotten a lucky draw. The number was simply better.

## Transferring this: LLMs, RL, and everything else

Nothing in this chapter is biological. The seed problem is structural and it shows up wherever training is stochastic and evaluation is small, which is to say everywhere.

**In LLM work**, the seed decomposes into at least three separate knobs and people routinely control only the first. There is the **fine-tuning seed**, which does what any training seed does (initialization of any new parameters, data order, dropout). There is the **decoding seed and temperature**, which is the generation-time sampling knob, and it means that even a frozen model, evaluated twice on the same prompt set, will not give you the same score. And there is **prompt ordering**, which for in-context learning is a seed in everything but name: the order of few-shot examples changes the result, sometimes dramatically, and a benchmark that fixes one arbitrary order is reporting one draw from a distribution it never characterized. If you are comparing two fine-tunes on a 200-item eval and you ran each once at temperature $0.7$, you have measured almost nothing, and the fix is the same as ours: sweep, average, and report the spread.

**In RL**, the situation is worse and it is well documented. The seed controls environment initialization, exploration noise, and policy initialization, and their interaction is chaotic in the technical sense: small differences compound over an episode and then over a training run. It is entirely routine for two seeds of the same algorithm on the same environment to produce learning curves that look like different algorithms. Three seeds is our budget here and I would not defend it in an RL paper. The literature's rule of thumb starts around ten and people who care use more, and even then the right summary is the distribution of outcomes rather than the mean, because RL seed distributions are frequently multi-modal (some seeds solve the task, some never do, and the average of those two populations describes no run that ever happened).

**In time series and forecasting**, the analogous exposure is the split: which windows landed in your test period. A model retrained on a different random initialization, evaluated on a handful of held-out horizons, has exactly our problem, and the additional trap is that the horizons are not independent of each other, which makes the effective sample size smaller than the nominal one.

The common structure across all of them, and the thing to carry away, is this. Every experiment has a **noise floor** that you can measure for free by doing the same thing twice. Measure it before you interpret anything. And when it comes back larger than you expected, ask the diagnostic question from the previous section before you reach for your wallet: *is my model noisy, or is my ruler?*

## What to do on Monday

Concretely, the practice that follows from this chapter is small.

Run every configuration at three seeds minimum before it is allowed onto a scoreboard, and record all three individual scores in the ledger, not just their mean, because the individual scores are your noise floor and you will want them later. Never compare two single-seed runs and never let a single-seed number into a slide. When a difference is smaller than your measured spread, write "within noise" and move on, no matter how much you want it to be real. And when the spread looks bad, audit the metric before you buy more seeds, because the metric is free to fix and the seeds are not.

The next chapter takes up the other half of a trustworthy comparison. A seed sweep tells you whether a difference is real. It does not tell you what the difference is *attributable to*, and for that you need to have changed exactly one thing.

---

*[Series index](index.md) | Previous: [The scoring seam](02-the-scoring-seam.md) | Next: [Ablations and controls](04-ablations-and-controls.md)*
