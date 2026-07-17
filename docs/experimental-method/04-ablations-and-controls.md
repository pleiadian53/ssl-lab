# Chapter 4 — Ablations and controls

*An A/B test changes exactly one thing. A **control** changes everything and asks a different question. They are not two flavors of the same experiment, and reading one as if it were the other is how a whole-method gap gets blamed on a single component. Plus the structural property that makes ablations cheap, and the one-lever rule we broke and then could not un-break.*

## Two experiments that look identical on a scoreboard

Open any results table in any method paper. You will see a column of configuration names and a column of numbers. Some of those rows differ from the headline method by one component. Some of them are entirely different systems. On the page they look the same: a name, a number, a confidence interval. Nothing in the typography tells you which is which.

But they answer different questions, they license different conclusions, and the difference between them is the single most useful distinction in experimental design. This chapter is about drawing it sharply enough that you cannot accidentally cross it.

## An ablation isolates

An **ablation**, which is the same thing as a properly-run A/B test, changes **exactly one lever** and holds everything else fixed. Same data, same splits, same encoder, same evaluator, same seeds, same everything, with one component swapped or removed. Because everything else is held constant, any difference in the outcome is *attributable* to that one lever. That attribution is the entire point, and it is bought purely by the discipline of holding the rest still.

Our project ran several, and they are worth naming because they show the range of what "one lever" can mean.

**Flow source.** The generative model (Stage B) transports a starting point to a target latent. In one arm, the starting point is a draw from a standard Gaussian, which is the textbook choice. In the other, it is a *real control-cell latent*, so the model learns to transport an actual unperturbed cell to its perturbed counterpart rather than conjuring it from noise. That is one lever: where the trajectory starts. Everything downstream is identical, and the difference between $0.612$ and $0.648$ is therefore attributable to it.

**Minibatch coupling.** Within a batch, the pairing between source points and target points can be assigned independently (a random matching) or by solving an optimal-transport problem to pair each source with a nearby target. One lever, one line of config. The result was instructive and is discussed in Chapter 6: OT coupling *lowered the training loss* and *hurt the metric*, dropping the score from $0.648$ to $0.627$.

**Decoder flags.** The count decoder (Stage C) has a handful of independent switches governing how it forms its mean and its dispersion. Each one is its own arm.

**Operator versus flow.** Replacing the conditional flow with an action operator (a model that predicts how a perturbation *moves* a latent, rather than sampling the destination directly) while keeping the same frozen encoder, the same decoder, and the same conditioning. One lever, and a large one, but still one: the transition model.

Every one of these tells you something you can act on, because in every one of them there is exactly one candidate explanation for the difference.

## A control calibrates

A **control** is a different animal entirely, and calling it an ablation is a category error.

Our control is the **from-scratch conditional NB-VAE**: a negative-binomial variational autoencoder trained end to end on the same data, with the same conditioning information, evaluated by the same harness on the same twenty held-out combinations. It scores $0.766$, which beats our best method arm ($0.648$) by $0.118$.

Now look at what actually differs between that model and ours. It has a **different encoder** (learned end to end for this task, rather than a frozen self-supervised JEPA representation). It has a **different prior** (a Gaussian variational posterior, rather than a learned flow). It has a **different training objective** (an ELBO with a reconstruction term and a KL term, rather than flow matching against a velocity field). It has different capacity, different inductive biases, and a different relationship between its representation and its reconstruction, because it is *allowed to adapt the representation to the reconstruction task* and we are not.

That is not one lever. That is every lever, pulled simultaneously.

So the control **cannot** tell us which of those differences produced the $0.118$ gap. It is structurally incapable of it, and no amount of statistical machinery layered on top will extract an attribution from an experiment that did not vary things one at a time. What it *can* tell us is something else, and something we needed to know more than we needed an attribution:

> **Does all of this machinery beat a simple, honest alternative?**

The answer was no. That is not an attribution, it is a **calibration**. It locates our entire method on an absolute scale, and it tells us whether the scale is one we should be proud to be on.

The distinction, compressed to something you can keep in your head:

> **Ablations isolate. Controls calibrate.**

An ablation answers *which part of my method is doing the work*. A control answers *is my method worth having at all*. You need both, they are not substitutes, and the failure mode is specific.

## The failure mode: reading a control as an ablation

Here is the mistake in its natural habitat, because it is tempting and it sounds reasonable when you say it out loud.

The NB-VAE beats us by $0.118$. The NB-VAE does not use a JEPA encoder. *Therefore the JEPA encoder is what is costing us $0.118$.*

That inference is invalid, and it is invalid in a way that will send a research program down a hole for a month. The gap is a **joint** property of every difference between the two systems. It might be the frozen encoder. It might be that a flow prior over a frozen latent is a harder learning problem than an ELBO over a co-adapted one. It might be that the decoder, trained against latents it did not shape, is the bottleneck. It might be capacity, or optimization, or three of these interacting. The control has entangled all of them and hands you one number.

The number is *true*. It is the honest measurement of the gap, and Chapter 5 shows the gap survives correction for the entire family of comparisons we ran. What is not true is any story about *which component* the gap belongs to, and the seductive thing about the mistake is that the number's credibility launders the story's. You did the statistics correctly on the gap, so the attribution feels like it inherited the rigor. It did not. **A rigorous measurement of the wrong quantity is still the wrong quantity.**

If you want the attribution, you have to earn it with ablations: hold the JEPA encoder and swap the prior, hold the prior and swap the encoder, and see which one moves. That is a different experiment. It is a *cheaper* experiment than you probably think, and the reason is the subject of the next section.

## The invariant that makes ablations cheap

Most people's mental model of an ablation is a full pipeline run: change one thing, retrain everything, wait a day. If that is the cost, then ablations are a luxury, you run three of them at the end of the project, and most of the questions you had go unanswered.

Our pipeline does not work that way, and the reason is a structural property that was worth more to the project than any modeling idea in it.

Recall the three stages. **Stage A** is the JEPA encoder, trained by self-supervision and then **frozen**. **Stage B** is the generative model over the frozen latents, either a conditional flow or an action operator. **Stage C** is the count decoder mapping a latent back to gene counts. Write $x$ for a cell's expression profile, $c$ for the perturbation condition, and $z$ for the frozen latent produced by the encoder. Stage B models $p(z \mid c)$. Stage C models $p(x \mid z)$.

The important part is the conditional independence. **Given the frozen latents, Stage B and Stage C are independent.** Neither one's training signal touches the other. Stage C learns to decode latents into counts using latents produced by the encoder from *real* data, and it never sees a sample from Stage B during training. Stage B learns to place probability mass in latent space, and it never sees the decoder. They meet for the first time at evaluation.

That independence is not a technicality. It is the thing that turns an experiment from a day into minutes, because it means:

> **An experiment retrains only the stage it changes, and reuses the rest.**

Which cashes out, concretely, like this.

A **decoder ablation** with four variants trains one encoder (already done, once, months ago) and one flow (already done) and then trains only four small decoders, each reusing the identical frozen latents and the identical flow samples. Everything upstream is fixed and cached. Four arms, and the marginal cost of an arm is one Stage-C fit.

An **operator experiment** reuses the encoder and reuses the decoder and retrains only the transition model. The evaluation pipeline, including the decoder that turns latents into the counts the metric actually scores, is byte-for-byte identical to the flow's. That is what makes "operator versus flow" a clean one-lever comparison rather than a confounded one: not luck, but the pipeline's structure.

The result is that an A/B stops being a budgeted event and becomes something you do casually, several times a day, while thinking. And *that* changes what questions you are willing to ask. When an experiment costs a day, you only run the ones you are fairly confident about, which means you only ever confirm things you already believed. When it costs ten minutes, you run the stupid idea, and occasionally the stupid idea is the OT-coupling result: a thing that lowers the training loss, that every instinct says should help, and that measurably hurts.

### The general principle

Lift this out of our pipeline, because it is the most transferable thing in this chapter.

> **Design your system so that its components are separable, because separability is what makes experimentation cheap, and cheap experimentation is what makes a research program fast.**

Separability means: a clean interface between stages, an artifact you can freeze and cache at that interface, and no training signal crossing the boundary. When you have it, the cost of an ablation is the cost of the *changed* component, not the cost of the system. When you do not, every question costs a full run and you will ask fewer questions.

The usual argument for modularity is engineering hygiene: easier to debug, easier to test, easier for two people to work on at once. All true, and all beside the point here. The research argument is stronger and it is rarely made. **A monolithic end-to-end system is not just harder to debug. It is harder to learn from.** It has one knob, called "the model", and when the number moves you get no information about why, because everything moved together. Every question you ask it costs a full training run, so you ask few questions, so you learn slowly. The end-to-end system may well be the better *artifact* (our own control is end-to-end and it wins, which is exactly the honest irony of this project). But the modular system is the better *instrument*, and during the phase of a project where you are still trying to find out what is true, you want the instrument.

There is a real tension here and it deserves naming rather than hiding: the separability that makes our pipeline a good instrument (freezing the encoder, so Stage B and C cannot reshape it) is *plausibly among the reasons it loses* to a model that has no such constraint. That is a legitimate hypothesis and it is exactly the kind of thing an ablation could test, by unfreezing Stage A and re-running. The point is not that separable is always better. The point is that separability is what lets you find out.

## One lever per arm, and the arm where we broke it

The rule is simple enough to state in six words and it is broken constantly, usually under deadline: **one lever per arm.** If an arm differs from its reference in two ways, it cannot attribute its own result, and no analysis performed afterward can rescue it.

We broke it. Here is what happened, told as what it is, which is a mistake we made and had to pay for.

In the action-operator experiment, one arm was configured with **two** changes relative to its reference at the same time. It introduced a **stochastic coefficient distribution** (making the operator's coefficients random variables rather than point estimates, so that the operator itself contributes variance to the predicted latent distribution). And, in the same arm, it added a **per-condition residual displacement** (a learned vector, one per perturbation condition, added to the operator's output).

The arm regressed, and it regressed specifically on the distributional axis, which is the set of metrics that ask whether the *spread* of the predicted cells matches the spread of the real ones rather than merely whether the mean is in the right place. The share of predicted variance contributed by the latent distribution fell from $0.140$ to $0.081$. Interval coverage, which measures how often the true value lands inside the model's predicted central interval, dropped by $0.037$. Both moves are in the direction of a model that has become *more deterministic*: it is still putting the mean roughly in the right place, but the cloud it draws around that mean has collapsed inward.

And we have a mechanism, and the mechanism is genuinely plausible. A per-condition residual is a **constant shift**: for a given condition $c$, it adds the same vector to every cell. A constant adds **zero variance**. So the residual is a very cheap way to capture the *mean* effect of a perturbation, and gradient descent will happily take a cheap route. If the residual absorbs the mean effect, the operator no longer needs to produce it, and the operator is free to relax back toward the identity. An operator near the identity moves cells very little, which contributes very little variance, which is exactly the collapse we observed in both numbers.

That story is coherent, it fits every number in the table, and it is the kind of explanation that gets written into a paper as a finding.

**And the experiment cannot test it.** It changed two things. Every number that arm produced is consistent with at least three stories: the residual is at fault (as above), the stochastic coefficients are at fault (perhaps the added coefficient noise destabilized training in a way that drove the operator toward a conservative solution), or the two interact, and the combination is bad in a way that neither is alone.

There is no clever post-hoc analysis that separates them. The information is not in the data. It was never collected, because the design did not collect it, and this is what makes a two-lever arm such an expensive mistake: it is not that the result is *wrong*, it is that the result is **uninterpretable**, and you do not find out until you try to write down what it means.

So the finding is not a finding. It is a **hypothesis**, and it is recorded in our ledger as one, and the arm has to be re-run split into two single-lever arms before anything can be concluded. That is the cost of the mistake: an experiment that already ran, that produced clean-looking numbers, that fooled us for a while, and that has to be run again from scratch.

The general form of the lesson, since it will visit you in a different costume: **a two-lever arm cannot attribute its own result.** When you are tempted to bundle (and you will be tempted, because bundling saves an arm and the two changes "obviously go together"), remember that you are not saving an experiment. You are spending one and getting nothing back.

## The fair-comparison checklist

Everything above assumes the arms are comparable in the first place. That assumption is doing a lot of work and it is worth making explicit, because the most common way to get a wrong answer is not a statistical error but an unfair comparison that nobody noticed.

**Same evaluators.** One scoring harness, one implementation of the metric, one code path, run on every arm. Not "the same metric", which is a statement about intent. The same *code*, which is a statement about fact. If the baseline is scored by the script that shipped with the baseline's repository and your method is scored by the script you wrote, you have measured two different things and the comparison is void.

**Same test set.** The identical twenty held-out combinations, with the identical split, for every arm. Any arm that saw any test perturbation during any of its three training stages is disqualified, and the frozen-encoder design means you must check *Stage A* for this too. A self-supervised encoder trained on all the data has seen the test cells, and if it saw them with their labels, the split has already leaked before Stage B started.

**Same seeds.** Chapter 3 covers why. Every arm gets a sweep, and where possible the arms share seed values so that the comparison is paired, which Chapter 5 shows is what makes a small test set tractable at all.

**Same reference.** The $\Delta$ in $\Delta$-correlation is computed against a control profile, and every arm must use the *same* control profile. Changing the reference changes the metric, and an arm scored against a more favorable reference is not scoring better, it is being graded differently.

**And the one that matters most: the baseline gets the same conditioning information the method does.**

This is where good-faith comparisons most often go quietly wrong, and it is easy to see why. You built an elaborate conditioning mechanism. Your baseline is "simple". It feels natural to give the simple baseline the simple conditioning, and it is fatal, because now the comparison confounds *the generative machinery* with *how the intervention is encoded*, and any win you post might be entirely attributable to the latter.

In our project the perturbation condition is encoded as a **compositional gene-set embedding**: the intervention is represented by combining embeddings of the individual genes being perturbed, which is what allows a model to say something sensible about a two-gene combination it has never seen by composing the two singles it has. That mechanism is powerful, and it would have been very easy to hand the NB-VAE a plain one-hot perturbation label instead, on the reasonable-sounding grounds that a one-hot is what a simple baseline uses.

Had we done that, we would have won. The baseline would have been unable to generalize to unseen combinations (a one-hot for a combination it never saw is a vector it has no embedding for), we would have beaten it comfortably, and we would have learned **nothing**, because the win would have been a win for the gene-set embedding and not for the flow, and we would have proceeded to attribute it to the flow.

So the VAE was given the **identical** gene-set embedding. Same conditioning, same information, same compositional structure. Which means the comparison isolates the thing we actually wanted to test (the generative machinery: flow over frozen JEPA latents versus an end-to-end VAE) rather than being confounded by the conditioning. And because the conditioning was held fixed and shared, we learned the most important structural fact the project produced: **the compositional gene-set embedding is what drives combination generalization, in both models.** Almost everything our method achieves on unseen combinations is achievable without the JEPA and without the flow.

We could not have learned that from a handicapped baseline. We would have learned the opposite, confidently.

> **A baseline you have handicapped is not a baseline. It is a strawman, and beating it teaches you nothing.**

## What a strong baseline is for

Which brings us to the thing this chapter is really arguing, and it is not a technical point.

The from-scratch NB-VAE, the "simple honest alternative", beat our elaborate method by $0.118$, a margin that survives a seed sweep on both sides, survives correction for the whole family of comparisons, and is roughly ten times either method's noise floor. By the ordinary incentives of research, that is the worst thing that happened to this project.

It is the single most valuable result the project produced.

It is valuable because it is **true**, and because it **redirected everything**. Before that number, the plausible next moves were incremental: tune the flow, add guidance, try a bigger velocity field, sweep the coupling. After it, those moves were visibly pointless, because none of them can close a $0.118$ gap in a family whose entire internal spread is a few hundredths. The number did not tell us which component to fix (it is a control, and controls do not do that). It told us something more useful: that the *premise* needed re-examination, and that the remaining hope has to be structural rather than incremental. That reorientation is worth more than any of the small wins we would otherwise have spent the next month collecting.

Now imagine the counterfactual, because it is the one that actually happens to most projects. Suppose we had built a weak baseline: a VAE with one-hot conditioning, undertrained, unswept, scored on a metric computed over a degenerate gene list. We beat it. We write the paper. The method "works". And every subsequent decision is made on a foundation we never tested, and the field gets one more result that does not replicate, and we spend a year building on top of a comparison that was never real.

The weak baseline does not just fail to inform you. It **actively misleads you**, and it does so in the most dangerous possible direction, which is the direction you were already hoping to go.

So the instruction, and it is the one thing I would most like a reader to take from this chapter:

> **Build the strongest, fairest baseline you can, and want it to be strong.**

Give it the same conditioning. Give it the same tuning budget. Give it the same seed sweep. Give it a metric that can actually see what it is doing. And then hope, genuinely, that it is hard to beat, because a baseline you cannot beat is telling you something true about your problem, and a baseline you crush is usually telling you something true only about your baseline.

If your method beats a strong baseline, you have a result. If it loses to one, you have a *better* result, which is a direction, and you have it now rather than a year from now.

The next chapter takes the last step. We have a difference, and we know it is bigger than the noise floor, and we know what it is attributable to. Turning that into a verdict you can defend requires one more piece of machinery, and it requires being honest about how many comparisons you ran before you found the one you liked.

---

*[Series index](index.md) | Previous: [Seeds, and the noise you cannot see](03-seeds-and-noise.md) | Next: [From a difference to a verdict](05-from-difference-to-verdict.md)*
