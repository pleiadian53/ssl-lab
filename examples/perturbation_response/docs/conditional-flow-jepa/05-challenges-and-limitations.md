# Chapter 5 — Challenges and limitations

*Chapter 4 ended on a loss: the full stack is beaten by a from-scratch conditional VAE, by $0.118$ in $\Delta$-correlation, and the margin survives a simultaneous correction across the whole family of comparisons. This chapter is the post-mortem, in five parts. The control cell meant to anchor each prediction was drawn at random and unpaired with the target, so conditioning on it could not move the predicted response, and the flow ended up modeling the very same object as the VAE. Optimal-transport coupling, the training trick that was supposed to sharpen the flow, did lower the training loss and made the effect-size score worse. Calibration, once measured on sampled counts rather than expected rates, says every model is over-confident: the predicted populations are far too narrow, and a variance decomposition puts most of that failure in the count decoder both models share rather than in the latent distribution that distinguishes them. And twenty held-out combinations, which share genes among themselves, remain a thin base to stand on even with the statistics tightened. Each cause is concrete and fixable, and Chapter 6 turns them into directions.*

## The baseline that wins

The central challenge is the one [Chapter 4](04-results.md) reports: a from-scratch conditional negative-binomial VAE, with no JEPA pretraining and no flow, beats the full stack on effect size by $0.118$, with a simultaneous 95% confidence interval on the difference of $[-0.228, -0.008]$. This is worth stating plainly rather than explaining away. The VAE is a standard conditional generator. Its encoder is a plain multilayer perceptron over the expression vector, its latent prior is a fixed standard Gaussian, and its decoder is the same negative-binomial head the flow uses. Conditioned on the same gene-set embedding, it learns to map a perturbation to a plausible response population directly.

Nor is this a lucky initialization that a rerun would erase. The VAE's three training seeds score $0.762$, $0.767$ and $0.768$, a spread of $0.006$, which is an order of magnitude smaller than the gap it opens. Reseeding will not close that gap, and the rest of this chapter is an attempt to understand why a simpler model is holding the lead.

## The condition formulation gave the flow nothing to work with

The first thing we found on inspection was that the original formulation quietly threw away the flow's structural advantage. The condition is $c = (z_b, z_p)$, where $z_b$ is a baseline cell state in the latent space and $z_p$ is an intervention embedding. In the code, for each target cell the baseline $z_b$ was drawn as a *random, unpaired* control-cell latent, sampled independently of the specific outcome. Single-cell perturbation is a destructive assay, so there is no true "same cell before and after" pairing to draw on, which is why the baseline was random. But that independence has a consequence. If $z_b$ carries no information about the target beyond the population it came from, then conditioning on it cannot move the predicted mean of the outcome. The baseline part of the condition is inert for the effect.

Worse, the flow was trained to transport Gaussian noise to the outcome, with $z_b$ entering only as a side input to the velocity field, the neural network $v_\theta(z, t, c)$ that says how fast and in which direction to move a latent $z$ at flow time $t$ under condition $c$. So the model reduced to learning the distribution of perturbed latents given the perturbation, which is exactly the object the VAE fits. The two-part condition that was meant to make this a baseline-anchored transport had degenerated into an unconditional-on-baseline generator. There was no structural reason the flow should beat the VAE, because it was modeling the same thing by a more elaborate route.

## The transport reformulation, and why it only half-worked

The fix follows directly from the diagnosis. Instead of transporting noise to the outcome with the baseline as a side input, transport a real control latent to the outcome, and let the condition be the intervention alone. The flow's source distribution becomes the control-cell population, its target becomes the perturbed population, and each generated outcome is anchored to a real baseline it flowed from. Formally the generated latent is $z_b + \int_0^1 v_\theta(z_t, t, z_p) \mathrm{d}t$, so the model now has to represent only the *displacement*, the effect, rather than the whole absolute state. The effect is where the lower-variance signal lives, and it is what the metric scores.

This is well founded even without cell-level pairing. Rectified-flow and optimal-transport flow-matching train transports between two marginal distributions under independent couplings, so pairing a random control to a random perturbed cell of a given perturbation still learns a valid map from the control population to that perturbation's population. The reformulation is enabled in the code by an optional source argument to the flow-matching loss and the sampler, selected by `--flow-base control`.

It helped, and the tightened statistics of [Chapter 4a](04a-reading-the-head-to-head.md) support it more strongly than a first look at the raw scores would suggest. Transporting from a control latent beats transporting from noise by $+0.036$ in $\Delta$-correlation, with a simultaneous 95% interval of $[+0.019, +0.052]$ that stays clear of zero even after the correction for the whole contrast family. Anchoring to a real baseline recovers a genuine advantage that the noise formulation had discarded, and it is one of the two findings inside the flow family that we keep.

What it did not do is clear the baseline. A $0.036$ improvement inside the flow family is a real improvement and still an order of magnitude short of the $0.118$ the VAE is ahead by. Fixing the condition formulation was necessary, and it was nowhere near sufficient.

## Optimal-transport coupling: a better loss, a worse model

The natural next lever was the coupling. Under independent coupling the displacement target is noisy, because a random control paired with a random perturbed cell can point in an unhelpful direction. Minibatch optimal transport reorders each batch so that every control is paired with its nearest perturbed target, which straightens the flow-matching paths and should give a lower-variance regression target. It did lower the training loss, from around $1.96$ to around $1.48$. And it made the effect-size score significantly *worse*, by $-0.021$ with a simultaneous interval of $[-0.039, -0.002]$.

The lesson is that a lower flow-matching loss is not the objective. Our best reading of the mechanism is that global optimal transport within a batch pairs each perturbed cell to its nearest control across *mixed* perturbations, ignoring which perturbation the target belongs to, and that this over-concentrates the learned map. Straighter paths to nearest neighbors reduce the diversity of control-to-outcome directions the flow sees, and the effect-size metric, which compares population means, is sensitive to that collapse. Coupling by proximity optimizes geometry the metric does not reward, and it is a clean reminder that a training objective and an evaluation objective can disagree in the direction that matters.

## Calibration is harder to measure than it looks

We turned to calibration precisely because effect size grades only the mean, and a flow's promise is the full distribution. Measuring it well proved subtle, and the subtleties are themselves findings.

The first thing to get right is *what a generated cell is*. If each cell is generated as the decoder's expected rate for its latent, then a whole predicted population has almost no spread, because real per-cell variation in single-cell data is dominated by technical count noise and expected rates omit it entirely. Asking what fraction of true cells fall inside an interval built from expected rates gives an answer near zero for every model, not because the models are bad but because the question is malformed. An expected-rate population cannot be calibrated against real cells. Calibration has to be measured on *sampled counts* drawn from the negative-binomial head, which is the population a real experiment would be compared against. That measurement rule is a prerequisite for everything below.

Measured that way, the calibration axis says something sharp and uncomfortable, and it says it about every model at once. These are secondary endpoints, reported as numbers without a significance verdict, as [Chapter 3b](3b-reading-the-calibration-metrics.md) sets out.

| model | spread correlation ↑ | coverage (nominal 0.80) | 1-Wasserstein ↓ | joint energy ↓ |
|---|---|---|---|---|
| Gaussian flow | 0.205 | 0.357 | 1.013 | 3.794 |
| transport flow | 0.234 | 0.375 | 0.982 | **3.578** |
| transport + OT | 0.214 | 0.365 | 1.010 | 3.746 |
| NB-VAE | **0.522** | 0.328 | **0.956** | 3.962 |

**Every model is over-confident.** Coverage sits between $0.33$ and $0.38$ against a nominal $0.80$, which means each model's predicted $80\%$ interval captures only about a third of the real cells. The predicted populations are too *narrow*, not too wide. The models are under-dispersed on exactly the top differentially-expressed genes the effect-size metric scores. This is the most robust fact on the calibration axis, it holds for the flow and the VAE alike, and its uniformity across architectures is the clue: a failure this consistent is a property of the machinery the models share rather than of the latent distributions that distinguish them.

**The models do track which genes vary, and the VAE tracks it better.** The spread correlation, which asks whether a model puts its variance on the genes that actually vary in the real response, is positive for every model here. The VAE's $0.522$ against the transport flow's $0.234$ is a large margin, and its contrast interval, $[-0.429, -0.136]$, does not go near zero. On the question of which genes are variable, the plain conditional generator is substantially better informed than the flow.

**The flow's apparent edge on joint structure does not resolve.** The transport flow posts the best joint energy distance, $3.578$ against the VAE's $3.962$, and the energy distance is precisely the metric built to see the gene-gene structure a flow should be able to capture and a marginal metric cannot. It is tempting to read this as the flow finally earning its keep. The interval forbids it: the contrast is $-0.383$ with a 95% interval of $[-0.960, +0.172]$, which crosses zero. On twenty perturbations this difference is not resolvable, and it sits on a secondary endpoint besides. We record it as a hypothesis for a larger held-out set, and we decline to call it a finding. Discipline here is not pedantry, because the joint-structure story is exactly the story we would most like to be true, and that is the story a small sample is most likely to hand us by accident.

## Where the predicted spread actually goes

Under-dispersion this uniform invites one more measurement, and it is the one that reallocates the blame. By the law of total variance, the per-gene variance of a generated population splits *exactly* into two terms:

$$\mathrm{Var}(x_g) = \underbrace{\mathbb{E}_z[\mathrm{Var}(x_g \mid z)]}_{\sigma^2_{\text{dec}}} + \underbrace{\mathrm{Var}_z(\mathbb{E}[x_g \mid z])}_{\sigma^2_{\text{bio}}}$$

Here $x_g$ is the generated count of gene $g$ in a cell, $z$ is that cell's latent, $\sigma^2_{\text{dec}}$ is the count noise the negative-binomial decoder adds around each cell's own mean, and $\sigma^2_{\text{bio}}$ is the variance of the decoded mean *across* the latent cloud. The second term is the latent distribution's entire contribution to the predicted spread, and it is the only part the flow and the VAE do differently. Everything else is the shared decoder. The script `11_diagnose_variance.py` computes both terms on the top-DE genes.

| | real variance | predicted total | $\sigma^2_{\text{dec}}$ (decoder) | $\sigma^2_{\text{bio}}$ (latent) | latent's share |
|---|---|---|---|---|---|
| transport flow | 0.824 | 0.678 (0.84×) | 0.538 | 0.140 | 22% |
| NB-VAE | 0.824 | 0.355 (0.46×) | 0.226 | 0.128 | 38% |

Both models under-produce total spread, which is the coverage failure seen from a different angle, and the VAE under-produces it more severely: it generates less than half the variance of the real cells. But the decisive column is the last pair. The two models' latent contributions are $0.140$ and $0.128$, which is nearly the same number. Whatever separates a learned conditional flow from a Gaussian-prior VAE, it is *not* showing up as a difference in how much spread their latent distributions inject into the readout. The gap between the two models on the calibration axis is almost entirely a gap between the two decoders they happened to learn.

That has a direct consequence for what to fix. The decoder is not merely a nuisance component to be held fixed while the flow improves. It is where most of the predicted variance is made, it is making too little of it, and it is doing so under both models. Any attempt to read a flow's distributional quality through this readout is reading the decoder first and the flow second. The direction to push is *more* dispersion, not less, and [Chapter 8](08-modeling-the-readout-count-decoder.md) takes that up.

## The statistical-power ceiling

Every combination number rests on twenty held-out combinations. That is few, and it bounds what any of this can claim.

The metric is now well conditioned, and the consequence is that seed noise is much smaller than a first pass through this project would suggest. The Gaussian flow's three seeds span only $0.606$ to $0.618$, and the VAE's span $0.762$ to $0.768$. Configurations no longer trade places from one initialization to the next. On top of that, the analysis averages seeds before testing and then runs a *joint* paired bootstrap: the twenty perturbations are resampled once per iteration, every contrast in the family is evaluated on that shared resample, and a max-$t$ critical value is taken across the family so that the intervals hold simultaneously at 95%. That critical value comes out at $2.95$ against $1.96$ for a single unadjusted test, which is a materially higher bar than testing each comparison on its own. [Chapter 4a](04a-reading-the-head-to-head.md) develops the procedure in full.

The differences we care about clear that bar. All three primary contrasts, transport over Gaussian, OT under transport, and the baseline over the flow, survive the simultaneous correction. This is the sense in which the negative result is clean rather than ambiguous.

The humility is still warranted, and it lives in the test set rather than in the estimator. Twenty perturbations is a small sample, and those twenty combinations *share genes* with one another, so they are not strictly independent draws from a population of perturbations. A bootstrap that resamples them as if they were will, if anything, report intervals that are too narrow. The seed averaging also removes training randomness before the bootstrap rather than propagating it through, which pushes the same way. So the intervals above should be read as the optimistic end of the honest range, and any future claim of a flow overtaking the baseline will want more held-out combinations, and ideally a second dataset, before it can be trusted.

## The structural limitations behind the numbers

Stepping back from the individual fights, three design choices bound what this method can currently show.

The encoder is frozen and condition-blind. It is pretrained on states alone and never sees the intervention, so all conditioning lives downstream in the flow. This buys modularity and a reusable representation, but it means the representation itself is not shaped by the perturbation task, and any advantage from perturbation-aware features is left on the table.

The decoder is a shared bottleneck, and it is under-dispersed. The flow and the VAE both read out through a negative-binomial head, that head produces too little spread on exactly the genes the metric cares about, and the variance decomposition above shows it is responsible for most of the predicted variance in both models. When two models share a component that is miscalibrated in the same direction, differences in the components they do not share get compressed. Improving the decoder could matter more than improving the flow, and it is the one intervention that would help both models at once.

And the metrics we can compute reward the mean and the marginals, which are the parts a simple conditional generator already handles well. The flow's distinctive capability, a rich and possibly multimodal joint distribution over the response, is the hardest thing to measure and the least rewarded by the current scoreboard. The joint energy distance is our best attempt to see it, and at twenty perturbations it cannot resolve the difference. It is entirely possible the flow is capturing structure the evaluation cannot yet see. That possibility is not a defense of the current results, but it is a clear pointer for what to measure next.

[Chapter 6](06-beyond-the-current-limit.md) takes these limitations as a to-do list and lays out the directions most likely to move the result.

---

*Previous: [Chapter 4 — Results](04-results.md). Up: [the method series](index.md). Next: [Chapter 6 — Beyond the current limit](06-beyond-the-current-limit.md). The audit: [Chapter 4a](04a-reading-the-head-to-head.md).*
