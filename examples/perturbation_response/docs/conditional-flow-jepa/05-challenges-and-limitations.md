# Chapter 5 — Challenges and limitations

*What fought us, and what each fight taught. A condition formulation that gave the flow no structural edge, a coupling that improved the training loss yet hurt the metric, calibration metrics confounded by the decoder, and the statistical-power ceiling of a small held-out set. These are the seams where the next idea gets in.*

## The baseline that would not lose

The central challenge is the one [Chapter 4](04-results.md) reports: a from-scratch conditional negative-binomial VAE, with no JEPA pretraining and no flow, ties or slightly beats the full stack on both effect size and calibration. This is worth stating plainly rather than explaining away. The VAE is a standard conditional generator. Its encoder is a plain multilayer perceptron over the expression vector, its latent prior is a fixed standard Gaussian, and its decoder is the same negative-binomial head the flow uses. Conditioned on the same gene-set embedding, it learns to map a perturbation to a plausible response population directly. That it matches a JEPA representation plus a learned conditional flow is the finding the rest of this chapter tries to understand.

## The condition formulation gave the flow nothing to work with

The first thing we found on inspection was that the original formulation quietly threw away the flow's structural advantage. The condition is $c = (z_b, z_p)$, a baseline state and an intervention. In the code, for each target cell the baseline $z_b$ was drawn as a *random, unpaired* control-cell latent, sampled independently of the specific outcome. Single-cell perturbation is a destructive assay, so there is no true "same cell before and after" pairing to draw on, which is why the baseline was random. But that independence has a consequence. If $z_b$ carries no information about the target beyond the population it came from, then conditioning on it cannot move the predicted mean of the outcome. The baseline part of the condition is inert for the effect.

Worse, the flow was trained to transport Gaussian noise to the outcome, with $z_b$ entering only as a side input to the velocity field. So the model reduced to learning the distribution of perturbed latents given the perturbation, which is exactly the object the VAE fits. The two-part condition that was meant to make this a baseline-anchored transport had degenerated into an unconditional-on-baseline generator. There was no structural reason the flow should beat the VAE, because it was modeling the same thing by a more elaborate route.

## The transport reformulation, and why it only half-worked

The fix follows directly from the diagnosis. Instead of transporting noise to the outcome with the baseline as a side input, transport a real control latent to the outcome, and let the condition be the intervention alone. The flow's source distribution becomes the control-cell population, its target becomes the perturbed population, and each generated outcome is anchored to a real baseline it flowed from. Formally the generated latent is $z_b$ plus the integral of the learned velocity, so the model now has to represent only the *displacement*, the effect, rather than the whole absolute state. The effect is where the lower-variance signal lives, and it is what the metric scores.

This is well founded even without cell-level pairing. Rectified-flow and optimal-transport flow-matching train transports between two marginal distributions under independent couplings, so pairing a random control to a random perturbed cell of a given perturbation still learns a valid map from the control population to that perturbation's population. The reformulation is enabled in the code by an optional source argument to the flow-matching loss and the sampler, selected by `--flow-base control`.

It helped. As [Chapter 4](04-results.md) shows, transporting from a control latent beats transporting from noise on both axes, at borderline significance after seed averaging. But it did not clear the baseline. Anchoring to a baseline recovers a real advantage the noise formulation had discarded, yet the VAE, whose decoder also reconstructs a full response, was already capturing most of what that advantage buys on these metrics.

## Optimal-transport coupling: a better loss, a worse model

The natural next lever was the coupling. Under independent coupling the displacement target is noisy, because a random control paired with a random perturbed cell can point in an unhelpful direction. Minibatch optimal transport reorders each batch so that every control is paired with its nearest perturbed target, which straightens the flow-matching paths and should give a lower-variance regression target. It did lower the training loss, from around $1.96$ to around $1.48$. And it made the effect-size score significantly *worse*, the only clearly significant effect in the seed sweep.

The lesson is that a lower flow-matching loss is not the objective. Our best reading of the mechanism is that global optimal transport within a batch pairs each perturbed cell to its nearest control across *mixed* perturbations, ignoring which perturbation the target belongs to, and that this over-concentrates the learned map. Straighter paths to nearest neighbors reduce the diversity of control-to-outcome directions the flow sees, and the effect-size metric, which compares population means, is sensitive to that collapse. Coupling by proximity optimizes geometry the metric does not reward.

## Calibration is harder to measure than it looks

We turned to calibration precisely because effect size grades only the mean, and a flow's promise is the full distribution. Measuring it well proved subtle, and the subtleties are themselves findings.

The first attempt generated each cell as the decoder's expected rate for its latent, and asked what fraction of true cells fell inside the model's predicted interval. Coverage came out at zero for every model. The reason is that a population of decoded rates has almost no spread. Real per-cell variation in single-cell data is dominated by technical count noise, and expected rates omit it entirely, so the predicted interval was razor thin and contained no real cells. An expected-rate population cannot be calibrated against real cells.

Sampling actual counts from the negative-binomial fixed that and revealed the opposite problem. Coverage jumped to one, meaning the sampled populations are now too wide on the top differentially-expressed genes. The negative-binomial decoders are over-dispersed there. And because the two flow variants share a decoder, they score almost identically on the marginal metrics, which told us those metrics were reading the decoder rather than the latent distribution. The flow's real edge, if it has one, is in the joint structure across genes, so we added a multivariate energy distance. That metric confirmed the same ranking as effect size, with the VAE best. So the calibration axis, once measured carefully, agreed with the effect-size axis rather than rescuing the flow.

There is a deeper identifiability problem lurking here, which bounds how much any of these metrics can say. A real cell's variation is biological plus technical, the flow controls only the biological part through the latent distribution, and the decoder adds the technical part. Disentangling the two from held-out cells is genuinely hard, and until it is done, marginal calibration will keep reporting mostly on the decoder.

## The statistical-power ceiling

Every combination number rests on twenty held-out combinations. That is few. The paired bootstrap puts the resolution at roughly $0.05$ in $\Delta$-correlation, and several of the differences we care about are that size or smaller. The seed sweep made the point vivid: the same configuration retrained at three seeds swung by more than the transport reformulation's effect, and at one seed the ranking of Gaussian versus transport flow reversed. Single-seed conclusions, including an early one that had the VAE decisively beating the flow, did not survive the added power. The honest consequence is that we can rank configurations only coarsely, and any future claim of the flow overtaking the baseline will need more held-out combinations, more seeds, or ideally more datasets before it can be trusted.

## The structural limitations behind the numbers

Stepping back from the individual fights, three design choices bound what this method can currently show.

The encoder is frozen and condition-blind. It is pretrained on states alone and never sees the intervention, so all conditioning lives downstream in the flow. This buys modularity and a reusable representation, but it means the representation itself is not shaped by the perturbation task, and any advantage from perturbation-aware features is left on the table.

The decoder is a shared bottleneck. The flow and the VAE both read out through a negative-binomial head, and that head is over-dispersed on exactly the genes the metric cares about. When two models share a component that is miscalibrated, differences in the components they do not share get compressed. Improving the decoder could matter more than improving the flow.

And the metrics we can compute reward the mean and the marginals, which are the parts a simple conditional generator already handles well. The flow's distinctive capability, a rich and possibly multimodal joint distribution over the response, is the hardest thing to measure and the least rewarded by the current scoreboard. It is entirely possible the flow is capturing structure the evaluation cannot yet see. That possibility is not a defense of the current results, but it is a clear pointer for what to measure next.

[Chapter 6](06-beyond-the-current-limit.md) takes these limitations as a to-do list and lays out the directions most likely to move the result.

---

*Previous: [Chapter 4 — Results](04-results.md). Up: [the method series](index.md). Next: [Chapter 6 — Beyond the current limit](06-beyond-the-current-limit.md).*
