# Chapter 3a — The models in the head-to-head: Gaussian flow, transport flow, and the VAE

*A companion to [Chapter 3](03-training-and-evaluation.md). [Chapter 4](04-results.md) pits three generative models against each other and [Chapter 4a](04a-reading-the-head-to-head.md) audits the numbers. This chapter is the primer between them: what those three models actually are, why two of them are really one machinery with a single knob, how a flow differs from the VAE baseline, how the comparison is run as a controlled A/B, and the handful of statistical ideas needed to read the result. Nothing here is a number to defend; it is the vocabulary that makes the numbers legible.*

> **Where this sits.** Read [Chapter 3](03-training-and-evaluation.md) first for how the three stages train and how effect size and calibration are scored. This note names the models Chapter 4 compares and the way it compares them, so that the scoreboard and the statistics in [Chapter 4a](04a-reading-the-head-to-head.md) read as confirmation rather than as new machinery. The statistics here are conceptual; Chapter 4a applies them with the actual figures.

## 1. Three models, one metric

Chapter 4 compares three generative models on a single metric, the effect-size $\Delta$-correlation of [Chapter 3](03-training-and-evaluation.md). The three are the **Gaussian flow**, the **transport flow**, and a from-scratch **conditional negative-binomial VAE** used as the baseline. All three end by turning a latent into gene counts through the *same* negative-binomial decoder, so what genuinely separates them is one thing only: how each builds the distribution over the latent that the decoder reads. Get that straight and the whole results chapter is a comparison of three answers to one question.

## 2. Gaussian and transport flow are one machinery

The first simplification is that the two flow variants are not two methods. They are the identical velocity field, interpolant, loss, and sampler, run with a different **source distribution**. Flow matching trains a velocity field $v_\eta(z, t, c)$ to carry a source point $z_0$ to a target latent $z_1$ along the straight path

$$z_t = (1-t) z_0 + t z_1, \qquad u_t = z_1 - z_0,$$

by regressing the network onto the constant target velocity, $\mathcal{L} = \lVert v_\eta(z_t, t, c) - u_t \rVert^2$ ([Chapter 2](02-implementation.md) walks the code). The only choice that distinguishes the two variants is what $z_0$ is, an argument the flow-matching loss and the sampler both take:

- **Gaussian flow** draws $z_0 \sim \mathcal{N}(0, I)$, plain noise, and fuses both parts of the condition, $c = (z_b, z_p)$. The field learns to carry *noise to the outcome latent*. This is the canonical rectified-flow setup, the noise-to-data prior, and the name refers to the Gaussian **source**, not a different algorithm.
- **Transport flow** sets $z_0$ to a *real control-cell latent* and makes the condition the intervention alone, $c = z_p$. The field now carries the *control population to the perturbed population*, so it models the **displacement** the intervention produces, with the baseline anchoring each sample: $z_{\text{outcome}} = z_b + \int v \mathrm{d}t$.

The reason the source matters is the grading. Effect size scores the *change*, and a large intervention-independent baseline dominates a cell's absolute state, so a model that regenerates the whole state from noise spends its capacity on the part the metric ignores. Transporting from a real baseline lets the field model only the small displacement, where the signal is. That is the intuition; whether it actually helps, and by how much, is [Chapter 4](04-results.md)'s to report.

So whenever the results say "Gaussian flow" versus "transport flow," read it as one flow-matching model under two source choices, not as a contest between two architectures.

## 3. A flow versus the VAE

The baseline is a different kind of generative model, and the contrast is the point of having it. Both a flow and a VAE are latent-variable generators, drawing a latent and decoding it, but they construct the distribution over latents in fundamentally different ways.

| | **flow matching** (Gaussian or transport) | **conditional VAE** (baseline) |
|---|---|---|
| how a latent is drawn | integrate an ODE $\dot z = v_\eta(z, t, c)$ over many small steps | one draw from a fixed, *unconditional* prior $z \sim \mathcal{N}(0, I)$; the condition enters later, through the decoder |
| what is trained | a velocity field $v_\eta(z, t, c)$ | an encoder $q(z \mid x)$ over a cell's expression profile $x$, and a decoder $p(x \mid z, c)$ that reads both the latent and the condition, jointly |
| the objective | a plain regression, mean-squared error onto $u_t = z_1 - z_0$; no encoder, no KL, no adversary | the ELBO: a reconstruction term plus $\mathrm{KL}\big(q \Vert \text{prior}\big)$ |
| where the flexibility lives | in the latent: the flow can bend its starting distribution into an arbitrary shape, multimodal and correlated, before anything is decoded | in the decoder: the latent is a plain Gaussian, so any multimodal or correlated structure in the output has to be built by the decoder's nonlinearity |
| density | implicit, no closed form | an explicit lower bound |

The fourth row is easy to overstate, so it is worth being precise about it. A Gaussian prior does **not** force the VAE's generated expression to be Gaussian or unimodal. For a fixed condition, the VAE's output distribution is the decoder's output averaged over every latent the prior can produce, and a nonlinear decoder can make that average multimodal. A decoder that sends one half of the latent space toward one cell fate and the other half toward another turns a single Gaussian into two fates; in general, almost any distribution can be produced by pushing a Gaussian through a suitable map, which is exactly how GANs and normalizing flows generate data. Gene-to-gene correlation arises the same way: the negative-binomial counts are independent across genes for a *fixed* latent, but the latent varies from cell to cell, and genes that respond to the same latent direction move together. Nor is each cell's Gaussian posterior the same thing as the population. Averaged over all cells, those posteriors form a mixture, not a Gaussian. What the Gaussian prior changes is therefore *where* the flexibility has to live, not *what* can be represented. With a simple prior, every sharp boundary between fates must be learned by the decoder, and smooth, weight-decayed networks are biased against sharp boundaries. A flow moves that work into the latent. That is a difference in inductive bias, not a hard limit, and it is the precise sense in which the flow carries a theoretical promise here.

Two facts specific to this project turn that generic contrast into the exact comparison Chapter 4 runs.

First, **the VAE is the no-JEPA control.** Its encoder is a plain multilayer perceptron over the raw expression vector, with its own latent space and a fixed standard-Gaussian prior. It does not use the JEPA encoder at all. The flow, by contrast, models $p(z \mid c)$ over the *frozen JEPA* latent. So putting the two side by side is precisely the test of whether the JEPA representation and the learned flow buy anything over a standard conditional generator built from scratch.

Second, **both read out through the same kind of negative-binomial count head.** Each uses the same likelihood and the same basic design: two hidden layers of width 512, a softmax over genes scaled by the cell's library size to give the expected counts, and one dispersion value per gene. The layer details differ slightly, so the heads are matched rather than identical, but neither model wins by having a better-suited noise model for counts. The two heads are not the same decoder, though, and the difference matters for what the comparison can and cannot isolate.

**What this comparison does and does not isolate.** Two asymmetries separate the models beyond the choice of latent generator.

- **Where the perturbation enters.** The VAE's decoder receives the condition directly, reading $(z, c)$, so it can express a perturbation's effect as a direct shift in expression space. The flow pipeline's decoder reads the latent alone. Every bit of the perturbation's effect has to be written into the latent's *position* by the flow, then read back out by a decoder that never sees which perturbation was applied.
- **Whether the representation adapts to its readout.** The VAE learns its encoder and decoder together, so its latent is shaped to be easy for its own decoder to read. The flow pipeline freezes the JEPA encoder, and its decoder is fitted afterward to latents that were never shaped for it.

Both asymmetries load onto the same step, reading the effect back out of a frozen latent, and that is where later measurements found the loss. The [ceiling analysis](../../../../docs/experimental-method/07-the-ceiling.md) showed that a plain linear readout of the frozen latents scores $0.852$ in effect size, above the VAE, while the trained count decoder given the same real latents scores $0.679$. So the comparison is a fair test of the method *as designed*, which claims that a frozen self-supervised representation plus a flow prior beats a from-scratch generator. It is *not* a clean test of a flow prior against a Gaussian one, because prior type is confounded with how the encoder is trained and with where the condition enters. Freezing was deliberate, since it is what lets a difference between two arms be attributed to the one stage that changed, but it costs exactly this. The [co-adaptation program](../../co-adaptation/README.md) tests the second asymmetry directly by training the encoder jointly with the decoder. A cleaner prior-versus-prior comparison would give the flow pipeline's decoder the condition too, or place a flow prior inside a jointly trained VAE.

This is also why the VAE is the *right* control and not a straw man. It is the simplest standard thing that solves the same task. If the elaborate stack cannot beat it, the elaboration is not, on this metric, earning its place, and that is a conclusion worth being able to state cleanly.

## 4. How the comparison is run: a controlled A/B

To attribute any difference in score to a single design choice, you hold everything else fixed and change only that choice. The Gaussian-versus-transport comparison is run exactly this way, by [`run_flow_compare_pod.sh`](../../run_flow_compare_pod.sh): both arms reuse the same frozen JEPA encoder, the same negative-binomial decoder, the same `combo` split, the same gene-set condition, the same number of epochs, and the same evaluation. The *only* variable that changes is the flow's source, `--flow-base gaussian` versus `--flow-base control`. Because nothing else moves, any gap in the resulting $\Delta$-correlation is attributable to the source choice alone. That is what makes it a controlled A/B test rather than two loosely comparable runs.

Run that way at one fixed random seed, the transport flow scores $0.636$ and the Gaussian flow $0.609$ on the held-out combinations, the cleanest isolation of what the reformulation buys. Averaged over three training seeds, the same two configurations score $0.648$ and $0.612$. But a single seed is only the start of the story, which is where the statistics come in.

## 5. Reading a small-sample comparison: the statistical background

The results rest on twenty held-out combinations, and that is a small sample. A few ideas, stated here in the abstract, are what let [Chapter 4a](04a-reading-the-head-to-head.md)'s tables be read at a glance.

**A point estimate is one number with a wide error bar.** The mean $\Delta$-correlation over twenty combinations is a single figure, but if you had held out a *different* twenty it would land somewhere else. So "model A has the higher mean" is not yet "model A is better." The question is always whether the gap survives the noise, not whether it exists in one run.

**There are two independent noise sources, and both are large here.** One is *which* combinations you happened to hold out, sampling noise. The other is *which random seed* you trained with, optimization noise. For the flow variants, both turn out to be comparable to or larger than the effects of interest, which sit around $0.02$ to $0.04$ in $\Delta$-correlation. When the noise is bigger than the signal, a single number cannot rank the models.

**Seeds can flip a single-seed ranking, so you average over them.** Retraining the very same flow configuration under different random seeds can move its score by more than the differences between flow variants. The transport flow's three seeds span $0.623$ to $0.673$, and at one seed the optimal-transport-coupled flow scores $0.634$ against the plain transport flow's $0.623$, the reverse of their averaged order. A conclusion drawn from one seed can therefore reverse under another. The remedy is to retrain each configuration at several seeds and compare the seed-averaged scores. Averaged that way, the contrasts between flow variants are stable, and the VAE's lead over every flow holds at every seed, since its own three seeds span only $0.762$ to $0.768$.

**A paired bootstrap turns a gap into a verdict.** To ask whether model A really beats model B, compute their difference *combination by combination* first, then resample the twenty combinations with replacement many times and look at the spread of the mean difference. The pairing matters: combinations differ enormously in difficulty, some easy for every model and some hard for every model, and taking the per-combination difference cancels that shared difficulty so the test sees only the disagreement between the two models. If the resulting confidence interval excludes zero, the difference is real; if it straddles zero, the two models are indistinguishable within the noise. Chapter 4a runs exactly this, on all the contrasts at once so that the intervals hold simultaneously, and finds that the VAE's lead over the transport flow is real, that the transport reformulation's gain is real, and that optimal-transport coupling's harm is real.

None of this is special to flows or to biology; it is the ordinary discipline of comparing models on a small test set. Meeting it here, before the numbers, means the numbers arrive already interpretable.

## 6. What to carry into the results

Three models, then, on one metric and one shared decoder. The two flow variants are a single flow-matching machinery differing only in whether they transport from noise or from a real control cell. The VAE is the no-JEPA control, a standard conditional generator that shares the decoder so the comparison isolates the generative model over the latent. The comparison is a controlled A/B, read not off a single run but off seed-averaged scores with a paired bootstrap to separate real gaps from noise. With that vocabulary in hand, Chapter 4's arc, the VAE beating the flow by $0.118$, transport beating the Gaussian source, and optimal-transport coupling hurting, reads as a set of clean, checkable claims rather than a wall of figures.

---

*Previous: [Chapter 3: Training and evaluation](03-training-and-evaluation.md). Up: [the method series](index.md). Next: [Chapter 3b: Reading the calibration metrics](3b-reading-the-calibration-metrics.md).*
