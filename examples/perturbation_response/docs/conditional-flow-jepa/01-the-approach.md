# Chapter 1 — The approach: two gaps, and a conditional flow to close them

*The key ideas behind generating single-cell perturbation responses: what a plain JEPA leaves undone, the two-part fix, and why the fix takes the shape of a conditional flow over frozen latents with a count decoder.*

> **Where this sits.** The [design-space survey](../../../../docs/generative_jepa/index.md) argued from theory that the first serious method to build is a conditional flow prior with a count decoder over a JEPA representation. The [Reading Perturb-seq](../reading-perturb-seq/index.md) series grounds the biology in the real Norman 2019 data. This chapter is the bridge: it explains the approach in words and equations, so that [Chapter 2](02-implementation.md) can walk through the code without re-arguing why the code looks the way it does. No results here. Those are [Chapters 4](04-results.md) and [5](05-challenges-and-limitations.md).

## 1. The problem, and how it is graded

We want to predict how a single cell's gene expression responds when we intervene on its genome. The intervention is a CRISPR activation that switches on one gene, or two genes at once. Give the model a control cell and a target intervention, and it should tell us what the perturbed transcriptome looks like.

The subtlety is in how the field grades that prediction, and it decides everything downstream. Success is not measured by how well the model reconstructs the perturbed cell's absolute expression state. It is measured by the **effect size**: the change the intervention produces relative to unperturbed control. Concretely, take the mean expression of a population of perturbed cells and subtract the mean expression of control cells, gene by gene:

$$
\Delta = \operatorname{mean}(\text{perturbed}) - \operatorname{mean}(\text{control}).
$$

$\Delta$ is a vector with one entry per gene, the *differential expression* of the perturbation. The score that matters is the agreement between the predicted $\Delta$ and the true $\Delta$, computed on the top differentially-expressed genes, the genes the intervention actually moved.

This grading is not a technicality. A cell's absolute state is dominated by a large, intervention-independent baseline that the cell already carried before we touched it. The effect of a single-gene activation is a comparatively small shift riding on top of that baseline. A model can reproduce the after-state beautifully by nailing the baseline and still get the change badly wrong, which is exactly the part the benchmark cares about. The [design-space survey calls this baseline dominance](../../../../docs/generative_jepa/05-two-gaps-four-routes.md), and it is why reconstructing absolute state is not enough. The field grades the change, so the method must be built to recover the change.

## 2. Why a plain JEPA is not enough — two gaps

JEPA gives us a strong encoder. Trained self-supervised on cell states, it maps a cell to a latent that is semantically organized and robust. That is a genuinely good starting substrate. But an encoder is not a generative model, and for this task it leaves two distinct things undone. The [design-space survey](../../../../docs/generative_jepa/05-two-gaps-four-routes.md) names them G1 and G2, and the important point is that they are independent: closing one does nothing for the other.

> **Gap G1 — a point latent, but the answer is a distribution.** A JEPA encoder, and the deterministic predictor bolted on top of it, returns exactly one latent per input. One cell in, one latent out. But a perturbation does not produce one outcome. Identical cells given the identical intervention respond differently, and the responding population is sometimes multimodal, splitting into two distinct cell fates. What we need is not a single predicted latent but a *distribution* over outcome latents, a whole cloud we can draw a population from. We need a generative model over latents, not a point.

> **Gap G2 — a latent is not data.** JEPA's loss lives entirely in latent space: it predicts an embedding and matches it to a target embedding. There is no map from a latent back to gene counts. And effect size lives in gene counts. You cannot subtract two latents and read a differential-expression vector off the result in units the benchmark understands. To recover $\Delta$ we need a *decoder* that turns a latent into an expression profile.

A worked contrast makes the independence concrete. Suppose we close only G2, bolting a decoder onto the single predicted latent. We get one expression profile per intervention, a point estimate wearing a generative model's clothes, with no population to compute a proper $\Delta$ from. Now suppose we close only G1, learning a distribution over latents. We can sample many plausible cell-state latents, but none of them is data, so there is still no gene-count number to score. A usable method has to close both.

## 3. The answer — a G1/G2 decomposition

The approach is exactly this pair of closures, one mechanism for each gap.

**G1 is closed by a conditional flow prior.** We learn a generative model $p(z \mid c)$ over the JEPA latent $z$, conditioned on a description $c$ of the perturbation we are asking about. The model is realized as a *rectified-flow velocity field* $v_\eta(z, t, c)$, a network with weights $\eta$ that takes a latent position $z$, a time $t \in [0, 1]$, and the condition $c$. To draw an outcome, we start from a noise sample and follow the velocity field's arrows through time until we land on a latent. Different noise draws land on different latents, so a thousand draws through the same $c$ simulate a thousand cells from the responding population. Because a flow can bend a noise cloud into almost any shape, that population can be genuinely multimodal, which a single Gaussian could never represent.

**G2 is closed by a negative-binomial count decoder.** We learn a decoder that maps a sampled latent $z$ to gene counts, so the model emits actual data and effect sizes become recoverable. The decoder uses a negative-binomial likelihood rather than a Gaussian one because that is what the data are. Single-cell RNA-seq measurements are counts, and they are *overdispersed*: their variance grows faster than their mean, more than a Poisson would allow. The [Reading Perturb-seq series](../reading-perturb-seq/index.md) develops why the negative binomial is the right noise model for this count structure. Here we simply inherit that conclusion and put an NB head on the decoder.

Put the two together and the full generative story reads left to right. Fix a condition $c$. Draw noise. Integrate it through the conditional flow to a latent $z^* \sim p(z \mid c)$. Decode $z^*$ to a gene-count profile. Repeat with fresh noise to build a population, then take its mean, subtract the control mean, and read off the predicted $\Delta$.

## 4. What the condition is

Everything the flow needs to know about the question lives in the condition $c$. It has two parts:

$$
c = (z_b, z_p).
$$

$z_b$ is the **baseline state**: the latent of a control cell, encoded by the frozen JEPA encoder. It is the "before", the state the intervention acts on. Carrying the baseline in the condition is what lets the generated outcome be anchored to a specific starting cell rather than to the average cell.

$z_p$ is the **intervention**: an embedding of which gene or genes were activated. It is the "what we did". A naive way to build $z_p$ is a lookup table with one learned vector per perturbation, but that cannot represent a combination the model never saw during training. A better route builds $z_p$ compositionally from per-gene parts, so that a two-gene activation is assembled from its single-gene pieces and an unseen combination still has a vector. [Chapter 2](02-implementation.md) details both encoders and why the compositional one matters for generalization.

## 5. Why a flow

A flow is one choice among several for closing G1, and the [design-space survey devotes a full chapter to the argument](../../../../docs/generative_jepa/09-conditional-flow-prior.md). Three properties make it the one this series builds.

It is **expressive**. Rectified flow learns an arbitrary noise-to-latent transport, so it can bend a Gaussian noise cloud into a multimodal, correlated population that a single diagonal Gaussian cannot. That directly matches the two-fates structure a real perturbation response can have.

It has a **dead-simple objective**. Training is one mean-squared regression: predict the velocity that carries noise toward data. There is no adversary, no sampling step buried inside the loss, and no noise schedule to tune. Among expressive generators it is about the gentlest to train.

It has **near-straight sampling paths**. Rectified flow's training targets are straight lines in time, so the learned transport stays close to straight, and a near-straight trajectory needs only a handful of integration steps to sample. The cost of this expressiveness, as everywhere, is that the flow gives an implicit distribution with no closed-form density. [Design-space Part 9](../../../../docs/generative_jepa/09-conditional-flow-prior.md) makes the full case and is honest about that trade.

## 6. The three-stage build

The decomposition dictates the build. Three components are trained, and the order follows the G1/G2 story rather than a strict data-flow order.

**Stage A — pretrain the JEPA encoder.** Train the encoder self-supervised on cell states alone, with no knowledge of any intervention, then *freeze* it. From that point on it only produces latents. It is never touched again by a generative gradient.

**Stage C — train the negative-binomial decoder.** On the frozen latents, fit the count decoder that maps a latent back to gene counts. This closes G2.

**Stage B — train the conditional flow.** Also on the frozen latents, fit the velocity field $v_\eta(z, t, c)$ so that integrating noise under condition $c$ lands on the distribution of perturbed-cell latents for that baseline and intervention. This closes G1.

Freezing the encoder is a deliberate design choice, not a shortcut. It keeps the representation a pure, reusable, modular substrate that no generative gradient can disturb, so the decoder and the flow are both trained against a fixed target. The [design-space survey discusses the freeze-versus-joint dial](../../../../docs/generative_jepa/09-conditional-flow-prior.md) and where giving that up might pay off. This series starts from the frozen, decoupled end.

One naming note, flagged once so it does not confuse you later. The A/B/C letters above follow the *conceptual* order of the G1/G2 story: encoder first, then the decoder that closes G2, then the flow that closes G1. The *pipeline scripts* in the example folder are numbered by their own convention, roughly `01` for the encoder, `03` for the decoder, and `04` for the flow. So the script numbers do not line up with the A/B/C letters. When [Chapter 2](02-implementation.md) walks the code and [Chapter 3](03-training-and-evaluation.md) walks the workflow, read the letters as the idea and the numbers as the files. They describe the same three stages.

---

*Up: [the method series](index.md). Next: [Chapter 2 — Implementation](02-implementation.md).*
