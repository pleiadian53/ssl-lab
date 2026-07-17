# Modeling the Readout: Count Decoders for Perturbation Response

*The decoder is where a latent becomes counts and where the effect is finally measured. Two of its three jobs are already right, the third points the opposite way from intuition, and following it honestly cuts the decoder lever down to its true size.*

> **Recap: where this sits.** [Beyond the current limit](06-beyond-the-current-limit.md) nominated two levers for closing the gap to the from-scratch NB-VAE, which beats the stack by $0.118$. One was *modeling the transition*: how a perturbation moves a cell's latent. [Modeling the transition](07-modeling-the-transition-action-operators.md) opened that thread with the action operator. The other lever is *modeling the readout*: how a latent becomes gene counts, and how that measurement can quietly corrupt or clarify the score. This chapter opens that second thread. It is the decoder handoff that [the operator chapter](07-modeling-the-transition-action-operators.md) named but deferred.
>
> The two levers are complementary, but they are not equal partners. The operator makes the *transition* model the change directly. The readout work here makes the *measurement* honest, so that an improvement in the latent distribution can reach the scoreboard at all. Only the first of those is a route to a better model, and one job of this chapter is to say exactly why.

> **Prerequisites and notation.** We reuse the `CountDecoder` from [Implementation](02-implementation.md), the per-gene effect vector $\Delta$ and the calibration metrics from [Results](04-results.md), and the diagnosis of the decoder as a shared, under-dispersed bottleneck from [Challenges and limitations](05-challenges-and-limitations.md). Every symbol is defined on first use.

---

## 1. The idea in one line

The metric grades the *change*, and the decoder is where that change is measured. If the readout spends its capacity fitting the baseline instead of the deviation, or reports a spread that does not match the spread of real cells, then the transition can be modeled perfectly and the score will still not move.

So the decoder is a lever in its own right. But it is a precise lever, and a smaller one than it looks. It has three knobs, and they do not all act on the axis you would guess. One knob governs effect size, a second governs calibration, and a third is already set correctly. Knowing which is which is what keeps a decoder change from being wasted effort. Knowing how far each one reaches, which is where the chapter ends up, is what keeps a decoder change from quietly making the real problem worse.

---

## 2. The decoder as it stands, and its three knobs

Recall the decoder from [Chapter 2](02-implementation.md), stated in its own symbols. A latent $z \in \mathbb{R}^{256}$ runs through a small network to a **relative gene-rate profile**

$$
\rho = \mathrm{softmax}(\mathrm{net}(z)), \qquad \sum_{g=1}^{G} \rho_g = 1,
$$

where $G$ is the number of genes and $\rho_g$ is the fraction of a cell's transcripts assigned to gene $g$. The profile lives on the probability simplex: its entries are non-negative and sum to one. The **negative-binomial mean** is then

$$
\mu = \ell \cdot \rho,
$$

with $\ell$ the cell's library size, a given covariate rather than a prediction. The **dispersion** is one learned scalar per gene,

$$
\kappa = \mathrm{softplus}(\texttt{log\_kappa}) + 10^{-4}, \qquad \texttt{log\_kappa} \in \mathbb{R}^{G},
$$

held in a single parameter vector shared across every cell and every condition, and the count of gene $g$ is drawn as $x_g \sim \mathrm{NB}(\mu_g, \kappa_g)$ with variance $\mu_g + \mu_g^2 / \kappa_g$. Small $\kappa_g$ means heavy over-dispersion, and large $\kappa_g$ approaches Poisson, where the variance collapses to the mean. Hold on to that direction, because §5 shows the fitted $\kappa$ sits too far toward the Poisson end.

Three knobs govern this decoder, and the rest of the chapter is organized around them:

- **The mean head**, meaning how $\rho$ is parameterized. Today it is a bare softmax over an unconstrained network output.
- **The dispersion**, meaning how $\kappa$ is set. Today it is one constant per gene, independent of the cell and the perturbation.
- **The readout**, meaning which decoded quantity the metric actually consumes. This is not a network choice but an evaluation choice, and it is the one already set correctly.

[Chapter 5](05-challenges-and-limitations.md) called the decoder a shared, under-dispersed bottleneck. That is true, but the phrase hides a distinction the next section draws out, because the three knobs do not act on the same axis.

---

## 3. The readout is already clean, and that relocates the problem

The decoder exposes two readouts, and [Chapter 2](02-implementation.md) wired each to the axis it belongs to. The effect-size metric consumes `predicted_expression`, the population mean of the rate profile,

$$
\widehat{\text{expr}} = \frac{1}{n} \sum_{i=1}^{n} \mathrm{log1p}\big(10^4 \cdot \rho^{(i)}\big),
$$

computed over $n$ generated cells with **no count sampling**, because $\rho$ is library-free and already a clean estimate of the mean response. The calibration metrics instead consume `predicted_population`, which does draw integer counts through the negative binomial, because per-cell spread is real technical noise that a population of bare rates does not carry.

This split has a consequence that reorganizes the whole decoder discussion. The dispersion $\kappa$ enters the *sampling* path only. It never touches the effect-size number, because effect size reads the mean rate and never samples. So on the axis where the loss to the NB-VAE was actually measured, the dispersion is a red herring. Moving $\kappa$ in either direction moves the coverage numbers and leaves the $\Delta$-correlation exactly where it was.

The binding decoder component is therefore different for each axis:

- For **effect size**, the mean head is the component the score responds to, and the dispersion never enters it. The mean head *attenuates* the score rather than gating it: it does not cap what is achievable, since the reparameterization below is the same function class, but it decides how readily the fit resolves the scored genes.
- For **calibration**, the dispersion is the decoder's only knob and the mean head barely enters it. Here the readout does impose a genuine cap. No amount of fitting the mean head will make a generated population cover the real cells if the counts drawn around each mean are spread too tightly. What §5 adds is that the readout supplies only one of the *two* sources of spread the metric sees, and it is not the one whose growth would tell us anything.

This refines the "shared bottleneck" reading into something you can act on. It also settles a loose end from [the operator chapter](07-modeling-the-transition-action-operators.md), which listed "state-dependent dispersion, scoring $\Delta$ on the expected rate" as prerequisites for seeing an operator improvement. Scoring $\Delta$ on the expected rate is already done. The handoff reduces to two targeted changes: an identity-anchored mean head for effect size, and a dispersion raised, carefully, toward the spread the sampled counts should actually be carrying. The rest of the chapter takes them in that order, and then asks how much the second one is really worth.

---

## 4. Lever one: an identity-anchored mean head

The mean head is the effect-size lever, and its weakness is the simplex it lives on.

Because $\rho = \mathrm{softmax}(\cdot)$ sums to one, all $G$ genes compete for a single fixed unit of probability mass. Raising a low-abundance gene's rate means taking mass away from the housekeeping genes that hold most of it. That is exactly the wrong coupling for this task, because the genes the metric scores are the top differentially-expressed ones, and those are often low in absolute abundance yet moved by the perturbation. The simplex makes the decoder fight its own normalization to resolve the very genes it is graded on.

The aligned reparameterization is to predict a **log-fold-change on a learned baseline profile**. Let $\rho_{\text{base}} \in \mathbb{R}^{G}$ be a learned control-cell rate profile on the simplex, and let $\delta(z) \in \mathbb{R}^{G}$ be a per-gene deviation head. Set

$$
\rho \propto \rho_{\text{base}} \odot \exp\big(\delta(z)\big), \qquad \rho = \mathrm{softmax}\big(\log \rho_{\text{base}} + \delta(z)\big),
$$

where $\odot$ is the elementwise product. The two forms are the same object, which is worth being honest about: this is not a more expressive decoder. A full linear $\delta$ head can absorb $\log \rho_{\text{base}}$ into its own bias, so the function class is unchanged.

What changes is the inductive bias and the conditioning. Initialize $\delta$ at zero. Then the decoder returns the baseline profile $\rho_{\text{base}}$ before it has learned anything, so the readout is *anchored at no effect* and the network only has to learn the deviation the perturbation causes. This is the readout-side twin of the operator's near-identity initialization: there the transition starts at "do nothing" and earns each departure from identity; here the readout starts at "the control profile" and earns each departure from baseline. Both encode the same fact about the data, that effects are small shifts on a large, intervention-independent baseline, and both put that fact where the gradient can use it rather than leaving the network to rediscover it.

There is a more aggressive version worth naming, with a real trade. The simplex is what couples the genes; dropping it and predicting an independent per-gene log-mean would decouple them entirely, so raising one gene no longer costs another. The price is that $\mu = \ell \rho$ no longer separates sequencing depth from expression shape, which is the property that keeps cells comparable across library sizes. The log-fold-change reparameterization is the conservative move that keeps the depth-shape separation, and the decoupled per-gene rate is the further step for when gene competition is demonstrably the binding constraint.

---

## 5. Lever two: raising the dispersion, and what that cannot buy

The dispersion is the calibration lever, and it points the opposite way from where a first guess lands. It is also the place where the decoder story stops being about the decoder.

Start with the measured symptom, on the genes the metric scores. From [Chapter 4](04-results.md), coverage sits between $0.33$ and $0.38$ against a nominal $0.80$: the predicted $80\%$ interval captures only about a third of the real held-out cells. The generated populations are **too narrow**, not too wide, and this holds for the flow and the VAE alike. Every model is over-confident. In the decoder's own symbols that means $\kappa$ is too *large*, sitting too near the Poisson limit, so the sampling path adds too little count noise around each cell's mean. The direction to push is **more** over-dispersion, which is to say a smaller $\kappa$.

How much more? The answer is not a matter of taste, because a piece of exact algebra names the target. Both models generate a cell in the same two steps: draw a latent from the model's own distribution, $z \sim p(z \mid \text{pert})$, then draw counts from the decoder, $x \mid z \sim \mathrm{NB}(\mu(z), \kappa)$. The law of total variance splits the per-gene variance of the resulting population into exactly two pieces, one contributed by each step:

$$\underbrace{\mathrm{Var}[x_g]}_{\sigma^2_{\text{obs}}} = \underbrace{\mathbb{E}_z\big[\mu_g(z) + \mu_g(z)^2/\kappa_g\big]}_{\sigma^2_{\text{dec}}} + \underbrace{\mathrm{Var}_z\big[\mu_g(z)\big]}_{\sigma^2_{\text{bio}}}.$$

This is an identity, not a modeling approximation, and it assigns every unit of observed spread to an owner. Here $x_g$ is the count of gene $g$, $\mu_g(z)$ is the mean the decoder emits for a cell at latent $z$, and $\kappa_g$ is that gene's dispersion. The first term, $\sigma^2_{\text{dec}}$, is the count noise the decoder adds around each cell's own mean, averaged over the latent cloud. It belongs to the shared readout and it is the *only* thing the dispersion knob controls. The second term, $\sigma^2_{\text{bio}} = \mathrm{Var}_z[\mu_g(z)]$, is the variance of the decoded *mean* across the cloud: cells differ from one another because the model placed them at different latents. That term is the latent distribution's entire contribution to the readout, and it is precisely the part the flow and the VAE do differently.

Measured on the transport flow, over the genes the metric scores:

| | real variance | predicted total | $\sigma^2_{\text{dec}}$ (decoder) | $\sigma^2_{\text{bio}}$ (latent) | latent's share |
|---|---|---|---|---|---|
| transport flow | 0.824 | 0.678 (0.84$\times$) | 0.538 | 0.140 | 22% |
| NB-VAE | 0.824 | 0.355 (0.46$\times$) | 0.226 | 0.128 | 38% |

Two things fall out at once. The flow produces only $0.84$ of the spread it should, which is the under-dispersion restated as a number. And the decoder owns $78\%$ of the spread it does produce, outweighing the latent distribution by nearly four to one. **That ratio is the compression.** The metric sees $\sigma^2_{\text{obs}}$, the shared readout dominates it, and so two models with genuinely different latent distributions land in nearly the same place. Note also that the two models' *latent* contributions are almost equal, $0.140$ for the flow against $0.128$ for the VAE, so what separates them on this axis is not their latent distributions at all. It is the decoder each of them happened to learn.

The identity also names the residual the dispersion knob should be aiming at. Leave the latent cloud as it is, and the honest target for the readout is

$$\sigma^2_{\text{dec}} \longrightarrow \mathrm{Var}[x_g] - \sigma^2_{\text{bio}} = 0.824 - 0.140 = 0.684,$$

against the $0.538$ it currently delivers. That is a concrete, measurable ask: fit $\kappa$ so the sampled counts carry the variance the latent cloud does not already account for, no more and no less.

### The two ways to close the shortfall, and why only one of them helps

The total shortfall is $0.824 - 0.678 = 0.146$ of missing variance. The identity has exactly two terms, so there are exactly two places to find it. They repair coverage *equally well*, and they have **opposite** consequences for everything this project cares about. [Chapter 6](06-beyond-the-current-limit.md) develops the argument in full; here it is in the decoder's own terms.

**Fix A, give the decoder more noise.** Raise $\sigma^2_{\text{dec}}$ from $0.538$ to $0.684$, which is exactly the residual target above, and the total lands on $0.824$. Coverage is repaired. But the decoder's share of the spread rises from $78\%$ to $83\%$, and the latent's share falls from $22\%$ to $17\%$. The shared component now speaks even louder than it did. A better flow would be *harder* to detect after the fix than before it. You would have repaired the metric and deepened the very problem the metric was supposed to reveal.

**Fix B, give the latent distribution more spread.** Raise $\sigma^2_{\text{bio}}$ from $0.140$ to $0.286$, leave the decoder alone, and the total lands on $0.824$ just the same. Coverage is repaired, and the latent's share rises to $35\%$. The generative model becomes a larger fraction of what the metric sees, which is the only way its quality can ever register.

Both fixes produce identical coverage, and only one of them makes the flow legible. That reframes the entire lever: **the calibration failure is not primarily a readout problem, it is a generative one.** The latent cloud the flow produces is simply too tight. Its decoded means do not vary enough from cell to cell to account for the biological heterogeneity that is actually there, and no setting of $\kappa$ can manufacture that variety, because the count head sits downstream of where the variety would have to come from. Widening the cloud is the transition lever's job, and [Chapter 7](07-modeling-the-transition-action-operators.md)'s operator, with its dialable residual velocity field, is the most direct way to do it in a structured way rather than merely a noisier one.

### The credit-assignment trap, and why any dispersion flexibility needs an anchor

Fix A is not only something you might choose. It is something maximum likelihood will choose *for* you if you let it, and that is the risk that governs how the dispersion is allowed to move.

The direct way to raise the dispersion is to let it depend on the state rather than sit as one constant per gene. Either make it a small head on the decoder trunk, $\kappa = \mathrm{softplus}(\texttt{kappa\_head}(h))$ with $h$ the trunk's hidden activation, so dispersion can vary per cell and per condition, or tie it to the mean through a learned mean-dispersion trend $\kappa_g = f(\mu_g)$, which follows the empirical relationship that higher-expressed genes are relatively less over-dispersed. Both give $\kappa$ the freedom to be right in more places than one shared scalar can.

But a flexible $\kappa$ fitted by free maximum likelihood opens a trap, and it is the identifiability concern from [Chapter 5](05-challenges-and-limitations.md) made precise. A real cell's variation is biological plus technical. The biological part is what the latent distribution is supposed to own, whether that distribution comes from the flow or from the operator. The technical part is what the decoder's count noise is supposed to add. Free maximum likelihood has no reason whatsoever to respect that division. It will happily grow $\sigma^2_{\text{dec}}$ to explain variation that $\sigma^2_{\text{bio}}$ should have carried, because the likelihood cannot tell the two apart and only sees the sum. That is Fix A arrived at by gradient descent rather than by decision, and it is worse than choosing Fix A knowingly, because it can also *shrink* the latent's contribution: a decoder that absorbs biological variance strips the generative model of the very spread it exists to model, and improves calibration while doing it.

So the fix has to come with an anchor. Rather than let maximum likelihood place $\kappa$ freely, fit it by moment-matching against the observed per-gene variance, or regularize it toward the residual the identity names, so the decoder is allowed to model the spread the latent cloud does not account for and no more. Decoder flexibility and generative credit are in genuine tension here, and the anchor is what keeps them separable. This is the one place in the decoder where a change that lowers the training loss can silently defeat the purpose of the whole stack.

Which leaves the decoder lever with an honest and rather narrow scope. Raise the dispersion, under an anchor, so the readout stops being over-confident about a population it never had the right to be confident about. Anchor the mean head so it models the deviation rather than the whole absolute profile. Then stop. Both changes are worth making, and neither is the main event.

---

## 6. Two guardrails

Two smaller cautions keep the decoder work from overreaching.

**Reweight the loss only by answer-agnostic signals.** It is tempting to upweight the per-gene negative log-likelihood on the responsive genes so the fit does not neglect them. That is fine only if the weighting is computed from something the model is allowed to know, such as a gene's overall variability across all cells. Weighting by the *known* differential-expression identity of the test perturbations leaks the graded answer into training, and any gain from it is an artifact rather than a method.

**Do not reflexively enable ZINB.** The decoder supports a zero-inflated variant, and dropout-heavy single-cell data makes it tempting. Modern UMI-based counts are generally not zero-inflated, though, and switching on the dropout gate can *mask* a dispersion problem rather than fix it, by explaining away zeros that a correctly-dispersed negative binomial would produce on its own. That failure mode is especially easy to walk into here, where the decoder is already producing too little spread and a gate that eats zeros will look, on the loss curve, like progress. The code leaves ZINB optional for this reason. Leave it off unless the zeros in the specific dataset genuinely demand it.

---

## 7. How the two levers compose, and what neither of them can do

The two levers sit at two stages and act on two axes, and reading them together tells you both the order to pull them and where the ceiling is.

The **mean head** sharpens effect size directly, by anchoring the readout at the baseline so the decoder models the deviation the metric scores. It touches the axis on which the loss was measured, it costs nothing in expressiveness, and it can land immediately, alongside the operator, without waiting on anything else. Of everything in this chapter, it is the change most likely to move the number the project is graded on.

The **anchored dispersion** makes the readout honest. It is a correction, not an improvement: it stops the model claiming a confidence it has not earned, which is a real thing to fix, since a coverage of $0.35$ against a nominal $0.80$ means the calibration column is currently reporting on a population no one should trust. But it must be pulled with its own limits in view. Raising $\sigma^2_{\text{dec}}$ to the residual target repairs coverage and, by the arithmetic of §5, *lowers* the latent's share of the spread from $22\%$ to $17\%$. That is Fix A. It buys an honest readout at the price of making the generative model an even smaller fraction of what the metric sees. So the dispersion fix does not make the flow visible, and it should never be advertised as though it does.

What actually has to happen is $\sigma^2_{\text{bio}}$ growing, and that is not a decoder change at all. It is the transition lever's job. The operator of [Chapter 7](07-modeling-the-transition-action-operators.md), whose residual velocity field is dialable, is the direct route to a latent cloud that is wider in a structured way rather than merely a noisier one. So the ordering is: the mean head now, on the effect-size axis, where it stands on its own; the dispersion as a bounded, anchored correction that makes the calibration column mean something; and the operator as the one lever aimed at the quantity that actually distinguishes one generative model from another. If a distributional win against the baseline ever arrives, it will arrive through $\sigma^2_{\text{bio}}$, and the decoder's contribution will have been to make it visible rather than to produce it.

The throughline that connects [the transition chapter](07-modeling-the-transition-action-operators.md) to this one is a single sentence in two halves. **Model the change**, with a transition that starts at identity and a readout that starts at baseline; and **measure it without corrupting it**, with a mean readout that carries no sampling noise and a spread that carries only the variance the latent distribution does not own. The operator earns each departure from identity. The mean head earns each departure from baseline. The anchored dispersion refuses to take credit for biology. Each is the same discipline applied at a different stage, and the last of the three is the one that keeps the other two measurable.

---

## 8. The generalization, in one note

The readout lever generalizes the same way the operator does. Nothing in the identity-anchored mean head or the variance-anchored dispersion is specific to a single control-to-perturbed transition. Any count or rate readout in a latent world model benefits from a mean that anchors at the previous state's profile and a dispersion that models only measurement noise. The cell case decodes one latent; a temporal rollout decodes each latent it steps to, through the same head. The readout is the $T$-agnostic corner of the same construction the operator chapter laid out for the transition, which is what keeps the cell-first build one dial-turn away from the temporal world model on the decoder side as well as the operator side. The law of total variance is $T$-agnostic too, and the same trap waits at every horizon: a decoder given free rein will absorb the dynamics it was supposed to read out.

> **Throughline.** The decoder is not one bottleneck but three knobs on two axes. Scoring the mean rate is already right, so the effect-size lever is an identity-anchored mean head, and the calibration lever is a dispersion raised, under an anchor, toward the variance the latent cloud does not supply. That second lever repairs the readout and, on its own, makes the generative model *less* visible rather than more. The variance that would make a flow worth having has to come from the latent distribution, which is the operator's job, and the decoder's honest role is to stop distorting the measurement of it.

---

*Previous: [Modeling the transition: action operators](07-modeling-the-transition-action-operators.md). Next: [Why the operator is linear — the Koopman argument](09-why-the-operator-is-linear-koopman.md), which resumes the transition thread with the first of the four questions Chapter 7 deferred. Up: [the method series](index.md). The generator basis and two-gene composition still follow; this chapter answers the decoder handoff those chapters depend on.*
