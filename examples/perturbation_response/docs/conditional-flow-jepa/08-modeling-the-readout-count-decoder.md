# Modeling the Readout: Count Decoders for Perturbation Response

*The decoder is where a latent becomes counts and where the effect is finally measured. Two of its three jobs are already right, and the third decides a different axis than intuition suggests.*

> **Recap — where this sits.** [Beyond the current limit](06-beyond-the-current-limit.md) nominated two levers for pushing past the tie against the from-scratch NB-VAE. One was *modeling the transition*: how a perturbation moves a cell's latent. [Modeling the transition](07-modeling-the-transition-action-operators.md) opened that thread with the action operator. The other lever is *modeling the readout*: how a latent becomes gene counts, and how that measurement can quietly corrupt or clarify the score. This chapter opens that second thread. It is the decoder handoff that [the operator chapter](07-modeling-the-transition-action-operators.md) named but deferred, and it turns out to need one correction to how the earlier chapters described the decoder.
>
> The two levers are complementary and they compose. The operator makes the *transition* model the change directly; the readout fixes here make the *measurement* trustworthy, so that an operator improvement in the latent distribution can actually reach the scoreboard.

> **Prerequisites and notation.** We reuse the `CountDecoder` from [Implementation](02-implementation.md), the per-gene effect vector $\Delta$ and the calibration metrics from [Results](04-results.md), and the diagnosis of the decoder as an over-dispersed shared bottleneck from [Challenges and limitations](05-challenges-and-limitations.md). Every symbol is defined on first use.

---

## 1. The idea in one line

The metric grades the *change*, and the decoder is where that change is measured. If the readout injects noise into the measurement, or spends its capacity fitting the baseline instead of the deviation, then the transition can be modeled perfectly and the score will still not move.

So the decoder is a lever in its own right. But it is a precise lever, not a blanket one. It has three knobs, and the surprise this chapter delivers is that they do not all act on the axis you would guess. One knob governs effect size, a second governs calibration, and a third is already set correctly. Knowing which is which is what keeps a decoder change from being wasted effort.

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

held in a single parameter vector shared across every cell and every condition, and the count of gene $g$ is drawn as $x_g \sim \mathrm{NB}(\mu_g, \kappa_g)$ with variance $\mu_g + \mu_g^2 / \kappa_g$. Small $\kappa_g$ means heavy over-dispersion; large $\kappa_g$ approaches Poisson.

Three knobs govern this decoder, and the rest of the chapter is organized around them:

- **The mean head** — how $\rho$ is parameterized. Today it is a bare softmax over an unconstrained network output.
- **The dispersion** — how $\kappa$ is set. Today it is one constant per gene, independent of the cell and the perturbation.
- **The readout** — which decoded quantity the metric actually consumes. This is not a network choice but an evaluation choice, and it is the one already set correctly.

[Chapter 5](05-challenges-and-limitations.md) called the decoder a shared, over-dispersed bottleneck. That is true, but the phrase hides a distinction the next section draws out, because the three knobs do not act on the same axis.

---

## 3. The readout is already clean, and that relocates the problem

The decoder exposes two readouts, and [Chapter 2](02-implementation.md) wired each to the axis it belongs to. The effect-size metric consumes `predicted_expression`, the population mean of the rate profile,

$$
\widehat{\text{expr}} = \frac{1}{n} \sum_{i=1}^{n} \mathrm{log1p}\big(10^4 \cdot \rho^{(i)}\big),
$$

computed over $n$ generated cells with **no count sampling**, because $\rho$ is library-free and already a clean estimate of the mean response. The calibration metrics instead consume `predicted_population`, which does draw integer counts through the negative binomial, because per-cell spread is real technical noise that a population of bare rates does not carry.

This split has a consequence that reorganizes the whole decoder discussion. The over-dispersed dispersion $\kappa$ enters the *sampling* path only. It never touches the effect-size number, because effect size reads the mean rate and never samples. So on the axis where the tie against the NB-VAE was actually measured, the dispersion is a red herring. Widening or tightening $\kappa$ moves the coverage-of-one calibration symptom and leaves $\Delta$-correlation untouched.

The binding decoder component is therefore different for each axis:

- For **effect size**, the mean head is the component the score responds to, and the dispersion never enters it. The mean head *attenuates* the score rather than gating it: it does not cap what is achievable, since the reparameterization below is the same function class, but it decides how readily the fit resolves the scored genes.
- For **calibration**, the dispersion is what limits the score, and the mean head barely enters it. Here the limit is a genuine cap: an over-dispersed decoder cannot be made to cover correctly by any amount of fitting elsewhere.

This refines the earlier "shared bottleneck" reading into something you can act on. It also settles a loose end from [the operator chapter](07-modeling-the-transition-action-operators.md), which listed "state-dependent dispersion, scoring $\Delta$ on the expected rate" as prerequisites for seeing an operator improvement. Scoring $\Delta$ on the expected rate is already done. The handoff reduces to two targeted changes: an identity-anchored mean head for effect size, and an anchored dispersion for calibration. The rest of the chapter takes them in that order.

---

## 4. Lever one — an identity-anchored mean head

The mean head is the effect-size lever, and its weakness is the simplex it lives on.

Because $\rho = \mathrm{softmax}(\cdot)$ sums to one, all $G$ genes compete for a single fixed unit of probability mass. Raising a low-abundance gene's rate means taking mass away from the housekeeping genes that hold most of it. That is exactly the wrong coupling for this task, because the genes the metric scores are the top differentially-expressed ones, and those are often low in absolute abundance yet moved by the perturbation. The simplex makes the decoder fight its own normalization to resolve the very genes it is graded on.

The aligned reparameterization is to predict a **log-fold-change on a learned baseline profile**. Let $\rho_{\text{base}} \in \mathbb{R}^{G}$ be a learned control-cell rate profile on the simplex, and let $\delta(z) \in \mathbb{R}^{G}$ be a per-gene deviation head. Set

$$
\rho \propto \rho_{\text{base}} \odot \exp\big(\delta(z)\big), \qquad \rho = \mathrm{softmax}\big(\log \rho_{\text{base}} + \delta(z)\big),
$$

where $\odot$ is the elementwise product. The two forms are the same object, which is worth being honest about: this is not a more expressive decoder. A full linear $\delta$ head can absorb $\log \rho_{\text{base}}$ into its own bias, so the function class is unchanged.

What changes is the inductive bias and the conditioning. When $\delta(z) = 0$ the decoder returns the baseline profile $\rho_{\text{base}}$, so the readout is *anchored at no effect* and the network only has to learn the deviation the perturbation causes. This is the readout-side twin of the operator's near-identity initialization: there the transition starts at "do nothing" and earns each departure from identity; here the readout starts at "the control profile" and earns each departure from baseline. Both encode the same fact about the data, that effects are small shifts on a large, intervention-independent baseline, and both put that fact where the gradient can use it rather than leaving the network to rediscover it.

There is a more aggressive version worth naming, with a real trade. The simplex is what couples the genes; dropping it and predicting an independent per-gene log-mean would decouple them entirely, so raising one gene no longer costs another. The price is that $\mu = \ell \rho$ no longer separates sequencing depth from expression shape, which is the property that keeps cells comparable across library sizes. The log-fold-change reparameterization is the conservative move that keeps the depth-shape separation; the decoupled per-gene rate is the further step for when gene competition is demonstrably the binding constraint.

---

## 5. Lever two — state-aware dispersion, and the credit-assignment trap

The dispersion is the calibration lever, and it is where the subtlety lives.

The coverage-of-one symptom from [Chapter 4](04-results.md) has a clean cause. Because $\kappa_g$ is one constant per gene, it must simultaneously fit the control population and every perturbed condition. On the top differentially-expressed genes, whose true spread genuinely varies from condition to condition, one shared value cannot be right everywhere, and the compromise lands wide. Sampled populations then over-cover the real cells on exactly the genes the calibration metric watches.

The direct fix is to let dispersion depend on the state. Either make it a small head on the decoder trunk, $\kappa = \mathrm{softplus}(\texttt{kappa\_head}(h))$ with $h$ the trunk's hidden activation, so dispersion can vary per cell and per condition; or tie it to the mean through a learned mean-dispersion trend $\kappa_g = f(\mu_g)$, which follows the empirical relationship that higher-expressed genes are relatively less over-dispersed.

But a flexible $\kappa$ opens a trap that is the deepest point in this chapter, and it is the identifiability concern from [Chapter 5](05-challenges-and-limitations.md) made precise. A real cell's variation is biological plus technical. The biological part is what the latent distribution is supposed to own, whether that distribution comes from the flow or from the operator. The technical part is what the decoder's count noise is supposed to add. A state-dependent $\kappa$ trained by free maximum likelihood has no reason to respect that division. It will happily absorb biological variance into the decoder if that lowers the likelihood, and when it does, calibration improves while the generative model is quietly stripped of the very variance it exists to model. You would then have a well-calibrated population and no way to see a distributional win from the transition lever, because the decoder ate it.

So the fix has to come with an anchor. Rather than let maximum likelihood place $\kappa$ freely, fit it by moment-matching against the observed per-gene variance, or regularize it toward that observed dispersion, so the decoder is allowed to model technical noise and no more. Decoder flexibility and generative credit are in genuine tension here, and the anchor is what keeps the two separable. This is the one place in the decoder where a change that lowers the training loss can silently defeat the purpose of the whole stack, so it is worth doing carefully or not at all.

---

## 6. Two guardrails

Two smaller cautions keep the decoder work from overreaching.

**Reweight the loss only by answer-agnostic signals.** It is tempting to upweight the per-gene negative log-likelihood on the responsive genes so the fit does not neglect them. That is fine only if the weighting is computed from something the model is allowed to know, such as a gene's overall variability across all cells. Weighting by the *known* differential-expression identity of the test perturbations leaks the graded answer into training, and any gain from it is an artifact rather than a method.

**Do not reflexively enable ZINB.** The decoder supports a zero-inflated variant, and dropout-heavy single-cell data makes it tempting. Modern UMI-based counts are generally not zero-inflated, though, and switching on the dropout gate can *mask* a dispersion problem rather than fix it, by explaining away zeros that a correctly-dispersed negative binomial would produce on its own. The code leaves ZINB optional for this reason. Leave it off unless the zeros in the specific dataset genuinely demand it.

---

## 7. How the two levers compose, and the honest measurement

The two levers sit at two stages and act on two axes, and reading them together tells you the order to pull them.

The **mean head** sharpens effect size directly, by anchoring the readout at the baseline so the decoder models the deviation the metric scores. It touches the axis on which the tie was measured, and it can land alongside the operator without waiting on anything else.

The **anchored dispersion** makes calibration trustworthy, which is the prerequisite for *seeing* a distributional win at all. If the operator or the flow captures a richer, possibly multimodal response, that structure lives in the latent distribution and shows up only in the calibration and joint-structure metrics. Those metrics are worthless while the decoder over-disperses on the scored genes, and dangerous while an unanchored $\kappa$ can absorb the structure outright. So the dispersion anchor is what lets the transition lever's benefit become visible, and it should precede any claim that the operator beats the baseline on distribution.

The throughline that connects [the transition chapter](07-modeling-the-transition-action-operators.md) to this one is a single sentence in two halves. **Model the change**, with a transition that starts at identity and a readout that starts at baseline; and **measure it without corrupting it**, with a mean readout that carries no sampling noise and a spread that carries only the variance the latent distribution does not own. The operator earns each departure from identity; the mean head earns each departure from baseline; the anchored dispersion refuses to take credit for biology. Each is the same discipline applied at a different stage.

---

## 8. The generalization, in one note

The readout lever generalizes the same way the operator does. Nothing in the identity-anchored mean head or the variance-anchored dispersion is specific to a single control-to-perturbed transition. Any count or rate readout in a latent world model benefits from a mean that anchors at the previous state's profile and a dispersion that models only measurement noise. The cell case decodes one latent; a temporal rollout decodes each latent it steps to, through the same head. The readout is the $T$-agnostic corner of the same construction the operator chapter laid out for the transition, which is what keeps the cell-first build one dial-turn away from the temporal world model on the decoder side as well as the operator side.

> **Throughline.** The decoder is not one bottleneck but three knobs on two axes. Scoring the mean rate is already right, so the effect-size lever is an identity-anchored mean head and the calibration lever is a state-aware but variance-anchored dispersion. Together with the operator, the stack models the change at the transition and at the readout, and measures it without letting either stage take credit for what belongs to the other.

---

*Previous: [Modeling the transition — action operators](07-modeling-the-transition-action-operators.md). Up: [the method series](index.md). The operator form, generator basis, and two-gene composition still follow; this chapter answers the decoder handoff those chapters depend on.*
