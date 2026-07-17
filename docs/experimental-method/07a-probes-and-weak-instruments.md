# Chapter 7a — Probes, and other deliberately weak instruments

*A companion to [Chapter 7](07-the-ceiling.md). The ceiling used a linear readout to prove that the loss lived in the decoder rather than the representation. That instrument was not invented for the occasion; it is a standard tool with a long history, and it belongs to a family. This chapter is about the family, and about the property that makes any of its members worth trusting: a good diagnostic is one that **cannot be right for the wrong reason**.*

---

## 1. Two probes, one instrument

This project ran two linear probes without ever noticing they were the same tool.

The first has been in the pipeline since Stage A. [`02_probe_cell_encoder.py`](../../examples/perturbation_response/02_probe_cell_encoder.py) freezes the encoder and fits a logistic regression from a cell's latent $z$ to the identity of the perturbation applied to it, one of $237$ classes. It scores $5.2\%$.

The second is the `linear` arm of the ceiling. It freezes the same encoder and fits a ridge regression from the same $z$ to the cell's $5000$-gene expression vector. It scores $0.852$ in $\Delta$-correlation.

Set them side by side and the shared skeleton is obvious:

| | probe (`02`) | ceiling's linear arm (`14`) |
|---|---|---|
| what is frozen | the encoder | the encoder |
| the input | the latent $z$ | the latent $z$ |
| the head | logistic regression | ridge regression |
| the target | which of 237 perturbations | per-gene expression |
| the question | does $z$ know *which* perturbation happened? | does $z$ know *what the perturbation did*? |
| the floor to beat | chance, $1/237 = 0.42\%$ | the trained decoder, $0.679$ |

Same instrument, two heads. Freeze the representation, fit the simplest map you can to a target you care about, and read what comes out. A classification head asks whether the latent carries an identity. A regression head asks whether it carries a quantity. Nothing else changes.

This is the **linear evaluation protocol**, and it is the standard way to grade a frozen representation. The idea traces to Alain and Bengio's 2016 note on using linear classifier probes to understand intermediate layers, and it became the headline benchmark of self-supervised learning: CPC, MoCo, SimCLR, BYOL, and DINO all report "freeze the encoder, fit a linear classifier, quote the accuracy." Natural-language processing developed the same habit independently, with probing classifiers asking what syntax is linearly recoverable from a frozen language model's activations.

Worth being precise about the lineage, because it is easy to mis-file: probing comes from **representation learning**, not from generative modeling. Generative models proper are graded by likelihood, by sample-quality scores, or by two-sample tests. The probe is how you grade an *encoder*.

## 2. The rule that makes a diagnostic worth trusting

Here is the property shared by every instrument in this chapter, and it is the same property [Chapter 2](02-the-scoring-seam.md) called *self-guarding*:

> **A diagnostic is only informative if it is structurally unable to give you a good answer for the wrong reason.**

The Wilcoxon $z$ of Chapter 2 earns its place because a silent gene *cannot* post a large rank statistic. The oracle of Chapter 7 earns its place because a perfect stage *cannot* score better than perfect. And the linear probe earns its place because a linear map **cannot manufacture structure that is not already in the representation**.

That is why the probe's weakness is the argument rather than a limitation of it. Three consequences follow, and the second is the one people miss.

**It makes the claim a lower bound.** When a linear map extracts $0.852$, the honest reading is "at least this much signal is present, and it is accessible to the dumbest possible readout." The weaker the head, the stronger that statement. A more powerful head scoring $0.852$ would license a much vaguer claim.

**It attributes capacity.** The trained NB decoder is *already* a nonlinear map, an MLP with a softmax rate head, and on identical latents it scores $0.679$. A linear map beats it by $0.173$. If the decoder's problem were insufficient capacity, the map with *less* capacity would have scored *worse*. It scored better, which rules out capacity as the explanation and points at misdirection: the decoder's likelihood objective weights genes by abundance, so it spends its capacity on the genes the metric does not score. No flexible probe could have established this, because a flexible probe winning is equally consistent with "capacity was the problem" and "alignment was the problem."

**It avoids the confound that destroys flexible probes.** Give the head enough capacity and it starts doing the *representation's* job. A sufficiently powerful decoder can recover the target from almost any injective encoding, including a random projection. At that point the score measures the probe rather than the encoder, and the measurement no longer discriminates between a good representation and a bad one. A diagnostic that returns a high number for every input has told you nothing.

## 3. Reading a probe without fooling yourself

**Always read a probe against a floor, never in absolute terms.** The perturbation probe scores $5.2\%$. In isolation that number looks like a failure, and it is in fact a strong result: chance is $0.42\%$, so the latent is about twelve times better than guessing at a $237$-way problem. Absolute probe numbers are meaningless. Only the ratio to a floor carries information, and choosing that floor honestly is most of the work.

**A probe is a lower bound on accessible information, not an upper bound on information.** This is the limitation of the ceiling's own loss budget, and it is worth stating plainly. The gap from $1.000$ to $0.852$ was described as "the encoder's ceiling," but it is really *the encoder's ceiling given a linear readout*. Those missing $0.148$ are either information the encoder genuinely destroyed when compressing $5000$ genes into $256$ dimensions, or information it kept but stored **nonlinearly**, where a linear map cannot reach. A linear probe cannot tell those apart, and the distinction matters: the first says the encoder is a hard wall, the second says even that residue is recoverable with a better-shaped readout.

**A probe is not a model.** The ridge was handed the *real* held-out latents, which a deployed system cannot produce, and it emits only a mean, so it cannot be scored on calibration at all. "The linear probe beats the baseline" is a statement about headroom, not a system you can ship. Confusing the two is the failure mode Chapter 7 named as *reporting a skyline as a result*, and it is tempting precisely because the number is good.

**Fit on train, apply to held-out.** A probe is a supervised model and leaks like one. Ours is fit on the training cells, which exclude the twenty held-out combinations, and applied to held-out latents.

## 4. The rest of the toolbox

Diagnostics are worth organizing by the question they answer, because reaching for the wrong one is how you get a confident number about something you did not want to know. Below, each entry names the question, the mechanism, and what the instrument is unable to fake.

### Is the representation any good?

**The linear probe.** Covered above. Cannot invent structure that is not linearly present.

**The $k$-nearest-neighbour evaluation.** Classify a held-out latent by the labels of its nearest neighbours among the training latents. It has *zero* fitted parameters, so it removes the probe-capacity confound entirely rather than merely limiting it, and it asks a purely geometric question: does the latent space *place* similar things near each other? DINO reports $k$-NN alongside a linear probe for exactly this reason, and a large gap between the two is itself informative, since it means the classes are linearly separable but not locally clustered.

**The nonlinear probe.** The complement to the linear one, and the right tool for the $0.148$ question above. Fit a regularized MLP from frozen latents to the target. If it substantially beats the linear probe, the encoder preserved the information and stored it nonlinearly. If it does not, the information is genuinely gone. Use it to *bound the encoder*, never to grade it against another encoder, since that is where the capacity confound returns.

**Effective rank.** How many dimensions is the representation actually using? Take the eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_D$ of the latent covariance, normalize them into a distribution $p_i = \lambda_i / \sum_j \lambda_j$, and report the exponential of its Shannon entropy:

$$\mathrm{erank} = \exp\left(-\sum_{i=1}^{D} p_i \log p_i\right).$$

A representation of nominal width $256$ whose variance lives in $10$ directions has an effective rank near $10$, and it will disappoint every downstream stage no matter how they are built. This project tracks it, and the value near $176$ to $213$ against a nominal $256$ is what says Stage A did not collapse. The instrument cannot be faked because it reads the spectrum directly and never consults a label.

**Feature variance and covariance.** The cheap continuous cousin of effective rank, and the basis of the [collapse guard](../../examples/perturbation_response/docs/conditional-flow-jepa/3c-the-vicreg-collapse-guard.md) already in Stage A. Per-dimension standard deviation catches a representation shrinking toward a constant; off-diagonal covariance catches dimensions duplicating each other. Both are computable every epoch for nearly nothing, which makes them a monitor rather than a post-mortem.

**Representational similarity, via CKA or RSA.** Compare two representations to each other rather than to a label. Useful for questions of the form "did this fine-tuning change anything?" or "are these two encoders learning the same thing?" Centered kernel alignment is the common modern choice.

### Where is the bottleneck?

**The oracle ladder.** The subject of [Chapter 7](07-the-ceiling.md). Replace a stage with the ground truth it was trying to produce, and read the headroom. Cannot be faked because a perfect stage cannot score better than perfect.

**Ablations and controls.** [Chapter 4](04-ablations-and-controls.md). An ablation removes one component; a control changes the setup to answer a different question. Confusing them is how a whole-method gap gets charged to one part.

### Is the generator actually generating?

This group is specific to generative models, and the failures it catches are invisible to every diagnostic above.

**Posterior collapse and dead units.** A latent-variable model can learn to *ignore its own latent*, letting the decoder do all the work while the latent carries nothing. The classic diagnostic is the per-dimension KL divergence between the approximate posterior and the prior: dimensions with KL near zero are dead, and counting them tells you the model's real latent width. This project met a close relative of the same pathology and named it differently, when the conditioning latent $z_b$ turned out to be **inert** for the mean effect, which is posterior collapse wearing a conditional model's clothes.

**Prior versus aggregate-posterior mismatch.** A model can reconstruct beautifully and generate garbage, and this is usually why. Reconstruction decodes latents drawn from the posterior, the region where real data actually lands. Generation decodes latents drawn from the *prior*. If those two distributions disagree, generation samples from regions the decoder never trained on, and the output is junk despite a healthy reconstruction loss. Diagnose it by comparing prior samples against the aggregated encoded latents of real data, with a two-sample test or simply by looking. This diagnostic deserves emphasis here because it is the entire motivation for **this project's central design choice**: a learned conditional flow prior exists precisely so the prior can be shaped to match the aggregate posterior instead of being assumed Gaussian and hoping.

**The reconstruction-versus-generation gap.** The blunt version of the above, and worth reporting as a matter of routine. Score reconstruction and generation with the same metric. A large gap is a prior problem, not a decoder problem, and it redirects the work.

**Two-sample tests.** How do you compare a cloud of generated samples against a cloud of real ones with no correspondence between them? Energy distance and maximum mean discrepancy both do it, needing no pairing. This project uses the energy distance in two distinct roles, as the operator's training objective and as a calibration metric. They cannot be faked into reporting zero for mismatched clouds, though they *can* be low while the model is worse downstream, which is the trap the next subsection is about.

**Variance decomposition.** When a generative model's spread is wrong, *which part* of it is wrong? The law of total variance splits the predicted variance of gene $g$ into the decoder's own noise and the variance induced by the latent distribution:

$$\mathrm{Var}[x_g] = \underbrace{\mathbb{E}_z[\mathrm{Var}(x_g \mid z)]}_{\sigma^2_{\text{dec}}} + \underbrace{\mathrm{Var}_z[\mathbb{E}(x_g \mid z)]}_{\sigma^2_{\text{bio}}}.$$

The first term is the readout's noise; the second is the generator actually doing its job. This project's [`11_diagnose_variance.py`](../../examples/perturbation_response/11_diagnose_variance.py) computes exactly this, and the split turned a vague "calibration is bad" into a sharp claim: coverage can be fixed by growing either term, but growing $\sigma^2_{\text{dec}}$ makes the generator *less* visible while growing $\sigma^2_{\text{bio}}$ makes it more so. A single coverage number could never have said that.

**Calibration and coverage.** Does the model's claimed uncertainty match reality? Form a central interval at nominal level and count how much real data falls inside. Read [Chapter 2](02-the-scoring-seam.md) before trusting it, since coverage is the metric that was pinned at exactly $1.00$ by a broken gene list for months.

### Is the training doing what you think?

**Track the target metric next to the loss, always.** The single most repeated lesson in this project is that **a better training loss is not a better model**, which happened three separate times: optimal-transport coupling lowered the flow-matching loss and hurt the score, the dispersion anchor improved the likelihood and hurt the score, and the stochastic operator improved the energy distance and got worse downstream. A loss that falls while the metric stalls is not a curiosity to be explained away. It is a report that your objective and your goal have come apart.

**Check the model at initialization.** If a construction claims to start from a no-op, verify the no-op numerically before training anything. The action operator is built so that a zero-initialized policy gives $A = \exp(0) = I$, and the check is a one-line assertion that $\lVert A - I \rVert = 0$. It is trivial, it is fast, and it is the difference between a design property and an aspiration.

**Shuffle the labels.** Train or fit the same probe against *permuted* targets. Performance should collapse to the floor. If it does not, you have a leak, and you have just found it for the price of one extra run. The same logic generalizes to a **permutation null** for any correlation you plan to claim, which is what the operator round's endpoint eval uses before reporting that a bracket predicts epistasis.

## 5. Choosing an instrument

The selection rule is a single question, and it is the rule this whole chapter is an argument for:

> **What must this instrument be unable to fake?**

Answer that first, then pick the tool whose *structure* enforces it, rather than a tool that merely tends to behave. A linear probe cannot fake structure. A $k$-NN evaluation cannot fake geometry. An oracle cannot fake being better than perfect. An effective rank cannot fake a spectrum. A permutation null cannot fake a correlation. In each case the guarantee comes from the shape of the instrument, not from the care of the person holding it, and that is what makes it survive a refactor, a new dataset, and your own future enthusiasm.

The corollary, and the reason this companion sits next to the ceiling chapter rather than inside it: **the most informative instrument is usually the weakest one that can still answer the question.** Reaching for a more powerful diagnostic feels like rigor and is usually the opposite, because power is exactly what lets an instrument return the answer you were hoping for regardless of the truth.

---

*Up: [Running an experiment you can trust](index.md). The chapter this companion serves: [The ceiling](07-the-ceiling.md). The scoring-seam chapter that first argued for structurally self-guarding criteria: [Chapter 2](02-the-scoring-seam.md).*
