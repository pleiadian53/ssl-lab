# The results ledger

*A living, append-only record of every experimental round on this method. [Chapter 4](04-results.md) is the narrative of the first round. This ledger is where each subsequent round lands, so that progress against the baseline can be read in one place rather than reconstructed from prose. Each round says what changed, what was measured, and what the honest verdict was, with a link from every number back to the run that produced it.*

> **How to read this.** The method is improved in rounds. [Chapter 6](06-beyond-the-current-limit.md) names the levers, [Chapter 7](07-modeling-the-transition-action-operators.md) develops the operator (the transition) and [Chapter 8](08-modeling-the-readout-count-decoder.md) develops the decoder (the readout). A round is one pass of building a lever and measuring it. The [pipeline guide](../running-the-pipeline.md) is how to run one.
>
> **What this ledger covers.** Every round evaluated on the same benchmark: the twenty held-out Norman two-gene combinations, scored on effect size and calibration, against the same from-scratch NB-VAE. That is what makes rounds commensurable and it is the ledger's only scope claim. When the method stops answering that question, for instance by rolling forward in time ($T > 1$) rather than making a single control-to-perturbed transition, it has become a world model rather than a perturbation-response model, and its results belong elsewhere.

---

## The standing verdict

**The from-scratch conditional NB-VAE beats every generative configuration we have built, on the primary endpoint, by about $0.12$.** It scores $0.766$ in $\Delta$-correlation against the transport flow's $0.648$ and the action operator's $0.645$. Both gaps are significant under simultaneous intervals covering the whole contrast family, and the VAE's seed-to-seed spread is $0.006$, so this is not a lucky draw and reseeding will not close it.

**Three structural levers have now been measured and none closes the gap.** The decoder (round 2) targets the readout and is bounded by construction. The action operator (round 3) targets the transition, which was the deeper bet, and it lands in a dead tie with the flow it was meant to replace ($-0.003$, not significant). The operator *algebra* (round 4) gave the operator the one mechanism round 3 never used, composition in the group, and its structural claim is refuted outright. A fourth lever, the metric itself, was corrected and made the picture *worse* for the method rather than better.

**And a ceiling now explains why the transition rounds could not have won.** Handing the pipeline the *real* held-out perturbed latents, a Stage B that is perfect by construction, scores $0.679$ against the flow's $0.648$. The transition is at $96\%$ of the best it can do given this encoder and this decoder, so **at most $0.03$ was ever available there**. The same ladder relocates the loss: a plain linear readout of the frozen latents scores $0.852$, above the baseline, so the representation is information-rich and the **decoder** is where the effect is lost. See [Chapter 7 of the methodology series](../../../../docs/experimental-method/07-the-ceiling.md).

Three findings do survive and are worth keeping. Transporting from a real control latent beats transporting from noise ($+0.036$, significant). Optimal-transport coupling hurts ($-0.021$, significant). And the operator is the only model that has ever raised the latent distribution's share of the predicted spread, which is the mechanism the calibration axis actually needs, even though it does not raise it nearly enough to matter.

Every model remains badly **under-dispersed**: coverage sits at $0.33$ to $0.38$ against a nominal $0.80$, so predicted populations are far too narrow.

---

## Two rules of this ledger

**Compare within a round.** Rounds are not run under identical conditions, so a difference is meaningful only against the other arms of its own round. The single quantity carried across rounds is the **NB-VAE reference**, held fixed as the standing bar.

**One primary endpoint, everything else exploratory.** The $\Delta$-correlation is the only metric on which significance is claimed, and its contrasts are tested with a **joint bootstrap and max-$t$ simultaneous intervals** across the whole contrast family (see [Chapter 4a](04a-reading-the-head-to-head.md)). Calibration metrics are reported as intervals with no verdict. A difference found on a secondary endpoint after the fact is a hypothesis, and it does not become a result until it is confirmed on data that did not suggest it.

---

## Running scoreboard

Every configuration evaluated on the twenty held-out Norman combinations, on the current gene selection ([Chapter 3e](3e-the-genes-the-metric-scores.md)). Primary metric first; the rest are exploratory.

| round | configuration | seeds | $\Delta r$ | spread_r ↑ | coverage (nom. 0.80) | W1 ↓ | energy ↓ |
|---|---|---|---|---|---|---|---|
| 1 | Gaussian flow (noise to outcome) | 3 | 0.612 | 0.205 | 0.357 | 1.013 | 3.794 |
| 1 | transport flow + OT coupling | 3 | 0.627 | 0.214 | 0.365 | 1.010 | 3.746 |
| 1 | **transport flow** (control to outcome) | 3 | **0.648** | 0.234 | 0.375 | 0.982 | **3.578** |
| 3 | **action operator** (deterministic) | 3 | 0.645 | 0.215 | **0.382** | 1.005 | 3.658 |
| 3 | action operator (stochastic + residual) | 3 | 0.646 | 0.208 | 0.344 | 1.005 | 3.746 |
| 4 | operator algebra, weak prior ($\lVert M\rVert$ $12.4$, **not near identity**) | 1 | 0.629 | | | | |
| 4 | operator algebra, near-identity prior held ($\lVert M\rVert$ $1.10$) | 1 | 0.497 | | | | |
| 1 | **conditional NB-VAE** (the bar) | 3 | **0.766** | **0.522** | 0.328 | **0.956** | 3.962 |
| — | *oracle Stage B (real held-out latents, same decoder)* | — | *0.679* | | | | |
| — | *linear readout of the frozen latents (diagnostic, not a model)* | — | *0.852* | | | | |

The last two rows are not models and cannot be run; they are the [ceiling](../../../../docs/experimental-method/07-the-ceiling.md). The first says a *perfect* Stage B scores $0.679$ through this decoder, so every transition row above is within $0.03$ of the best it could possibly do. The second says the effect is sitting in the frozen representation at $0.852$, above the baseline, and the decoder is losing it.

Read the bolded rows against each other and that is the project so far. Every generative configuration we have built clusters between $0.61$ and $0.65$, and the baseline sits alone at $0.766$. **No model's coverage is anywhere near nominal**, and the best coverage of any model belongs to the operator, which is the one model that widened its own latent cloud.

---

## Round 1 — the head-to-head: does the flow beat a from-scratch VAE?

**What was tested.** The full stack (frozen JEPA encoder, conditional flow prior, NB count decoder) against a from-scratch conditional NB-VAE with no JEPA and no flow, conditioned on the same gene-set embedding so the comparison isolates the generative machinery. Along the way, two flow variants (transporting from Gaussian noise versus from a real control latent) and optimal-transport coupling.

**Narrative:** [Chapter 4](04-results.md). **Audit and method:** [Chapter 4a](04a-reading-the-head-to-head.md). **Post-mortem:** [Chapter 5](05-challenges-and-limitations.md).

**Primary endpoint, max-$t$ simultaneous intervals over the 3-contrast family (critical value $2.95$, against $1.96$ unadjusted):**

| contrast | difference | simultaneous 95% CI | reading |
|---|---|---|---|
| transport − Gaussian | $+0.036$ | $[+0.019, +0.052]$ | **significant**: the transport reformulation is real |
| OT − transport | $-0.021$ | $[-0.039, -0.002]$ | **significant**: OT coupling hurts |
| **transport − NB-VAE** | $\mathbf{-0.118}$ | $\mathbf{[-0.228, -0.008]}$ | **significant**: the baseline wins |

**Secondary endpoints (intervals only, no verdicts).** The VAE tracks per-gene variability far better (spread_r $0.522$ against $0.234$; contrast $[-0.429, -0.136]$). The transport flow posts the best joint energy distance, which is exactly the metric built to see the structure a flow should capture, but the contrast against the VAE is $-0.383$ with an interval of $[-0.960, +0.172]$ that **crosses zero**. On twenty perturbations it does not resolve. It is a hypothesis for a larger test set, not a claim.

**Where the predicted spread goes** (`11_diagnose_variance.py`, by the law of total variance):

| | real variance | predicted total | $\sigma^2_{\text{dec}}$ | $\sigma^2_{\text{bio}}$ | latent's share |
|---|---|---|---|---|---|
| transport flow | 0.824 | 0.678 (0.84×) | 0.538 | 0.140 | 22% |
| NB-VAE | 0.824 | 0.355 (0.46×) | 0.226 | 0.128 | 38% |

Both under-produce spread. The two models' *latent* contributions are close ($0.140$ against $0.128$), so the gap between them is not a gap in what their latent distributions do. It is mostly in the decoder each learned.

**Verdict.** The flow loses to the VAE. Transporting from a real control latent is the right way to build the flow and OT coupling is the wrong way, and both of those are solid. But the method as a whole does not clear its baseline, and the compositional gene-set embedding, which both models share, is what actually drives combination generalization.

---

## Round 2 — the decoder levers (superseded, being re-run)

**What was tested.** The two levers of [Chapter 8](08-modeling-the-readout-count-decoder.md), as opt-in flags on `CountDecoder`: **B1**, an identity-anchored mean head targeting effect size, and **B2**, a state-aware dispersion with a moment-matching anchor targeting calibration. Four arms, single seed, retraining only Stage C.

**Status: the numbers from this round are not carried forward.** The round was designed against a reading of the calibration axis that the current gene selection does not support. B2 was built to *narrow* a decoder believed to be over-dispersed, and the decoder is in fact **under-dispersed** on every model measured (coverage $0.33$ to $0.38$, far below nominal). A lever pushing in that direction is aimed the wrong way, so its result carries no information about the design it was meant to test. B1, which targets effect size and is unaffected by the dispersion question, is the part worth re-measuring.

**Re-running:** the four arms on the current gene selection, seeded, so that B1's contribution can be read against the same simultaneous-interval standard as Round 1. The arms' checkpoints persist on the pod volume.

**Carry forward regardless:** B2's failure mode is instructive independent of its sign. Its NB likelihood *improved* while its target metric did not, which is the same trap optimal-transport coupling sprang in Round 1. **A better training loss is not a better model**, and that has now happened twice.

---

## Round 3 — the action operator: does modeling the *transition* beat modeling the destination?

**What changed.** The lever of [Chapter 7](07-modeling-the-transition-action-operators.md), built for the first time. Instead of learning a free velocity field that samples where a perturbed cell *lands*, learn the operator that carries a control cell to its perturbed counterpart:

$$c = e(p), \qquad \alpha = \pi(c), \qquad M = \sum_i \alpha_i B_i, \qquad A_p = \exp(M), \qquad z' = A_p z_b .$$

Here $e(p)$ is the gene-set embedding of perturbation $p$ (so an unseen combination composes from its single-gene parts), $\pi$ is a policy emitting coefficients $\alpha$, $\{B_i\}$ is a learned basis of $16$ generators, and $A_p$ is the operator. Three properties are in the parameterization rather than in a hope. The policy is zero-initialized, so $A_p = \exp(0) = I$ at the start: the operator begins at "this perturbation does nothing" and must earn every departure from the identity. A Frobenius penalty on $M$ is a least-action prior pulling it back toward identity. And $\alpha$ depends on the *condition*, not the cell, so one operator serves every cell of a perturbation, which is both the right model and far cheaper.

Because sequencing destroys the cell, there is no "same cell before and after" and the per-pair equivariance loss cannot be computed at all. So each step pushes the control cloud through $A_p$ and matches its **marginal** against the real perturbed cloud with an **energy distance**, which needs no correspondence. Only Stage B is retrained; the frozen encoder and the count decoder are reused.

**Two arms, three seeds each**, on an A40. Training energy fell from $0.68$ to $0.45$ and $\lVert A - I \rVert_F$ rose from $0$ to about $6$, so the operator did depart from the identity and stay there, rather than collapsing back into an expensive no-op.

| contrast | difference | simultaneous 95% CI | reading |
|---|---|---|---|
| operator − transport flow | $-0.003$ | $[-0.030, +0.024]$ | **not significant: a dead tie** |
| stochastic − deterministic operator | $+0.001$ | $[-0.039, +0.042]$ | not significant |
| **operator − NB-VAE** | $\mathbf{-0.121}$ | $\mathbf{[-0.228, -0.014]}$ | **significant: the baseline still wins** |

**Verdict on the primary endpoint: the operator does not pay off.** It ties the flow it was meant to replace and loses to the from-scratch VAE by the same margin the flow does. This was the deeper of the two levers, the one that changes what the *transition* is rather than what happens downstream of it, and it moves the number by $-0.003$.

**But one mechanism did work, and it is the first time anything has.** The variance decomposition ([Chapter 6](06-beyond-the-current-limit.md)) says the calibration axis needs $\sigma^2_{\text{bio}}$, the latent distribution's share of the predicted spread, to *grow*. The deterministic operator is the only model that has ever done it:

| | $\sigma^2_{\text{dec}}$ | $\sigma^2_{\text{bio}}$ | latent's share | coverage |
|---|---|---|---|---|
| transport flow | 0.538 | 0.140 | 22.4% | 0.375 |
| **operator (deterministic)** | 0.533 | **0.174** | **26.0%** | **0.382** |
| operator (stochastic + residual) | 0.538 | 0.081 | 15.4% | 0.344 |

A near-identity-initialized, least-action-penalized operator produces a *structurally* wider latent cloud than a free flow does, and its coverage is the best of any model built. This is Chapter 6's "Fix B" actually happening. It is simply nowhere near large enough to matter: closing the coverage gap would need $\sigma^2_{\text{bio}}$ near $0.29$, and the operator delivers $0.17$.

**A failure, and an honest limit on what we can say about it.** The stochastic arm was designed to widen the cloud further, by letting the perturbation induce a *distribution* over operators. It did the opposite: $\sigma^2_{\text{bio}}$ fell to $0.081$ and coverage dropped by $0.037$ (an interval excluding zero). The likely mechanism is the per-condition **residual displacement**, which is a constant shift and therefore adds *zero* variance, so it can absorb the mean effect cheaply and let $A_p$ relax back toward the identity. But that remains a hypothesis, **because the arm changed two things at once** (a stochastic $\alpha$ *and* a residual). A two-lever arm cannot attribute its own result. The next round separates them.

**Carry forward.** The energy distance *fell* for the stochastic arm ($0.428$ against the deterministic $0.447$), meaning it matched the latent cloud better in latent space while getting *worse* downstream. A better training objective is not a better model, and that has now happened three times in this project (optimal-transport coupling, the dispersion anchor, and now this).

## Round 4 — the operator algebra: is epistasis non-commutativity?

**What changed.** Round 3's operator applied $\exp$ once, to a single generator, and composed two genes in the *additive gene-set embedding*, which is the same object the NB-VAE uses. The matrix exponential was doing the work of a reparameterization and never the work of a capability, and the operator's own algebra sat idle. This round gives each single gene its own dense generator $M_g$ and composes a combination **in the group**, through the symmetric product

$$A_{A+B} = \exp(\tfrac{1}{2}M_A) \exp(M_B) \exp(\tfrac{1}{2}M_A).$$

The motivation is a theorem rather than an analogy. That product collapses to $\exp(M_A + M_B)$ exactly when the generators commute, so **non-commutativity is the departure from additivity**, and departure from additivity is what epistasis means. See [the algebra of composition](../../../../docs/action_operator/03-the-algebra-of-composition.md). The product is symmetrized because both guides are delivered at once: the observation carries no order, while the leading BCH term $\tfrac12[M_A,M_B]$ is swap-odd, so an ordered product would predict two answers for a thing that has one.

**The pre-committed primary endpoint was NOT $\Delta r$.** The [ceiling analysis](../../../../docs/experimental-method/07-the-ceiling.md) had already shown Stage B is saturated: an oracle handed the real perturbed latents scores $0.679$ against the flow's $0.648$, so no Stage-B lever can move effect size by more than about $0.03$. Grading this round on $\Delta r$ would have measured the one thing already known to be pinned. The endpoint was instead a structural claim, benchmark-independent and needing no comparison to the baseline: does $\lVert [M_A, M_B] \rVert$ predict a pair's measured genetic interaction? The decision rule was fixed in advance: the **in-sample** arm carries the statistical power, and a null there kills the idea.

**Verdict: refuted.** One seed, A40, 105 generators, 91 training combinations and 20 held out.

| test | in-sample ($n=91$) | held-out ($n=20$) |
|---|---|---|
| **raw bracket against rel_GI** (the pre-committed endpoint) | $\mathbf{-0.070}$ (perm-$p$ $0.75$) | $-0.262$ (perm-$p$ $0.87$) |
| raw bracket against the directional component (post-hoc) | $+0.114$ | $+0.042$ |
| scale-normalized bracket against rel_GI (post-hoc) | $-0.141$ | $-0.344$ |
| scale-normalized bracket against the directional component (post-hoc) | $-0.023$ | $-0.120$ |

Every variant, every target, both splits, all $111$ pairs: nothing. The two post-hoc refinements were genuine attempts to rescue the hypothesis and both failed, which makes the negative stronger rather than weaker.

**A confound that nearly produced a fake result, and the gate that now prevents it.** The first run used a least-action weight of $10^{-4}$ and returned the same null. That null was **uninformative**, and reading it as a refutation would have been wrong. The generators had wandered to a median $\lVert M_g \rVert_F$ of $12.4$, with $\lVert A - I \rVert_F$ near $11$: nowhere close to the identity. This matters because the equivalence the whole round rests on is a *near-identity* statement. Far from the identity, two generic large matrices fail to commute merely because they are large, and indeed the bracket was $0.916$ rank-correlated with $\lVert M_A \rVert \cdot \lVert M_B \rVert$. The bracket had degenerated into a readout of generator magnitude and could not have carried pair-specific information whatever the answer.

Sweeping the penalty fixed it. At $10^{-1}$ the median $\lVert M_g \rVert$ falls to $1.10$ and the near-identity premise holds. **The endpoint is unchanged.** So the refutation is measured in the regime where the claim's mathematics actually applies, which is what makes it credible. Stage B now **gates itself** on this premise: a run whose median generator norm exceeds a threshold reports that it cannot test the claim, rather than returning a null that reads like a refutation.

**What is refuted, and what is not.** Not the mathematics: "generators commute $\iff$ composition is additive" is a theorem, verified numerically. What is refuted is that *these learned generators carry it*. The likely cause is **identifiability**, and it is visible in the numbers. Each $M_g$ is $65{,}536$ parameters fit only to make $\exp(M_g)$ push the control cloud onto one perturbation's marginal, which constrains what the operator does to one cloud rather than what the matrix is. The bracket therefore lives largely in directions the loss never touched, and even at near-identity it remains $0.63$ rank-correlated with the product of the generator norms. It measures how strong the two perturbations are individually, not how they interact.

**The finding worth more than the refutation: the two requirements are in direct conflict.** Grading both arms on the standing benchmark exposes a tension the design never anticipated.

| arm | $\lVert M_g \rVert$ (median) | near-identity gate | training energy | $\Delta r$ |
|---|---|---|---|---|
| weak prior ($10^{-4}$) | $12.40$ | **failed** | $0.482$ | $0.629$ |
| strong prior ($10^{-1}$) | $1.10$ | **passed** | $0.777$ | $0.497$ |

The arm that models the perturbation well sits far outside the regime where its bracket means anything. The arm whose bracket is interpretable is too constrained to model the perturbation, and pays $0.13$ of effect size for the privilege. **The operator cannot simultaneously fit the response and stay where its own algebra applies**, because fitting the response demands a generator far from zero and the bracket-is-epistasis equivalence demands one near zero.

That indicts a premise this series has leaned on since [Chapter 7](07-modeling-the-transition-action-operators.md), which motivates the near-identity prior by asserting that effects are *small shifts on a large, intervention-independent baseline*. Measured in this latent geometry they are not small: fitting the response wants $\lVert M \rVert \approx 12$, and constraining it to $1.1$ costs a fifth of the score. Baseline dominance is a true statement about *expression*, where a perturbation moves a handful of genes against thousands that do not move. It does not survive the trip through the encoder into *latent* coordinates, where the response is evidently a large rotation rather than a small nudge. The prior was imported from the wrong space.

**Carry forward.** The empirical epistasis this round targeted is real and well-structured, which is why the null is informative rather than vacuous. Across the twenty held-out pairs the interaction spans $0.17$ to $0.54$ of the effect, the additive scale $\lambda$ spans $0.655$ to $1.266$ (nine super-additive, seven sub-additive, four with the magnitude right), and a median $76\%$ of each interaction is **directional**, meaning the pair moves genes the additive model does not predict. There was a real, varied signal to find. The bracket did not find it.

---

## Open, and what comes next

Three structural levers have now been built and measured, and none closes a gap of $0.12$. The ceiling explains why two of them never could: the transition they targeted had at most $0.03$ of headroom. That is worth stating plainly rather than softening, because it changes what the remaining questions are, and it retires some of them.

**Retired.** *Stage-B levers, for effect size.* The ceiling bounds them at $0.03$, and a fourth transition design would be a fourth round spent on three percent. The operator thread is closed on the primary endpoint, and round 4 closes its structural claim as well.

1. **The readout is where the loss actually is, and it is the one lever never aimed correctly.** The linear rung of the ceiling scores $0.852$ on the frozen latents, above the baseline's $0.766$, while the trained decoder scores $0.679$ on the identical latents. That $0.173$ is the largest single term in the loss budget, it is six times the transition's, and round 2 aimed at the decoder's *dispersion* rather than at this. An un-attenuated mean head is the only route on the table that could plausibly beat the baseline on $\Delta r$. Note the honest caveat: the linear rung is a diagnostic, not a model, since it is handed real latents and emits no distribution.
2. **Data efficiency is still the one premise never tested.** Self-supervised pretraining does not claim to win at full data. It claims that a representation learned on abundant *unlabeled* cells pays off when *labeled* examples are scarce. Every number in this ledger is a full-data number, so not one of them tests the actual claim. The experiment is a subsampling ladder: shrink the training cells per perturbation, retrain the flow and the VAE at each rung, and plot $\Delta$-correlation against cells. It needs only a subsampling flag, and it is independent of everything above.
3. **The frozen, condition-blind encoder** remains the deepest structural constraint, and three separate findings now converge on it. The ceiling says the representation was never shaped for the decoder that consumes it. The baseline's advantage survives with an architecturally *identical* decoder, which leaves co-adaptation as the difference. And [Chapter 9](09-why-the-operator-is-linear-koopman.md) says the operator's linearity is licensed only by an encoder trained to supply Koopman coordinates, which ours was not. Relaxing it is the biggest swing still available, and it is the point at which the method stops being "a flow prior over frozen JEPA latents" and becomes something else.
4. **If the generator identifiability of round 4 is ever revisited**, the diagnosis names the fix: constrain the generators to a shared low-rank basis, $M_g = \sum_i \beta_{g,i} B_i$, so a bracket lives in the span of $[B_i, B_j]$ rather than in $65{,}536$ directions the marginal-matching loss never touched. This is a hypothesis with a named mechanism, not a rescue of the refuted claim, and it should be pre-registered like any other round.
5. **Or accept the negative result.** "A frozen self-supervised representation plus a learned conditional prior does not beat a from-scratch conditional VAE at Perturb-seq effect size; here is the ceiling that says why the transition never could; here are three structural levers that failed; and here is the metric-selection trap that nearly hid all of it" is a real contribution, and more useful to the field than a fifth lever that also fails.

---

## Adding a round

- **State what changed, and which chapter specified it.** A round executes a design; it does not improvise one.
- **Pre-commit the primary endpoint and the decision rule** before looking at numbers.
- **Report the arms of that round together**, comparing only within it, and carry the NB-VAE reference forward as the fixed bar.
- **Say the seed count.** One seed is directional. A claim needs seed averaging and a simultaneous interval that excludes zero.
- **Report failures with their diagnosis.** B2 is more useful written up as a failure with a named cause than omitted.
- **Link every number to its script and report** so a reader can regenerate it.

---

*Up: [the method series](index.md). Round 1's narrative is [Chapter 4](04-results.md); its audit is [Chapter 4a](04a-reading-the-head-to-head.md). To run a round, see [Running the pipeline](../running-the-pipeline.md).*
