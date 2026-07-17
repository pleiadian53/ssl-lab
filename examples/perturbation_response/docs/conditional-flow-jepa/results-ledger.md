# The results ledger

*A living, append-only record of every experimental round on this method. [Chapter 4](04-results.md) is the narrative of the first round. This ledger is where each subsequent round lands, so that progress against the baseline can be read in one place rather than reconstructed from prose. Each round says what changed, what was measured, and what the honest verdict was, with a link from every number back to the run that produced it.*

> **How to read this.** The method is improved in rounds. [Chapter 6](06-beyond-the-current-limit.md) names the levers, [Chapter 7](07-modeling-the-transition-action-operators.md) develops the operator (the transition) and [Chapter 8](08-modeling-the-readout-count-decoder.md) develops the decoder (the readout). A round is one pass of building a lever and measuring it. The [pipeline guide](../running-the-pipeline.md) is how to run one.
>
> **What this ledger covers.** Every round evaluated on the same benchmark: the twenty held-out Norman two-gene combinations, scored on effect size and calibration, against the same from-scratch NB-VAE. That is what makes rounds commensurable and it is the ledger's only scope claim. When the method stops answering that question, for instance by rolling forward in time ($T > 1$) rather than making a single control-to-perturbed transition, it has become a world model rather than a perturbation-response model, and its results belong elsewhere.

---

## The standing verdict

**The from-scratch conditional NB-VAE beats every generative configuration we have built, on the primary endpoint, by about $0.12$.** It scores $0.766$ in $\Delta$-correlation against the transport flow's $0.648$ and the action operator's $0.645$. Both gaps are significant under simultaneous intervals covering the whole contrast family, and the VAE's seed-to-seed spread is $0.006$, so this is not a lucky draw and reseeding will not close it.

**Two structural levers have now been measured and neither closes the gap.** The decoder (round 2) targets the readout and is bounded by construction. The action operator (round 3) targets the transition, which was the deeper bet, and it lands in a dead tie with the flow it was meant to replace ($-0.003$, not significant). A third lever, the metric itself, was corrected and made the picture *worse* for the method rather than better.

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
| 1 | **conditional NB-VAE** (the bar) | 3 | **0.766** | **0.522** | 0.328 | **0.956** | 3.962 |

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

## Open, and what comes next

Both structural levers have now been built and measured, and neither closes a gap of $0.12$. That is worth stating plainly rather than softening, because it changes what the remaining questions are.

1. **Data efficiency is the one premise never tested.** Self-supervised pretraining does not claim to win at full data. It claims that a representation learned on abundant *unlabeled* cells pays off when *labeled* examples are scarce. Every number in this ledger is a full-data number, so not one of them tests the actual claim. The experiment is a subsampling ladder: shrink the training cells per perturbation, retrain the flow (or operator) and the VAE at each rung, and plot $\Delta$-correlation against cells. If the pretrained stack degrades more gracefully, the method has a real and practically important niche even at full-data parity. **This is now the first thing to run**, and it needs only a subsampling flag.
2. **Separate the two changes in the stochastic operator arm.** Run stochastic-$\alpha$ alone and residual-alone, so the regression in $\sigma^2_{\text{bio}}$ can be attributed. One lever per arm.
3. **The frozen, condition-blind encoder** remains the deepest structural constraint, and relaxing it (joint training, or a conditional pretext task) is the biggest swing still available. It is also the point at which the method stops being "a flow prior over frozen JEPA latents" and becomes something else, which is the fork this ledger's scope note names.
4. **Or accept the negative result.** "A frozen self-supervised representation plus a learned conditional prior does not beat a from-scratch conditional VAE at Perturb-seq effect size, and here is the careful measurement, the two structural levers we tried, and the metric-selection trap that nearly hid all of it" is a real contribution. It is more useful to the field than a fourth lever that also fails.

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
