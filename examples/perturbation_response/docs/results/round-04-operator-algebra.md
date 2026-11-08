# Round 4 — the operator algebra: is epistasis non-commutativity?

*Part of [the results ledger](index.md), which holds the standing verdict, the scoreboard and the settings every round shares. Previous: [round 3](round-03-action-operator.md) · Next: [round 5](round-05-koopman-probe.md).*

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

That indicts a premise this series has leaned on since [Chapter 7](../conditional-flow-jepa/07-modeling-the-transition-action-operators.md), which motivates the near-identity prior by asserting that effects are *small shifts on a large, intervention-independent baseline*. Measured in this latent geometry they are not small: fitting the response wants $\lVert M \rVert \approx 12$, and constraining it to $1.1$ costs a fifth of the score. Baseline dominance is a true statement about *expression*, where a perturbation moves a handful of genes against thousands that do not move. It does not survive the trip through the encoder into *latent* coordinates, where the response is evidently a large rotation rather than a small nudge. The prior was imported from the wrong space.

**Carry forward.** The empirical epistasis this round targeted is real and well-structured, which is why the null is informative rather than vacuous. Across the twenty held-out pairs the interaction spans $0.17$ to $0.54$ of the effect, the additive scale $\lambda$ spans $0.655$ to $1.266$ (nine super-additive, seven sub-additive, four with the magnitude right), and a median $76\%$ of each interaction is **directional**, meaning the pair moves genes the additive model does not predict. There was a real, varied signal to find. The bracket did not find it.

---

*Up: [the results ledger](index.md). Previous: [round 3](round-03-action-operator.md) · Next: [round 5](round-05-koopman-probe.md).*
