# 7. The application, and what it taught

*Six chapters built the framework. This one runs it, on real single-cell perturbation data, and reports that its central empirical claim was refuted. That sentence is the point of the whole series, because a refutation with a diagnosis is worth more than an assertion without one, and because the diagnosis turns out to say something the framework's proponents, ourselves included, did not want to hear: the setting we chose to test it in is the one where it had the least to offer, and the finding that survived the refutation is about geometry, not about cells.*

> **Where this sits.** The closing chapter of the theory. It assumes all six before it and it is deliberately concrete. The round-by-round record with the exact numbers is the [results ledger](../conditional-flow-jepa/results-ledger.md); this chapter is the interpretation.

---

## 1. Perturbation response is the $T = 1$ corner

Every dial the earlier chapters introduced sits, in this application, at its simplest setting.

| construct | this application |
|---|---|
| number of applications $T$ | $1$: control to perturbed, a single step |
| the context $c$ | a gene-set embedding of the perturbation |
| composition | unordered, because both guides arrive at once |
| training coupling | none: cells come unpaired |
| the encoder | frozen, pretrained by masked prediction |

That is the framework at $T = 1$, simultaneous, unpaired, on a frozen encoder. It is a legitimate corner and a natural first target, and it is also, as this chapter will argue, close to the *least* favourable corner in which to test the framework's distinctive machinery. Three of those five rows are not simplifications so much as amputations.

## 2. Three things the domain breaks

**Unpaired breaks the objective.** [Chapter 6](06-training.md) showed the equivariance loss, the thing that makes the operator *mean* the transformation, cannot be computed without pairs, and sequencing destroys the cell. We trained against a marginal-matching surrogate that identifies the operator's effect on a distribution but not the operator. This is amputation of the objective that gives the operator its meaning.

**Simultaneous breaks half the algebra.** [Chapter 5](05-composition.md) showed that when both interventions arrive at once, the observation is invariant under swapping them, so only the swap-*even* part of the interaction is observable. The commutator, which carries the *direction* of non-commutativity, is swap-odd and therefore unobservable by construction. The operator's most distinctive structure, the part that would make a plan's order matter, is exactly the part this experiment cannot see.

**The frozen encoder breaks the premise.** [Chapter 2](02-why-linear-koopman.md) showed the operator's linearity is licensed only if the encoder supplies coordinates in which the dynamics close, and our encoder was trained by masked prediction, which mentions no dynamics, and then frozen. We adopted the Koopman *form* without the Koopman *training*.

Three of the framework's load-bearing assumptions were unavailable, weakened, or unearned before a single operator was trained. Hold that thought; §6 is what it implies.

## 3. What we built, in two rounds

**Round 3, the operator that never used its algebra.** The first operator composed two genes not in the group but in the *embedding*: a gene-set encoder mapped a pair to a summed embedding, a policy turned that into coefficients, and $\exp$ was applied once to the result. No product of operators was ever formed, so no bracket was ever computed. The matrix exponential bought invertibility and a near-identity start and nothing else, which is [Chapter 3](03-why-exp-of-m.md) §8's failure mode exactly. It tied the free transport flow it was meant to replace, which is the arithmetic you should expect when a restriction removes capacity and a prior adds bias in equal measure. Worse, composition-in-the-embedding is what the from-scratch baseline also does, so the operator had no combination mechanism the baseline lacked.

**Round 4, the operator algebra.** The second round fixed the first. Each gene got its own generator, and a pair composed in the group through the symmetric product of [Chapter 5](05-composition.md), so that non-commutativity carried the interaction. Its central prediction was the framework's distinctive claim made testable: $\lVert [M_A, M_B] \rVert$, computed from the two single-gene generators, should track the pair's measured genetic interaction. This is benchmark-independent. It needs no comparison to the baseline; it asks only whether the algebra recovers a biological quantity it was never trained to predict.

## 4. The refutation

It does not. Across four tests, both a within-training-set split that carries the statistical power and a held-out split, and every variant tried, the correlation between the bracket and measured epistasis was null or negative. The pre-registered decision rule, that a null on the powered split kills the idea, was met, and two post-hoc attempts to rescue the hypothesis by correlating against sub-components of the interaction also failed, which strengthens the negative rather than weakening it.

The refutation nearly did not survive contact with a confound, and how it survived is the methodologically important part. The first run returned its null with the generators sitting at a median norm of about $12$, far outside the near-identity regime. [Chapter 5](05-composition.md) §4 measured what that means: at $\lVert M \rVert \approx 12$ the bracket picture has broken so badly that its correction is worse than assuming additivity, and the bracket is dominated by generator *magnitude* rather than by genuine non-commutativity. A null in that regime is *uninformative*, not a refutation, because the quantity being correlated no longer measures what it is supposed to. Only a one-line diagnostic on the generators' own norms distinguished the two cases. Constraining the generators back to $\lVert M \rVert \approx 1$, where [Chapter 5](05-composition.md) §4 shows the bracket picture is quantitatively sound, left the null unchanged. That is what makes the refutation credible: it was measured in the regime where the claim's own mathematics applies.

## 5. The finding that outlived the hypothesis

Grading the two regimes on the standard benchmark exposed a conflict the design never anticipated, and it is worth more than the refutation.

| generator norm | near-identity regime | effect-size recovery |
|---|---|---|
| $\lVert M \rVert \approx 12$ | violated | $0.63$ |
| $\lVert M \rVert \approx 1$ | held | $0.50$ |

The operator that *fit the response well* sat far outside the regime where its bracket means anything. The operator whose bracket is *interpretable* could not fit the response, and paid an eighth of the effect-size score for staying small. These two demands pull in opposite directions, and no setting of the penalty satisfies both, because fitting the response wanted a large generator and the bracket picture wanted a small one.

That conflict indicts a premise this whole method series had carried since it first motivated the near-identity prior: that an intervention's effect is a *small* shift on a large, intervention-independent baseline. In *expression* space that is true, and it is why a perturbation moves a handful of genes among thousands. It does **not** survive the trip through the encoder into *latent* coordinates, where fitting the response wanted a generator of norm $12$, a large rotation rather than a small nudge.

> **What outlived the hypothesis.** The near-identity prior was imported from the wrong space. It is a true statement about expression and a false one about this latent geometry, and the operator inherited it by assumption rather than by measurement. This is not a fact about brackets or epistasis. It is a fact about what masked-prediction pretraining does and does not shape, and it survives the specific refutation intact.

The connection to the rest of the corpus is exact. [Chapter 1](01-what-an-operator-is.md) noted that retaining the perturbation signal is necessary but not sufficient; the encoder must also keep it in coordinates where the intervention is a small motion. The [ceiling analysis](../../../../docs/experimental-method/07-the-ceiling.md) confirmed the signal was retained, since a linear readout of the frozen latents beats the baseline. Round 4 confirmed the second condition failed, since the operator that fit was large. The two conditions are independent, our encoder met the first and failed the second, and *nothing in masked prediction had any reason to make it meet the second*, because that objective preserves information without shaping geometry.

## 6. What the refutation does and does not license

**It does not refute the framework.** Everything in §2 says why. The objective was a surrogate, half the algebra was unobservable, and the encoder premise was unearned, all before training began. A negative obtained under three amputated assumptions is strong evidence about *this application* and weak evidence about the framework, and pretending otherwise would be exactly the overreach this series was written to avoid.

**It does refute a specific claim in a specific place.** In the $T = 1$, simultaneous, unpaired, frozen-encoder setting, the non-commutativity of learned per-gene generators does not predict genetic interaction, and it does not for reasons that are now understood rather than mysterious: the objective under-identifies the operator ([Chapter 6](06-training.md)), the parameterization compounds that ([Chapter 4](04-basis-policy-least-action.md)), and the regime that fits the data is the regime where the bracket is meaningless ([Chapter 5](05-composition.md)). Each of those is a diagnosis, and each names a fix.

**And it tells you where to test the framework properly.** Every one of the three breaks is repaired by moving to the setting the framework was designed for. A temporal world model is paired, so the equivariance loss computes and the operator recovers its meaning. It is sequential, so the swap-odd part of the algebra becomes observable and composition is a real question rather than a symmetrized shadow. And it can train the encoder jointly, so the Koopman premise is earned rather than assumed. The odd part of the bracket, unobservable here, is precisely "the order of your plan matters," which is a first-class quantity in a world model and absent by construction in a perturbation screen. The honest reading of this chapter is not that the operator failed. It is that we tested it in the one corner where its distinctive machinery is invisible, and the corner told us so.

---

*Previous: [Training an operator](06-training.md). Next: [what comes next](08-what-comes-next.md). Up: [the series index](index.md). The numbers: [the results ledger](../conditional-flow-jepa/results-ledger.md).*
