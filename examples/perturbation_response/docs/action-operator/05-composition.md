# 5. Composition: what happens when two interventions meet

*Everything so far concerned one operator. This chapter is about two, and it is where the framework earns or loses its most distinctive claim: that the **interaction** between two interventions is an algebraic property of their generators, computable without ever observing the pair. The claim is true, with a radius of validity that we can measure, and the measurement turns out to explain a great deal about why this project's central experiment failed.*

> **Where this sits.** [Chapter 4](04-basis-policy-least-action.md) built the generator out of a basis and showed that this confines every bracket the model can produce. This chapter says what a bracket *is* and what it measures. The domain-general version of the algebra, with the proofs worked in full, is [the algebra of composition](../../../../docs/action_operator/03-the-algebra-of-composition.md); here the emphasis is on what a *simultaneous, unpaired* experiment can and cannot see of it.

---

## 1. Two interventions, and the question

Apply intervention $A$, then intervention $B$. Chapter 1 established composition is the product, read right to left:

$$z'' = \exp(M_B)\exp(M_A) z .$$

The question this chapter answers is what that composite is *as an object*. Specifically: is there a generator for it, and if so, how does it relate to $M_A$ and $M_B$?

The naive hope is that generators add, $M_{AB} = M_A + M_B$. If that were always true, composing interventions would be bookkeeping, and two interventions together would do exactly what the two do separately. It is sometimes true, and **when it fails is precisely the interesting part**, because a departure from additivity is what a geneticist calls **epistasis** and what a clinician calls the interaction between two treatments.

## 2. The commutator is the exact measure of non-additivity

Define the **commutator** of two matrices:

$$[X, Y] = XY - YX .$$

It is zero exactly when $X$ and $Y$ commute. And it is the precise answer to our question:

$$[M_A, M_B] = 0 \quad \Longrightarrow \quad \exp(M_A)\exp(M_B) = \exp(M_A + M_B).$$

The proof is the binomial theorem. Multiplying the two exponential series and collecting terms of degree $n$ gives $\sum_{i+j=n} \frac{1}{i!j!}X^iY^j$, which assembles into $\frac{1}{n!}(X+Y)^n$ only if you may reorder $X$ past $Y$ while gathering. For $n = 2$ it is visible without bookkeeping: $(X+Y)^2 = X^2 + XY + YX + Y^2$ equals $X^2 + 2XY + Y^2$ only when $XY = YX$.

Near the identity the implication runs both ways, so:

> **Commuting generators compose additively. Non-commutativity IS the departure from additivity.** The interaction between two interventions is not merely *measured by* the bracket; in this framework it *is* the bracket.

That is the framework's most distinctive claim, and it is worth appreciating what it would buy. Non-commutativity is computable from the two single-intervention generators alone. If it holds, you can predict how two interventions will interact **without ever having observed them together**.

## 3. Baker-Campbell-Hausdorff, and why only brackets appear

When the generators do not commute, the composite is still an operator, and near the identity it is still the exponential of something. The Baker-Campbell-Hausdorff formula says what:

$$\log\big(\exp(X)\exp(Y)\big) = X + Y + \tfrac{1}{2}[X,Y] + \tfrac{1}{12}\big([X,[X,Y]] + [Y,[Y,X]]\big) - \cdots$$

Read it as the naive sum plus an infinite tail of corrections. Every correction is built from nested commutators, so if $[X,Y] = 0$ the entire tail vanishes and §2 is recovered.

The structural fact worth carrying is that the series contains **only** nested commutators. Not $XY$, not $X^2$, not $XY + YX$. The reason is that generators live in a **Lie algebra**, a vector space closed under the bracket but *not* under plain matrix multiplication, and the logarithm of a group element has to land back in that algebra. The bracket is the only product the algebra is closed under, so it is the only one BCH may use.

The skew-symmetric case makes this concrete. If $X^\top = -X$ and $Y^\top = -Y$ then $[X,Y]^\top = -[X,Y]$, so the bracket of two rotation generators is again a rotation generator. But $(XY)^\top = YX \ne -XY$ in general: the plain product has fallen out of the algebra and no longer describes a rotation at all. Multiplication destroys the structure that made the generator meaningful; the bracket preserves it.

## 4. The radius of validity, measured

BCH is a series, and series have radii of convergence. This matters far more than it usually gets credit for, because the entire "interaction is a bracket" story is a statement about the *generator of the composite*, and outside the radius that generator is not given by any bracket expansion.

The composite itself always exists: $\exp(M_A)\exp(M_B)$ is a perfectly good invertible matrix for any generators whatsoever. What breaks down is the claim that its **interaction term is a bracket**.

Here is the measurement ([`verify_bch_radius.py`](../../../../dev/planning/action_operator/verify_bch_radius.py)). For random generators at a given norm, compare the true interaction, $\log(\text{composite}) - (M_A + M_B)$, against the second-order bracket prediction:

| $\lVert M \rVert$ | interaction, as a fraction of the composite | error after the bracket correction | improvement |
|---|---|---|---|
| $0.1$ | $0.0002$ | $0.0000$ | $3480\times$ |
| $0.5$ | $0.0071$ | $0.0001$ | $120\times$ |
| $1.1$ | $0.0298$ | $0.0013$ | $22\times$ |
| $2.0$ | $0.101$ | $0.015$ | $6.8\times$ |
| $5.0$ | $0.744$ | $0.705$ | $1.1\times$ |
| $12.4$ | $0.942$ | $3.224$ | $0.3\times$, **worse than assuming additivity** |

Read the two ends. Below about $\lVert M \rVert \approx 1$ the bracket describes the interaction essentially exactly. Above about $5$ it describes it not at all, and by $12$ the correction *overshoots by more than three times the whole composite*, so applying it is worse than pretending the interventions are additive.

> **The uncomfortable structural fact.** Look down the second column as well. Where the bracket picture is valid, the interaction is **tiny**: at $\lVert M \rVert = 1.1$ the composite is $97\%$ additive. Where the interaction is large, at $\lVert M \rVert = 12$ it is $94\%$ of the composite, the bracket picture has already broken. **The window in which the bracket is trustworthy and the window in which the interaction matters barely overlap.**

That is not a fact about our data. It is a fact about the algebra, and it says something sobering about the whole programme: an operator model tuned to fit large effects has, by that very act, left the regime in which its interaction story means anything.

## 5. What the basis confines

Chapter 4 §2 showed that with $M = \sum_i \alpha_i B_i$,

$$[M_A, M_B] = \sum_{i,j} \alpha_i \beta_j [B_i, B_j],$$

so every interaction the model can express lies in the span of the basis's pairwise brackets. Two consequences now visible:

**A small basis is a strong prior on interactions.** If $\mathrm{span}\{[B_i,B_j]\}$ is low-dimensional, the model can only represent a correspondingly restricted set of interactions, whatever the data says. If the basis is closed under brackets, composition never leaves the family.

**No basis at all is no prior at all.** Give each intervention its own unconstrained dense generator and the bracket can point anywhere in $D^2$ dimensions, including directions the training objective never constrained. That is not a hypothetical failure mode; §8 is about the round in which it happened.

## 6. What a simultaneous experiment can see

Now the part specific to this domain, and the sharpest thing in the chapter.

Every matrix product splits into two halves:

$$XY = \underbrace{\tfrac{1}{2}(XY + YX)}_{\text{anticommutator } \{X,Y\}} + \underbrace{\tfrac{1}{2}(XY - YX)}_{\text{commutator } [X,Y]} ,$$

and they behave oppositely under swapping the two arguments. Writing $\sigma$ for the swap $A \leftrightarrow B$:

| object | under $\sigma$ | |
|---|---|---|
| $[X,Y]$ | $\mapsto -[X,Y]$ | **odd** |
| $\{X,Y\}$ | $\mapsto \{X,Y\}$ | **even** |

Now look at the experiment. In Perturb-seq both guides are delivered **at the same time**. The measured object is "cells that received $A$ and $B$," and there is no first and no second. **The observation is invariant under the swap.**

But the leading BCH correction is $\tfrac{1}{2}[M_A, M_B]$, which is swap-*odd*. A naive ordered product would therefore predict two different answers, one for each labelling, for a physical event that has only one. The model must be made swap-invariant, and how the invariance is achieved determines what survives.

Average the two orderings and the odd part cancels exactly:

$$\tfrac{1}{2}\Big(\log(e^{M_A}e^{M_B}) + \log(e^{M_B}e^{M_A})\Big) = M_A + M_B + \tfrac{1}{12}\big([M_A,[M_A,M_B]] + [M_B,[M_B,M_A]]\big) + \cdots$$

The first-order bracket is gone. What remains at leading order is a **double** commutator, which is swap-even, and whose size is governed by $\lVert [M_A,M_B] \rVert^2$ rather than by the bracket itself.

> **A simultaneous experiment can observe the magnitude of non-commutativity, never its sign.** The odd part, the raw bracket that carries the *direction* of non-commutativity, is unobservable by construction: seeing it would require doing $A$ then $B$ and comparing against $B$ then $A$, and the experiment does neither.

This is the precise sense in which perturbation response is the *simple* corner of the framework, and the reason is not "at most two genes." It is that **the experiment's symmetry group is larger**, so it probes a *smaller* part of the algebra. The even part is epistasis; the odd part is the path-dependence of a plan, and it becomes observable only when interventions are sequential.

**On implementation.** The symmetric (Strang) product $\exp(M_A/2)\exp(M_B)\exp(M_A/2)$ is the cheap way to build a swap-motivated composite, and it also cancels the first-order bracket. It is worth noting that it is not the same object as the even-average above and does not share its coefficients: its own expansion is $M_A + M_B - \tfrac{1}{24}[M_A,[M_A,M_B]] + \tfrac{1}{12}[M_B,[M_B,M_A]] + \cdots$. Both are second-order accurate and both kill the odd term; conflating their correction terms is an easy and real mistake.

## 7. More than two, and the shape of a plan

For $T$ interventions the honest object is the **time-ordered product**

$$z_T = \Big(\prod_{k=T-1}^{0} \exp(M_{c_k})\Big) z_0 = \exp(M_{c_{T-1}}) \cdots \exp(M_{c_0}) z_0 ,$$

read right to left, earliest first. This is still a single matrix you can form and inspect, which is what makes a plan a first-class object.

What you lose relative to a single intervention is that the *generator* of a plan is no longer a simple sum. Writing it as $\exp\big(\sum_k M_{c_k}\big)$ is the **commutative, or Lie-Trotter, approximation**: exact when all the generators commute, or in the continuous limit of infinitesimally fine steps, and wrong by the bracket series otherwise. Any model that aggregates a sequence of interventions by summing their coefficients inside one exponential has silently assumed they all commute, which is to say it has assumed away every interaction it might have reported.

That distinction matters for planning, since it is exactly the difference between "a week of these interventions" as a bookkeeping sum and as an ordered object whose value depends on the order you chose.

## 8. The premise, and the two ways it failed here

The premise of this chapter:

> Interventions combine by composition in the group, so their interaction is the bracket of their generators.

It failed twice in this project, in different ways, and both are instructive.

**First failure: the algebra was never engaged.** The initial operator round composed two genes not in the group but in the *embedding*: the gene-set encoder mapped a pair to a summed embedding, a policy turned that into coefficients, and $\exp$ was applied once to the result. No product of operators was ever formed, so no bracket was ever computed, and the interaction was carried by a black-box network rather than by the algebra. The matrix exponential was doing the work of a reparameterization and never the work of a capability, which is Chapter 3 §8's failure mode showing up at the composition layer. Worse, composition-in-the-embedding is exactly what the from-scratch baseline also did, so the operator had no combination mechanism the baseline lacked.

**Second failure: the algebra was engaged outside its radius.** The round that fixed the first failure gave each gene its own generator and composed in the group, and its central prediction, that $\lVert [M_A, M_B] \rVert$ should track measured genetic interaction, was refuted across every variant tried. The post-mortem is Chapter 7, but §4 of this chapter supplies the part that is pure algebra: the arm that fit the data best had $\lVert M \rVert \approx 12.4$, where the bracket correction is *worse than assuming additivity*, so its brackets could not have carried interaction information whatever the model had learned. Constraining the generators back to $\lVert M \rVert \approx 1.1$ restored the regime where the bracket picture is quantitatively sound, and the prediction still failed, which is what makes the refutation credible rather than an artifact.

The general lesson is the one §4 already stated and is worth repeating as a design rule:

> **If you intend to read interactions off brackets, you must keep the generators in the regime where brackets mean something, and you must check that you have.** The check is one line, a median generator norm, and it distinguishes "my hypothesis is wrong" from "my run could not test my hypothesis." Those two look identical in the endpoint and they call for opposite responses.

---

*Previous: [The anatomy](04-basis-policy-least-action.md). Next: [training an operator](06-training.md), and what changes when the data comes unpaired. Up: [the series index](index.md). The full algebra with its proofs: [the algebra of composition](../../../../docs/action_operator/03-the-algebra-of-composition.md).*
