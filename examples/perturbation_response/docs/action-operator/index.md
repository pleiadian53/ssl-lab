# Action operators: a development

*A careful build-up of the action-operator framework, from "what is an action, really" to a working model and its failure modes. Written from the far side of a refutation, which is the only reason it can be honest about the parts that usually go unstated.*

---

## Why this subject is hard

The action-operator idea can be said in one sentence: *an action is not a label you pick from a menu, it is a transformation you apply to a state.* People nod at that sentence and then get lost, and the reason is worth naming up front, because it shapes how this series is organized.

The idea originates in the sibling **GRL** project, *Generalized Reinforcement Learning: Actions as Operators on State Space*, where the policy constructs an operator rather than choosing an index, and classical reinforcement learning falls out as the special case where the operator family is a finite set of displacements. What follows carries that framework into a setting it was not designed for, one with no reward, no trajectory, and a system that is destroyed by being measured.

**The framework is a stack of premises, and every one of them is easy to state and easy to leave unchecked.**

| the premise | what it buys | what happens if it silently fails |
|---|---|---|
| the latent operator is a faithful image of the real one | the operator *means* something | you have fit an arbitrary matrix that predicts well and explains nothing |
| the encoder supplies coordinates where dynamics are linear | linearity is exact, not approximate | your linear map is the crude approximation it looks like |
| $A = \exp(M)$ comes from a flow | invertibility, an identity start, clean composition | $\exp$ is a reparameterization doing no work |
| effects are small, so $M$ stays near zero | the bracket measures interaction | the bracket degenerates into a measure of *magnitude* |
| the training objective can actually be computed | the operator learns what you intended | you have quietly trained a weaker surrogate |

Each row is a real thing that can go wrong, and here is the property that makes them dangerous: **a violated premise does not crash.** The model trains, the loss falls, the metric returns a plausible number, and nothing anywhere reports that the object you built has stopped meaning what you think it means.

This project violated the second and the fourth. We did not notice for a long time, and finding out is most of what this series has to teach.

## How this differs from the other operator writing

Three places in this repository talk about operators, and they are pitched differently on purpose.

- **[Action operators, the foundation](../../../../docs/action_operator/00-from-actions-to-operators.md)** is the short, domain-general on-ramp: what an operator is, a gallery of them, and the algebra of composing two. Read it first if the whole idea is new. It assumes no background and it deliberately does not go deep.
- **[Operator world models](../../../../docs/operator_world_models/index.md)** takes the same machinery forward in time: rollout, action-conditioned dynamics, planning. That is where sequences of actions live.
- **This series** develops the theory rigorously with a single application driving it, and states each premise together with the diagnostic that checks it. It is the long version, and it is where the mathematics is derived rather than asserted.

The application is perturbation response: what a cell does when you activate a gene. That setting turns out to be an unusually *hostile* environment for the framework, and hostility is pedagogically useful, because every premise that a friendly domain would let you skip is one this domain forces you to confront.

## The chapters

1. **[What an operator is, and what makes it mean anything](01-what-an-operator-is.md)** — the two-space picture (the real transformation you cannot write down, and the latent one you build), why moving to latent space is what lets you choose something simple, and the **commuting square** that is the difference between an operator and an arbitrary matrix. The conceptual crux of the whole framework, and the one most treatments mention in a sentence and move past.

2. **[Why the operator is linear](02-why-linear-koopman.md)** — the Koopman argument: you do not linearize the dynamics, you change what you track until they are already linear. The Koopman operator is *exactly* linear on observables for any nonlinear system, an invariant subspace turns it into a matrix, and that subspace **is** the encoder. Shows a system where the linearization is exact, explains why such coordinates exist (eigenfunctions), and proves that Chapter 1's commuting square and Koopman invariance are **the same equation**. Ends on the premise our own encoder never satisfied, and the cheap test that would have caught it.

3. **[Why $\exp(M)$](03-why-exp-of-m.md)** — four reasons a freely learned matrix is the wrong object, and the reframe that dissolves all four: $A$ is the **time-one map of a flow**, $\dot z = Mz$, so $M$ is the generator and $\exp$ is the solution of the differential equation rather than a parameterization. The one-parameter semigroup makes that structural. Invertibility, the identity at the origin, a flat space to emit into, collapsing repetition, meaningful interpolation, a readable spectrum, and the least-action penalty then all arrive as *consequences*. Includes the Lie picture (constrain $M$, constrain $A$ to a subgroup), the honest limits, and the failure mode this project actually hit: writing $\exp$ and never using the algebra it buys.

4. **[The anatomy: basis, policy, and least action](04-basis-policy-least-action.md)** — $M$ has $65{,}536$ entries, so where does it come from? The three objects between an intervention and its operator: the **basis** (the real design decision, since it fixes what is representable at all *and* confines every bracket the model can produce), the **policy** (direct or learned, on the state or on the condition alone, deterministic or a mixture), and the **energy**. Includes the gauge freedom that makes free-basis generators uninterpretable, why a deterministic operator *cannot* split a population, and a table of where this project set every dial.

5. **[Composition](05-composition.md)** — what happens when two interventions meet, and the framework's most distinctive claim: that their **interaction is the bracket of their generators**, computable without ever observing the pair. Establishes it, then measures its **radius of validity**, which yields the uncomfortable structural fact that the window where the bracket is trustworthy and the window where the interaction matters barely overlap. Then the symmetry argument: a *simultaneous* experiment sees the magnitude of non-commutativity but never its sign, because the raw bracket is swap-odd. Ends on the two ways this premise failed here.

6. **[Training an operator](06-training.md)** — the paired case, where the equivariance loss computes directly, and the unpaired case, where it cannot. Matching *distributions* rather than pairs is not a patch but the more general objective, of which per-pair error is the corner where the coupling is known. The energy distance, why it contains the $\Delta$ metric as its crudest case, and the objective-side origin of the identifiability failure: marginal matching identifies the operator's *effect*, not the operator.
7. **[The application, and what it taught](07-the-application.md)** — perturbation response as the $T=1$ corner, and the three things the domain amputates before training begins. What we built across two rounds, the refutation of the bracket-epistasis claim, and how it survived a confound that nearly faked it. The finding that outlived the hypothesis: the near-identity prior is true of expression and false of this latent geometry, so it was imported from the wrong space. Why a negative here is strong evidence about *this application* and weak evidence about the framework.
8. **[What comes next](08-what-comes-next.md)** — the experiments the diagnosis named, in order: the data-efficiency ladder that finally tests the self-supervised premise, the three ways to unfreeze the encoder (co-adapt with the decoder, with the operator, or fully) and what each costs in attribution, the low-rank basis that fixes identifiability without unfreezing, and the temporal world model where the framework's amputated assumptions are all restored at once.

## The honest status of the framework, as of this writing

Stated here rather than buried, because a reader deserves to know what they are being taught before they invest in it.

**The mathematics is sound and is not in question.** That commuting generators compose additively, that $\exp$ is the time-one map of a linear flow, that BCH produces only brackets: these are theorems, and where this series makes such a claim it is [checked numerically](../../../../dev/planning/action_operator/verify_bch.py).

**The framework's central empirical claim, in this domain, was refuted.** We built an operator whose non-commutativity should have predicted genetic interaction, and it did not, across four tests and every variant we tried. The [results ledger](../conditional-flow-jepa/results-ledger.md) records it as Round 4.

**And the refutation is scoped.** Perturbation response is a $T=1$, simultaneous, unpaired, frozen-encoder setting, which is the corner of the framework where its distinctive machinery has the least to offer and its training objective is a degraded surrogate. A negative there is weak evidence about the framework and strong evidence about *this* application of it. [Chapter 7](07-the-application.md) is careful about the difference, and [Chapter 8](08-what-comes-next.md) turns the diagnosis into the experiments worth running next.

That combination, sound mathematics plus a refuted application plus a diagnosis of why, is exactly what makes this worth writing down carefully. A tutorial written before the experiment would have asserted the premises. This one can tell you which ones held.

---

*Up: [perturbation response](../index.md). The domain-general on-ramp: [action operators](../../../../docs/action_operator/00-from-actions-to-operators.md). Forward in time: [operator world models](../../../../docs/operator_world_models/index.md). The deep formalism and the reinforcement-learning setting: the [GRL project](https://github.com/pleiadian53/GRL).*
