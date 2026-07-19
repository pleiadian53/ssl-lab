# 4. The anatomy: basis, policy, and least action

*Chapter 3 ended with $A = \exp(M)$, which is elegant and, taken literally, absurd: at $D = 256$ the generator has $65{,}536$ entries and you would need all of them for every intervention. Nobody emits that. This chapter is about the three objects that actually stand between an intervention and its operator, what each one decides, and which of them is the real design choice.*

> **Where this sits.** [Chapter 1](01-what-an-operator-is.md) gave the operator meaning, [Chapter 2](02-why-linear-koopman.md) gave it linearity, [Chapter 3](03-why-exp-of-m.md) gave it the exponential form. All three were about what the operator **is**. This chapter is about where it **comes from**, and it is the one with knobs on it.

---

## 1. The pipeline

Between the intervention and the operator sit three objects:

```
   c   ────→   α   ────→        M        ────→   A = exp(M)   ────→   z' = Az
        π            M = Σᵢ αᵢ Bᵢ
  the      the policy      the basis          the exponential
intervention              {B₁ ... B_m}          (Chapter 3)
```

Reading left to right: an intervention $c$ (a gene, a dose, a logged action) goes through a **policy** $\pi$ that emits a short coefficient vector $\alpha \in \mathbb{R}^m$; those coefficients combine a fixed set of **basis generators** $\{B_i\}$ into the generator $M$; the exponential turns $M$ into the operator.

The three carry very different weight:

| object | decides | how much it matters |
|---|---|---|
| the **basis** $\{B_i\}$ | what transformations are *representable at all* | **the main design decision of the framework** |
| the **policy** $\pi$ | how an intervention *selects* among them | matters, but it is a modeling choice within the basis |
| the **energy** $E(M)$ | how far the operator may *depart from doing nothing* | a prior, and the thing that keeps everything else honest |

Everything below is about why the first row is in bold.

## 2. Why a basis at all

The expansion

$$M = \sum_{i=1}^{m} \alpha_i B_i$$

says the generator lives in an $m$-dimensional subspace of the $D^2$-dimensional space of matrices. With $m = 16$ and $D = 256$ that is sixteen numbers per intervention instead of sixty-five thousand. But parameter count is the least interesting of the three things it buys.

**It bounds what can be represented.** Chapter 3 §5 showed that constraining $M$ constrains $\exp(M)$ to a subgroup. The basis is how you apply that constraint in practice: span a subspace of skew-symmetric matrices and every operator the model can express is a rotation, however the data pushes. Structure enters as a *fact about the architecture* rather than a penalty the optimizer may trade away.

**It can make $\alpha$ readable.** If $B_i$ is "the generator for sleep" and $B_j$ is "the generator for metformin," then $\alpha$ is not an opaque code, it is a description of what was done. That is a property of the basis, not of the network.

And a fourth consequence that only shows up when you compose:

**It confines the brackets.** Because the expansion is linear in the coefficients, the commutator of two generators expands as

$$[M_A, M_B] = \sum_{i,j} \alpha_i \beta_j [B_i, B_j],$$

so every bracket the model can produce lies in the span of the pairwise brackets of the basis. If that span is small, interactions are strongly constrained; if the basis is closed under brackets, meaning each $[B_i, B_j]$ is itself a combination of the $B_k$, then the generator space is a **Lie subalgebra** and the operator family is closed under composition. Composing two interventions can then never leave the family you designed. That is a strong and useful guarantee, and it is available only through the basis.

## 3. The four bases, and the dial they sit on

The implementations in [`context_operator.py`](../../../../src/ssllab/action_operator/context_operator.py) span the range deliberately.

| basis | construction | $\exp(M)$ lands in | what it assumes |
|---|---|---|---|
| **Free** | $m$ learnable dense matrices | $GL^{+}(D)$ | nothing beyond invertibility; the model discovers its own structure |
| **Named** | one learnable $B_i$ per labeled intervention | $GL^{+}(D)$ | that you can *name and log* the interventions that matter |
| **Skew** | $B_i^\top = -B_i$ | $SO(D)$ | the intervention rotates the state and changes no lengths |
| **$\mathfrak{se}(3)$** | six *fixed* generators, three rotation and three translation | $SE(3)$ | the state is a rigid body in space |

Read the table as a single **expressiveness against structure** dial. At the free end you can represent any invertible transformation, you need no domain knowledge, and you get very little back: the basis elements mean nothing individually (§4), and an over-expressive operator can absorb phenomena you would rather it reported. At the $\mathfrak{se}(3)$ end the generators are not even learned, every operator is guaranteed to be a physically valid rigid motion, and the model simply cannot express anything else, which is exactly right when that is true and fatal when it is not.

The two domains this corpus cares about sit near opposite ends, which is the point: a behavioral or biological system needs the free or named end because its "physics" is unknown, while a molecular one can use $\mathfrak{se}(3)$ because the symmetry is known exactly.

**Where a hybrid belongs.** A realistic system often wants named generators for the interventions it can log, plus a small free residual for everything unmeasured, with the residual's size tuned against how much interpretability degrades. That is the natural next experiment in this part of the design space and it has not been run here.

## 4. The gauge freedom, and why named bases are identifiable

Here is a property of the free basis that is easy to miss and that decides whether you can interpret anything.

The expansion $M = \sum_i \alpha_i B_i$ is unchanged if you mix the basis with any invertible $m \times m$ matrix $S$ and un-mix the coefficients:

$$B_i \mapsto \sum_j S_{ji} B_j, \qquad \alpha \mapsto S^{-1}\alpha .$$

Both produce exactly the same $M$, the same operator, and the same loss. So a learned free basis is determined only **up to a linear change of basis**, and the individual $B_i$ carry no meaning of their own: what the optimizer hands you is one arbitrary representative of a whole family of equivalent parameterizations.

> **The consequence.** Inspecting a single generator of a *free* basis and reporting what it does is not a valid readout. Only quantities invariant under the gauge, such as the span itself or $M$ as a whole, are meaningful.

A **named** basis breaks the gauge by fixing in advance what each $B_i$ is attached to. That is the deeper reason to prefer it when you can: not that it is more convenient to read, but that it makes the basis elements *identified objects at all* rather than an arbitrary frame.

## 5. The policy: three choices, each with teeth

The policy $\pi$ maps the intervention to coefficients. Three decisions live here.

**Direct, or learned.** At one extreme, `DirectInterventionPolicy` sets $\alpha = c$ outright: the quantified intervention log *is* the coefficient vector, and there is no network at all. A day's generator is then literally

$$M = n_{\text{sleep}} B_{\text{sleep}} + n_{\text{exercise}} B_{\text{exercise}} + n_{\text{meds}} B_{\text{meds}} + \cdots$$

with the counts read off the log. At the other, `MLPPolicy` learns $\alpha = \mathrm{MLP}(c)$ with a **zero-initialized final layer**, so $\alpha = 0$, $M = 0$, and $A = I$ at the start of training and the operator must earn every departure from doing nothing. The direct policy needs a named basis to make sense; the learned one works with any.

**On the state, or on the condition alone.** Does $\alpha$ depend on $z$ as well as $c$?

- $\alpha(z, c)$ is more expressive and is the general form: the same medication can do different things depending on where the system currently is.
- $\alpha(c)$ gives **one operator per intervention**, shared by every state it acts on.

This project chose $\alpha(c)$, and the reasoning is worth stating because it is a modeling claim and not a shortcut: an operator is a property of the *intervention*, not of the thing it acts on. The practical payoff is large, one matrix exponential per intervention rather than per sample, but the assumption is real. It says the intervention's effect is context-independent in latent coordinates, and a temporal world model may well want to relax it.

**Deterministic, or stochastic.** A deterministic $\alpha$ gives one operator per intervention. Making it Gaussian, $\alpha \sim \mathcal{N}(\mu(c), \sigma(c))$, gives a *mixture* of operators, and the reason to want one is structural rather than cosmetic:

> $\exp(M)$ is always invertible, so it is a diffeomorphism, so it **preserves the number of modes** of any cloud it transports. A deterministic operator can rotate, scale, and shear a population but it **cannot split it**, and it barely widens it. If the response you are modeling is genuinely multimodal, some cells going one way and some another, a deterministic operator cannot represent that no matter how it is trained.

A distribution over $\alpha$ is the cheapest repair, since a mixture of diffeomorphisms need not be one. It is not a complete one: the mixture only separates into clean modes if the distribution over $\alpha$ is itself multimodal and its branches push to well separated regions, which is fragile.

## 6. Least action, and a gauge trap

The energy is the third object, and Chapter 3 §6 already gave it two justifications: it encodes "an intervention is a small structured change," and it is what bounds a composed plan before you run it. Here is how it interacts with the basis.

The natural penalty is on the generator itself,

$$E(M) = \lVert M \rVert_F^2 ,$$

and this is the right choice for a reason that §4 makes obvious once seen: **it is gauge-invariant in the way that matters.** Penalizing $\lVert \alpha \rVert$ instead would not be, because the gauge transformation rescales $\alpha$ and $B$ against each other. Shrink every basis element by ten and inflate every coefficient by ten and the model is *identical* while $\lVert \alpha \rVert$ has grown a hundredfold. A penalty on coefficients is therefore a penalty on an arbitrary frame, and the optimizer can escape it for free by rescaling the basis.

> **Implementation note.** Penalize $M$, or fix the basis scale and then penalize $\alpha$. Penalizing $\alpha$ alone, with a learnable basis and nothing pinning its scale, regularizes nothing.

## 7. The premise, and its silent failure

The premise of this chapter:

> The basis spans the transformations that actually occur, and its elements mean what you think they mean.

Both halves fail quietly.

**Span failure.** If the true transformation lies outside $\mathrm{span}\{B_i\}$, the model does not report this. It fits the best available element of the subspace, the loss falls to whatever floor the subspace allows, and every metric returns a number. A structurally incapable model looks exactly like a model that has found the data hard. The only way to see it is to *vary the basis* and watch whether the floor moves: if enlarging the span improves the fit, the smaller span was the binding constraint.

**Identification failure.** With a free basis, §4's gauge freedom means individual generators are not identified, so any per-generator interpretation is reading structure into an arbitrary frame. This one is worse than the span failure because the output looks *more* meaningful, not less: eigenvalues of $B_3$ are perfectly computable and perfectly arbitrary.

There is a third failure that this project actually hit, and it is a cousin of both. Our operator-algebra round used **no basis at all**: it gave each gene its own dense $65{,}536$-parameter generator and fit each one to make a single cloud match. Under that arrangement nothing pins the generator except its action on one population, so most of the matrix lives in directions the loss never touched, and quantities built from it, brackets in particular, were dominated by whatever happened to fill those directions. A low-rank shared basis is exactly the structural fix, since it would have confined the brackets to $\mathrm{span}\{[B_i, B_j]\}$ by §2. Chapter 7 tells that story with its numbers.

## 8. Where this project set every dial

Stated plainly, because the settings explain a good deal of what happened.

| dial | our setting | consequence |
|---|---|---|
| basis | **free**, $m = 16$ (round 3); then **none**, one dense generator per gene (round 4) | no structure imposed, and in round 4 no low-rank constraint either, which is where identifiability failed |
| policy input | **condition only**, $\alpha(c)$ | one operator per intervention: cheap, and the right model for a single intervention |
| policy form | learned MLP, zero-initialized head | starts at the identity, as designed |
| stochasticity | deterministic by default | cannot split a cloud, so multimodal responses are out of reach by construction |
| energy | Frobenius on $M$ | gauge-correct, and the dial whose setting turned out to conflict with fidelity |

The pattern is that we sat near the **expressive, unstructured** end of every dial. That is the defensible default when the domain's structure is genuinely unknown, and it is also the end of the dial that gives you the least back: no interpretable generators, no confined brackets, no representable multimodality. Round 4 is what it looks like to ask an unstructured operator for a structural answer.

---

*Previous: [Why $\exp(M)$](03-why-exp-of-m.md). Next: [composition](05-composition.md), and what the algebra says about interacting interventions. Up: [the series index](index.md).*
