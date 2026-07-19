# 2. Why the operator is linear

*A cell's response to an intervention is a tangled nonlinear cascade, and we propose to model it with a matrix. That should bother you. The defence turns on an idea from 1931: you do not linearize the dynamics, you change what you are tracking until they are already linear. This chapter derives that carefully, shows a case where it is exact, and then states the premise it rests on, which our own encoder never satisfied.*

> **Where this sits.** [Chapter 1](01-what-an-operator-is.md) established *what* the latent operator is and *what makes it mean anything*: the commuting square, which forces the built operator to be the faithful image of the real one. It left the operator's **form** open. This chapter closes that: why $f_\theta$ should be linear at all. The two turn out to be the same equation read twice, which is §6.

---

## 1. The objection, at full strength

Activating a gene sets off a cascade. Transcription factors bind, feedback loops engage, downstream targets rise and fall on their own timescales, and hours later the cell settles somewhere new. There is no honest sense in which that process is linear.

A linear map is the most restrictive interesting function there is. It cannot saturate. It cannot threshold. It cannot produce a response that depends on the starting state in any way except proportionally. Double the input and you exactly double the output, always. Biology violates all three constantly, and any biologist reading $z' = Az$ is entitled to stop there.

The answer is not "it is a reasonable approximation." The answer is that **the objection is aimed at the wrong space**, and seeing why requires changing what we track.

## 2. Koopman's move: track functions, not states

In 1931 Bernard Koopman pointed out something that still reads like sleight of hand. Take any dynamical system on a state $s$,

$$s_{t+1} = F(s_t),$$

with $F$ as nonlinear as you like. Instead of tracking the *state*, track **observables**: functions that read some number off the state,

$$g : \mathcal{S} \to \mathbb{R}.$$

An observable is any measurement you could make. "The expression of gene 12" is an observable. So is "the total UMI count," and so is "the square of the expression of gene 12," and so is any nonlinear function of the state you care to name.

Now define the **Koopman operator** $\mathcal{K}$, which advances an observable by one step of the dynamics:

$$(\mathcal{K}g)(s) = g\big(F(s)\big).$$

Read that slowly, because the types matter. $\mathcal{K}$ eats a *function* $g$ and returns a *function*: the one that answers "what will $g$ read after one step, as a function of where we are now."

**Claim: $\mathcal{K}$ is linear. Exactly, always, for every $F$, however nonlinear $F$ is.** The proof is one line. Take two observables $g_1, g_2$ and two scalars $\alpha, \beta$:

$$\big(\mathcal{K}(\alpha g_1 + \beta g_2)\big)(s) = (\alpha g_1 + \beta g_2)\big(F(s)\big) = \alpha g_1\big(F(s)\big) + \beta g_2\big(F(s)\big) = \big(\alpha \mathcal{K}g_1 + \beta \mathcal{K}g_2\big)(s).$$

Nothing was assumed about $F$. The linearity comes from the fact that $\mathcal{K}$ acts by *composition with $F$ on the inside*, and composition distributes over sums of functions for free.

The distinction that trips everyone, worth stating flatly:

> **$\mathcal{K}$ is linear in the observable, not in the state.** The nonlinearity of $F$ has not been approximated away, ignored, or linearized about a fixed point. It has been **moved**, out of the operator and into the arguments the operator acts on.

Nothing is free, and here is the bill. The space of all observables is a *function space*, and it is infinite-dimensional. Koopman trades a nonlinear finite-dimensional problem for a linear infinite-dimensional one. Whether that is a good trade depends entirely on the next step.

## 3. The bargain: a finite invariant subspace

An infinite-dimensional linear operator is not something you can store in a checkpoint. But you may not need all of it.

Suppose you can find a *finite* set of observables $g_1, \dots, g_D$ whose span is **closed** under $\mathcal{K}$: applying $\mathcal{K}$ to any one of them returns a combination of the same $D$ functions and never anything outside. In symbols, for each $i$ there are coefficients $A_{ij}$ with

$$\mathcal{K}g_i = \sum_{j=1}^{D} A_{ij} g_j .$$

Such a span is a **Koopman-invariant subspace**, and on it $\mathcal{K}$ is no longer abstract. It is the $D \times D$ matrix $A$.

Now watch what those observables are when you stack them into a vector:

$$z = \big(g_1(s), g_2(s), \dots, g_D(s)\big) = E(s).$$

**That is an encoder.** And the dynamics of $z$ follow immediately:

$$z'_i = g_i(s') = g_i\big(F(s)\big) = (\mathcal{K}g_i)(s) = \sum_j A_{ij} g_j(s) = (Az)_i,$$

so $z' = Az$, exactly. The latent operator of Chapter 1 *is* the Koopman operator restricted to the span of the encoder's coordinates.

> **The reframe this chapter exists for.** Linearity is not an assumption about the biology. It is a **demand on the encoder**. You are not claiming the cascade is linear; you are asking the representation to be a set of coordinates in which it already is. The nonlinearity is not approximated, it is absorbed into the definition of the coordinates.

A note on conventions, since sources differ and the mismatch causes real confusion: with the convention above, $\mathcal{K}g_i = \sum_j A_{ij} g_j$, the latent evolves as $z' = Az$ with no transpose. Some treatments expand in coordinates the other way and carry an $A^\top$. The content is identical; only the bookkeeping differs.

## 4. A case where it is exact

The reframe is a promise until you see it pay. The classic demonstration is small enough to check by hand.

Take the system

$$\dot{s}_1 = \mu s_1, \qquad \dot{s}_2 = \lambda\big(s_2 - s_1^2\big),$$

with constants $\mu, \lambda$ and the dot meaning a time derivative. This is genuinely nonlinear: the $s_1^2$ term bends every trajectory around a parabola, and no change of *state* variables removes it.

Now add **one** observable. Track $s_1$, $s_2$, and the square:

$$y_1 = s_1, \qquad y_2 = s_2, \qquad y_3 = s_1^2 .$$

Differentiate each, using the chain rule on the third:

$$\dot{y}_1 = \mu s_1 = \mu y_1,$$
$$\dot{y}_2 = \lambda(s_2 - s_1^2) = \lambda y_2 - \lambda y_3,$$
$$\dot{y}_3 = 2 s_1 \dot{s}_1 = 2 s_1 (\mu s_1) = 2\mu s_1^2 = 2\mu y_3 .$$

Every right-hand side is a linear combination of $y_1, y_2, y_3$. In matrix form:

$$\frac{d}{dt}\begin{pmatrix} y_1 \\ y_2 \\ y_3 \end{pmatrix} = \begin{pmatrix} \mu & 0 & 0 \\ 0 & \lambda & -\lambda \\ 0 & 0 & 2\mu \end{pmatrix} \begin{pmatrix} y_1 \\ y_2 \\ y_3 \end{pmatrix}.$$

A nonlinear system in two dimensions became an **exactly linear** system in three. Not approximately, not to first order, not near a fixed point: exactly, everywhere, for all time. The price was one extra coordinate, and the coordinate we added was precisely the nonlinearity that was causing the trouble.

The system did not change. The coordinates did. That is the entire Koopman argument, and everything else is about when you can find such coordinates and what it costs when you cannot.

## 5. Why finite invariant subspaces exist at all

The worked example might look like a lucky accident. It is not, and the reason is worth knowing because it tells you what kind of object you are asking the encoder to learn.

A **Koopman eigenfunction** is an observable $\varphi$ that the dynamics merely rescale:

$$\mathcal{K}\varphi = e^{\lambda}\varphi .$$

Such a function evolves as a pure exponential along any trajectory, $\varphi(s_t) = e^{\lambda t}\varphi(s_0)$, no matter how tangled the underlying motion is. Eigenfunctions are the dynamics' own natural coordinates.

Two facts follow. The span of any finite collection of eigenfunctions is automatically invariant, since each is mapped to a multiple of itself. And products of eigenfunctions are eigenfunctions, with the exponents adding. That is exactly why $s_1^2$ closed the example above: $s_1$ was an eigenfunction with rate $\mu$, so its square is one with rate $2\mu$.

So "find a Koopman-invariant subspace" is really "find enough of the dynamics' eigenfunctions, or enough of their products, to close." When such a finite set exists, the linear model is exact. When it does not, the honest statement is the next section.

## 6. The premise and the bridge are the same equation

Here is the unification that justifies developing these two chapters in this order, and it is the sharpest thing in the series.

Chapter 1 said the operator means something only when the commuting square holds:

$$E\big(\hat{O}(s)\big) = f_\theta\big(E(s)\big).$$

Now impose the *form* this chapter argues for, $f_\theta(z) = Az$, and read the square coordinate by coordinate. The $i$-th component of the left side is $g_i(\hat{O}(s))$, which is $(\mathcal{K}g_i)(s)$ by the definition of the Koopman operator. The $i$-th component of the right side is $\sum_j A_{ij} g_j(s)$. So the square commutes for every $s$ precisely when

$$\mathcal{K}g_i = \sum_j A_{ij} g_j \quad \text{for every } i,$$

which is **the definition of Koopman invariance**.

> **One equation, two readings.** *Equivariance* asks whether the operator is a faithful image of the real transformation. *Koopman invariance* asks whether the encoder's coordinates close under the dynamics. With a linear operator, these are the same statement. Chapter 1's bridge and this chapter's premise are not two conditions to satisfy separately; satisfying either with a linear $f_\theta$ is satisfying both.

That is also why training against the equivariance loss *is* training toward Koopman coordinates, and why an encoder trained without it has no reason to supply them.

## 7. What "approximately invariant" actually costs

Real encoders will not be exactly invariant, so the useful question is what the error looks like. Write the residual explicitly:

$$\mathcal{K}g_i = \sum_j A_{ij} g_j + r_i ,$$

where $r_i$ is whatever part of $\mathcal{K}g_i$ falls **outside** the span. Then the linear latent model is wrong by exactly $r$, and $\lVert r \rVert$ is the precise meaning of "how non-Koopman are these coordinates."

Two consequences worth carrying:

**The error is a property of the encoder, not of the operator.** No amount of cleverness in choosing $A$ removes $r$, because $r$ lives in directions the span cannot represent. If your latent is badly non-invariant, a better matrix will not save you and a *nonlinear* latent operator is treating the symptom.

**Errors compound under rollout.** One step costs $r$; $k$ steps of feeding the model's own output back in cost roughly $k$ accumulations of it, with no observation to re-anchor you. This is why the $T = 1$ setting of this series is forgiving and a long temporal rollout is not.

## 8. Control: a family of operators, one per intervention

Classical Koopman theory is for **autonomous** systems: one fixed $F$, one $\mathcal{K}$. Our setting has an intervention, and honesty requires saying that this is a generalization rather than the classical theorem.

Each intervention induces its own dynamics $F_c$, hence its own Koopman operator $\mathcal{K}_c$, hence its own matrix on the invariant subspace:

$$z' = A(c) z .$$

So what the model learns is a **family of Koopman operators indexed by the intervention**, which is exactly what a policy emitting coefficients from a condition produces. Two caveats belong here rather than in a footnote. The subspace must be invariant for *every* $c$ in the family, which is a stronger demand on the encoder than invariance for a single dynamics. And the theory for controlled Koopman systems is less clean than the autonomous case, so this is a well-motivated extension, not a theorem we are inheriting.

## 9. The premise our own encoder never satisfied

Everything above is conditional. Strip the conditional away and the framework is exactly the crude approximation §1 accused it of being. So it matters a great deal *how* a project comes to have Koopman coordinates.

Methods built on this premise satisfy it **by construction**. Deep Koopman approaches train the encoder *jointly* with the linear operator, so gradient descent pushes the representation toward coordinates in which the dynamics close. The linearity is not hoped for, it is optimized for. That joint training is the entire point, and by §6 it is the same thing as training against the equivariance loss.

**Our encoder was never asked.** Stage A trains a JEPA encoder by masked prediction: infer the embedding of held-out gene groups from the groups you can see. It is a good objective and it produces a good representation, and **nothing in it mentions dynamics, interventions, or decoding**. There is no term rewarding a latent in which perturbation responses compose linearly. We then froze that encoder and attached an operator to it.

> **The honest status.** We adopted the Koopman *form* without the Koopman *training*. The operator is linear because we made it linear, and the argument licensing that choice assumes a property of the encoder our pipeline never optimized for and never checked.

This is one of the two premises the [series index](index.md) warns can fail silently, and it failed silently here for months.

## 10. How you would actually check it

The premise is testable, and cheaply, which makes the omission worse rather than better.

**Do not test it on the downstream benchmark.** We have direct evidence that the benchmark cannot answer this question: the linear operator tied a free nonlinear velocity field, which *looks* like evidence that the latent dynamics are close to linear. It is worth almost nothing, because a [ceiling analysis](../workflows/03-diagnose.md) later showed that stage had at most $0.03$ of headroom. A metric with three points of room cannot distinguish a linear transition from a nonlinear one, so the tie is equally consistent with "the coordinates are nearly invariant" and "this benchmark could not have noticed either way." That is a genuine confound and it should not be argued around.

**Test it on the latent transition itself.** Fit a linear map from control latents to perturbed latents, fit a flexible nonlinear map on the same pairs, and compare how well each predicts $z'$ rather than how well each scores downstream. This is the probe logic applied to dynamics: the linear map is the weak instrument, the residual gap is an estimate of $\lVert r \rVert$, and neither answer can be manufactured by the readout.

- If the linear map does nearly as well, the coordinates are close to invariant and the operator's central assumption is **earned**.
- If the nonlinear map wins decisively, the latent is not a Koopman embedding, the linearity really is the crude approximation of §1, and the honest fix is to train an encoder that makes it true rather than to add expressiveness downstream.

## 11. What this chapter licenses, and what it does not

**It licenses the form.** A linear latent operator is not a crude approximation of a nonlinear cascade. It is *exact* on a Koopman-invariant subspace, and the price of exactness is paid in coordinates rather than in fidelity. When such coordinates exist, "linear" costs nothing at all.

**It does not license our instance.** The argument is conditional on an encoder that supplies those coordinates. Ours was trained by masked prediction, with no dynamical objective, and then frozen. The form is defensible; the instance is unearned, and unearned in a way that is measurable rather than merely arguable.

**And it says what to do about it.** By §6, the fix is not a vaguer hope for a better representation. Koopman names the objective: train the encoder so that the intervention's dynamics close linearly in its coordinates, which is the equivariance loss of Chapter 1 applied to the encoder rather than only to the operator.

---

*Previous: [What an operator is](01-what-an-operator-is.md). Next: why $\exp(M)$, and what the generator buys. Up: [the series index](index.md).*
