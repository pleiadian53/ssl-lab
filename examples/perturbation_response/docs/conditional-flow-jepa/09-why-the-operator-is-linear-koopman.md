# Why the Operator Is Linear: the Koopman Argument

*A cell's response to a perturbation is a tangled nonlinear cascade, and we propose to model it with a matrix. That should bother you. This chapter is the defense, and it turns on a hundred-year-old idea: you do not linearize the dynamics, you choose coordinates in which they already are linear. It also names the premise our own encoder never satisfied.*

> **Recap: where this sits.** [Modeling the transition](07-modeling-the-transition-action-operators.md) built the action operator $A_p = \exp(M_p)$, which transports a control cell's latent to its perturbed counterpart, and showed in its §6 that this operator is not a rival to the transport flow but the *same flow with its velocity field restricted* to be linear in the state and constant in time. [Modeling the readout](08-modeling-the-readout-count-decoder.md) then answered the decoder handoff. Chapter 7 §8 left four things open, and this chapter takes the first: **why should the latent operator be linear at all, and why $\exp(M)$ specifically?**
>
> **Prerequisites and notation.** We reuse the encoder $E$, which maps a cell's expression state $s$ to a latent $z = E(s) \in \mathbb{R}^{256}$, and the operator $A_p$ from Chapter 7. The algebra of composing two operators is developed separately in the foundation series, [The algebra of composition](../../../../docs/action_operator/03-the-algebra-of-composition.md). Every symbol is defined at first use.

---

## 1. Take the objection seriously

Activating a gene sets off a regulatory cascade. Transcription factors bind, feedback loops engage, downstream targets rise and fall on their own timescales, and the whole thing settles into a new state hours later. There is no honest sense in which that process is linear.

And we are proposing to model it as $z' = A z$: multiply by a matrix.

If that does not feel like a cheat, you have not looked at it hard enough. So let us state the objection in its strongest form. **A linear map is the most restrictive interesting function there is.** It cannot saturate, it cannot threshold, it cannot produce a response that depends on where the cell started in any way except proportionally. Biology does all three constantly. On its face, a matrix is the wrong object.

The answer is not "it is a reasonable approximation." The answer is that **the objection is aimed at the wrong space.**

## 2. Koopman's move: change what you track

In 1931 Bernard Koopman pointed out something that still sounds like a trick. Take any nonlinear dynamical system on a state $s$:

$$s_{t+1} = F(s_t),$$

where $F$ is the update rule, as nonlinear as you like. Instead of tracking the *state*, track **observables**: functions $g$ that read some scalar off the state, $g : \mathcal{S} \to \mathbb{R}$. An observable is any measurement you could make, such as "the expression of gene 12," or "the total UMI count," or "the square of the expression of gene 12."

Now define the **Koopman operator** $\mathcal{K}$, which advances an observable by one step:

$$(\mathcal{K}g)(s) = g\big(F(s)\big).$$

Read it slowly: $\mathcal{K}$ takes the *function* $g$ and returns a new *function*, the one that says "what $g$ will read after one step, as a function of where you are now."

Here is the claim. **$\mathcal{K}$ is linear. Exactly, always, for every $F$, no matter how nonlinear $F$ is.** The proof is one line:

$$\big(\mathcal{K}(\alpha g_1 + \beta g_2)\big)(s) = (\alpha g_1 + \beta g_2)\big(F(s)\big) = \alpha g_1\big(F(s)\big) + \beta g_2\big(F(s)\big) = \big(\alpha \mathcal{K}g_1 + \beta \mathcal{K}g_2\big)(s).$$

Nothing was assumed about $F$. The linearity comes from the fact that $\mathcal{K}$ acts by *composition with $F$*, and composition distributes over sums of functions for free.

The crucial distinction, and the one that trips everybody: **$\mathcal{K}$ is linear in the observable, not in the state.** The nonlinearity of $F$ has not been approximated away or ignored. It has been *moved*, out of the operator and into the observables the operator acts on.

Nothing is free, and here is the bill. The space of all observables is a **function space**, and it is infinite-dimensional. So Koopman trades a nonlinear finite-dimensional problem for a linear infinite-dimensional one. That is a real trade, not a free lunch, and whether it is a good one depends entirely on the next step.

## 3. The bargain: an invariant subspace, which is exactly what an encoder is

An infinite-dimensional linear operator is not something you can put in a checkpoint. But you may not need all of it.

Suppose you can find a *finite* collection of observables $g_1, \dots, g_D$ whose span is **closed** under $\mathcal{K}$: applying $\mathcal{K}$ to any combination of them returns another combination of the same $D$ functions, never anything outside. Such a span is a **Koopman-invariant subspace**. On it, $\mathcal{K}$ is no longer an abstract infinite-dimensional object. It is a $D \times D$ **matrix**.

And now look at what those $D$ observables are, collected into a vector:

$$z = \big(g_1(s), g_2(s), \dots, g_D(s)\big) = E(s).$$

**That is an encoder.** The observable map and the encoder are the same object. Which delivers the reframe this whole chapter exists for:

> **Linearity is not an assumption about the biology. It is a demand on the encoder.** You are not claiming the cascade is linear. You are asking the representation to be a set of coordinates in which it *already* is. The nonlinearity does not get approximated. It gets absorbed into the definition of the coordinates.

### A worked example, so this is not just a promise

The classic demonstration is two-dimensional and completely explicit. Take the system

$$\dot{s}_1 = \mu s_1, \qquad \dot{s}_2 = \lambda\big(s_2 - s_1^2\big),$$

where $\mu$ and $\lambda$ are constants and the dot means the time derivative. This is genuinely nonlinear: the $s_1^2$ term is a parabola, and trajectories bend around it.

Now add **one** observable. Track $s_1$, $s_2$, and the square:

$$y_1 = s_1, \qquad y_2 = s_2, \qquad y_3 = s_1^2.$$

Differentiate each, using the chain rule on the third:

$$\dot{y}_1 = \mu s_1 = \mu y_1,$$
$$\dot{y}_2 = \lambda(s_2 - s_1^2) = \lambda y_2 - \lambda y_3,$$
$$\dot{y}_3 = 2 s_1 \dot{s}_1 = 2 s_1 (\mu s_1) = 2\mu s_1^2 = 2\mu y_3.$$

Every right-hand side is a linear combination of $y_1, y_2, y_3$. In matrix form:

$$\frac{d}{dt}\begin{pmatrix} y_1 \\ y_2 \\ y_3 \end{pmatrix} = \begin{pmatrix} \mu & 0 & 0 \\ 0 & \lambda & -\lambda \\ 0 & 0 & 2\mu \end{pmatrix} \begin{pmatrix} y_1 \\ y_2 \\ y_3 \end{pmatrix}.$$

A nonlinear system in two dimensions became an **exactly linear** system in three. Not approximately. Not to first order. Exactly, everywhere, for all time. The price was one extra coordinate, and the coordinate we added was precisely the nonlinearity that was causing the trouble.

This is the entire Koopman argument in one example. The system did not change. The coordinates did.

## 4. Why $\exp(M)$, and not just any matrix

Chapter 7 §6 answered this from the flow side: restrict a velocity field to $v(z) = Mz$, integrate $\dot z = Mz$, and the time-1 map is $z(1) = \exp(M) z(0)$. The Koopman picture gives the same answer from the other direction, and the two turn out to be one story.

In continuous time you do not get a single Koopman operator but a **family**, one for each elapsed time $t$, written $\mathcal{K}_t$. That family has an obvious property: evolving for time $t$ and then for time $u$ is the same as evolving for time $t+u$, and evolving for no time does nothing:

$$\mathcal{K}_{t+u} = \mathcal{K}_t \mathcal{K}_u, \qquad \mathcal{K}_0 = I.$$

A family like that is called a **one-parameter semigroup**, and every such family is generated by a single object. There is an operator $L$, the **Koopman generator**, with

$$\mathcal{K}_t = \exp(t L).$$

Restrict to a $D$-dimensional invariant subspace and $L$ becomes a $D \times D$ matrix, which is our $M$. The time-1 operator is $\exp(M)$.

> **The punchline.** $\exp$ is not a parameterization trick chosen to make optimization convenient. **It is what the Koopman family is.** $M$ is the generator of the dynamics; $A = \exp(M)$ is the evolution those dynamics produce. Writing $A = \exp(M)$ is not a way of representing an operator. It is the statement that the operator *came from* a flow.

Every property Chapter 7 leaned on now arrives as a consequence rather than a design choice:

| property | why it follows |
|---|---|
| $A$ is invertible | $\exp(M)$ is invertible for every $M$, with inverse $\exp(-M)$. Dynamics run backwards. |
| $A = I$ at initialization | $\exp(0) = I$. A zero generator is a system that does not evolve, so "this perturbation does nothing" is the natural starting point rather than an imposed one. |
| effects are small departures | $\lVert M \rVert$ small means the generator is weak, so the least-action penalty is a statement about the *dynamics*, not a regularizer bolted on. |
| $M$ is easy to emit | generators live in a flat vector space, where a network can output anything; operators live on a curved manifold, where it cannot. |
| composition is clean | $\exp(M)^k = \exp(kM)$, and two different operators compose through the algebra developed in [the algebra of composition](../../../../docs/action_operator/03-the-algebra-of-composition.md). |

## 5. But there is no time in a Perturb-seq experiment

An honest objection to everything above: the Koopman story is about **dynamics**, and our setting has no clock. We do not observe a trajectory. We observe a control population and a perturbed population, and that is all.

The resolution is that the dynamics are real but unobserved. A cell hit with a CRISPR guide does not jump to its new state; it *runs* there, over hours of cascade, and then settles. The process exists. We simply measure only its endpoint, once, destructively.

So the operator models the **time-1 map of the response dynamics**, where "time 1" means "after the response has settled" and the normalization to $1$ is bookkeeping. This is the same move the transport flow already makes: its interpolation parameter $t$ running from $0$ to $1$ is not physical time either, just a dial that carries a control latent to a perturbed one. Chapter 7's generalization map calls this the $T=1$ corner of a temporal world model, and the Koopman framing is why that corner is a corner of anything: the same generator, evolved for longer or applied repeatedly, *is* the temporal case.

One more piece follows. Different perturbations induce different dynamics, so each gets its own generator. Writing $c = e(p)$ for the gene-set embedding of perturbation $p$, the model learns a **family of Koopman generators indexed by the intervention**, $M(c)$, which is exactly what the policy $\alpha = \pi(c)$ produces. This is action-conditioned Koopman, and it is the same construction the [operator world models](../../../../docs/operator_world_models/03-conditioning-jepa-on-actions.md) series applies to interventions in time.

## 6. The premise we never satisfied

Here is the part that matters most, and it is a limitation rather than a defense.

The Koopman argument has a **premise**, and §3 stated it plainly: the encoder must supply observables spanning a subspace that is (approximately) closed under $\mathcal{K}$. Everything above is conditional on that. A linear operator on coordinates that are *not* Koopman-invariant is exactly the crude approximation the §1 objection accused us of, with no defense at all.

Methods built on this premise satisfy it *by construction*. Deep Koopman approaches train the encoder **jointly** with the linear operator, so the encoder is pushed toward coordinates in which the dynamics close. The linearity is not hoped for; it is optimized for. That joint training is the whole point.

**Our encoder was never asked.** Stage A trains the JEPA encoder by masked prediction: infer the embedding of held-out gene groups from the groups you can see. It is a good objective and it produces a good representation, and *nothing in it mentions dynamics at all*. There is no term that rewards a latent in which perturbation responses compose linearly. We then froze that encoder and bolted an operator on top.

So the honest status is: **we adopted the Koopman form without the Koopman training.** The operator is linear because we made it linear, and the argument that licenses linearity assumes a property of the encoder that our pipeline never optimized for and has never checked.

This connects to the two findings that dominate the current ledger. The [ceiling analysis](../../../../docs/experimental-method/07-the-ceiling.md) showed the frozen representation is *information-rich*, since a linear readout of it recovers the effect better than the from-scratch baseline does, but also that the representation was never shaped for the thing consuming it. That is the same complaint in a different stage: the encoder was not co-adapted with the decoder, and it was not co-adapted with the dynamics either. It was trained alone, for a different objective, and then asked to serve two masters that had no say in it.

Which is why the ledger's standing open question, **unfreezing the encoder**, is the biggest swing left. The Koopman argument does not merely say "try joint training." It says *what to train it for*: coordinates in which the intervention's dynamics close linearly.

## 7. What the evidence says, and what it does not

Does our latent happen to be Koopman-ish anyway?

There is one suggestive data point, and it is weaker than it looks. The action operator, which restricts the velocity field to be linear, landed in a **dead tie** with the free transport flow, whose velocity field is an unrestricted neural network. If the latent dynamics were badly nonlinear, forcing them through a linear map should have *hurt*. It did not.

That reads like evidence for the premise, and it is worth almost nothing, because the ceiling explains it away. Stage B is **saturated**: an oracle that produces the real perturbed latents scores $0.679$, and the flow already scores $0.648$. There was only $0.031$ of room for *any* Stage B to distinguish itself. A metric with three points of headroom cannot tell a linear transition from a nonlinear one, so the tie is equally consistent with "the latent is approximately Koopman-invariant" and with "this benchmark could not have noticed either way." That is a genuine confound and it should not be argued around.

Which leaves the premise **untested**, and testable. A direct check does not need the benchmark at all: fit a linear map from control latents to perturbed latents, fit a flexible nonlinear one on the same pairs, and compare how well each predicts *the latent transition itself* rather than the downstream score. The [probe logic](../../../../docs/experimental-method/07a-probes-and-weak-instruments.md) applies unchanged: if the linear map does nearly as well, the coordinates are close to invariant and the operator's central assumption is earned. If the nonlinear map wins decisively, the JEPA latent is not a Koopman embedding, the linearity really is the crude approximation §1 feared, and the honest fix is to train an encoder that makes it true.

## 8. What this chapter does and does not license

**It licenses the form.** A linear latent operator is not a crude approximation of a nonlinear cascade. It is exact on a Koopman-invariant subspace, and the price of exactness is coordinates rather than fidelity. The matrix exponential is not a trick; it is the semigroup the dynamics generate, and invertibility, the identity start, the least-action prior, and clean composition are all consequences of that rather than separate choices.

**It does not license our instance.** Everything above is conditional on an encoder that supplies Koopman coordinates, and ours was trained by masked prediction with no dynamical objective, then frozen. The form is defensible. The instance is unearned, and it is unearned in a way that is measurable rather than merely arguable.

> **Throughline.** You do not linearize the dynamics, you choose coordinates in which they are already linear, and the encoder is that choice. The Koopman operator is exactly linear on observables and infinite-dimensional; an invariant subspace makes it a matrix; the semigroup makes that matrix an exponential; and the generator is the object the model should learn. All of which holds precisely to the degree that the encoder was built for it, and ours was not.

---

*Previous: [Modeling the readout — count decoders](08-modeling-the-readout-count-decoder.md). Up: [the method series](index.md). The algebra of composing two operators, and what it says about epistasis, is developed in the foundation series' [algebra of composition](../../../../docs/action_operator/03-the-algebra-of-composition.md); its application to two-gene perturbations continues the transition thread. Current state of play: [the results ledger](results-ledger.md).*
