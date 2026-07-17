# A Gallery of Operators: What θ Does to a State

*A tour of concrete operators — pick a parameter, watch it build a transformation, watch the state move.*

The action-operator foundation makes a recurring claim: a parameter $\theta$ does not *look up* an operator from a shelf, it **configures** one from scratch, and that operator then transforms the state. This note makes that claim tangible. It is a gallery — a handful of small, fully worked operators, each showing what its parameter means and what it does to a state.

> **Where this fits.** A gentle introduction to action operators is in [From actions to operators](00-from-actions-to-operators.md), and the move to JEPA in [Augmenting JEPA with Action Operators](01-jepa-action-operators.md). This page is the worked-examples reference they lean on; the state-vs-latent mechanics are developed in the downstream [Operator World Models](../operator_world_models/index.md) series. The deeper formalism (operator families, algebra, learning) lives in the [GRL project](https://github.com/pleiadian53/GRL).

Throughout, the operators act on a vector state $z \in \mathbb{R}^d$ — read it as a latent, a position, or any small state you like. We focus on **linear** operators, because they are the workhorse of the formalism and because you can see exactly what they do.

---

## 1. The family we will tour: an operator built from a generator

Every operator in this gallery has the same shape:

$$
\hat O_\theta(z) = A_\theta z, \qquad A_\theta = \exp(M_\theta).
$$

Three objects, defined in order:

- $M_\theta$ — the **generator**, a square $d \times d$ matrix. This is what the parameter $\theta$ actually fills in; it is the object you configure.
- $\exp$ — the **matrix exponential**, the series $\exp(M) = I + M + \tfrac{1}{2}M^2 + \tfrac{1}{6}M^3 + \cdots$ (real matrix powers, *not* applied entry by entry). $I$ is the identity matrix.
- $A_\theta = \exp(M_\theta)$ — the **operator matrix**, the thing you multiply the state by.

The single most useful way to read the generator: **$M_\theta$ is a velocity field, and $A_\theta$ is where you arrive after letting it run for one unit of time.** Formally, $M_\theta$ defines the linear flow

$$
\dot z = M_\theta z \quad\Longrightarrow\quad z(t) = \exp(t M_\theta) z(0),
$$

where $\dot z$ is the time-derivative of the state. So $M_\theta$ is the *instantaneous rule of motion* and $A_\theta = \exp(M_\theta)$ is the *one-step* map it integrates to.

### Why build the operator as $\exp(M_\theta)$

It would seem simpler to let $\theta$ fill in $A_\theta$ directly. Building it as $\exp(M_\theta)$ buys three properties for free, each a fact about the exponential:

- **Invertibility, with no constraints.** For *any* matrix $M$, $\exp(M)$ is invertible and its inverse is $\exp(-M)$. So $\theta$ can fill $M_\theta$ with any numbers at all and still produce a valid, undoable operator. (Filling $A_\theta$ directly, you would have to *enforce* invertibility.)
- **A sane starting point.** $\exp(0) = I$. Start with $M_\theta \approx 0$ and the operator begins as "do almost nothing," $A_\theta \approx I$ — exactly the right prior for dynamics, where the next state resembles the current one. Training then pushes the operator away from the identity only as far as the data demands.
- **A flat space to work in.** The generators $M_\theta$ live in an ordinary flat vector space (you can add them, scale them, average them); the operators $A_\theta$ live on a curved surface where most matrices are *not* valid operators. It is far easier to have a model emit a point in a flat space and *then* exponentiate into the curved one. Configure flat, apply curved.

So $\theta$ configures the operator by filling the generator $M_\theta$; the rest is the exponential. The whole gallery below is just *different choices of $M$, and what each one does.*

---

## 2. How to read a generator: stretch and spin

You can predict an operator's behavior from its generator's **eigenvalues** without computing anything. If $M$ has an eigenvalue $\lambda = a + bi$ (with real part $a$ and imaginary part $b$), then $A = \exp(M)$ has the eigenvalue $e^\lambda = e^a e^{ib}$, and the two parts read cleanly:

- the **real part $a$** sets *growth or decay* — $a > 0$ amplifies that direction each step, $a < 0$ shrinks it;
- the **imaginary part $b$** sets *rotation* — it spins the state at frequency $b$.

Two pure cases are worth memorizing, because every generator is a blend of them:

- **Skew-symmetric** $M$ (meaning $M^\top = -M$): eigenvalues are purely imaginary, so $\exp(M)$ is a **rotation** — it turns the state without changing its length.
- **Symmetric** $M$ (meaning $M^\top = M$): eigenvalues are purely real, so $\exp(M)$ is a **pure stretch** — it lengthens or shrinks along fixed directions without rotating.

Every matrix splits uniquely into a symmetric part plus a skew part, $M = \tfrac{1}{2}(M + M^\top) + \tfrac{1}{2}(M - M^\top)$, so *any* linear operator is "stretch along some directions while rotating in some planes." That is the whole vocabulary; the gallery is just particular sentences in it.

---

## 3. The gallery

Each entry is a $2 \times 2$ generator (small enough to see), the operator it builds, what it does to a state, and how to read it in the digital-phenotyping setting — where $z$ is a person's behavioral-state latent and the operator is its day-to-day dynamics.

### Pure rotation — a cycle that neither grows nor fades

$$
M = \omega \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}
\quad\Longrightarrow\quad
A = \exp(M) = \begin{pmatrix} \cos\omega & -\sin\omega \\ \sin\omega & \cos\omega \end{pmatrix}.
$$

The generator is skew-symmetric, so $A$ is a rotation by angle $\omega$; the eigenvalues of $M$ are $\pm i\omega$ (pure imaginary — no growth). Starting at $z(0) = (1, 0)$ and applying $A$ repeatedly with $\omega = 90^\circ$ walks the state around a circle: $(1,0) \to (0,1) \to (-1,0) \to (0,-1) \to (1,0)$. **Phenotyping reading:** the natural model of a *circadian or weekly cycle* — the state orbits a baseline, returning without drifting away.

### Mean reversion — a perturbation relaxing to baseline

$$
M = \begin{pmatrix} -0.1 & 0 \\ 0 & -0.5 \end{pmatrix}
\quad\Longrightarrow\quad
A = \begin{pmatrix} e^{-0.1} & 0 \\ 0 & e^{-0.5} \end{pmatrix} \approx \begin{pmatrix} 0.90 & 0 \\ 0 & 0.61 \end{pmatrix}.
$$

The generator is symmetric with both eigenvalues real and negative, so every coordinate **decays toward zero** (the personal baseline), the second faster than the first. Starting at $z(0) = (1, 1)$: the state moves to $(0.90, 0.61)$, then $(0.81, 0.37)$, then $(0.73, 0.23)$, settling back to the origin. **Phenotyping reading:** a *healthy* operator — a mood perturbation that fades over days rather than persisting.

### Damped oscillation — a perturbation that rings before settling

$$
M = \begin{pmatrix} -0.1 & -1 \\ 1 & -0.1 \end{pmatrix}, \qquad \text{eigenvalues } -0.1 \pm i.
$$

This generator has *both* a symmetric part (the $-0.1$ on the diagonal → decay) and a skew part (the off-diagonal $\pm 1$ → rotation). The state therefore **spins inward** — a damped harmonic oscillator. **Phenotyping reading:** a perturbation that *overshoots, oscillates, and settles* — often a better fit for mood than plain mean reversion, which cannot capture the overshoot.

### Instability — a direction that amplifies (the decompensation flag)

$$
M = \begin{pmatrix} 0.05 & 0 \\ 0 & -0.3 \end{pmatrix}
\quad\Longrightarrow\quad
A = \begin{pmatrix} e^{0.05} & 0 \\ 0 & e^{-0.3} \end{pmatrix} \approx \begin{pmatrix} 1.05 & 0 \\ 0 & 0.74 \end{pmatrix}.
$$

One eigenvalue of $M$ is now **positive** ($0.05$), so the first coordinate *grows* by about 5% each step while the second still decays. A small displacement in that direction does not fade — it compounds. **Phenotyping reading:** this is the signature to watch for. If a model's operator starts showing a positive-real-part eigenvalue for a given person, that is a principled, interpretable read on *destabilizing dynamics* — far more meaningful than a threshold crossing on a raw feature.

### Near-identity — the operator that does almost nothing

$$
M \approx 0 \quad\Longrightarrow\quad A = \exp(M) \approx I.
$$

The state is left essentially unchanged. This is where every operator *starts* before training (the "sane starting point" from Section 1), and it is the reference against which all the others are departures.

---

## 4. Stacking operators: repetition and order

Operators compose, and the generator form makes composition especially clean.

**Repeating one operator is just scaling its generator.** Applying the *same* operator $k$ times multiplies the generator by $k$:

$$
A^k = \exp(M)^k = \exp(k M).
$$

So "a week of one daily operator" is the single clean matrix $\exp(7M)$ — no need to multiply seven copies. (For the rotation above, $\exp(7M)$ is simply a rotation by $7\omega$, which matches intuition exactly.)

**Order matters when the operators differ.** For two *different* operators, $\exp(M_1)\exp(M_2)$ generally does **not** equal $\exp(M_2)\exp(M_1)$ — the operators do not commute. Applying a "stress" operator then a "medication" operator lands somewhere different from the reverse order, and the gap is governed by the **commutator** $[M_1, M_2] = M_1 M_2 - M_2 M_1$. This non-commutativity is a feature: real interventions are order-dependent, and the operator algebra captures that for free.

---

## 5. Where θ comes from in practice

Two ways to let a parameter $\theta$ fill the generator $M_\theta$:

- **Dense.** Let $\theta$ be all $d^2$ entries of $M_\theta$ directly. Maximally expressive, but a lot of numbers and no built-in structure.
- **Basis-coefficient (the usual choice).** Fix a small set of generators $\{B_1, \dots, B_m\}$ — a **basis** — and let $\theta = (\alpha_1, \dots, \alpha_m)$ be just the coefficients:

$$
M_\theta = \sum_{i=1}^{m} \alpha_i B_i.
$$

Now $\theta$ is a short vector, and the choice of basis injects meaning. Pick the $B_i$ skew-symmetric and every operator is a rotation (the protein-style, symmetry-respecting case). Or make each $B_i$ a **named intervention** — $B_{\text{sleep}}, B_{\text{stress}}, B_{\text{meds}}$ — and let $\alpha$ be the *quantified intervention log* itself: how much sleep, how much stress, whether medication was taken. Then "a week" is $\exp(\sum_i \alpha_i B_i)$ with the week's totals as coefficients, and the order-dependence of Section 4 comes along automatically.

That basis-coefficient form is the bridge from this gallery to running code, and to the two application poles — free-and-learned generators for behavior, fixed symmetry-respecting generators for proteins — developed in the world-model series.

---

## Where to go next

- **The narrative on-ramp:** [From actions to operators](00-from-actions-to-operators.md).
- **Next, composing two operators:** [The Algebra of Composition](03-the-algebra-of-composition.md) — what the commutator above actually measures, why $\exp(A)\exp(B) = \exp(A+B)$ exactly when the generators commute, the BCH correction series, and why its swap-symmetric twin (the anticommutator) answers a different question.
- **The JEPA connection:** [Augmenting JEPA with Action Operators](01-jepa-action-operators.md).
- **State vs latent operators:** [State and latent operators](../operator_world_models/01-state-and-latent-operators.md), in the downstream world-model series.
- **More operator families (deep reference):** the GRL *Action-Operator Formalization* note in the [GRL project](https://github.com/pleiadian53/GRL).
