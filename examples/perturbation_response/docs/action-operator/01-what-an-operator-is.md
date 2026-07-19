# 1. What an operator is, and what makes it mean anything

*Two objects wear the name "action operator" and they live in different spaces. Keeping them apart is the first thing, and the bridge between them is the second. That bridge is the whole subject: without it, what you have built is not an operator, it is a matrix that happens to fit.*

> **Prerequisites.** None beyond knowing what a function and a matrix are. Every symbol is defined at first use.
>
> **Origin.** The framework comes from the sibling **GRL** project, *Generalized Reinforcement Learning: Actions as Operators on State Space*, where it is developed as a strict generalization of reinforcement learning. There the policy does not select an action from a menu; it **constructs** an operator $\hat{O} : \mathcal{S} \to \mathcal{S}$, and the environment merely adds noise,
> $$s' = \hat{O}(s) + \xi, \qquad \xi \sim \mathcal{N}(0, \sigma^2 I),$$
> with the pipeline $s \xrightarrow{\pi_\psi} \theta \xrightarrow{\Phi} \hat{O}_\theta \xrightarrow{\text{apply}} s'$. Classical RL is recovered exactly when the operator family is a finite set of displacements $s \mapsto s + b_i$, which makes discrete-action reinforcement learning a special case rather than a different subject. The monoid and Lie-group structure, the generalized Bellman equation with its least-action term, and the operator families are formalized in [`GRL/docs/action_operator`](https://github.com/pleiadian53/GRL). **This series takes that framework out of the reinforcement-learning setting** and asks what it does when the "action" is a biological intervention, the system is measured once and destroyed, and there is no reward anywhere.

---

## 1. The generalization: from a label to a transformation

Most people meet the word *action* through reinforcement learning, where it means an element of a finite set. Press left. Press right. Buy, hold, sell. The agent's job is to pick one, and the environment does something in response.

That framing has a hidden commitment: **the action is an index, and all of its meaning lives outside it, in the environment's transition function.** "Press left" is just the label `2`. Nothing about the number `2` tells you what it does. The dynamics live in $P(s' \mid s, a)$, and the action is a key into it.

The operator view moves the meaning into the action itself:

$$s' = \hat{O}(s).$$

Read $\hat{O}$ as a **function from states to states**. The hat marks it as an operator, a function that acts on the whole state space, rather than a number. Under this reading an action is not an index into a table of consequences; it *is* the consequence, packaged as a transformation you can apply.

Three things follow immediately, and they are why the generalization is worth anything:

**Actions compose, in a strong sense.** Labels can of course be *sequenced*: "press right, then press left" is a perfectly good instruction, and a trajectory is a list of them. But the list is not itself an action. There is no entry in the menu for "right-then-left," and the only way to find out what the sequence does is to **execute** it, pushing a state through the transition function one step at a time. Composition exists operationally, as something you run.

Operators are **closed** under composition: the composite $\hat{O}_2 \circ \hat{O}_1$ is *itself an operator*, the same kind of object as the two it was built from. Together with the identity operator $I(s) = s$, the family forms a **monoid**: closure, associativity, an identity element. That closure is the whole difference, because it turns composition from a procedure you execute into an object you can compute with. When the operators are matrices, the composite is a matrix you can form, store, and inspect **without ever touching a state**. Repeating one $k$ times collapses to $\exp(M)^k = \exp(kM)$, a single matrix whose eigenvalues say what $k$ steps do without simulating $k$ steps. And "does the order matter?" stops being a question you answer empirically, by running both orders and comparing outcomes, and becomes the commutator $[M_1, M_2] = M_1M_2 - M_2M_1$, computable directly from the two generators. 

**Actions become continuous and parameterized.** You are not restricted to a menu. An operator can be *constructed* from parameters, and nearby parameters give nearby transformations, so "a bit more of this intervention" is a meaningful and differentiable thing to ask for.

**Actions become inspectable.** If the operator is a matrix, you can look at its eigenvalues and read off what it does: what it amplifies, what it damps, what it rotates. A label has no interior.

Classical reinforcement learning is recovered as the special case where the family of available operators is a finite set of fixed maps. Nothing is lost; a great deal is made expressible.

## 2. The two spaces, and why they must be kept apart

Now the distinction that everything else depends on, and the one that causes the most confusion when it is skipped.

Take a concrete intervention: **activate gene X in a cell.**

What physically happens is a tangled biological cascade. The cell's expression state, call it $s$, a vector of some twenty thousand gene counts, becomes a new state $s'$. Transcription factors bind, feedback loops engage, downstream targets rise and fall over hours. Nobody can write that transformation as a formula. Call the real, meaningful, **intractable** transformation the **state operator**:

$$s' = \hat{O}_\theta(s),$$

where the subscript $\theta$ names *which* intervention this is ("activate gene X" is one $\theta$, "activate gene Y" another). You never hold this object. It exists implicitly, in the before-and-after states it produced.

Now encode. An encoder $E$ maps a state to a compact latent, $z = E(s)$, a vector of a few hundred numbers capturing the cell's *meaning* rather than its raw counts. The **same** intervention, viewed in latent space, is a second object:

$$z' = f_\theta(z), \qquad z = E(s), \quad z' = E(s').$$

This is the **latent operator** $f_\theta$, and here is the entire reason for moving to latent space: **you get to choose what $f_\theta$ is.** The state operator is handed to you by nature and is unwritable. The latent operator is yours to design, and you can design it to be something you can compute with, inspect, compose, and invert. The usual choice is affine,

$$f_\theta(z) = A_\theta z + b_\theta,$$

with $A_\theta$ a matrix and $b_\theta$ a shift vector.

> **Keep these straight.** $\hat{O}_\theta$ is the operator you *mean*: real, meaningful, intractable. $f_\theta$ is the operator you *build*: a matrix you chose to be simple. They are the same transformation seen in two spaces, and **the encoder is the bridge between them**.

A worked instance, small enough to see. Let the latent be two-dimensional, let a control cell sit at $z = (2, 1)$, and let

$$A_\theta = \begin{pmatrix} 1.2 & 0 \\ 0 & 0.8 \end{pmatrix}, \qquad b_\theta = 0.$$

Then $z' = A_\theta z = (2.4,\ 0.8)$. This operator amplified the first latent direction and damped the second, and the intervention's **effect** is the displacement $z' - z = (0.4,\ -0.2)$.

Now the anchor to carry through the entire series. Suppose $A_\theta = I$, the identity matrix, with $b_\theta = 0$. Then $z' = z$: the operator does nothing, and the intervention has no effect.

> **The anchor.** An intervention's effect is exactly *how far its operator departs from the identity*. Modeling the operator is modeling the change.

That is not a slogan, it is a design constraint that will reappear as an initialization (start at $I$), as a regularizer (stay near $I$), and eventually as a premise that can fail.

## 3. The hole: a matrix is not an operator

Here is where a lot of treatments move too fast, and where the framework is actually won or lost.

Look again at $f_\theta(z) = A_\theta z + b_\theta$. On its own, this is **just some matrix acting on vectors.** Pick any $A_\theta$ you like and you get *a* map. It transforms latents. It is invertible if you built it that way. It composes with other such maps.

And it has nothing whatever to do with activating gene X.

Nothing in the definition ties $f_\theta$ to $\hat{O}_\theta$. You have named it "the latent operator for intervention $\theta$," but the name is doing all the work. Something has to *force* $f_\theta$ to be the faithful latent image of the real transformation, and until something does, the object is an arbitrary matrix wearing a suggestive label.

This is the premise that, when it fails, produces a model that predicts acceptably and explains nothing, because the thing you are inspecting the eigenvalues of was never pinned to the biology.

## 4. The bridge: the commuting square

What forces the correspondence is a training signal, and the cleanest way to see it is as a diagram:

$$
\begin{array}{ccc}
s & \xrightarrow{\ \hat{O}_\theta\ } & s' \\[4pt]
{\scriptstyle E}\big\downarrow & & \big\downarrow{\scriptstyle E} \\[4pt]
z & \xrightarrow{\ f_\theta\ } & z'
\end{array}
$$

There are two routes from the top-left state $s$ to the bottom-right latent $z'$:

- **Transform, then encode** (across the top, then down the right): apply the real operator to get $s'$, then encode it. In symbols, $E(\hat{O}_\theta(s))$. This is *what actually happened*, encoded.
- **Encode, then transform** (down the left, then across the bottom): encode first, then apply your built operator. In symbols, $f_\theta(E(s))$. This is *your model's prediction*, computed without redoing the biology.

If those two routes always land on the same point, the square **commutes**, and $f_\theta$ is a faithful shadow of $\hat{O}_\theta$. You do not get this for free. You *make* it hold, by penalizing the gap:

$$
\mathcal{L}_{\text{equiv}} = \big\lVert \underbrace{E(\hat{O}_\theta(s))}_{\text{encode what truly happened}} - \underbrace{f_\theta(E(s))}_{\text{the operator's prediction}} \big\rVert^2 .
$$

Spelled out: the left term encodes the genuinely transformed state, the ground truth of where the system went. The right term is the operator's prediction of that landing point. The squared norm is the sum of squared coordinate differences. Driving this to zero forces $A_\theta$ to become *whatever matrix "activate gene X" is* in latent coordinates.

This property has a name, **equivariance**: the operator commutes with the encoding. And it is not decoration.

> **The crux.** Equivariance is the *precise condition* that makes $f_\theta$ mean anything as a stand-in for the real transformation. Without the commuting square, the operator is an arbitrary matrix. With it, the operator is the intervention, written in coordinates you can compute with.

In practice the left term is a **stop-gradient** target, written $\mathrm{sg}(\cdot)$, so no gradient flows back through it. This is the same anti-collapse device as the target encoder in JEPA pretraining: the goalpost must not chase the predictor, or the cheapest solution is for both to collapse to a constant, at which point the square commutes perfectly and says nothing.

Notice also what the diagram quietly requires of the encoder. If $E$ discards the very information the intervention changes, then $E(s)$ and $E(s')$ are nearly identical, the square commutes with $f_\theta = I$, and the honest conclusion "this intervention does nothing in these coordinates" is indistinguishable from a good fit. The bridge constrains the operator *given* an encoder; it cannot rescue an encoder that threw the signal away.

## 5. The crack this opens, and where it leads

Look hard at what $\mathcal{L}_{\text{equiv}}$ *requires*. For a single training example you need **both** $s$ and its own $s'$: the same system, before and after. Without that pair, "encode what truly happened" is undefined.

In a temporal setting this is free. The state $s$ is a person at time $t$, $s'$ is *that same person* at $t+1$, and you observe the whole trajectory, so every consecutive pair is a training example. This is the **paired** regime the framework was designed in.

Now the setting this series is about. Reading a cell's transcriptome by sequencing **destroys** it. You measure a control cell *or* a perturbed cell, never the same cell before and after. The pair $(s, s')$ does not exist.

> **The crack.** For destructive measurements the equivariance loss cannot be computed *at all*, because its basic ingredient is missing. The bridge that gives the operator its meaning does not build in the unpaired form.

This is the first of the three things perturbation response breaks, and it is not a detail to be patched. Chapter 6 shows that the repair is also a generalization: the object we actually care about is a *distribution*, pairing is merely how much we happen to know about it, and matching distributions recovers the paired case exactly when the pairing is available. But it is worth sitting with the problem before reaching for the fix, because a great deal follows from the fact that our operator was trained against a **surrogate** for the objective that would have given it meaning.

---

*Next: why the operator is linear, and the Koopman argument that licenses it. Up: [the series index](index.md). The short domain-general version of this chapter: [From actions to operators](../../../../docs/action_operator/00-from-actions-to-operators.md).*
