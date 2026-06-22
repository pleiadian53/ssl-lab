# Part 1 — The operator-learning problem

*Making "a map between function spaces" precise — and seeing why an ordinary network is the wrong shape for it.*

> **Where we are.** [Part 0](00-what-is-a-neural-operator.md) gave the concept: a learned map from an input field to an output field. This chapter states the problem properly. We build up three ideas in plain language first — a *function space*, the *solution map* between two of them, and *discretization invariance* — then collect the formal statements, including the theorem that says the whole enterprise is possible, into a marked math section at the end. Nothing here requires more than the idea of a function and a neural network.

---

## 1. The setup: two spaces of functions, and a map between them

Recall the job: turn an input field $a$ into an output field $u$. To say this precisely we need to name *where $a$ lives* and *where $u$ lives*.

An input field — say a drug concentration over a tissue sheet — is one particular function. But it is one of *many possible* such fields: a different experiment applies a different pattern. The collection of *all the input fields you might encounter* is a **function space**, call it $\mathcal{A}$. Each point of $\mathcal{A}$ is an entire function. Likewise, all the possible output fields form another function space $\mathcal{U}$, and each point of $\mathcal{U}$ is a whole output function.

The thing you want to learn is the **map** that takes any input field to its corresponding output field:

$$
G : \mathcal{A} \to \mathcal{U}, \qquad u = G(a).
$$

Read it slowly: $G$ is a function whose *inputs are themselves functions* (members of $\mathcal{A}$) and whose *outputs are functions* (members of $\mathcal{U}$). For a system governed by a differential equation, $G$ is the **solution operator** — the rule "given this input field, here is the field that solves the equation." Our goal is to approximate $G$ by a neural network $G_\theta$ with trainable weights $\theta$, learning from example pairs $(a, u)$.

The defining difficulty, and the whole reason a new model family exists: $\mathcal{A}$ and $\mathcal{U}$ are **infinite-dimensional**. A vector in ordinary deep learning has a fixed, finite number of components. A *function* has a value at every one of infinitely many points — you cannot write it as a fixed-length list without throwing information away. Learning a map between infinite-dimensional spaces is a genuinely different problem, and pretending otherwise is exactly the mistake the next section diagnoses.

---

## 2. Why the obvious approach is the wrong shape

The tempting shortcut: pick a grid, sample every input field at those grid points to get a fixed-length vector, sample every output field the same way, and train an ordinary network — a multilayer perceptron or a CNN — to map the input vector to the output vector. Flatten the functions into pixels and use standard tools.

This *can* be trained, but it has quietly changed the problem, and the change costs you the three properties that made neural operators worth wanting (Part 0, §"Why a new model family"):

- **It is welded to one resolution.** The network's input and output sizes *are* the grid you chose. Train on a $64\times 64$ grid and the network simply cannot accept a $128\times 128$ input — the shapes do not match. You have learned a map between two *finite* vector spaces, not between the function spaces $\mathcal{A}$ and $\mathcal{U}$. Change the discretization and you must retrain from scratch.
- **It cannot take scattered inputs or answer new locations.** Real sensors are not on your grid, and you often want the output somewhere you never sampled. A fixed-grid network has no notion of "evaluate the underlying function here"; it only knows its grid.
- **It does not transfer across discretizations.** Two researchers sampling the same physical field at different resolutions would train two unrelated networks, even though the underlying operator $G$ is one and the same.

The diagnosis: an ordinary network learns a map between *vectors*, but the truth we want is a map between *functions*. A neural operator is the architecture redesigned so that the object it learns is genuinely $G : \mathcal{A} \to \mathcal{U}$, with the grid demoted to an implementation detail you can change at will. The property that captures "the grid is just a detail" is **discretization invariance**, and it is the technical heart of the method.

---

## 3. Discretization invariance — the heart of the method

Here is the property stated as a goal: a neural operator should learn a single object that can be **fed inputs at any sampling, and queried for outputs at any sampling**, and as the sampling gets finer the answers should converge to a consistent, resolution-independent result.

Three consequences make this concrete and are worth holding onto:

1. **Train coarse, evaluate fine (zero-shot super-resolution).** Because the learned object targets the underlying function, you can train on cheap low-resolution data and then evaluate on a finer grid the model never saw, and it produces a sensible finer field. This is not a trick; it falls out of having learned the function rather than the pixels.
2. **Any input mesh.** Scattered, irregular, mesh-free sampling of the input is fine — the architecture is built to consume "the function, as sampled here," not "the array of this exact shape."
3. **Query anywhere.** The output is a function you evaluate at arbitrary locations, decoupled entirely from where the input was sampled.

A useful litmus test you will see used to judge these models: take a trained operator, evaluate it at several resolutions, and check that the prediction *converges* as resolution increases rather than drifting or degrading. An ordinary fixed-grid network fails this test by construction — it has nothing to say at a resolution other than its own. A genuine neural operator passes it. Keep this test in mind; it is how the field separates real operator learning from a CNN in disguise.

---

## 4. Is this even possible? The universal approximation theorem for operators

Before building architectures, one foundational worry: ordinary neural networks are famously *universal approximators* of continuous **functions** — but we are now asking them to approximate continuous **operators**, maps between infinite-dimensional spaces. Is there any guarantee that a neural network can do *that*?

There is, and it is older than the modern field. A classical result — the **operator universal-approximation theorem** (Chen & Chen, 1995) — established that a neural network with a single hidden layer can approximate, to arbitrary accuracy, any continuous nonlinear *operator*, not merely any continuous function. In words: the expressive power that lets networks fit functions extends, in principle, to fitting maps *between* functions.

This theorem is more than reassurance. Its *structure* — how it represents an operator as a combination of two pieces, one encoding the input function and one encoding the output location — is precisely the blueprint that [DeepONet](02-deeponet.md) turns into an architecture in the next chapter. So the theory does double duty: it tells us operator learning is possible at all, and it hands us a concrete design. The theorem guarantees *existence* of a good approximating operator; it does not promise it is *easy to find*, *data-efficient*, or *robust out of distribution* — the practical caveats from Part 0 remain, and we return to them in the planned training-and-limits chapter.

---

## 5. The general shape of a neural-operator layer (a first look)

How do you actually build a layer that maps a function to a function? The general recipe, which both architectures specialize, is worth previewing in plain language now and stating formally in the math section.

An ordinary network layer transforms a vector by a matrix multiply plus a pointwise nonlinearity. A neural-operator layer transforms a *function* by two pieces:

- a **local, pointwise** part — at each location, apply a small linear transform to the current field's value there (this is the familiar matrix multiply, applied point by point); and
- a **global, mixing** part — let every location's new value depend on the field's values *everywhere*, through an **integral**: a weighted average of the whole field, with a learned weighting that says how strongly each location influences each other.

That global mixing term is what lets information travel across the whole domain in a single layer — essential, because in a real field what happens at one point (a wave passing) depends on the state far away. Stack a few such layers with nonlinearities between them, and you have a network that maps an input field to an output field while respecting the global, spatial nature of the problem.

The two landmark architectures are two different answers to the question *"how do you make that global integral both expressive and cheap to compute?"* — [DeepONet](02-deeponet.md) factorizes the operator following the 1995 theorem; the [Fourier Neural Operator](03-the-fourier-neural-operator.md) computes the integral as a multiplication in frequency space. Everything in the next two chapters is a way of filling in this one shape.

---

## 6. The math, collected

*Deferred per the series' top-down contract. The objects below make §§1–5 precise; skip on a first pass.*

**Function spaces and the operator.** Let $\mathcal{A}$ and $\mathcal{U}$ be spaces of functions on a spatial domain $D \subseteq \mathbb{R}^d$ (for example $D$ a patch of tissue, $d = 2$). An element $a \in \mathcal{A}$ is a function $a : D \to \mathbb{R}^{d_a}$ (the input field, $d_a$ components); an element $u \in \mathcal{U}$ is $u : D \to \mathbb{R}^{d_u}$ (the output field). The target is the operator $G : \mathcal{A} \to \mathcal{U}$, approximated by $G_\theta$. Training data are pairs $\{(a^{(j)}, u^{(j)})\}_{j=1}^N$ with $u^{(j)} = G(a^{(j)})$, and we minimize a field-space loss, typically the relative $L^2$ error

$$
\mathcal{L}(\theta) = \frac{1}{N} \sum_{j=1}^N \frac{\lVert G_\theta(a^{(j)}) - u^{(j)} \rVert_{L^2}}{\lVert u^{(j)} \rVert_{L^2}},
$$

where $\lVert f \rVert_{L^2} = \big( \int_D \lvert f(x) \rvert^2 dx \big)^{1/2}$ is the size of a field measured over the domain. The relative form makes the loss scale-free across inputs of different magnitudes.

**The general neural-operator layer.** Write the hidden field at layer $t$ as $v_t : D \to \mathbb{R}^{n}$ (an $n$-channel field). One layer updates it by

$$
v_{t+1}(x) = \sigma\Big( W v_t(x) + \big(\mathcal{K}_\phi v_t\big)(x) + b \Big),
$$

where $\sigma$ is a pointwise nonlinearity, $W$ is the local linear transform applied at each point $x$ (the pointwise part), $b$ is a bias, and $\mathcal{K}_\phi$ is the **kernel integral operator** (the global mixing part):

$$
\big(\mathcal{K}_\phi v\big)(x) = \int_D \kappa_\phi(x, y) v(y) dy,
$$

with $\kappa_\phi$ a learned kernel — a function saying how strongly the field at $y$ contributes to the update at $x$. The first and last layers lift the input field into the $n$-channel hidden representation and project back to the $d_u$-component output. This integral is the global term promised in §5; the next two chapters are two ways to make it tractable.

**Discretization invariance, stated.** The construction above is defined on the *function* $v_t$, independent of any sampling. Evaluating it requires discretizing the integral — but the *operator it defines* does not depend on that choice, so refining the discretization yields a consistent limit. This is the formal sense in which the same learned $G_\theta$ accepts and produces fields at any resolution.

**The operator universal-approximation theorem (Chen & Chen, 1995), informally.** For a continuous operator $G$ and a continuous nonlinearity, there exist parameters such that an expression of the form

$$
G(a)(y) \approx \sum_{k=1}^{p} \underbrace{b_k\big(a(x_1), \dots, a(x_m)\big)}_{\text{encodes the input function }a} \cdot \underbrace{t_k(y)}_{\text{encodes the query point }y}
$$

approximates $G(a)(y)$ uniformly to any desired accuracy. The two factors — one reading the input field sampled at $m$ points $x_1,\dots,x_m$, the other reading the output location $y$ — are exactly the **branch** and **trunk** of DeepONet in [Part 2](02-deeponet.md).

---

## 7. Where we go next

We now have the problem stated correctly: learn an operator $G : \mathcal{A} \to \mathcal{U}$ between function spaces, in a way that is discretization-invariant, and we know from the 1995 theorem that such learning is possible — with a hint at the form the solution takes.

[Part 2](02-deeponet.md) builds **DeepONet**, the architecture that reads the theorem's two-factor form literally: a *branch* network that encodes the input function and a *trunk* network that encodes the query location, combined by a dot product. [Part 3](03-the-fourier-neural-operator.md) builds the **Fourier Neural Operator**, which takes the kernel-integral view of §5 and evaluates that global integral cheaply in frequency space. Two answers, one question.

> **One-paragraph recap.** We want to learn the solution operator $G : \mathcal{A} \to \mathcal{U}$ mapping an input field to an output field, where both live in infinite-dimensional function spaces. An ordinary fixed-grid network learns a map between *vectors* and is welded to its resolution; a neural operator learns the map between *functions* and stays discretization-invariant — train coarse, query anywhere, converge as resolution refines. The 1995 operator universal-approximation theorem guarantees such operators are learnable and even hints at the architecture. A neural-operator layer combines a pointwise transform with a global integral; the next two chapters are two ways to make that integral practical.
