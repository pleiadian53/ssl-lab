# State Operators and Latent Operators

*One transformation, two spaces — and the bridge that ties them together.*

> **Prerequisite.** The [Action Operators](../action_operator/00-from-actions-to-operators.md) foundation, which introduces the action operator $\hat O_\theta$ (an action read as a *function that transforms the state*, $s' = \hat O_\theta(s)$, configured by a parameter $\theta$) and why JEPA benefits from one. This note picks up exactly where the foundation's closing question left off. New to the symbols ($\hat O_\theta$, $f_\theta$, $E$, $\Theta$)? Keep the [notation reference](notation.md) open alongside this page.

The foundation's primer, [From actions to operators](../action_operator/00-from-actions-to-operators.md), ended on a question. An action operator $\hat O_\theta$ is defined on the *raw* state $s$ — but machine-learning models work on compact **latent** encodings $z$, not raw states. Can the *same* transformation be described on the latent instead of the raw state, and made simpler in the bargain?

The answer is yes, and it splits the single idea "the action operator" into **two distinct objects** that are easy to conflate but must be kept apart. Pulling them apart is the keystone the rest of this series rests on: temporal prediction, conditioning on interventions, and the runnable operator code all hang off this one distinction. This note introduces the two objects, the single bridge that connects them, and why a world model built on JEPA needs to carry both.

The running thread uses the two application domains from the foundation — a person's day-to-day behavioral state, and a protein's 3D structure — which turn out to be the *same* construction with one dial turned to opposite extremes.

---

## 1. An action lives in two spaces

Picture a transformation you would like a model to understand. Two concrete ones, which we will carry the whole way:

- **Behavioral.** "A week of more sleep" applied to a person. It changes their underlying physiological and cognitive state — mood, alertness, stress reactivity — in ways that are real but that you can never write down in closed form.
- **Protein.** A rigid rotation-and-translation applied to a protein structure: every atom's coordinate moves by $x \mapsto Rx + t$. This one you *can* write down exactly.

Each transformation can be described in two different places, and the distinction is the heart of everything:

- **The state operator** $\hat O_\theta$. It acts on the system state $s$ in the original observation space: $s' = \hat O_\theta(s)$. This is the *physically meaningful* object — the actual transformation on the real system. Advance physical time. Rotate the protein. Apply a week of more sleep to the person. The subscript $\theta$ names *which* operator from a family; the hat marks it as an operator (a function), not a number.
- **The latent operator** $f_\theta$. It acts on the encoded latent $z = E(s)$ in representation space: $z' = f_\theta(z)$. Here $E$ is the **encoder** — a neural network that maps a state to a latent vector capturing its *meaning* rather than its surface form. This $f_\theta$ is the object you actually *compute with*: a concrete map on vectors, typically $f_\theta(z) = A_\theta z + b_\theta$ with $A_\theta = \exp(M_\theta)$, a matrix you can multiply, compose, and differentiate.

> **Key distinction.** $\hat O_\theta$ is the operator you *mean*; $f_\theta$ is the operator you *build*. They are the same transformation, viewed in two different spaces.

What links them is the encoder $E$, and the relationship is exact enough to draw as a diagram.

---

## 2. The encoder is the bridge: a commuting square

Put the two spaces on top of each other. The top row is observation space; the bottom row is latent space. The encoder $E$ runs downward, turning states into latents. The defining requirement that ties $\hat O_\theta$ to $f_\theta$ is that this square **commutes**:

$$
\begin{array}{ccc}
s & \xrightarrow{\ \hat O_\theta\ } & s' \\[4pt]
{\scriptstyle E}\big\downarrow & & \big\downarrow{\scriptstyle E} \\[4pt]
z & \xrightarrow{\ f_\theta\ } & z'
\end{array}
\qquad\Longleftrightarrow\qquad
E\big(\hat O_\theta(s)\big) = f_\theta\big(E(s)\big).
$$

Read the square as a promise about two paths. Starting from the top-left state $s$, there are two ways to reach the bottom-right latent $z'$:

- **Top-then-right-then-down:** transform the state first ($s \to s'$), then encode it ($s' \to z'$). That is the left-hand side, $E(\hat O_\theta(s))$.
- **Down-then-right:** encode first ($s \to z$), then transform the latent ($z \to z'$). That is the right-hand side, $f_\theta(E(s))$.

Commuting means the two paths land in the same place. When they do, $f_\theta$ is the faithful *latent image* of $\hat O_\theta$ — the very same operator, pushed through the encoder. Conceptually, $f_\theta = E \circ \hat O_\theta \circ E^{-1}$: decode, transform, re-encode. (We write $E^{-1}$ here only to express the idea; a key payoff in §6 is that you never actually need it.)

---

## 3. The commuting square *is* the equivariance loss

The square is not just a picture — it is a training objective. A perfectly commuting square is the ideal; in practice you *make* it commute by penalizing the gap between its two paths:

$$
\mathcal{L}_{\text{equiv}} = \big\lVert E(\hat O_\theta(s)) - f_\theta(E(s)) \big\rVert^2.
$$

This is the **operator equivariance loss**. The left term encodes the *actually-transformed* state; the right term is the model's *prediction* of where that transformation lands in latent space, computed without redoing it from scratch. Driving the gap to zero forces $f_\theta$ to track $\hat O_\theta$ through the encoder.

> **Key insight.** Equivariance is not a decorative side property. It is the precise condition that makes $f_\theta$ *mean* anything as a stand-in for $\hat O_\theta$. Without the commuting square, $f_\theta$ is just some map on vectors with no connection to a real transformation.

This is also exactly the shape of JEPA's own loss — *(embedding of the true outcome)* minus *(a prediction of that outcome)*, squared. That correspondence is what lets a JEPA encoder double as the bridge in an action-operator world model, and it is developed in the [bridge note](../action_operator/01-jepa-action-operators.md).

---

## 4. Why carry two objects instead of one

If $\hat O_\theta$ and $f_\theta$ are the same transformation, why not keep just one? Because they have **opposite virtues**, and the encoder is precisely the device that lets you enjoy both at once.

- $\hat O_\theta$ is **meaningful but intractable.** It is the real transformation, so it carries the semantics you care about — but it is generally nonlinear, high-dimensional, and often *you have no explicit access to it at all*. You cannot write the operator that maps a person's full physiological state through a week of sleep; you only observe sensor streams before and after. The operator lives implicitly in the data, never on paper.
- $f_\theta$ is **tractable by construction.** It is a concrete map on latent vectors that you choose to be simple — linear, composable, near the identity. And here is the leverage: *you get to choose the encoder $E$* so that $f_\theta$ comes out simple **even when $\hat O_\theta$ is wildly nonlinear.**

That last sentence is the whole game, and it has a name.

---

## 5. The Koopman move: choose the encoder so the latent operator is linear

The claim that a nonlinear transformation can have a *linear* latent image is not wishful thinking — it is a classical result. **Koopman's theorem** says that a nonlinear dynamical system $s' = \hat O(s)$ in state space becomes a *linear* operator when lifted to a suitable (typically higher-dimensional) space of latent observables. You trade "nonlinear in the original coordinates" for "linear in the right latent coordinates."

This reframes the encoder's job entirely:

> **The encoder's job is not to compress.** It is to *find coordinates in which the physical operator becomes a clean linear latent operator.* Compression is incidental; linearization is the point.

It also legitimizes the default parameterization $f_\theta(z) = \exp(M_\theta) z + b_\theta$, where $M_\theta$ is a **flow generator** (a matrix in a flat vector space) and $A_\theta = \exp(M_\theta)$ is the operator obtained by matrix exponential. Choosing $\exp(M_\theta)$ is not merely convenient — it is *justified exactly when* an encoder exists under which the dynamics linearize, and Koopman theory guarantees that broad classes of systems admit such an encoder. The generator $M_\theta$ is where the structure of the transformation lives; the exponential lifts it into an operator you can apply and compose. (The series builds $M_\theta = \sum_i \alpha_i B_i$ from a *generator basis* $\{B_i\}$ in a later part; the [operator gallery](../action_operator/02-operator-gallery.md) shows concrete $M$'s and exactly what each does to a state. For now, read $\exp(M_\theta)$ as "a tractable linear operator the encoder was chosen to make valid.")

---

## 6. The JEPA payoff: you only ever touch the latent operator

Everything so far is general. What makes it *operational* — rather than philosophical — is a property specific to JEPA-style world models: **you never instantiate $\hat O_\theta$, and you never need $E^{-1}$.**

There is no decoder and no reconstruction anywhere. The state operator $\hat O_\theta$ is the *data-generating* process — it is what physically produced the observed pair $(s, s')$ — but you work entirely with the **encoded** quantities. You predict $f_\theta(E(s))$ and compare it against $E(s')$, both living in latent space. The state operator stays implicit, present only through the data it generated.

This is exactly why a JEPA world model can tolerate an inaccessible, nonlinear $\hat O_\theta$: it asks *nothing* of the operator except that the **encoded** transitions be predictable by a simple $f_\theta$. The intractable object never has to be named, inverted, or evaluated. You compute only the shadow.

> **The throughline of this note.** $\hat O_\theta$ is the operator you mean; $f_\theta$ is the operator you build; the encoder plus the equivariance loss is what guarantees the second is a faithful shadow of the first; and JEPA is the regime where you can get away with only ever computing the shadow.

---

## 7. Two domains, two poles of one dial

The two domains we have been carrying differ in *where the structure of $f_\theta$ comes from* — and that difference is not a quirk, it is a dial you set. At one pole the operator's structure is **learned** from data; at the other it is **given** by physics. The same machinery spans both.

| | the state operator $\hat O_\theta$ | the latent operator $f_\theta$ | where equivariance comes from |
|---|---|---|---|
| **Behavioral** (digital phenotyping) | the real "+1 week of sleep" on a person's true state — *inaccessible*, nonlinear | $\exp(M_\theta) z + b_\theta$ on the behavioral latent, with $M_\theta$ **learned** | **learned** from encoded sensor streams |
| **Protein** (3D structure) | a literal rigid motion $x \mapsto Rx + t$ on coordinates — *exactly known* | the SE(3) operator on residue frames, with structure **given** by physics | **demanded** as a hard symmetry (SE(3)-equivariance) |
| **Image** (I-JEPA) | "shift attention to region $p$" on the image | the predictor conditioned on position | **learned** via the predictor |

Read the two main rows as two poles of a single **expressiveness ↔ structure dial**:

- **Behavioral, the learned pole.** You do not know what "a week of sleep" does to the latent, so you let the data teach you: $M_\theta$ is free (or organized by named interventions), equivariance is something the model *discovers*, and the operators it learns are *associational* dynamics — useful, but not automatically causal. The payoff is the ability to roll the latent forward under a chosen intervention and to read off when a person's dynamics depart from their own baseline.
- **Protein, the given pole.** You know exactly what a rigid motion is, so you *impose* it: $f_\theta$ is constrained to the SE(3) group, equivariance is *demanded* as a hard constraint rather than learned, and correctness is guaranteed by construction — every predicted structure respects rigid-motion symmetry. The action-operator framing bites hardest on the *dynamic and generative* protein tasks (placing residue frames one operator at a time, modeling conformational change, predicting the effect of an in-silico mutation), as distinct from merely representing a static structure equivariantly.

> **Key takeaway.** Behavioral and protein modeling are not example-and-foil. They are the *same operator construction* — state operator, encoder bridge, latent operator $\exp(M_\theta)$ — with the generator basis swapped: free and learned at one pole, fixed by the SE(3) algebra at the other. The breadth between those poles is the whole reason the formalism is worth carrying.

---

## Where this goes next

With the two operators and their bridge in hand, the rest of the series builds outward:

- **[JEPA as a temporal world model](02-jepa-as-a-temporal-world-model.md)** — reinterpret "predict the masked region" as "predict the next timestep," so $f_\theta$ becomes a *dynamics* operator on a latent trajectory.
- **Conditioning JEPA on actions** — replace the bare "how far ahead" query with $f_{\theta(c_t)}$, an operator chosen by the *intervention* $c_t$, unlocking counterfactual rollout and a sharpened surprise signal.
- **Generator bases and the operator in code** — the concrete $\{B_i\}$ that realize the two poles above, and the runnable module that turns this note into a forward pass.

> **See it in a real scenario.** For $\hat O_\theta$, $f_\theta$, and the commuting square made concrete on one person's metabolic data, see the [worked example — a personal world model for diabetes](05-worked-example-diabetes.md).

*Series home: [Operator World Models](index.md). Notation: [reference](notation.md).*
