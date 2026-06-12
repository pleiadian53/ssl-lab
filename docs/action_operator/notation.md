# Notation Reference — Action Operators (foundation)

A standalone glossary for the **Action Operators** foundation: [From actions to operators](00-from-actions-to-operators.md), [Augmenting JEPA with Action Operators](01-jepa-action-operators.md), and [A Gallery of Operators](02-operator-gallery.md). Every symbol used across those notes, grouped by role. Keep it open in a second tab while reading.

The encoder is written $E$ here (and $\varphi_\psi$ in the GRL literature — the same object). The downstream [Operator World Models](../operator_world_models/index.md) series uses the same conventions.

---

## Representation — the JEPA side

| Symbol | Meaning |
|---|---|
| $x$ | a raw input data point — an image, a behavioral-signal window, a sequence |
| $s$ | the underlying **system state** in state space $\mathcal{S}$ — the *physically real* thing an action transforms |
| $E$ | the **encoder**: a neural network mapping an input to an embedding (written $\varphi_\psi$ in GRL) |
| $z = E(x)$ | the **latent embedding** of $x$ — a vector capturing *meaning*, not surface form. Lives in latent space $\mathcal{Z}$ |
| $E_{\text{target}}$ | the **target encoder**: a slow exponential-moving-average (EMA) copy of $E$, updated by averaging, not by gradients (stop-gradient). Produces JEPA's prediction targets |
| $\mathcal{S},\ \mathcal{Z}$ | the **state space** and the **latent space**. When actions operate on latents, $\mathcal{S}=\mathcal{Z}$ |
| $\mathrm{Pred}(\cdot)$ | JEPA's **predictor** — maps a context embedding plus a query to the predicted embedding of the target |
| $q$ | a **query / position token** — tells the predictor *which* target to predict (where, or how far ahead) |

---

## Actions as operators — the GRL side

| Symbol | Meaning |
|---|---|
| $\hat{O}$ | an **action operator**: a function $\hat O:\mathcal{S}\to\mathcal{S}$ that turns a state into its successor. The hat marks "operator (a function), not a number" |
| $\hat{O}_\theta$ | the specific **state operator** configured by parameters $\theta$ — the physically meaningful (often inaccessible) transformation acting on $s$ |
| $\theta$ | **operator / action parameters** — the knobs that *configure* the operator (a complete description of how it transforms the state) |
| $\Theta$ | the **space** of all operator parameters $\theta$ |
| $f_\theta$ | the **latent operator**: how $\hat O_\theta$ acts on the latent $z = E(s)$ instead of on the raw state. The object you compute with |
| $\Phi(\theta, s)$ | the **operator generator**: builds an operator from parameters, $\hat O_\theta(s)=\Phi(\theta,s)$ |
| $E(\hat{O})\ge 0$ | the **energy functional**: how *large* a transformation the operator is — a least-action / parsimony penalty. (Distinct from the encoder $E$; the argument disambiguates) |

---

## Building the latent operator

| Symbol | Meaning |
|---|---|
| $M_\theta$ | the **generator** — a $D \times D$ matrix; the "velocity field" the parameter $\theta$ fills in |
| $A_\theta = \exp(M_\theta)$ | the **operator matrix** — the group element from exponentiating the generator. $\exp$ is the matrix exponential |
| $b_\theta$ | an optional **affine bias** in latent space |
| $f_\theta(z) = A_\theta z + b_\theta$ | the latent operator in its concrete linear form |
| $B_i$ | a **generator basis** — matrices spanning the allowed operator directions; $M_\theta = \sum_i \alpha_i B_i$ |
| $\alpha$ | the **coefficient vector** over the basis; $\alpha = \theta$ |

---

## Policy and exploration

| Symbol | Meaning |
|---|---|
| $\pi$ | the **operator policy**: given a state, it chooses operator parameters |
| $\pi_\psi:\mathcal{Z}\to\Delta(\Theta)$ | the policy (weights $\psi$) maps a latent to a **distribution over** operator parameters |
| $\Delta(\Theta)$ | the set of **probability distributions** over $\Theta$. A *stochastic* policy emits one of these; a deterministic policy emits a single $\theta$ |
| $\Delta z = E(\hat O_\theta(x)) - E(x)$ | the **change in latent** caused by applying an operator — the operationalized *meaning* of a perturbation |

---

## Symbol operators

| Symbol | Meaning |
|---|---|
| $\lVert v \rVert^2$ | squared **Euclidean ($\ell_2$) norm** — measures magnitude / distance |
| $\exp(M)$ | the **matrix exponential** $I + M + \tfrac{1}{2}M^2 + \cdots$ (matrix powers, not elementwise) |
| $\circ$ | **function composition**: $(\hat O_2 \circ \hat O_1)(s) = \hat O_2(\hat O_1(s))$ — apply $\hat O_1$, then $\hat O_2$ |
| $\operatorname{Re}(\lambda)$ | the **real part of an eigenvalue** of $M_\theta$ — positive in any mode flags locally growing dynamics |
| $\approx$ | "approximately equals" — holds up to a small learned error |
| $\sim$ | "is distributed as" / "is sampled from" |

---

## One overload to keep straight

The symbol $E$ does double duty: the **encoder** $E$ (and its EMA copy $E_{\text{target}}$) and the **energy functional** $E(\hat O)$. They never appear in the same role, and the argument disambiguates — $E(x)$ encodes an observation, $E(\hat O)$ scores an operator.

---

*Foundation home: [From actions to operators](00-from-actions-to-operators.md). Next: [Augmenting JEPA with Action Operators](01-jepa-action-operators.md).*
