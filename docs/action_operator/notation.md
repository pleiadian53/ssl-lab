# Notation Reference — Action Operators on JEPA

A standalone glossary for [*JEPA as an Action-Operator World Model*](01-jepa-action-operators.md).
Every symbol used in that write-up, grouped by role, with a "read as" column so you know how to
*say* each one. Keep this open in a second tab while reading the main piece.

---

## Representation — the JEPA side

| Symbol | Read as | Meaning |
|---|---|---|
| $x$ | "x" | a raw input data point — an image, an RNA window, a protein sequence |
| $\varphi$ | *phi* | the **encoder**: a neural network that maps an input to an embedding |
| $\psi$ | *psi* | the **trainable weights** of a network. Subscript usage: $\varphi_\psi$ = "encoder with weights $\psi$." A generic label for "learned parameters" |
| $z = \varphi_\psi(x)$ | "z equals phi-psi of x" | the **latent embedding** of $x$ — a vector that captures *meaning*, not surface pixels/nucleotides |
| $\bar\psi$ | *psi-bar* | a slow **exponential-moving-average (EMA) copy** of $\psi$. The *target encoder* $\varphi_{\bar\psi}$ is updated by averaging, not by gradients (stop-gradient) |
| $\mathcal{S},\ \mathcal{Z}$ | "script S / script Z" | the **state space** and the **latent space**. When actions operate on latents, $\mathcal{S}=\mathcal{Z}$ |
| $\mathrm{Pred}(\cdot)$ | "pred" | JEPA's **predictor** — maps a context embedding (plus a position) to the predicted embedding of the target region |
| $p$ | "p" | a **position / mask token** — tells JEPA *which* region to predict |

---

## Actions as operators — the GRL side

| Symbol | Read as | Meaning |
|---|---|---|
| $\hat{O}$ | "O-hat" | an **action operator**: a function $\hat O:\mathcal{S}\to\mathcal{S}$ that turns a state into its successor. The hat marks "operator (a function), not a number" |
| $\hat{O}:\mathcal{S}\to\mathcal{S}$ | "O-hat from S to S" | its type: takes a state, returns a state |
| $\theta$ | *theta* | **operator / action parameters** — the knobs that select *which* operator |
| $\hat{O}_\theta$ | "O-hat-theta" | the specific operator chosen by parameters $\theta$ |
| $\Phi(\theta, s)$ | *capital Phi* | the **operator generator**: builds an operator from parameters, $\hat O_\theta(s)=\Phi(\theta,s)$ |
| $\Theta$ | *capital Theta* | the **space** of all operator parameters $\theta$ |
| $f_\theta$ | "f-theta" | the **feature-space form** of $\hat O_\theta$ — how the operator acts on latents $z$ instead of on raw states |
| $E(\hat{O})\ge 0$ | "energy of O-hat" | the **energy functional**: how *large* a transformation the operator is — a least-action / parsimony penalty |

---

## Policy and exploration — the coupling layer

| Symbol | Read as | Meaning |
|---|---|---|
| $\pi$ | *pi* | the **operator policy**: given a state, it chooses operator parameters |
| $\pi:\mathcal{Z}\to\Delta(\Theta)$ | "pi from Z to the simplex over Theta" | the policy maps a latent to a **distribution over** operator parameters |
| $\Delta(\Theta)$ | "simplex over Theta" | the set of **probability distributions** over $\Theta$. A *stochastic* policy emits one of these; a deterministic policy emits a single $\theta$ |
| $\Delta z = \varphi_\psi(\hat O_\theta(x)) - \varphi_\psi(x)$ | "delta-z" | the **change in latent** caused by applying an operator — the operationalized *meaning* of a perturbation |
| $X_{\text{struct}}$ | "X-structural" | a **disjoint, independent view** (e.g. conservation, structure, measured expression) used to *justify* a hypothesis |

---

## Symbol operators

| Symbol | Meaning |
|---|---|
| $\lVert v \rVert^2$ | squared **Euclidean ($\ell_2$) norm** — the sum of squared components; measures magnitude / distance |
| $\approx$ | "approximately equals" — holds up to a small learned error |
| $\circ$ | **function composition**: $(\hat O_2 \circ \hat O_1)(s) = \hat O_2(\hat O_1(s))$ — apply $\hat O_1$, then $\hat O_2$ |
| $\nabla$ | **gradient** — the vector of partial derivatives |
| $\sim$ | "is distributed as" / "is sampled from" |

---

## One overload to keep straight

$\psi$ labels *learned weights* generically, so both $\varphi_\psi$ (the encoder) and $\pi_\psi$ (the
policy) wear it — but they are **separate parameter sets**, trained together, not the same numbers.
Where the distinction matters, the main text writes $\psi_\varphi$ and $\psi_\pi$.

---

*Back to the main write-up: [JEPA as an Action-Operator World Model](01-jepa-action-operators.md).*
