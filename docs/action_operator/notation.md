# Notation Reference — Action Operators on JEPA

A standalone glossary for [*JEPA as an Action-Operator World Model*](01-jepa-action-operators.md). Every symbol used in that write-up, grouped by role. Keep this open in a second tab while reading the main piece.

---

## Representation — the JEPA side

| Symbol | Meaning |
|---|---|
| $x$ | a raw input data point — an image, an RNA window, a protein sequence |
| $\varphi$ | the **encoder**: a neural network that maps an input to an embedding |
| $\psi$ | the **trainable weights** of a network. Subscript usage: $\varphi_\psi$ = "encoder with weights $\psi$." A generic label for "learned parameters" |
| $z = \varphi_\psi(x)$ | the **latent embedding** of $x$ — a vector that captures *meaning*, not surface pixels/nucleotides |
| $\bar\psi$ | a slow **exponential-moving-average (EMA) copy** of $\psi$. The *target encoder* $\varphi_{\bar\psi}$ is updated by averaging, not by gradients (stop-gradient) |
| $\mathcal{S},\ \mathcal{Z}$ | the **state space** and the **latent space**. When actions operate on latents, $\mathcal{S}=\mathcal{Z}$ |
| $\mathrm{Pred}(\cdot)$ | JEPA's **predictor** — maps a context embedding (plus a position) to the predicted embedding of the target region |
| $p$ | a **position / mask token** — tells JEPA *which* region to predict |

---

## Actions as operators — the GRL side

| Symbol | Meaning |
|---|---|
| $\hat{O}$ | an **action operator**: a function $\hat O:\mathcal{S}\to\mathcal{S}$ that turns a state into its successor. The hat marks "operator (a function), not a number" |
| $\hat{O}:\mathcal{S}\to\mathcal{S}$ | its type: takes a state, returns a state |
| $\theta$ | **operator / action parameters** — the knobs that select *which* operator |
| $\hat{O}_\theta$ | the specific operator chosen by parameters $\theta$ |
| $\Phi(\theta, s)$ | the **operator generator**: builds an operator from parameters, $\hat O_\theta(s)=\Phi(\theta,s)$ |
| $\Theta$ | the **space** of all operator parameters $\theta$ |
| $f_\theta$ | the **feature-space form** of $\hat O_\theta$ — how the operator acts on latents $z$ instead of on raw states |
| $E(\hat{O})\ge 0$ | the **energy functional**: how *large* a transformation the operator is — a least-action / parsimony penalty |

---

## Policy and exploration — the coupling layer

| Symbol | Meaning |
|---|---|
| $\pi$ | the **operator policy**: given a state, it chooses operator parameters |
| $\pi:\mathcal{Z}\to\Delta(\Theta)$ | the policy maps a latent to a **distribution over** operator parameters |
| $\Delta(\Theta)$ | the set of **probability distributions** over $\Theta$. A *stochastic* policy emits one of these; a deterministic policy emits a single $\theta$ |
| $\Delta z = \varphi_\psi(\hat O_\theta(x)) - \varphi_\psi(x)$ | the **change in latent** caused by applying an operator — the operationalized *meaning* of a perturbation |
| $X_{\text{struct}}$ | a **disjoint, independent view** (e.g. conservation, structure, measured expression) used to *justify* a hypothesis |

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

$\psi$ labels *learned weights* generically, so both $\varphi_\psi$ (the encoder) and $\pi_\psi$ (the policy) wear it — but they are **separate parameter sets**, trained together, not the same numbers. Where the distinction matters, the main text writes $\psi_\varphi$ and $\psi_\pi$.

---

*Back to the main write-up: [JEPA as an Action-Operator World Model](01-jepa-action-operators.md).*
