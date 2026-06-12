# Notation Reference — Operator World Models

A standalone glossary for the **Operator World Models** series. Every symbol used across the series, grouped by role, with a "read as" column. Keep it open in a second tab while reading.

This series writes the encoder as $E$ (to match the temporal world-model formulation, where an online encoder and its slow target copy are both in play). The companion [Action Operators](../action_operator/01-jepa-action-operators.md) bridge writes the same encoder as $\varphi_\psi$. They are **the same object**: $E \leftrightarrow \varphi_\psi$. Each series stays internally consistent; this note is the bridge between the two conventions.

---

## Representation — the JEPA side

| Symbol | Read as | Meaning |
|---|---|---|
| $x$ | "x" | a raw observation — an image, a window of behavioral sensor signal, a protein structure |
| $s$ | "s" | the underlying **system state** in observation/state space $\mathcal{S}$ — the *physically real* thing an action transforms (a person's full physiological state, a protein's atomic coordinates) |
| $E$ | "E" | the **encoder**: a neural network mapping an observation to a latent. Same object as $\varphi_\psi$ in the bridge docs |
| $z = E(x)$ | "z equals E of x" | the **latent embedding** — a vector capturing *meaning*, not surface form. Lives in latent space $\mathcal{Z}$ |
| $z_t$ | "z at t" | the latent at time $t$, $z_t = E_\xi(x_{\le t})$ — the encoded history up to $t$ |
| $E_\xi$ | "E-xi" | the **online encoder**, trainable weights $\xi$ (Greek *xi*) |
| $E_{\bar\xi}$ | "E-xi-bar" | the **target encoder**: a slow exponential-moving-average (EMA) copy of $E_\xi$, used to produce prediction targets. Stop-gradient — no backprop flows into it |
| $\bar\xi \leftarrow \tau\bar\xi + (1-\tau)\xi$ | — | the **EMA update** of the target weights; $\tau$ close to 1 makes the target drift slowly |
| $\tau$ | *tau* | the **EMA rate** (e.g. $0.999$): the fraction of the old target weights kept each update |
| $\operatorname{sg}$ | "stop-grad" | **stop-gradient**: treat the argument as a constant during backprop |

---

## Operators — state space and latent space

| Symbol | Read as | Meaning |
|---|---|---|
| $\hat O$ | "O-hat" | an **action operator**: a function $\hat O:\mathcal{S}\to\mathcal{S}$ that turns a state into its successor. The hat marks "operator (a function), not a number" |
| $\hat O_\theta$ | "O-hat-theta" | the specific **state operator** chosen by parameters $\theta$ — the *physically meaningful* transformation acting on the real state $s$. Often inaccessible: you only observe $s$ and $s'$, never the operator itself |
| $f_\theta$ | "f-theta" | the **latent operator**: how $\hat O_\theta$ acts on the latent $z=E(s)$ instead of on $s$. This is the object you actually *compute with* |
| $\theta$ | *theta* | **operator parameters** — the knobs that select *which* operator |
| $\Theta$ | *capital Theta* | the **space** of all operator parameters |

---

## Building the latent operator

| Symbol | Read as | Meaning |
|---|---|---|
| $B_i$ | "B-i" | the **generator basis** — a set of $m$ matrices (each $D\times D$, where $D=\dim\mathcal{Z}$) that span the allowed operator directions. The *basis choice* sets what kind of operator is possible |
| $\alpha$ | *alpha* | the **coefficient vector** over the basis; $\alpha = \theta$. Component $\alpha_i$ says "how much of generator $B_i$" |
| $M_\theta = \sum_i \alpha_i B_i$ | "M-theta" | the **flow generator** — a Lie-algebra element; the "infinitesimal" form of the operator, living in a *flat* vector space |
| $A_\theta = \exp(M_\theta)$ | "A-theta" | the **operator matrix** — the group element obtained by exponentiating the generator. $\exp$ is the matrix exponential |
| $b_\theta$ | "b-theta" | an optional **affine bias** term in latent space |
| $f_\theta(z) = A_\theta z + b_\theta$ | — | the latent operator in its concrete linear form, $\exp(M_\theta) z + b_\theta$ |

---

## Conditioning and policy

| Symbol | Read as | Meaning |
|---|---|---|
| $c_t$ | "c at t" | the **context / intervention covariates** at time $t$ — the *known* causes of state change (hours slept, medication taken, a stressor; a mutation, a ligand) |
| $\pi_\psi$ | "pi-psi" | the **context policy**: emits the operator coefficients from the latent and context, $\theta \sim \pi_\psi(z_t, c_t)$, with weights $\psi$ |
| $\Delta(\Theta)$ | "simplex over Theta" | the set of **probability distributions** over $\Theta$. A stochastic policy emits one of these; a deterministic policy emits a single $\theta$ |
| $g_\phi$ | "g-phi" | JEPA's **predictor** with weights $\phi$ — the query-conditioned operator the action operator generalizes |
| $q_{\Delta t}$ | "query at delta-t" | the predictor's **query**: in vanilla temporal JEPA it carries only the *time offset* $\Delta t$; the action operator replaces it with $\theta(c_t)$ |

---

## Energy, spectrum, loss

| Symbol | Read as | Meaning |
|---|---|---|
| $E(\hat O)\ge 0$ | "energy of O-hat" | the **energy functional**: how *large* a transformation the operator is. Here $E(\hat O)=\lVert M_\theta\rVert_F^2$ — a least-action / parsimony penalty that keeps the operator near identity unless the data demands otherwise. (Distinct from the encoder $E$; context disambiguates) |
| $\lambda$ | *lambda* | the weight on the energy penalty |
| $\operatorname{Re}(\lambda_i)$ | — | the **real part of an eigenvalue** of the generator $M_\theta$. A positive real part in any mode flags locally *growing* (destabilizing) dynamics |
| $\mathcal{L}$ | "script L" | the training **loss** — latent-space squared error between the predicted and target latents |
| $\lVert v\rVert^2$ | — | squared **Euclidean ($\ell_2$) norm** |
| $\lVert M\rVert_F^2$ | — | squared **Frobenius norm** — the sum of squared matrix entries |

---

## One overload to keep straight

The symbol $E$ does double duty: the **encoder** $E$ (and its variants $E_\xi$, $E_{\bar\xi}$) and the **energy functional** $E(\hat O)$. They never appear in the same role, and the argument disambiguates — $E(x)$ encodes an observation, $E(\hat O)$ scores an operator.

---

*Series home: [Operator World Models](index.md). The conceptual bridge: [JEPA as an Action-Operator World Model](../action_operator/01-jepa-action-operators.md).*
