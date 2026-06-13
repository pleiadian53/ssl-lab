# Notation Reference — Time-Series JEPA

A standalone glossary for the **Time-Series JEPA** series, grouped by role, with a "read as" column. Keep it open in a second tab while reading.

The encoder is written $E$ here, matching the [Action Operators](../action_operator/00-from-actions-to-operators.md) foundation and the [Operator World Models](../operator_world_models/index.md) series; it is the same object written $\varphi_\psi$ in the I-JEPA literature.

---

## The signal and time

| Symbol | Read as | Meaning |
|---|---|---|
| $x$ | "x" | an observation — one timestep of the signal (in the running example, one day's behavioral index) |
| $x_{\le t}$ | "x up to t" | the **context window**: the stretch of past signal ending at time $t$ |
| $x_{t+1}$ | "x at t plus 1" | the **target**: the next observation, the thing being predicted |
| $t$ | "t" | the current time index |
| $\Delta t$ | "delta t" | the **prediction offset** — how far ahead to predict ($\Delta t = 1$ is one step) |

---

## The four pieces

| Symbol | Read as | Meaning |
|---|---|---|
| $E_\xi$ | "E-xi" | the **online encoder**, trainable weights $\xi$ (Greek *xi*); maps the past window to a latent |
| $z_t = E_\xi(x_{\le t})$ | "z at t" | the **context latent** — a vector capturing where the system is now |
| $E_{\bar\xi}$ | "E-xi-bar" | the **target encoder**: a slow EMA copy of $E_\xi$, used to produce prediction targets; stop-gradient (no backprop into it) |
| $\bar\xi \leftarrow \tau\bar\xi + (1-\tau)\xi$ | — | the **EMA update** of the target weights |
| $\tau$ | *tau* | the **EMA rate** (e.g. $0.99$): how slowly the target encoder trails the online one |
| $g_\phi$ | "g-phi" | the **predictor**, weights $\phi$ (Greek *phi*); takes a context latent plus a query and returns the predicted target latent |
| $q_{\Delta t}$ | "query at delta-t" | the **query**: tells the predictor what to predict; here it carries the offset $\Delta t$ (*how far ahead*) |
| $\hat z_{t+1}$ | "z-hat at t plus 1" | the **predicted next latent**, $\hat z_{t+1} = g_\phi(z_t, q_{\Delta t})$ |

---

## Loss and signals

| Symbol | Read as | Meaning |
|---|---|---|
| $\mathrm{sg}$ | "stop-grad" | **stop-gradient**: treat the argument as a fixed constant during backprop (applied to the target) |
| $\mathcal{L}$ | "script L" | the training **loss**: latent-space squared error, $\lVert \hat z_{t+1} - \mathrm{sg}(E_{\bar\xi}(x_{t+1})) \rVert^2$ |
| residual / **surprise** | — | the size of the prediction gap, $\lVert z_{t+1} - \hat z_{t+1} \rVert$ — small when the signal follows its usual structure, large when it does not |
| $\lVert v \rVert^2$ | — | squared **Euclidean ($\ell_2$) norm** |
| $R(\omega)$ | "R of omega" | a **rotation** by angle $\omega$ — the latent dynamics of a clean cycle in the running example (the gallery's rotation operator) |

---

## Multimodal (Part 2)

| Symbol | Read as | Meaning |
|---|---|---|
| $m$ | "m" | a **modality index** — one channel of the signal (heart rate, sleep, phone activity, …) |
| $E^{(m)}$ | "E-m" | a **per-modality encoder** for channel $m$, before fusion into the shared latent $z_t$ |

---

*Series home: [Time-Series JEPA](index.md). Start at [Part 1 — From I-JEPA to Time-Series JEPA](01-from-ijepa-to-tjepa.md).*
