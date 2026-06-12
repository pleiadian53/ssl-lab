# Notation Reference — Generative JEPA

A standalone glossary for the **Generative JEPA** series. Every symbol used across the chapters, grouped by stage, with a "read as" column. Keep it open in a second tab while reading.

This series uses the standard I-JEPA convention: the encoder is $f_\theta$ and the predictor is $g_\phi$. The companion [Operator World Models](../operator_world_models/index.md) series reserves $f_\theta$ for a *latent operator* and writes the encoder as $E$. They are **the same encoder**, named to keep each series internally consistent.

---

## Stage 1 — the encoder

| Symbol | Read as | Meaning |
|---|---|---|
| $x$ | "x" | a raw observation — here a $28 \times 28$ image, but the core is modality-agnostic |
| $f_\theta$ | "f-theta" | the **encoder** (student), weights $\theta$: maps an observation, split into patch tokens, to per-patch embeddings |
| $f_{\bar\theta}$ | "f-theta-bar" | the **target encoder**: a slow exponential-moving-average (EMA) copy of $f_\theta$ that produces the prediction targets. Stop-gradient — no backprop flows into it |
| $\bar\theta \leftarrow m \bar\theta + (1-m)\theta$ | — | the **EMA update** of the target weights; momentum $m$ close to 1 makes the target drift slowly |
| $m$ | "m" | the **EMA momentum**, ramped on a cosine schedule from $0.996$ to $1$ over training |
| $g_\phi$ | "g-phi" | the **predictor**, weights $\phi$: from context embeddings and the *positions* of masked tokens, predicts the target embeddings. Used only during pretraining, then discarded |
| context / target | — | the visible patches the encoder sees vs. the held-out patches whose embeddings are predicted |
| $\mathrm{sg}$ | "stop-grad" | **stop-gradient**: treat the argument as a constant during backprop |
| $z$ | "z" | the **pooled latent** for an observation: the mean of $f_\theta(x)$ over patches, a vector in $\mathbb{R}^{D}$ with $D = 128$. The object the prior and decoder act on |

---

## Stage 2 — the latent prior

| Symbol | Read as | Meaning |
|---|---|---|
| $p(z)$ | "p of z" | the **prior**: a learned, sampleable distribution over latents, fit to the frozen encoder's $z$'s |
| $z_0$ | "z-naught" | a **noise** sample, $z_0 \sim \mathcal{N}(0, I)$ — the start of a generation path |
| $z_1$ | "z-one" | a **data latent**, $z_1 = z$ from a real observation — the end of a path |
| $t$ | "t" | the **flow time**, $t \in [0, 1]$: $t=0$ is noise, $t=1$ is data |
| $z_t$ | "z at t" | the point on the straight path between noise and data, $z_t = (1-t) z_0 + t z_1$ |
| $u_t$ | "u at t" | the **target velocity** of that path, $u_t = z_1 - z_0$ (constant in $t$ for rectified flow) |
| $v_\eta(z, t)$ | "v-eta" | the learned **velocity field**, weights $\eta$: predicts the direction to move a latent at time $t$. The trainable object of the prior |
| $\mathcal{L}_{\mathrm{CFM}}$ | "L-CFM" | the **conditional flow-matching loss**: mean squared error between $v_\eta(z_t, t)$ and $u_t$ |
| $\mu, \sigma$ | "mu, sigma" | per-dimension **mean and standard deviation** of the data latents, used to standardize $z$ before fitting the flow (and undone after sampling) |

---

## Stage 3 — the decoder

| Symbol | Read as | Meaning |
|---|---|---|
| $D_\omega$ | "D-omega" | the **decoder**, weights $\omega$: maps a latent to data space, $D_\omega: \mathbb{R}^{D} \to \mathcal{X}$. Trained on frozen $z$'s |
| $\mathcal{L}_{\mathrm{rec}}$ | "L-rec" | the **reconstruction loss**: binary cross-entropy between decoded pixel probabilities and the real image |

---

## Stage 4 — sampling

| Symbol | Read as | Meaning |
|---|---|---|
| $z^{*}$ | "z-star" | a **sampled latent**, drawn by integrating $v_\eta$ from $z_0 \sim \mathcal{N}(0,I)$ to $t=1$ |
| $\tilde x$ | "x-tilde" | a **generated observation**, $\tilde x = D_\omega(z^{*})$ — a new data point the model invented |
| $\mathbb{E}, \lVert \cdot \rVert$ | — | expectation, and the Euclidean ($\ell_2$) norm |

---

*Series home: [Generative JEPA](index.md).*
