# Notation Reference — Generative JEPA

A standalone glossary for the **Generative JEPA** series, with a "read as" column. Keep it open in a second tab while reading. It is in two halves: **Stages 1–4** cover the [Parts 0–4](index.md) starter (freeze the encoder, learn a prior, decode); the **design-space survey** sections below cover [Parts 5–13](05-two-gaps-four-routes.md), where the starter becomes a *conditional* generator and four routes for doing so are compared.

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

# The design-space survey — Parts 5–13

[Parts 5–13](05-two-gaps-four-routes.md) turn the starter into a *conditional* generator and survey four routes for it. The symbols below extend the starter's. A handful of letters are **overloaded across chapters** — keep these straight before reading:

- $\pi$ — three different objects: the **ZINB dropout probability** $\pi_g$ (Route A, [Part 6](06-route-a-latent-decoder-head.md)), the **learnable Gaussian prior** $\pi$ (Route B, [Part 7](07-route-b-variational-and-beyond-gaussian.md)), and the **policy** $\pi_\psi$ (the companion operator series).
- $t$ — the **flow time** $t \in [0, 1]$ (the prior, Parts [2](02-the-latent-prior.md)/[9](09-conditional-flow-prior.md)) vs. the **diffusion step index** $t \in \{0, \dots, T\}$ (Route C, [Part 8](08-route-c-conditioned-diffusion.md)).
- $\alpha$ — the diffusion **signal-retention** $\alpha_t, \bar\alpha_t$ ([Part 8](08-route-c-conditioned-diffusion.md)) vs. the **mixture weights** $\alpha_m$ of an MDN ([Part 7](07-route-b-variational-and-beyond-gaussian.md)).
- $f_\theta$ — the **encoder** in this series, but a **latent operator** in the companion [Operator World Models](../operator_world_models/index.md) series (the reconciliation is at the top of this page and in [Part 10 §5](10-route-d-world-model-planning.md)).

---

## Conditioning — the shared vocabulary (Parts 5–10)

| Symbol | Read as | Meaning |
|---|---|---|
| **G1**, **G2** | "gap one / gap two" | the two gaps a generative JEPA must close: **G1** turns the predictor's point estimate into a *distribution* over outcomes; **G2** adds a *decoder* from latent back to data |
| $z_b$ | "z-baseline" | the **context** / "before" latent — the encoded state you start from, $z_b = f_\theta(x_b)$. Parts 5–6 wrote it $z_{\text{ctx}}$; from Part 7 on it is $z_b$ |
| $p$ | "p" | the **intervention** itself — a drug, a gene knockout, a logged action (the thing applied to the baseline) |
| $e$ | "e" | a small **learned embedding** map that turns an intervention $p$ into a vector (the trick word embeddings use for tokens) |
| $z_p$ | "z-perturbation" | the **intervention embedding**, $z_p = e(p)$ — the "what we did," produced by $e$, *not* by the encoder |
| $c$ | "c" | the **condition** handed to a route, $c = (z_b, z_p)$ — context plus intervention |
| $\hat z$ | "z-hat" | the **predicted / sampled outcome latent** the predictor (or flow) produces under a condition |
| $x_{\text{out}}$ | — | the **real outcome** observation — the actually-perturbed cell, the realized next state |
| $z'$ | "z-prime" | the **EMA target latent** of the real outcome, $z' = f_{\bar\theta}(x_{\text{out}})$ — the goalpost the prediction is matched against |
| $g_\phi(z, c)$ | "g-phi" | the **conditioned predictor**: from a latent $z$ and a condition $c$, predict the next latent. Stage 1's predictor with its masked-position query generalized to an external condition |

---

## Route A — the count decoder (Part 6)

| Symbol | Read as | Meaning |
|---|---|---|
| $\rho$ | "rho" | the **gene-rate profile**, $\rho = \mathrm{softmax}(\text{decoder}(z))$ — a relative expression vector over genes that sums to one |
| $\ell$ | "ell" | the **library size**: a cell's total captured counts (sequencing depth), entering as a *given* covariate, not a prediction |
| $\mu$ | "mu" | the **NB mean** over genes, assembled as $\mu = \ell \rho$ (per gene $\mu_g = \ell \rho_g$) — *not* emitted directly |
| $\kappa$ | "kappa" | the **NB dispersion** (per gene $\kappa_g$): small $\kappa$ is heavy overdispersion, $\kappa \to \infty$ recovers the Poisson. Variance is $\mu + \mu^2/\kappa$ |
| $\mathrm{NB}, \mathrm{ZINB}$ | — | **negative binomial** / **zero-inflated NB** — the count likelihoods the decoder emits parameters for |
| $\pi_g$ | "pi-g" | the **ZINB dropout probability** for gene $g$ — the chance of a *structural* zero, mixed alongside the NB (distinct from Route B's prior $\pi$) |
| $\Gamma$ | "gamma" | the **gamma function**, the factorial generalized to reals ($\Gamma(n) = (n-1)!$) — the NB's combinatorial normalizer |
| $x_g$ | "x-g" | the observed **count** for gene $g$ in a cell |
| $\mathcal{L}_{\mathrm{NB}}, \mathcal{L}_{\text{decode}}$ | "L-NB / L-decode" | the **count negative-log-likelihood** (Part 6 §2 writes it out factor by factor) and the general **decode term** any route plugs in to reach data space |

---

## Route B — the variational predictor (Part 7)

| Symbol | Read as | Meaning |
|---|---|---|
| $q_\phi(z \mid z_b, z_p)$ | "q-phi" | the **posterior**: the predictor's emitted distribution over the outcome latent, used at *training* (pulled toward the true outcome). The $q$ marks it a learned approximation |
| $\pi(z \mid z_b, z_p)$ | "pi" | the **learnable conditional prior**: what you sample from at *generation*, when no outcome is available (distinct from the ZINB $\pi_g$) |
| $\mu_\phi, \sigma_\phi$ | "mu-phi / sigma-phi" | the posterior's **mean and per-dimension spread**, emitted by the predictor head |
| $\varepsilon, \odot$ | "epsilon / elementwise" | the **reparameterization noise** $\varepsilon \sim \mathcal{N}(0, I)$ and the elementwise product in $\hat z = \mu_\phi + \sigma_\phi \odot \varepsilon$ |
| $\mathrm{KL}(q_\phi \Vert \pi)$, $\mathcal{L}_{\mathrm{KL}}$ | "KL" | the **KL divergence** between posterior and prior — "how different are these two distributions" — closed-form for diagonal Gaussians (Part 7 §5), driven down so the sampled prior agrees with the learned posterior |
| $\mu_\pi, \sigma_\pi$ | "mu-pi / sigma-pi" | the **prior's** mean and spread in the closed-form KL, summed over the $D$ latent coordinates indexed by $i$ |
| $\mathcal{L}_{\text{predict}}$ | "L-predict" | the **representation-space term** $\lVert \mu_\phi - \mathrm{sg}(z') \rVert^2$ — vanilla JEPA's entire loss, kept intact inside Route B |
| $\lambda_{\mathrm{kl}}, \lambda_{\mathrm{dec}}$ | "lambda" | the **loss weights** balancing predictive vs. generative terms ($\lambda_{\mathrm{kl}}$ is the live knob — too high collapses the posterior onto the prior, too low lets generation drift) |
| $\sum_m \alpha_m \mathcal{N}(\mu_m, \sigma_m^2)$ | — | a **mixture density network (MDN)**: $K$ Gaussian components with weights $\alpha_m$ — the multimodal rung of the expressive-posterior ladder |

---

## Route C — conditioned diffusion (Part 8)

| Symbol | Read as | Meaning |
|---|---|---|
| $x_0, x_t, x_T$ | — | the **clean data**, the **noised point at step $t$**, and **pure noise** at the final step $T$ |
| $t, T$ | — | the **diffusion step index** $t \in \{0, \dots, T\}$ — here a discrete denoising step, *not* the flow time $t \in [0, 1]$ |
| $\beta_t$ | "beta-t" | the **noise schedule** — how much fresh Gaussian noise to inject at step $t$ |
| $\alpha_t, \bar\alpha_t$ | "alpha-t / alpha-bar-t" | $\alpha_t = 1 - \beta_t$ and the running product $\bar\alpha_t = \prod_{s=1}^{t} \alpha_s$ — "how much original signal survives to step $t$" (distinct from MDN weights $\alpha_m$) |
| $\epsilon_\theta(x_t, t, c)$ | "epsilon-theta" | the **denoiser**: from a noised point, its step, and the condition $c$, predict the noise $\epsilon$ that was added. Steering on $c$ = the JEPA latent is Route C's defining move |
| $\mathcal{L}_{\text{diff}}$ | "L-diff" | the **diffusion loss**: mean-squared error between the true noise $\epsilon$ and the denoiser's guess |
| $\nabla_x \log p_t(x)$ | "score" | the **score** — the gradient of the noised data's log-density; predicting the noise is, up to a known scale, predicting the score |

---

## The conditional flow prior (Part 9) and Route D — planning (Part 10)

| Symbol | Read as | Meaning |
|---|---|---|
| $v_\eta(z, t, c)$ | "v-eta" | the **conditioned velocity field**: Stage 2's $v_\eta(z, t)$ with a condition slot $c$ added — the single edit that turns the marginal prior into a conditional one |
| $p(z \mid c)$ | "p of z given c" | the **conditional latent distribution** the flow now samples (vs. the starter's marginal $p(z)$) |
| $z^{*}$ | "z-star" | the **sampled latent** from the conditional flow, $z^{*} = z(1)$, obtained by integrating $v_\eta(z, t, c)$ from noise at $t = 0$ to $t = 1$ |
| $z_{\text{goal}}$ | "z-goal" | the **goal** latent a planner aims for (a healthy phenotype, a target glucose pattern) |
| $\mathcal{E}(p)$ | "energy of p" | the **planning energy** $\lVert g_\phi(z_b, e(p)) - z_{\text{goal}} \rVert^2$ — how far action $p$'s predicted outcome lands from the goal (script $\mathcal{E}$ keeps it distinct from the encoder) |
| $p^{*}$ | "p-star" | the **chosen action** $p^{*} = \arg\min_p \mathcal{E}(p)$ — Route D's output (a *decision*, not data) |
| **CEM** | — | the **Cross-Entropy Method**: sample actions, score by energy, keep the **elites**, refit the distribution, repeat — derivative-free search for $p^{*}$ |

> **Crossing into Operator World Models.** Route D's conditioned predictor $g_\phi(z, c)$ is the companion series' **action operator** $f_{\theta(c)}(z) = \exp(M_{\theta(c)}) z + b$; its action $c$ is their $c_t$, its CEM search is their learned policy $\pi_\psi$, and its next latent $z'$ is their $z_{t+1}$. The full symbol-by-symbol reconciliation — including the $f_\theta$ encoder-vs-operator trap — is in [Part 10 §5](10-route-d-world-model-planning.md).

---

*Series home: [Generative JEPA](index.md).*
