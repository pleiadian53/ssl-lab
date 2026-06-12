# Part 2 — The latent prior

*Stage 2 of the [pipeline](index.md): learn a sampleable distribution over the frozen latents.*

After [Stage 1](01-the-jepa-encoder.md) we have an encoder and, for every training image, a latent $z \in \mathbb{R}^{128}$. We have a *cloud* of latents — but no way to produce a new one. JEPA's objective shaped the latent space to be *predictive*, never *distributional*: there is no prior, no density, no sampler. To generate, the first missing piece is a model $p(z)$ we can draw from. This chapter builds it with **rectified flow**, a flavor of flow matching.

## What we are actually trying to do

We want to convert easy-to-sample noise into latents that look like the data latents. Picture two clouds in $\mathbb{R}^{128}$: a standard Gaussian $\mathcal{N}(0, I)$, and the (unknown) distribution of real $z$'s. If we had a velocity field — an arrow at every point and every time telling a particle which way to drift — we could release a noise sample and let it flow until it lands in the data cloud. Learning that velocity field *is* learning the prior. Sampling is then just following the arrows.

## The straight-line construction

Flow matching makes this learnable by choosing, for each pair of endpoints, a path so simple its velocity is trivial to write down. Take a noise point $z_0 \sim \mathcal{N}(0, I)$ and a data latent $z_1$, and connect them with a straight line in time $t \in [0, 1]$:

$$
z_t = (1-t) z_0 + t z_1.
$$

The velocity of a point moving along this line is constant — it does not depend on $t$:

$$
u_t = \frac{d z_t}{dt} = z_1 - z_0.
$$

So for any sampled $(z_0, z_1, t)$ we know, exactly and for free, both where the particle is ($z_t$) and how fast it should be going ($u_t$). The network's whole job is to *predict that velocity from position and time alone* — without being told which endpoints it came from. Train a field $v_\eta(z, t)$ by regression:

$$
\mathcal{L}_{\mathrm{CFM}}(\eta) = \mathbb{E}\big\lVert v_\eta(z_t, t) - u_t \big\rVert^2 ,
$$

with the expectation over $t \sim U[0, 1]$, noise $z_0 \sim \mathcal{N}(0, I)$, and a data latent $z_1 \sim p_{\mathrm{data}}$, where $z_t$ and $u_t$ are the path and its velocity defined above.

This is the **conditional flow-matching** loss: "conditional" because each target $u_t = z_1 - z_0$ is conditioned on a specific endpoint pair. The quiet miracle of flow matching is that regressing against these *conditional* velocities yields, at the optimum, a field whose ODE transports the full noise distribution onto the full data distribution — even though no single straight line does. Many crossing straight paths average into one coherent flow. A plain mean-squared error is all it takes; there is no adversary, no sampling inside the loss, no likelihood to estimate.

## Sampling: follow the arrows

Once $v_\eta$ is trained, generating a latent is integrating an ordinary differential equation from noise to data:

$$
\frac{dz}{dt} = v_\eta(z, t), \qquad z(0) \sim \mathcal{N}(0, I), \qquad z^{*} = z(1).
$$

The starter uses the simplest integrator — fixed-step Euler, $z \leftarrow z + \Delta t \cdot v_\eta(z, t)$ — marching $t$ from $0$ to $1$ in a few dozen steps. Because rectified flow's paths are nearly straight, few steps suffice; this is its practical edge over a diffusion prior, whose curved reverse process typically wants more. The output $z^{*}$ is a fresh latent — the model's guess at "a plausible point in the data cloud" — handed to [Stage 3](03-the-decoder.md) for decoding.

## One practical detail: standardize first

JEPA latents are not centered or scaled in any friendly way — they are whatever the encoder happened to produce. A velocity field has an easier target if the data cloud is centered at the origin with unit per-dimension spread, matching the noise cloud's geometry. So before fitting the flow we standardize the latents,

$$
\hat z = \frac{z - \mu}{\sigma},
$$

using the per-dimension mean $\mu$ and standard deviation $\sigma$ of the training latents, fit the flow on the $\hat z$'s, and **undo** the transform after sampling ($z^{*} = \sigma \hat z^{*} + \mu$). The statistics $(\mu, \sigma)$ are saved alongside the prior so sampling can invert them.

## The network

$v_\eta$ is a small MLP. Time enters through a sinusoidal embedding of $t$, which then modulates the hidden activations by FiLM (a learned per-feature scale and shift) — a lightweight, standard way to make a network's behavior depend smoothly on $t$. Nothing here is JEPA-specific; the field only ever sees a vector and a time, which is exactly why the same prior code works for any modality's latents.

## In code

| Piece | Where |
|---|---|
| Train the prior on frozen latents | [`examples/generative_jepa/04_train_flow_prior.py`](https://github.com/pleiadian53/ssl-lab/blob/main/examples/generative_jepa/04_train_flow_prior.py) |
| Velocity field, interpolant, CFM loss, ODE sampler | [`src/ssllab/generative/flow.py`](https://github.com/pleiadian53/ssl-lab/blob/main/src/ssllab/generative/flow.py) |

The velocity net, `linear_interpolant`, `cfm_loss`, and `euler_sample` are the four functions; the chapter's math maps one-to-one onto them.

---

*Next: [Part 3 — The decoder](03-the-decoder.md), which turns a sampled latent back into an image.*
