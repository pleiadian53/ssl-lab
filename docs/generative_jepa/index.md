# Generative JEPA

*Turning a model that understands data into one that can generate it.*

JEPA learns by **predicting embeddings, not pixels**: hide part of an input, predict the *representation* of the hidden part from the representation of the visible part. The reward is a famously good encoder — but only an encoder. It has no decoder, defines no density, and cannot produce a new data point. This series is about the smallest honest set of additions that turn that encoder into a **sampleable generative model**, and about what each addition costs.

The recipe is two bolt-on pieces. JEPA gives you a map of the data — a latent space where *meaning* is organized. To generate, you need a way to **pick a point on that map at random** (a prior over latents) and a way to **walk from a latent back to a data point** (a decoder). Train the encoder once, freeze it, then learn those two pieces on top.

## The pipeline

```mermaid
flowchart TD
    X["observation x"] -->|"encoder f_θ (frozen)"| Z["latent z"]
    Z -.->|"train prior on the latents"| P["flow-matching<br/>prior p(z)"]
    Z -.->|"train decoder on the latents"| D["decoder D_ω"]
    P ==>|"sample z ~ p(z)"| Znew["latent z*"]
    Znew ==>|"decode"| Xnew["generated sample x̃"]
```

Read top-to-bottom in two passes. **Solid arrows** are the forward path at training time: encode observations into latents. **Dashed arrows** are the two things you fit on the frozen latents — a prior and a decoder. **Bold arrows** are sampling: draw a latent from the prior, decode it into a brand-new observation.

Stated as four stages:

1. **Encoder** $f_\theta$ — JEPA representation learning. Predict masked-region *embeddings* from context embeddings; no reconstruction. Freeze when done. *(Stage 1.)*
2. **Latent prior** $p(z)$ — a **rectified-flow** model fit to the frozen latents, so you can sample new latents from noise. *(Stage 2.)*
3. **Decoder** $D_\omega$ — maps a latent back to data space, trained on the frozen latents. *(Stage 3.)*
4. **Sampling** — draw $z \sim p(z)$, then $\tilde x = D_\omega(z)$. *(Stage 4, plus how to know whether the samples are any good.)*

## Why this shape

The design is deliberately the *simplest* thing that closes the loop, because the point of the starter is to make every joint visible. Three choices define the v0 route, each revisited in the chapters:

- **Train once, freeze, then add.** The encoder never sees the prior or decoder loss. This keeps the representation purely self-supervised and makes the generative head a clean post-hoc study — at the cost that the latents were never trained to be *decodable* (see [Stage 3](03-the-decoder.md)).
- **One latent per observation.** We mean-pool the patch embeddings into a single vector $z$, so the prior and decoder are small and the math is transparent. Per-patch latents are the obvious next step for sharper, more diverse samples.
- **Flow matching for the prior.** Rectified flow gives a simple regression objective and straight noise-to-data paths — a clean, modern alternative to a diffusion prior.

This is the recipe that, in vision, underlies "encode with a strong representation model, then learn a prior and decoder on top." JEPA's contribution is the *encoder objective*; everything downstream is the generative stack.

## Reading order

The series has **two halves**. **Parts 0–4** build one complete generative model end-to-end — the simplest thing that closes the loop (a flow prior + decoder on a frozen JEPA encoder), the *worked example* the rest opens from. **Parts 5–13** are the **design-space survey**: the full set of ways to make JEPA generative, the two applications they were built for, and a closing discussion of which route to build first.

*The starter (build one, end-to-end):*

| Part | Stage | What you get |
|---|---|---|
| **[0 — Generative models, and why JEPA](00-generative-models-and-why-jepa.md)** | background | what a generative model *is*, the latent-variable recipe, and why JEPA is a good substrate — no prior familiarity assumed |
| **[1 — The JEPA encoder](01-the-jepa-encoder.md)** | encode | predict embeddings not pixels; masking, the EMA target, and why collapse is the thing to watch |
| **[2 — The latent prior](02-the-latent-prior.md)** | sample latents | rectified flow over frozen $z$: the interpolant, the conditional flow-matching loss, ODE sampling |
| **[3 — The decoder](03-the-decoder.md)** | latents to data | $D_\omega: z \to x$ on frozen latents, and the decodability caveat this route makes honest |
| **[4 — Sampling and evaluation](04-sampling-and-evaluation.md)** | generate + judge | close the loop, then *measure* sample quality without eyeballing |

*The design-space survey (the full set of routes, and what they are for):*

| Part | Stage | What you get |
|---|---|---|
| **[5 — Two gaps, four routes](05-two-gaps-four-routes.md)** | the map | the two gaps (G1, G2) every generative JEPA must close, and the four routes that pair them — **start of the survey** |
| **[6 — Route A: a decoder on the latent](06-route-a-latent-decoder-head.md)** | route | the lowest-friction closure; count-aware NB/ZINB decoders (likelihood written out), and the CVAE-collapse honesty |
| &nbsp;&nbsp;↳ **[6a — From rate to counts: library size, worked through](06a-rate-library-size-and-counts.md)** | companion | one small cell end to end: softmax rate $\rho$ → scale by depth $\ell$ → $\mu = \ell\rho$ → NB draw; why depth is nuisance and a perturbation reshaping $\rho$ is effect size |
| **[7 — Route B: variational JEPA](07-route-b-variational-and-beyond-gaussian.md)** | route | the predictor becomes its own conditional prior; the Gaussian critique and the expressive-posterior ladder |
| &nbsp;&nbsp;↳ **[7a — JEPA two streams, rebuilt for Route B](07a-jepa-two-streams-and-route-b.md)** | companion | the architecture from scratch: which stream produces $z_b$, $z_p$, the EMA goalpost $z'$, $\mu/\sigma$, the sampled $\hat z$ — a vector-by-vector inventory |
| &nbsp;&nbsp;↳ **[7b — The learnable prior and the KL term, up close](07b-the-prior-and-the-kl-term.md)** | companion | why a learnable *conditional* prior (not $\mathcal{N}(0,I)$), why the posterior is held close to it (train/test consistency), and the ELBO + closed-form-KL derivations behind the §5 loss |
| &nbsp;&nbsp;↳ **[7c — Which network is the prior?](07c-which-network-is-the-prior.md)** | companion | prior (before outcome) vs posterior (after) without VAE background; amortized vs recognition forms + a recognition-form wiring diagram; which object the "predictor becomes the prior" slogan names; the action-operator generalization |
| **[8 — Route C: conditioned diffusion](08-route-c-conditioned-diffusion.md)** | route | diffusion from scratch, conditioned on the JEPA latent; the most modular, most expressive route |
| **[9 — The conditional flow prior](09-conditional-flow-prior.md)** | synthesis | the starter made conditional — one model that is Route B's flow posterior and Route C-with-flow at once |
| &nbsp;&nbsp;↳ **[9a — The three identities, made precise](09a-three-identities-formalized.md)** | companion | the formal version: all three are one conditioned push-forward $(\Phi_c)_\# \mathcal{N}(0,I)$ — exact as generative objects, coincident as training objectives only in the limit |
| &nbsp;&nbsp;↳ **[9b — Classifier-free guidance](09b-classifier-free-guidance.md)** | companion | a sampling-time knob that sharpens the conditional response; the CFG derivation (Bayes → delete the classifier), condition-dropout training, and the `--p-drop` / `--guidance` flags in the perturbation code |
| **[10 — Route D: world-model planning](10-route-d-world-model-planning.md)** | route | generate *decisions*, not data: plan over actions toward a goal; the bridge to operator world models |
| **[11 — Application: computational biology](11-application-computational-biology.md)** | application | the routes assembled for perturbation response, where **effect size** is the benchmark |
| **[12 — Application: digital phenotyping](12-application-digital-phenotyping.md)** | application | a personalized, controllable diabetes world model — sampling future trajectories under an intervention |
| **[13 — Discussion: choosing a route to build](13-choosing-a-route.md)** | discussion | which route to implement first, and why — a reasoned, *goal-contingent* recommendation, trade-offs kept live |
| **[Appendix — data-modalities primer](appendix-data-modalities.md)** | background | scRNA-seq, EHR codes, wearable streams — from scratch, for readers without a bio background |

New to the symbols? The [notation reference](notation.md) defines every one.

## The reference run

Each chapter points at the runnable code in [`examples/generative_jepa/`](https://github.com/pleiadian53/ssl-lab/tree/main/examples/generative_jepa) and the scripts `01`–`06`. On full MNIST (a proof-of-concept; the core is modality-agnostic), one A40 run produced: encoder effective rank **103.7 / 128** with **95.3%** linear-probe accuracy; generated digits that an independent classifier reads as **all ten classes** with high confidence and that sit **as far from the training set as real held-out images do** — that is, genuinely new, not memorized. The numbers, and how to read them, are in [Stage 4](04-sampling-and-evaluation.md).

---

*New to generative modeling or JEPA? Start with [Part 0 — Generative models, and why JEPA](00-generative-models-and-why-jepa.md). Already comfortable with latent-variable generative models? Jump to [Part 1 — The JEPA encoder](01-the-jepa-encoder.md). Already know the starter and want the full design space? Go straight to [Part 5 — Two gaps, four routes](05-two-gaps-four-routes.md).*
