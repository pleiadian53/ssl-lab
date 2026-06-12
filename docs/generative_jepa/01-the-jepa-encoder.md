# Part 1 — The JEPA encoder

*Stage 1 of the [pipeline](index.md): learn a representation by predicting embeddings, then freeze it.*

Everything downstream — the prior, the decoder, the samples — depends on one object: a frozen encoder $f_\theta$ that maps an observation to a latent where *meaning* is organized. This chapter is how JEPA learns that encoder without labels and without ever reconstructing a pixel, and how we check that it learned something real.

## Predict the representation, not the input

Self-supervised learning hides part of the input and asks the model to fill it back in. The families differ in *what* gets predicted. A masked autoencoder predicts the missing **pixels** — and so spends capacity modeling texture and noise that are fundamentally unpredictable. JEPA predicts the missing **embedding**: it asks "what is the *representation* of the hidden region?" A representation is free to ignore the unpredictable surface detail and keep only what is structurally meaningful. That is the whole bet, and it is why JEPA encoders tend to be semantically strong per unit of compute.

Concretely, an image $x$ is cut into patch tokens. On the $28 \times 28$ MNIST POC, that is a $4 \times 4$ grid of $7 \times 7$ patches, so 16 tokens. A random subset of tokens is the **target**; the rest are the **context**. Three small networks then play their roles:

- the **encoder** $f_\theta$ (a small vision transformer) embeds the context tokens;
- the **predictor** $g_\phi$ takes those context embeddings plus the *positions* of the target tokens and predicts what the target embeddings should be;
- the **target encoder** $f_{\bar\theta}$ embeds the full image and supplies the actual target embeddings.

The loss lives entirely in embedding space. Writing $\operatorname{sg}$ for stop-gradient and using subscripts $\mathrm{ctx}$ and $\mathrm{tgt}$ for the context and target token sets,

$$
\mathcal{L}(\theta, \phi) = \mathbb{E}_{x, \mathrm{mask}} \operatorname{SmoothL1}\Big( g_\phi\big(f_\theta(x_{\mathrm{ctx}}), \mathrm{tgt}\big), \operatorname{sg}\big(f_{\bar\theta}(x)_{\mathrm{tgt}}\big)\Big).
$$

There is no decoder anywhere in this objective and no term that touches pixels. The prediction is an embedding; the target is an embedding; the distance is measured between embeddings.

## The target encoder, and why it is a slow copy

A subtle question hides in that loss: where do the targets come from? If the targets came from the *same* network being trained, there is a trivial way to win — map every input to the same constant vector. Prediction becomes effortless and the representation becomes useless. This is **representation collapse**, the central failure mode of any predict-your-own-embedding scheme.

JEPA's main guard is to make the targets come from a **separate, slowly moving copy** of the encoder. The target encoder $f_{\bar\theta}$ is not trained by gradient descent; its weights track the student by an exponential moving average,

$$
\bar\theta \leftarrow m \bar\theta + (1-m)\theta,
$$

with momentum $m$ ramped from $0.996$ toward $1$ on a cosine schedule, and a stop-gradient so no learning signal flows into it. Because the target is a moving goal that the student cannot instantaneously match, the degenerate constant solution stops being a free lunch. As a second, optional belt, a VICReg-style variance–covariance penalty on the embeddings explicitly rewards spread and decorrelation across the batch.

## Knowing it worked: collapse diagnostics

"No collapse" is not a vibe — it is measurable, and you should measure it every epoch. Two cheap signals do the job, and they are distinct:

- **feature standard deviation** — the mean, over latent dimensions, of each dimension's spread across a batch. Drifting to zero means every input is mapping to the same point.
- **effective rank** — a soft count of *how many independent directions* the representation actually uses, computed as the exponential of the entropy of the (normalized) singular-value spectrum of the centered embeddings. It catches the sneakier failure where variance crams into a few axes while the rest go dead — which feature std alone can miss.

The full definitions and how to read them live in the companion note [Representation diagnostics](https://github.com/pleiadian53/ssl-lab/blob/main/examples/jepa_basics/docs/representation-diagnostics.md). The headline from the reference run: effective rank climbs from $\approx 74$ to $\mathbf{103.7}$ out of $128$ over training, with feature std steady around $0.75$ — a high-rank, non-collapsed latent. A separate sanity check, a **linear probe**, fits a logistic regression on the frozen latents and reaches **95.3%** test accuracy on digit classification: the representation is not merely high-variance, it is *semantically* organized, with classes linearly separable, despite never seeing a label.

## The handoff: one latent per observation

The predictor $g_\phi$ has done its job once training ends; we keep only the encoder. For the generative head we need a single vector per observation, so we **mean-pool** the encoder's patch embeddings into

$$
z = \operatorname{pool}\big(f_\theta(x)\big) \in \mathbb{R}^{128}.
$$

This pooled $z$ is the currency of the rest of the pipeline: [Stage 2](02-the-latent-prior.md) learns a distribution over these vectors, and [Stage 3](03-the-decoder.md) learns to turn one back into an image. Pooling to a single vector is the choice that keeps the prior and decoder small and the story clear; it also throws away spatial layout, which is exactly the lever the [next steps](04-sampling-and-evaluation.md#next-steps) reach for when chasing sharper, more diverse samples.

## In code

| Piece | Where |
|---|---|
| Train the encoder | [`examples/jepa_basics/01_train_jepa_mnist.py`](https://github.com/pleiadian53/ssl-lab/blob/main/examples/jepa_basics/01_train_jepa_mnist.py) |
| Encoder, predictor, EMA target, loss | [`src/ssllab/jepa/`](https://github.com/pleiadian53/ssl-lab/tree/main/src/ssllab/jepa), [`src/ssllab/objectives/jepa_loss.py`](https://github.com/pleiadian53/ssl-lab/blob/main/src/ssllab/objectives/jepa_loss.py) |
| Pooled latent | `JEPAEncoder.embed_pooled` in [`src/ssllab/models/encoder.py`](https://github.com/pleiadian53/ssl-lab/blob/main/src/ssllab/models/encoder.py) |
| Collapse diagnostics + probe | [`src/ssllab/eval/`](https://github.com/pleiadian53/ssl-lab/tree/main/src/ssllab/eval) |

---

*Next: [Part 2 — The latent prior](02-the-latent-prior.md), where a rectified-flow model learns to sample new $z$'s from noise.*
