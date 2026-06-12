# Part 3 — The decoder

*Stage 3 of the [pipeline](index.md): learn to walk from a latent back to a data point.*

The [prior](02-the-latent-prior.md) gives us new latents; it does not give us images. The second bolt-on piece is a **decoder** $D_\omega$ that maps a latent back to data space. It is the simplest network in the pipeline — and the place where the "freeze first" design has to be honest about a real cost.

## Trained on frozen latents

With the encoder frozen, every training image $x$ has a fixed pooled latent $z = \operatorname{pool}(f_\theta(x))$. The decoder learns to invert that map. For MNIST's pixel values in $[0, 1]$ we treat each pixel as a Bernoulli probability and minimize binary cross-entropy,

$$
\mathcal{L}_{\mathrm{rec}}(\omega) = \mathbb{E}_{x}\operatorname{BCE}\big(D_\omega(\operatorname{sg}(z)), x\big), \qquad z = \operatorname{pool}\big(f_\theta(x)\big),
$$

with stop-gradient on $z$ to make explicit that no signal reaches the encoder — $\theta$ is fixed, only $\omega$ moves. In code, $D_\omega$ is a small MLP that outputs $28 \times 28$ pixel logits.

## The decodability caveat (a feature, not a bug)

Here is the catch the route makes you confront. JEPA was trained to predict *embeddings*, and a good JEPA embedding deliberately **discards** the surface detail it judged unpredictable — the exact stroke texture, the pixel-level wiggle. That is the source of its semantic strength. But information the encoder threw away is information the decoder cannot recover. The latents were never trained to be *decodable*; the decoder is therefore asked to **hallucinate** a plausible appearance consistent with a semantic latent, not to recover the original pixels.

The visible consequence is soft, slightly averaged reconstructions and samples — sharp enough to read, not crisp. This is expected and instructive: it is the honest price of keeping the representation purely self-supervised. Two design choices compound it, and both are levers for later:

- **Freeze-then-add.** Because the encoder never sees the reconstruction loss, it has no incentive to retain decodable detail. A *hybrid* route co-trains a light decoder head during JEPA so the latents stay decodable — at the risk of dragging the encoder back toward modeling appearance, the very thing JEPA set out to avoid. It is a knob, not a free lunch.
- **One latent per image.** Mean-pooling to a single vector erases spatial layout, so the decoder must reinvent *where* things go from a global summary. Keeping a *per-patch* set of latents preserves that structure and is the most direct path to sharper, more varied output.

Both appear in the [next steps](04-sampling-and-evaluation.md#next-steps).

## Reconstruction as a diagnostic

Before trusting *samples*, check *reconstructions*. Feed real latents through the decoder: if it cannot reproduce images it was trained on, sampled latents have no hope. Reconstruction quality is the ceiling on sample quality. In the reference run the reconstructions are faithful and legible — the decoder inverts the pooled latent well — which is what licenses moving on to sampling. The reconstruction grid (real on top, decoded below) is saved each run for exactly this eyeball check, alongside the quantitative story in [Stage 4](04-sampling-and-evaluation.md).

## In code

| Piece | Where |
|---|---|
| Train the decoder on frozen latents | [`examples/generative_jepa/03_train_decoder.py`](https://github.com/pleiadian53/ssl-lab/blob/main/examples/generative_jepa/03_train_decoder.py) |
| The decoder network | [`src/ssllab/models/decoder.py`](https://github.com/pleiadian53/ssl-lab/blob/main/src/ssllab/models/decoder.py) |

---

*Next: [Part 4 — Sampling and evaluation](04-sampling-and-evaluation.md), where the loop closes and we measure whether the samples are any good.*
