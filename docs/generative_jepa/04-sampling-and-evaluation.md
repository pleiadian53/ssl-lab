# Part 4 — Sampling and evaluation

*Stage 4 of the [pipeline](index.md): close the loop, then find out whether the samples are any good.*

The pieces are in place: a frozen encoder, a [prior](02-the-latent-prior.md) over its latents, and a [decoder](03-the-decoder.md) back to images. Generation is now a two-line procedure — and the harder, more interesting half of this chapter is how to *judge* the result without simply looking at it, because "look at it" does not survive the jump to sequences or molecules.

## Closing the loop

To invent a new data point, sample a latent from the prior and decode it:

$$
z^{*} \sim p(z) \quad\text{(integrate the flow ODE from noise)}, \qquad \tilde x = D_\omega\big(\sigma z^{*} + \mu\big).
$$

That is the whole generator. Draw $z_0 \sim \mathcal{N}(0, I)$, follow $v_\eta$ from $t = 0$ to $t = 1$ to get a standardized latent, undo the standardization with the saved $(\mu, \sigma)$, and decode. In the reference run this produces a grid of **recognizable, varied digits** — JEPA, a model built only to *understand*, now *generating*. The pipeline that strings stages 1–6 together for a real (multi-epoch, GPU) run is [`run_pod_pipeline.sh`](https://github.com/pleiadian53/ssl-lab/blob/main/examples/generative_jepa/run_pod_pipeline.sh); the sampling step alone is [`05_sample_and_decode.py`](https://github.com/pleiadian53/ssl-lab/blob/main/examples/generative_jepa/05_sample_and_decode.py).

## Why evaluation is the real problem

For images you can eyeball a sample grid. For DNA, designed proteins, or gene-expression vectors you cannot — and "it looks plausible" is exactly the judgment that fails to scale. So sample evaluation has to become *quantitative*, and the metrics that matter are the ones that transfer across modalities. The methodology — intrinsic versus extrinsic evaluation, and a battery of metrics that all reduce to "compare the generated distribution to the real one in some feature space" — has its own full treatment in [Evaluating generated samples](https://github.com/pleiadian53/ssl-lab/blob/main/examples/generative_jepa/docs/evaluating-generated-samples.md). Here we read the verdict for this run.

The battery embeds both real and generated samples with an **independent** classifier (not the JEPA encoder, to avoid grading the work with its own marking scheme) and computes:

| Question | Metric | Reference value | Reads as |
|---|---|---|---|
| Do samples look real? | classifier confidence | **0.92** | the oracle sees confident digits |
| Are all modes present? | class coverage (entropy / count) | **0.996**, **10/10** | every digit generated — **no mode collapse** |
| Distributional distance | FID / KID | 10.4 / 0.41 | (comparable only at a fixed oracle + sample size) |
| Fidelity vs diversity | precision / recall | 0.65 / 0.41 | realistic, somewhat less diverse |
| Coverage of the real manifold | density / coverage | 0.96 / 0.80 | ~80% of real modes covered |
| Are samples *new*? | novelty: NN(gen→train) vs NN(test→train) | **2.73 vs 2.53** | generated points sit as far from training as real held-out data — **not memorized** |

The story these tell together: **high fidelity, full class coverage, and genuine novelty**, with diversity (recall) the clear headroom. That last point is not incidental — a generator can ace a distance metric by quietly memorizing the training set, and the novelty check is what rules that out. The diversity gap is the expected signature of the v0 route's two simplifications — a single pooled latent and a frozen, non-decodable encoder — which is precisely what the next steps attack.

## Next steps

The starter closes the loop honestly; the roadmap is about lifting the ceiling and pointing the machinery at problems that matter. Grouped by intent:

**Sharpen and diversify the samples (raise recall).**

- **Per-patch latents.** Replace the mean-pooled vector with the *set* of patch embeddings, and generate the set (an autoregressive or set-diffusion prior over patch latents). Preserving spatial structure is the most direct attack on blurry, low-diversity output.
- **Hybrid decoder.** Co-train a light reconstruction head *during* JEPA so the latents stay decodable, trading a little semantic purity for crispness — a knob to explore, not flip.
- **Scale.** A larger encoder and a richer dataset (CIFAR-10 as the next rung above MNIST) test whether the recipe holds beyond a POC.

**Vary the generative head.**

- **Diffusion prior.** Swap rectified flow for a diffusion prior over the same latents and compare sample quality, diversity, and step count — an apples-to-apples study of prior families on a fixed encoder.

**Make it controllable and scoreable.**

- **Conditional generation.** Condition the prior and decoder on a label or a context (a class, a partial observation). For biology this is the important one: condition on a covariate or perturbation to generate *under a specified intervention*.
- **Score, do not only sample.** The flow defines an exact likelihood via change-of-variables. Turning the prior into a *density* unlocks anomaly detection and variant scoring — directly relevant to genomics, where "how surprising is this sequence?" is a core question.

**Point it at a real modality.**

- **A bio-modality adapter.** The core is modality-agnostic; only the patch/tokenizer adapter is image-specific. A gene-count or DNA-window adapter is the path to **genomic and single-cell generative models** — a central motivation for this work.

**Mature the evaluation.**

- Independent, pretrained embedders for FID/KID; sweeps that compare prior and decoder variants on the full battery; reporting that pins the oracle and sample size so numbers stay comparable across runs.

A related but orthogonal thread — turning JEPA's predictor into an *action-conditioned* latent operator for world-modeling — lives in the [Operator World Models](../operator_world_models/index.md) series; it shares the encoder but answers a different question ("what happens next?" rather than "what is a plausible sample?").

## In code

| Piece | Where |
|---|---|
| Sample and decode | [`05_sample_and_decode.py`](https://github.com/pleiadian53/ssl-lab/blob/main/examples/generative_jepa/05_sample_and_decode.py) |
| Evaluate the samples | [`06_eval_samples.py`](https://github.com/pleiadian53/ssl-lab/blob/main/examples/generative_jepa/06_eval_samples.py), [`src/ssllab/eval/generative.py`](https://github.com/pleiadian53/ssl-lab/blob/main/src/ssllab/eval/generative.py) |
| Full results ledger | [`examples/generative_jepa/results/`](https://github.com/pleiadian53/ssl-lab/tree/main/examples/generative_jepa/results) |

---

*Back to the [series overview](index.md), or the [notation reference](notation.md).*
