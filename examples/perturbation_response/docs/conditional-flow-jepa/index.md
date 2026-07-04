# The conditional-flow + JEPA method — build, run, and findings

*A method series on generating single-cell perturbation responses with a conditional flow prior over frozen JEPA latents and a negative-binomial count decoder. It covers the idea, the implementation, the training and evaluation workflow, the experimental results against a from-scratch baseline, the challenges we hit, and the directions we believe can push past the current limits.*

> **Where this sits.** The [design-space survey](../../../../docs/generative_jepa/index.md) argued, from theory, that the first method to build is a **conditional flow prior** with a **count decoder** over a **JEPA** representation. The [Reading Perturb-seq](../reading-perturb-seq/index.md) series grounds the problem in the real Norman 2019 data. This series is the attempt itself: what we built, what happened, and what we learned. It reports negative results as carefully as positive ones, because the honest picture is what tells us where to push next.

## The one-paragraph summary

We built the full stack and ran it on held-out two-gene combinations from Norman 2019. The method works and generalizes to unseen combinations. But measured rigorously, with a from-scratch conditional negative-binomial VAE as the control, **the flow does not yet beat that simpler baseline** on either effect-size recovery or distributional calibration. Along the way we found that the way the baseline state enters the condition was leaving the flow no structural advantage, and fixing it (a control-to-outcome transport) is a real improvement to the flow, though not enough to clear the baseline. The compositional gene-set embedding, shared by both models, is what actually drives combination generalization. This series lays out that arc and where we think the breakthrough is.

## The chapters

1. **[The approach](01-the-approach.md)** — the key ideas: the two gaps a plain JEPA leaves, the G1/G2 decomposition (a prior over latents and a decoder to data), why a conditional flow, and how the baseline state and the intervention become the condition.
2. **[Implementation](02-implementation.md)** — the code, module by module: the frozen JEPA encoder, the NB/ZINB count decoder, the conditional velocity field and flow matching, the two condition encoders (a learned table and a compositional gene-set embedding), classifier-free guidance, and the control-to-outcome transport.
3. **[Training and evaluation](03-training-and-evaluation.md)** — the three stages (encoder, decoder, flow), the data and the combination-holdout split, the GPU-pod workflow that keeps the environment clean, and the evaluation harness for effect size and calibration.
4. **[Results](04-results.md)** — the numbers with confidence intervals: in-distribution effect size, held-out-combination generalization, the NB-VAE baseline, the transport reformulation, optimal-transport coupling, and the calibration axis. The honest head-to-head.
5. **[Challenges and limitations](05-challenges-and-limitations.md)** — what fought us: a condition formulation that gave the flow no edge, a coupling that lowered training loss yet hurt the metric, calibration metrics confounded by the decoder, and the statistical-power ceiling of a twenty-combination test set.
6. **[Beyond the current limit](06-beyond-the-current-limit.md)** — the directions we have not yet exhausted: data efficiency, native and parametric (operator-style) conditioning, better-calibrated decoders, joint training, more test combinations for power, and multimodal-structure metrics. Where a breakthrough would most plausibly come from.

## Companion code

Every claim maps to runnable code. The pipeline scripts are [`00`–`10` in this example folder](../../); the generative modules are in [`src/ssllab/generative`](../../../../src/ssllab/generative/) and the metrics in [`src/ssllab/eval`](../../../../src/ssllab/eval/).
