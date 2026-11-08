# Round 1 — the head-to-head: does the flow beat a from-scratch VAE?

*Part of [the results ledger](index.md), which holds the standing verdict, the scoreboard and the settings every round shares. Next: [round 2](round-02-decoder-levers.md).*

**What was tested.** The full stack (frozen JEPA encoder, conditional flow prior, NB count decoder) against a from-scratch conditional NB-VAE with no JEPA and no flow, conditioned on the same gene-set embedding so the comparison isolates the generative machinery. Along the way, two flow variants (transporting from Gaussian noise versus from a real control latent) and optimal-transport coupling.

**Narrative:** [Chapter 4](../conditional-flow-jepa/04-results.md). **Audit and method:** [Chapter 4a](../conditional-flow-jepa/04a-reading-the-head-to-head.md). **Post-mortem:** [Chapter 5](../conditional-flow-jepa/05-challenges-and-limitations.md).

**Primary endpoint, max-$t$ simultaneous intervals over the 3-contrast family (critical value $2.95$, against $1.96$ unadjusted):**

| contrast | difference | simultaneous 95% CI | reading |
|---|---|---|---|
| transport − Gaussian | $+0.036$ | $[+0.019, +0.052]$ | **significant**: the transport reformulation is real |
| OT − transport | $-0.021$ | $[-0.039, -0.002]$ | **significant**: OT coupling hurts |
| **transport − NB-VAE** | $\mathbf{-0.118}$ | $\mathbf{[-0.228, -0.008]}$ | **significant**: the baseline wins |

**Secondary endpoints (intervals only, no verdicts).** The VAE tracks per-gene variability far better (spread_r $0.522$ against $0.234$; contrast $[-0.429, -0.136]$). The transport flow posts the best joint energy distance, which is exactly the metric built to see the structure a flow should capture, but the contrast against the VAE is $-0.383$ with an interval of $[-0.960, +0.172]$ that **crosses zero**. On twenty perturbations it does not resolve. It is a hypothesis for a larger test set, not a claim.

**Where the predicted spread goes** (`11_diagnose_variance.py`, by the law of total variance):

| | real variance | predicted total | $\sigma^2_{\text{dec}}$ | $\sigma^2_{\text{bio}}$ | latent's share |
|---|---|---|---|---|---|
| transport flow | 0.824 | 0.678 (0.84×) | 0.538 | 0.140 | 22% |
| NB-VAE | 0.824 | 0.355 (0.46×) | 0.226 | 0.128 | 38% |

Both under-produce spread. The two models' *latent* contributions are close ($0.140$ against $0.128$), so the gap between them is not a gap in what their latent distributions do. It is mostly in the decoder each learned.

**Verdict.** The flow loses to the VAE. Transporting from a real control latent is the right way to build the flow and OT coupling is the wrong way, and both of those are solid. But the method as a whole does not clear its baseline, and the compositional gene-set embedding, which both models share, is what actually drives combination generalization.

---

*Up: [the results ledger](index.md). Next: [round 2](round-02-decoder-levers.md).*
