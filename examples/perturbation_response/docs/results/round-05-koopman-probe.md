# Round 5 — co-adaptation, Phase 0: are the frozen latents already linear?

*Part of [the results ledger](index.md), which holds the standing verdict, the scoreboard and the settings every round shares. Previous: [round 4](round-04-operator-algebra.md) · Next: [round 6](round-06-conditioned-encoder.md).*

**What changed.** Nothing was trained. Round 5 opens the [co-adaptation program](../../co-adaptation/README.md), which relaxes the frozen encoder one stage at a time, and its first step is a diagnostic. Round 4 found that fitting the response demanded generators of norm about $12.4$, far from the identity. That has two readings with different consequences. Either the response is *large but affine*, in which case the encoder already supplies Koopman coordinates and only the near-identity assumption failed, or it is *genuinely nonlinear*, in which case retraining the encoder together with the operator (Phase 2 of the program) is justified. [Chapter 2, §10](../action-operator/02-why-linear-koopman.md#10-how-you-would-actually-check-it) of the action-operator series specifies the test that separates them.

**The pre-committed endpoint was NOT $\Delta r$.** The ceiling bounds the transition stage at $0.03$ of effect size, which cannot distinguish a linear transition from a nonlinear one, so the test lives entirely in latent space. For each perturbation, an affine map $z \mapsto Az + b$ and a residual MLP $z \mapsto z + h(z)$ are fitted to carry the control cloud onto the perturbed cloud by energy distance, since the data is unpaired, and scored on held-out cells. The endpoint is the **calibrated nonlinear gap** $g_{\mathrm{excess}}$: the fraction of the measurable response, $E_{\mathrm{id}} - E_{\mathrm{floor}}$, that the nonlinear map explains beyond the affine one, minus the same quantity measured on a synthetic cloud that is exactly affine by construction. The rule, on the 103 single-gene perturbations: $g_{\mathrm{excess}} \geq 0.10$ with an interval excluding zero justifies Phase 2; below $0.03$ with an interval including zero does not; anything between is inconclusive. Phase 0 gates Phase 2 only. [Co-adaptation chapter 0](../../co-adaptation/docs/00-the-premise-as-a-number.md) builds the instrument and its three guards in full.

**Verdict: inconclusive by the letter, close to affine in substance.** Three seeds, CPU, frozen `norman_stage_a` encoder, top $20$ principal components of the latent, acceptance gate passed (median synthetic-null gap $0.008$ against a tolerance of $0.05$).

| stratum | $n$ | median $g_{\mathrm{excess}}$ | 95% interval |
|---|---|---|---|
| **singles** (primary) | 103 | $\mathbf{0.015}$ | $[0.003, 0.032]$ |
| training combinations | 76 | $0.004$ | $[-0.002, 0.012]$ |
| validation combinations | 18 | $0.010$ | $[-0.003, 0.028]$ |
| held-out combinations | 15 | $0.010$ | $[-0.011, 0.023]$ |

The singles interval excludes zero, so the nonlinear advantage is detectable, but its median is less than a sixth of the bar. At $32$ components the median is $0.021$, interval $[0.010, 0.044]$; at $48$ the acceptance gate fails, so no larger subspace can be trusted with this many cells.

**Where the large per-perturbation values come from (exploratory, after the rule was applied).** A quarter of the singles individually exceed $0.10$, but they are almost all weak responders. Among the half of the singles with the largest measurable response, the median $g_{\mathrm{excess}}$ is $0.005$ and only $2$ of $52$ exceed the bar. In the weaker half, the affine fit does worse than doing nothing in $29\%$ of cases, because there is almost no response to fit, while the zero-initialized MLP stays near the identity; dividing by a near-zero response magnifies the difference. The tail measures a failing fit, not curvature.

**What it means.** Round 4's generators were large because the response really is a large transformation in these coordinates, and that transformation is close to affine. The Koopman *linearity* premise is largely earned in the measurable subspace; the *near-identity* premise is what failed, consistent with round 4's diagnosis that it was imported from expression space. Phase 2 loses its linearization motive. Its other motive, re-coordinatizing the encoder so that perturbations become small, well-conditioned motions, is untested here and stays open. **Phase 1, co-adapting the encoder with the decoder, is next**, because it targets the $0.173$ the ceiling attributed to the readout and keeps the transition frozen.

**Limits.** Twenty components hold $17\%$ of the latent variance, so the conclusion is bounded to the subspace a few hundred cells can measure, which is also the subspace any operator fitted on these cells can use. Unpaired data shows distributions, not cells, so "affine" means an affine map is sufficient at the population level. And every number is conditional on the frozen encoder.

**Script and report.** [`co-adaptation/01_koopman_linearity_probe.py`](../../co-adaptation/01_koopman_linearity_probe.py), run with `--pca-dim 20 --seeds 3 --steps 300`, writes `output/norman_koopman_probe/reports/koopman_probe.json`.

---

*Up: [the results ledger](index.md). Previous: [round 4](round-04-operator-algebra.md) · Next: [round 6](round-06-conditioned-encoder.md).*
