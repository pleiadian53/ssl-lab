# Co-adaptation: a short series (Round 5+)

*How unfreezing the encoder changes the perturbation-response method, one stage at a time. This series is the narrative companion to the `co-adaptation/` scripts. The round-by-round numbers live in the parent [results ledger](../../docs/conditional-flow-jepa/results-ledger.md), and the motivating theory is the [action-operator series](../../docs/action-operator/index.md), especially [chapter 8](../../docs/action-operator/08-what-comes-next.md) on what comes next and [chapter 2](../../docs/action-operator/02-why-linear-koopman.md), whose §6 shows that equivariance and Koopman invariance are the same condition and whose §10 describes how to check it.*

## Where this sits

Four rounds held the encoder frozen, and by the end of them the frozen encoder had been measured against two separate requirements. It met the first: it **retained the signal**, since a linear readout of its latents scores 0.852 in effect-size correlation, above the NB-VAE baseline's 0.766. It appeared to fail the second: fitting the perturbation response demanded operators with generators of norm about 12.4, a large transformation rather than the small, near-identity motion the operator's design assumed.

This series records what happens when that freeze is relaxed. Each phase unfreezes one more stage, so each result stays as attributable as the design allows.

## Chapters

- **[0. The premise as a number](00-the-premise-as-a-number.md).** Before reshaping the encoder so that perturbations act linearly, check whether they already do. The Koopman-linearity probe (`01_koopman_linearity_probe.py`) fits an affine map and a flexible nonlinear map from control to perturbed latents and compares them by energy distance, since the data is unpaired. The chapter builds the instrument, explains its three guards (an unbiased distance, a principal subspace, and a synthetic affine null), and reads the result: a nonlinear map explains only about 1.5% more of the measurable response than the best affine map. The response is large but close to affine, so the round 4 failure was the near-identity assumption, not linearity. Phase 1 goes next.
- **1. Co-adapting the encoder with the decoder.** Phase 1 attacks the 0.173 that the ceiling analysis attributed to the readout, with the transition frozen so the result stays attributable. *(planned)*
- **2. Co-adapting the encoder with the operator.** Phase 2 covers the Koopman objective, the moving-target problem, and the machinery that keeps the encoder from collapsing. Phase 0 lowered its priority as a linearization project, but its other motive, making perturbations small and well-conditioned motions, remains open. *(planned)*

