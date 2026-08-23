# Co-adaptation — a short series (Round 5+)

*How unfreezing the encoder changes the perturbation-response method, one stage at a time. This series is
the narrative companion to the `co-adaptation/` scripts; the round-by-round numbers live in the parent
[results ledger](../../docs/conditional-flow-jepa/results-ledger.md), and the motivating theory is the
[action-operator series](../../docs/action-operator/index.md), especially [ch8](../../docs/action-operator/08-what-comes-next.md)
(what comes next) and ch2 §6/§10 (equivariance = Koopman invariance, and how to check the premise).*

## Where this sits

Three rounds held the encoder frozen. That froze one requirement met (the signal was retained: linear
readout 0.852) against one failed (it was not in linearizing coordinates: ‖M‖≈12.4). This series is the
record of relaxing the freeze.

## Planned chapters

- **0. The premise as a number.** The Koopman-linearity probe (`01_koopman_linearity_probe.py`): on the
  frozen latents, is control→perturbed affine or genuinely nonlinear? The instrument, the unpaired-data
  energy-distance grading, the confound guards, and the pre-registered decision rule that gates Phase 2.
  *(written once Phase 0 has run)*
- **1. Co-adapting the encoder with the decoder.** Phase 1: attacking the 0.173 the ceiling attributed to
  the readout, with the transition frozen so the result stays attributable. *(planned)*
- **2. Co-adapting the encoder with the operator.** Phase 2, only if Phase 0 justifies it: the Koopman
  objective, the moving-target problem, and the anti-collapse machinery. *(planned)*

*Public docs present the best accurate version; drafting fixes are narrated in `dev/`, not here.*
