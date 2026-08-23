# co-adaptation — unfreezing the encoder (Round 5+)

This sub-program continues the perturbation-response work by **unfreezing the JEPA encoder**. Every
prior round held the encoder frozen, which is the only reason the [results ledger](../docs/conditional-flow-jepa/results-ledger.md)
could attribute a difference between two arms to the one thing that changed. This program relaxes that,
**one stage at a time**, and its results **append to the same ledger as Round 5+** — it is still the
T=1 control→perturbed benchmark on the same Norman data, the same 20 held-out combinations, and the same
from-scratch NB-VAE bar (Δr = 0.766). It has not become a world model.

## Why unfreeze

The frozen encoder met one of two independent requirements and failed the other:

- **Retained the signal — MET.** A linear ridge readout of the frozen latents scores Δr = 0.852, above
  the NB-VAE bar ([`14_ceiling_analysis.py`](../14_ceiling_analysis.py), `linear` arm).
- **Kept it in linearizing coordinates — FAILED.** Fitting the latent response demanded generators of
  median ‖M‖≈12.4 (Round 4), a large rotation, not the small near-identity motion the operator's premise
  assumes. Masked-prediction pretraining preserves information but shapes no geometry.

The rationale is developed in the action-operator series, [ch8 "what comes next"](../docs/action-operator/08-what-comes-next.md),
and ch2 §6/§10 (equivariance = Koopman invariance; how to check the premise).

## The scripts (ordered; unfreeze one stage at a time)

| script | phase | what unfreezes | attributable? |
|---|---|---|---|
| `01_koopman_linearity_probe.py` | **Phase 0** — diagnostic | nothing (frozen latents) | n/a |
| `02_coadapt_decoder.py` | Phase 1 | encoder + NB decoder (transition frozen) | yes |
| `03_coadapt_operator.py` | Phase 2 | encoder + operator (Koopman objective) | no (gives it up) |

**Phase 0** is a near-free diagnostic that converts the program's central premise into a number *before*
we spend engineering on the delicate Phase 2. On the frozen latents, per perturbation, it fits a free
**affine** map and a **flexible nonlinear** map control→perturbed and asks — graded by energy distance,
because the data is unpaired — whether the nonlinear map explains response structure the affine map
cannot. Large gap ⇒ the frozen coordinates are not Koopman-invariant ⇒ Phase 2 is justified. Small gap ⇒
the geometry already closes ~linearly and the bottleneck is downstream (the decoder), so Phase 2 is low
priority. It is latent-space only (no decoder, no Δr): ch2 §10 forbids testing this premise on the
downstream benchmark, which the ceiling already showed cannot distinguish linear from nonlinear.

Phase 0 gates **Phase 2 only**. Phase 1 (E+decoder) runs regardless — it attacks the largest *measured*
loss (the 0.173 the ceiling attributed to the decoder) and stays attributable because the transition
stays frozen.

## Discipline

Every co-adapted encoder invalidates the old ceiling (the ceiling is conditional on the frozen encoder),
so **re-measure a new ceiling** (oracle Stage B + linear readout) for each one. Pre-commit the primary
endpoint (Δr on the 20 held-out combos) and the decision rule before looking at numbers; state the seed
count; report failures with their diagnosis and link every number to its script.
