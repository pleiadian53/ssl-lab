# co-adaptation: unfreezing the encoder (Round 5+)

This sub-program continues the perturbation-response work by **unfreezing the JEPA encoder**. Every earlier round held the encoder frozen, which is the only reason the [results ledger](../docs/conditional-flow-jepa/results-ledger.md) could attribute a difference between two arms to the one thing that changed. This program relaxes that freeze **one stage at a time**. Its results **append to the same ledger as Round 5 onward**, because the task has not changed: it is still the single-step control-to-perturbed benchmark on the same Norman data, scored on the same 20 held-out combinations, against the from-scratch NB-VAE bar. It has not become a world model. The bar was Δr = 0.766 through round 5. Round 6 let the baseline's encoder read the perturbation as well as the cell, which raised it to Δr = 0.858, and Phase 1 is judged against that.

The narrative companion is the [co-adaptation series](docs/index.md).

## Why unfreeze

By the end of round 4, the frozen encoder had been measured against two independent requirements.

- **It retained the signal.** A linear ridge readout of the frozen latents scores Δr = 0.852, above the original NB-VAE bar of 0.766 and just below the round-6 bar of 0.858. It is a diagnostic handed real latents, not a model. This is the `linear` arm of [`14_ceiling_analysis.py`](../14_ceiling_analysis.py).
- **It did not keep the signal in small motions.** Fitting the latent response demanded operator generators with a median norm of about 12.4 in round 4, a large transformation rather than the small, near-identity motion the operator's design assumed. Masked-prediction pretraining preserves information, but nothing in it shapes the geometry of how perturbations act.

The rationale is developed in the action-operator series, in [chapter 8, "what comes next"](../docs/action-operator/08-what-comes-next.md), and in [chapter 2](../docs/action-operator/02-why-linear-koopman.md), whose §6 shows that equivariance and Koopman invariance are one condition and whose §10 describes how to check it.

## The scripts, in order

Each phase unfreezes one more stage.

| script | phase | trained jointly with the encoder | attributable? | status |
|---|---|---|---|---|
| `01_koopman_linearity_probe.py` | Phase 0, diagnostic | nothing; the encoder stays frozen | not applicable | **done** |
| `02_coadapt_decoder.py` | Phase 1 | the NB count decoder, with the transition frozen | yes | planned, next |
| `03_coadapt_operator.py` | Phase 2 | the operator, under a Koopman objective | no, it gives attribution up | planned, lower priority |

## Phase 0: is the frozen latent space already linear?

Phase 0 is a cheap diagnostic that turns the program's central premise into a number before any engineering goes into the delicate Phase 2. On the frozen latents, for each perturbation, it fits a free **affine** map and a **flexible nonlinear** map from control to perturbed cells. Because sequencing destroys cells, there are no before-and-after pairs, so both maps are graded by how closely they carry the control cloud onto the perturbed cloud, measured by energy distance. The question is whether the nonlinear map explains response structure that the affine map cannot.

A large gap would mean the frozen coordinates are not Koopman coordinates, which would justify Phase 2. A small gap would mean the geometry already closes approximately linearly and the bottleneck is downstream, in the decoder. The probe stays entirely in latent space, with no decoder and no Δr, because chapter 2 §10 forbids testing this premise on the downstream benchmark: the ceiling showed that benchmark has too little room to tell linear from nonlinear.

### Result

Run on the frozen `norman_stage_a` encoder, 20 principal components, three seeds, with the acceptance gate passed. The full write-up is [chapter 0](docs/00-the-premise-as-a-number.md).

| stratum | perturbations | calibrated nonlinear gap, median | 95% interval |
|---|---|---|---|
| **single-gene** (primary) | 103 | **0.015** | **[0.003, 0.032]** |
| training combinations | 76 | 0.004 | [−0.002, 0.012] |
| validation combinations | 18 | 0.010 | [−0.003, 0.028] |
| held-out combinations | 15 | 0.010 | [−0.011, 0.023] |

The gap is the fraction of the measurable response that a nonlinear map explains beyond the best affine map, after subtracting the advantage it shows on a synthetic cloud that is exactly affine by construction.

- **Verdict by the pre-registered rule: inconclusive.** The singles interval excludes zero, so the advantage is detectable, but the median is far below the 0.10 that would justify Phase 2.
- **In substance: the response is close to affine** in the measurable subspace. Among the half of the singles with the largest responses, the median gap is 0.005. The scattered large values come from perturbations that barely move the cloud, where the affine fit mostly fits noise.
- **This reconciles round 4.** The response is a large but affine transformation. What failed was the near-identity assumption, not linearity.
- **Limits.** Twenty components cover 17% of the latent variance, and unpaired data shows distributions rather than individual cells. The conclusion is that an affine map is sufficient at the population level, within the measurable subspace.

The report is `output/norman_koopman_probe/reports/koopman_probe.json`, and the figure is `samples/koopman_gap.png` in the same experiment directory. To reproduce:

```bash
python examples/perturbation_response/co-adaptation/01_koopman_linearity_probe.py \
    --encoder output/norman_stage_a/checkpoints/encoder.pt --split combo \
    --pca-dim 20 --seeds 3 --steps 300
```

### What it decides

Phase 0 gates **Phase 2 only**. Phase 1 runs regardless, because it attacks the largest *measured* loss: the 0.173 the ceiling attributed to the decoder. It also stays attributable, because the transition stays frozen.

Phase 2 had two motives. Phase 0 largely removes the first, making perturbations act linearly, since they already do in the measurable subspace. It does not test the second, reshaping the encoder so that perturbations become small, well-conditioned motions in which the operator's near-identity prior and composition algebra would apply. That motive remains open.

## Discipline

Every co-adapted encoder invalidates the old ceiling, because the ceiling is conditional on the frozen encoder. So **re-measure a new ceiling**, the oracle Stage B and the linear readout, for each one. The same goes for the Phase 0 probe if a later phase needs to know whether the new coordinates are linear. Pre-commit the primary endpoint, Δr on the 20 held-out combinations, and the decision rule before looking at numbers. State the seed count. Report failures with their diagnosis, and link every number to its script.
