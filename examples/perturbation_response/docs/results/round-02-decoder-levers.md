# Round 2 — the decoder levers (superseded, not yet re-run)

*Part of [the results ledger](index.md), which holds the standing verdict, the scoreboard and the settings every round shares. Previous: [round 1](round-01-head-to-head.md) · Next: [round 3](round-03-action-operator.md).*

**What was tested.** The two levers of [Chapter 8](../conditional-flow-jepa/08-modeling-the-readout-count-decoder.md), as opt-in flags on `CountDecoder`: **B1**, an identity-anchored mean head targeting effect size, and **B2**, a state-aware dispersion with a moment-matching anchor targeting calibration. Four arms, single seed, retraining only Stage C.

## Experimental settings

| arm | flags | experiment |
|---|---|---|
| baseline | none | `norman_dec_baseline` |
| B1, anchored mean | `--anchored-mean` | `norman_dec_anchmean` |
| B2, state-aware dispersion | `--state-dispersion --anchor-weight 0.1` | `norman_dec_statedisp` |
| both | B1 and B2 | `norman_dec_both` |

- **Stage C only.** `03_train_count_decoder.py --split combo --epochs 30 --seed 0 <flags>`, batch 256, learning rate $10^{-3}$, on the frozen encoder `norman_stage_a` (md5 `38f51067`), run by [`run_decoder_ablation_pod.sh`](../../run_decoder_ablation_pod.sh). The transport flow is reused unchanged in every arm: the seed-42 `norman_flow_control`. B1 writes the rate as a softmax of the log control profile plus a learned offset that depends on the latent. B2 gives every cell its own dispersion and anchors it, with weight $0.1$, to moments fitted on control cells.
- **Grading.** `06_eval_effect_size.py` with 200 cells per pair, seed 0, top 20 genes, and `10_eval_calibration.py --model flow`. These reports predate the corrected gene selection: they ranked genes by fold change, the selection [Chapter 3e](../conditional-flow-jepa/03e-the-genes-the-metric-scores.md) replaced. That, and the direction of B2, is why the numbers are not carried forward. One seed per arm and no bootstrap comparison.
- **Compute.** One A40 GPU, about 22 minutes for the four arms.

**Status: the numbers from this round are not carried forward.** The round was designed against a reading of the calibration axis that the current gene selection does not support. B2 was built to *narrow* a decoder believed to be over-dispersed, and the decoder is in fact **under-dispersed** on every model measured (coverage $0.33$ to $0.38$, far below nominal). A lever pushing in that direction is aimed the wrong way, so its result carries no information about the design it was meant to test. B1, which targets effect size and is unaffected by the dispersion question, is the part worth re-measuring.

**The re-run.** The four arms are to be re-run on the current gene selection, seeded, so that B1's contribution can be read against the same simultaneous-interval standard as Round 1. It has not been run yet. A seeded runner exists, [`run_decoder_seed_sweep_pod.sh`](../../run_decoder_seed_sweep_pod.sh), and whether to run it as written or fold B1 into a decoder-only study of the readout gap is an open decision.

**Carry forward regardless:** B2's failure mode is instructive independent of its sign. Its NB likelihood *improved* while its target metric did not, which is the same trap optimal-transport coupling sprang in Round 1. **A better training loss is not a better model**, and that has now happened twice.

---

*Up: [the results ledger](index.md). Previous: [round 1](round-01-head-to-head.md) · Next: [round 3](round-03-action-operator.md).*
