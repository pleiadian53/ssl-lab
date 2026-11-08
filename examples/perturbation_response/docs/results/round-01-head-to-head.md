# Round 1 — the head-to-head: does the flow beat a from-scratch VAE?

*Part of [the results ledger](index.md), which holds the standing verdict, the scoreboard and the settings every round shares. Next: [round 2](round-02-decoder-levers.md).*

**What was tested.** The full stack (frozen JEPA encoder, conditional flow prior, NB count decoder) against a from-scratch conditional NB-VAE with no JEPA and no flow, conditioned on the same gene-set embedding so the comparison isolates the generative machinery. Along the way, two flow variants (transporting from Gaussian noise versus from a real control latent) and optimal-transport coupling.

**Narrative:** [Chapter 4](../conditional-flow-jepa/04-results.md). **Audit and method:** [Chapter 4a](../conditional-flow-jepa/04a-reading-the-head-to-head.md). **Post-mortem:** [Chapter 5](../conditional-flow-jepa/05-challenges-and-limitations.md).

## Experimental settings

| arm | what it is | experiments, seeds 0, 1, 2 |
|---|---|---|
| Gaussian flow | noise to outcome, `--flow-base gaussian --coupling independent` | `norman_sweep_gaussian_s{0,1,2}` |
| transport flow | control latent to outcome, `--flow-base control --coupling independent` | `norman_sweep_control_s{0,1,2}` |
| transport + OT | the transport flow with optimal-transport coupling, `--coupling ot` | `norman_sweep_control_ot_s{0,1,2}` |
| NB-VAE | from scratch, encoder blind to the condition | `norman_vae_s{0,1,2}` |

- **Stage A, shared by every flow.** `01_pretrain_stage_a.py --epochs 50 --reg-coef 0.04` on the 96,322 training cells of the combo split, run by [`run_stage_a_pod.sh`](../../run_stage_a_pod.sh): 50 tokens of 100 genes, embedding width 256, encoder depth 6, predictor depth 2, 4 heads, 12 masked target tokens, VICReg at weight 0.04, and an EMA teacher whose momentum rises from 0.996 to 1. The checkpoint is `norman_stage_a`, md5 `38f51067`.
- **Stage C, shared by every flow.** `03_train_count_decoder.py --split combo --epochs 30`, batch 256, AdamW at learning rate $10^{-3}$: a trunk of $256 \to 512 \to 512$, a rate head over the 5,000 genes, one dispersion per gene, and a negative-binomial likelihood. Trained once (md5 `52a44337`) and copied unchanged into every flow run.
- **Stage B, the flows.** `04_train_cond_flow.py --split combo --cond-type geneset --flow-base <base> --coupling <coupling> --seed <s> --epochs 60`, run by [`run_seed_sweep_pod.sh`](../../run_seed_sweep_pod.sh): batch 256, AdamW at learning rate $10^{-3}$, a velocity field of width 256 with 4 blocks, a 128-dimensional condition built from 64-dimensional gene vectors composed additively, and condition dropout of $0.1$ so that classifier-free guidance is available. Optimal-transport coupling is a Hungarian assignment within each minibatch. 96,256 training cells, with 11,720 control cells as the transport source.
- **The NB-VAE.** `08_train_cvae_baseline.py --split combo --epochs 60` at seeds 0, 1 and 2: latent dimension 256, hidden width 512, the same gene-set condition, KL weight $\beta = 1$, batch 256, AdamW at learning rate $10^{-3}$, gradient clipping at 5.0. It is the no-JEPA control and uses no pretrained part.
- **Grading.** `06_eval_effect_size.py` for flows and `09_eval_cvae_baseline.py` for the VAE: 200 generated cells per pair, generation seed 0, and for flows 100 Euler steps with guidance 1. The predicted profile is the mean of the decoder's expected normalized expression, without sampling counts. `10_eval_calibration.py` samples negative-binomial counts at the median library size of 13,921. Every run was graded again on a CPU after the gene selection was corrected, and the contrasts come from `12_compare_arms.py`, which writes `output/norman_vae_s0/reports/arm_comparison.json`.
- **The variance table** comes from `11_diagnose_variance.py` (200 latents, 64 repetitions, seed 0) on two single-seed runs, the seed-42 transport flow `norman_flow_control` and the original seed-42 NB-VAE `norman_combo`, not from the three-seed arms above.
- **Compute.** The flows on one A40 GPU, about 6 to 7 minutes per run and about an hour for all nine. The NB-VAE on a laptop CPU, three seeds in parallel, in about 50 minutes. Grading on a CPU.

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
