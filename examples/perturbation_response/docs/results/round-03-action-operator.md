# Round 3 — the action operator: does modeling the *transition* beat modeling the destination?

*Part of [the results ledger](index.md), which holds the standing verdict, the scoreboard and the settings every round shares. Previous: [round 2](round-02-decoder-levers.md) · Next: [round 4](round-04-operator-algebra.md).*

**What changed.** The lever of [Chapter 7](../conditional-flow-jepa/07-modeling-the-transition-action-operators.md), built for the first time. Instead of learning a free velocity field that samples where a perturbed cell *lands*, learn the operator that carries a control cell to its perturbed counterpart:

$$c = e(p), \qquad \alpha = \pi(c), \qquad M = \sum_i \alpha_i B_i, \qquad A_p = \exp(M), \qquad z' = A_p z_b .$$

Here $e(p)$ is the gene-set embedding of perturbation $p$ (so an unseen combination composes from its single-gene parts), $\pi$ is a policy emitting coefficients $\alpha$, $\{B_i\}$ is a learned basis of $16$ generators, and $A_p$ is the operator. Three properties are in the parameterization rather than in a hope. The policy is zero-initialized, so $A_p = \exp(0) = I$ at the start: the operator begins at "this perturbation does nothing" and must earn every departure from the identity. A Frobenius penalty on $M$ is a least-action prior pulling it back toward identity. And $\alpha$ depends on the *condition*, not the cell, so one operator serves every cell of a perturbation, which is both the right model and far cheaper.

Because sequencing destroys the cell, there is no "same cell before and after" and the per-pair equivariance loss cannot be computed at all. So each step pushes the control cloud through $A_p$ and matches its **marginal** against the real perturbed cloud with an **energy distance**, which needs no correspondence. Only Stage B is retrained; the frozen encoder and the count decoder are reused.

## Experimental settings

| arm | flags | experiments, seeds 0, 1, 2 |
|---|---|---|
| deterministic operator | none | `norman_operator_s{0,1,2}` |
| stochastic operator + residual | `--stochastic --residual-scale 1.0` | `norman_operator_stoch_s{0,1,2}` |

- **Stage B only.** `13_train_operator.py --split combo --epochs 60 --seed <s>` on latents of the frozen encoder `norman_stage_a` (md5 `38f51067`), run by [`run_operator_pod.sh`](../../run_operator_pod.sh). The count decoder is reused unchanged, the shared one of `norman_flow_control` (md5 `52a44337`).
- **The model.** A basis of 16 generators, each $256 \times 256$, initialized with random entries scaled by $1/\sqrt{256}$. A policy MLP of widths $128 \to 256 \to 16$ with SiLU maps the 128-dimensional gene-set condition to the coefficients $\alpha$, and its last layer starts at zero. The least-action penalty is $10^{-4} \lVert M \rVert_F^2$. The stochastic arm adds a head for the log standard deviation of $\alpha$, initialized at $-3$, and a residual displacement head of widths $128 \to 256 \to 256$ initialized at zero.
- **Training.** Each step samples 8 perturbations and clouds of 128 control and 128 perturbed cells, 100 steps per epoch, AdamW at learning rate $10^{-3}$, gradient clipping at 1.0. The loss is the sample energy distance between the pushed control cloud and the real perturbed cloud plus the penalty, averaged over perturbations. 196 of the 237 perturbations have enough cells to train on, over 96,256 training cells.
- **Grading.** `06_eval_effect_size.py --stage-b operator` and `10_eval_calibration.py --stage-b operator --model flow`, with 200 cells per pair and seed 0. There is no ODE: control samples are pushed through $A_p$ and decoded. The contrasts come from `12_compare_arms.py`, which writes `output/norman_operator_s0/reports/arm_comparison.json`.
- **The variance table** uses `11_diagnose_variance.py` on the seed-0 run of each operator arm and on the seed-42 transport flow `norman_flow_control`; its coverage column is the three-seed mean.
- **Compute.** One A40 GPU, about 6 minutes per deterministic seed and about 28 per stochastic seed, about 1 hour 40 minutes in all. Grading on a CPU.

**Two arms, three seeds each**, on an A40. Training energy fell from $0.68$ to $0.45$ and $\lVert A - I \rVert_F$ rose from $0$ to about $6$, so the operator did depart from the identity and stay there, rather than collapsing back into an expensive no-op.

| contrast | difference | simultaneous 95% CI | reading |
|---|---|---|---|
| operator − transport flow | $-0.003$ | $[-0.030, +0.024]$ | **not significant: a dead tie** |
| stochastic − deterministic operator | $+0.001$ | $[-0.039, +0.042]$ | not significant |
| **operator − NB-VAE** | $\mathbf{-0.121}$ | $\mathbf{[-0.228, -0.014]}$ | **significant: the baseline still wins** |

**Verdict on the primary endpoint: the operator does not pay off.** It ties the flow it was meant to replace and loses to the from-scratch VAE by the same margin the flow does. This was the deeper of the two levers, the one that changes what the *transition* is rather than what happens downstream of it, and it moves the number by $-0.003$.

**But one mechanism did work, and it is the first time anything has.** The variance decomposition ([Chapter 6](../conditional-flow-jepa/06-beyond-the-current-limit.md)) says the calibration axis needs $\sigma^2_{\text{bio}}$, the latent distribution's share of the predicted spread, to *grow*. The deterministic operator is the only model that has ever done it:

| | $\sigma^2_{\text{dec}}$ | $\sigma^2_{\text{bio}}$ | latent's share | coverage |
|---|---|---|---|---|
| transport flow | 0.538 | 0.140 | 22.4% | 0.375 |
| **operator (deterministic)** | 0.533 | **0.174** | **26.0%** | **0.382** |
| operator (stochastic + residual) | 0.538 | 0.081 | 15.4% | 0.344 |

A near-identity-initialized, least-action-penalized operator produces a *structurally* wider latent cloud than a free flow does, and its coverage is the best of any model built. This is Chapter 6's "Fix B" actually happening. It is simply nowhere near large enough to matter: closing the coverage gap would need $\sigma^2_{\text{bio}}$ near $0.29$, and the operator delivers $0.17$.

**A failure, and an honest limit on what we can say about it.** The stochastic arm was designed to widen the cloud further, by letting the perturbation induce a *distribution* over operators. It did the opposite: $\sigma^2_{\text{bio}}$ fell to $0.081$ and coverage dropped by $0.037$ (an interval excluding zero). The likely mechanism is the per-condition **residual displacement**, which is a constant shift and therefore adds *zero* variance, so it can absorb the mean effect cheaply and let $A_p$ relax back toward the identity. But that remains a hypothesis, **because the arm changed two things at once** (a stochastic $\alpha$ *and* a residual). A two-lever arm cannot attribute its own result. The next round separates them.

**Carry forward.** The energy distance *fell* for the stochastic arm ($0.428$ against the deterministic $0.447$), meaning it matched the latent cloud better in latent space while getting *worse* downstream. A better training objective is not a better model, and that has now happened three times in this project (optimal-transport coupling, the dispersion anchor, and now this).

---

*Up: [the results ledger](index.md). Previous: [round 2](round-02-decoder-levers.md) · Next: [round 4](round-04-operator-algebra.md).*
