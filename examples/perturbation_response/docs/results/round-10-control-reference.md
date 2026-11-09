# Round 10 — the control reference: is there a decoder gap once each readout is read against its own control?

*Part of [the results ledger](index.md), which holds the standing verdict, the scoreboard and the settings every round shares. Previous: [round 9](round-09-sigreg-regularizer.md).*

**What changed.** Nothing was trained. Every round so far scored a learned model's predicted change as its predicted profile for a pair minus the *measured* control mean, against a true change that is the measured pair mean minus the same control mean. A readout through the negative-binomial count decoder reports $\log(1 + 10^4 \rho)$, the log of an expected rate, while the measured profile is the mean of $\log(1 + \cdot)$ of noisy normalized counts. The two can differ by an offset that is the same for every condition, and subtracting the measured control mean leaves that offset inside every predicted change. This round rescored existing checkpoints under two conventions. For a readout with predicted pair profile $P_p$, its own prediction for the control condition $C$, and the measured control mean $m$:

$$
\text{measured:} \quad \hat{\Delta}_p = P_p - m, \qquad\qquad \text{own control:} \quad \hat{\Delta}_p = P_p - C .
$$

The true change is $\Delta_p = \bar{x}_p - m$ in both, where $\bar{x}_p$ is the mean of the pair's held-out cells. The own-control prediction $C$ uses only control cells, which are training data, so it reads nothing held out. The measured convention is the one every earlier round reported.

**How this round came about.** The plan for co-adaptation Phase 1 required a magnitude audit before launch: check the library-size, normalization and control-reference conventions, since agreement in correlation is not a check of magnitude. The audit found that the normalization is consistent, since both the truth and the decoder's readout are expressed per ten thousand counts over the same 5,000 genes. It also found that the decoder's prediction for control cells sits well below the measured control profile. Phase 1 was paused, and this round was registered to measure the consequence.

## Experimental settings

| readout | what it is | runs |
|---|---|---|
| readout ceiling, `roundtrip` | real held-out cells → frozen encoder → shared count decoder | `norman_ceiling_r10` |
| readout ceiling, `latent_mean` | the mean real latent → decoder (exploratory) | `norman_ceiling_r10` |
| linear readout | ridge from the frozen cell vector to expression, fit on training cells | `norman_ceiling_r10` |
| round-9 linear readouts | the same ridge on each round-9 encoder | `norman_ceiling_r10_{vicreg,sigreg_tok,sigreg_pool}_s{0,1,2}` |
| conditioned NB-VAE | round 6 | `norman_vae_cond_s{0,1,2}` |
| transport flow | round 1 | `norman_sweep_control_s{0,1,2}` |
| condition-blind NB-VAE, Gaussian flow, both action operators | exploratory | `norman_vae_blind_s*`, `norman_sweep_gaussian_s*`, `norman_operator_s*`, `norman_operator_stoch_s*` |
| additive reference | uses the measured control by construction, so its two conventions coincide | `norman_additive` |

- **Control predictions.** For the ceiling and linear readouts, `14_ceiling_analysis.py --save-profiles` applies the same readout to every control cell, the cells $m$ averages. For learned models, [`19_score_interaction.py`](../../19_score_interaction.py) regenerates each run's pair predictions exactly as its own evaluation did, with 200 cells, generation seed 0, and for flows 100 Euler steps with guidance 1. Flows and operators are read through the shared count decoder (md5 `52a44337`). It then generates the model's prediction for the control condition with the same settings and a separate generator seeded 2, so the scored predictions are unchanged.
- **Encoders.** The frozen Stage A encoder `norman_stage_a` (md5 `38f51067`) and the nine round-9 encoders.
- **Gates, all passed.**
  - Every identity arm scores 1.000, and its own control equals $m$.
  - The rerun ceiling reproduces the stored roundtrip ($0.67916$) and linear ($0.85214$) values to within $10^{-4}$.
  - Every round-9 rerun reproduces its stored linear readout.
  - Every learned run reproduces its stored per-pair $\Delta r$ exactly on the benchmark's own harness, all 19 jobs, and has a control prediction.
- **Compute.** The ten ceiling reruns on one A40 GPU, since encoding every training cell for the ridge takes about 18 minutes per encoder on a CPU. The 19 model rescorings on a laptop CPU, four at a time, in 35 minutes.
- **Statistics.** Per pair and seed, then the seed mean per pair, then the mean over the 20 pairs. Joint paired bootstrap over pairs, 10,000 resamples, seed 0. The primary uses a max-$t$ simultaneous 95% interval; the secondaries use percentile intervals.

## What was known beforehand

The protocol is the docstring of [`23_score_round10.py`](../../23_score_round10.py), committed before any round-10 run. The audit that prompted the round had already computed both conventions, with a scratch script, for the readout ceiling, the transport flow and the conditioned NB-VAE. The primary was therefore not blind. What the pre-registration protected was the endpoint, the rule and the arm list. Not seen beforehand: the linear readout under the own-control convention, and every exploratory arm.

Two corrections were made to the scoring script after it was first run and before any result was printed. Both are recorded in the protocol and in the commit history.
- **One gate's tolerance was amended.** The gate comparing the script's own recomputation of $\Delta r$ with the stored values had been set at $10^{-9}$. That tolerance is only valid for an identical computation path; the two paths differ by about $10^{-8}$ in every run. The exact reproduction is enforced, and was passed, by the scoring script on the harness itself. The cross-path gate uses $10^{-6}$.
- **An implementation error was fixed.** The code required every metric to be finite for every arm at once, which the additive reference's undefined interaction correlation violated. The rule as written applies per metric and per arms involved.

## The pre-committed endpoint

The readout gap under the own-control convention, on the frozen encoder: for each pair, the linear readout's $\Delta r$ minus the decoder roundtrip's $\Delta r$, both against their own control predictions, averaged over the 20 pairs. One contrast, with a max-$t$ simultaneous 95% interval. The gap is *established* if its estimate is at least $+0.02$ and its interval lies wholly above zero. That would establish a positive gap under the rule, not a true gap above $0.02$. If the interval lies wholly below zero, the decoder readout is better than the linear one; otherwise no gap is established.

## Verdict: no decoder gap is established once each readout is read against its own control

| readout of the frozen latents | $\Delta r$, measured control | $\Delta r$, own control |
|---|---|---|
| linear readout | $0.852$ | $0.831$ |
| decoder roundtrip | $0.679$ | $0.860$ |
| **gap, linear − decoder** | $+0.173$ $[+0.096, +0.245]$ | $\mathbf{-0.029}$ $[-0.098, +0.041]$ simultaneous |

All 20 pairs were evaluable, and the max-$t$ critical value is $3.227$. Under the published convention, the gap reproduces the $0.173$ the ceiling analysis reported. Under the own-control convention it vanishes: the estimate is slightly negative and its interval includes zero.

## Where the published gap came from

The control predictions show why the two conventions disagree. On the pairs' top genes:
- Every readout through the count decoder predicts the control profile between $0.52$ and $0.58$ below the measured control mean, in $\log(1 + \text{CP10K})$ units.
- Across all genes, the decoder's control prediction rises only about $0.67$ for each unit of the measured profile.
- The ridge readouts are unbiased by comparison: an offset of $+0.004$ and a slope of $0.997$.

The decoder's profile is therefore compressed and shifted against measured expression, the same way for every condition. Taking a change against the measured control leaves the whole offset in every predicted change; taking it against the decoder's own control removes the shared part.

## Secondary results, intervals only, no verdicts

| readout or model | $\Delta r$ measured | $\Delta r$ own | $e_p$ measured | $e_p$ own |
|---|---|---|---|---|
| decoder roundtrip, frozen encoder | 0.679 | 0.860 | 1.274 | 0.648 |
| linear readout, frozen encoder | 0.852 | 0.831 | 0.561 | 0.600 |
| decoder, mean latent (exploratory) | 0.706 | 0.849 | 1.195 | 0.556 |
| conditioned NB-VAE | 0.858 | 0.931 | 1.053 | 0.498 |
| transport flow | 0.648 | 0.826 | 1.261 | 0.672 |
| condition-blind NB-VAE | 0.775 | 0.779 | 1.207 | 0.790 |
| Gaussian flow | 0.612 | 0.796 | 1.300 | 0.767 |
| action operator, deterministic | 0.645 | 0.838 | 1.282 | 0.693 |
| action operator, stochastic + residual | 0.646 | 0.816 | 1.275 | 0.736 |
| round 9, VICReg linear readout | 0.853 | 0.832 | 0.543 | 0.578 |
| round 9, SIGReg on tokens | 0.747 | 0.734 | 0.806 | 0.923 |
| round 9, SIGReg on the cell vector | 0.756 | 0.703 | 0.794 | 0.907 |
| additive reference | 0.961 | 0.961 | 0.289 | 0.289 |

- **Round 8's question under the own-control convention.** Additivity still has the smaller error. The conditioned NB-VAE's $e_p$ is $0.209$ $[+0.138, +0.277]$ above additivity's, and the transport flow's is $0.383$ $[+0.290, +0.465]$ above. On $\Delta r$ the conditioned NB-VAE comes within $0.030$ $[-0.062, -0.009]$ of additivity, and the transport flow sits $0.134$ $[-0.219, -0.075]$ below it.
- **Round 9's question under the own-control convention.** SIGReg's linear readout stays worse than VICReg's: $-0.098$ $[-0.140, -0.056]$ and $-0.129$ $[-0.207, -0.068]$ in $\Delta r$.
- **Interaction recovery falls under the own-control convention.** The correlation between predicted and measured departure from additivity drops from $0.32$ to $0.14$ for the conditioned NB-VAE and from $0.22$ to $0.15$ for the transport flow. Part of what earlier rounds counted as interaction recovery appears to have been the offset.

## What it means

The ceiling analysis's main reading, that the frozen representation holds the effect and the trained decoder loses $0.173$ of it, does not survive a change of control reference. Read against its own control prediction, the decoder recovers the effect as well as the linear readout does. What the decoder gets wrong is the absolute level of expression: its profile is compressed and shifted against measured expression in a way shared by every condition. Every model built on this decoder inherits the shift.

Under this reading the learned models are much closer to the data than the published numbers suggested. The conditioned NB-VAE's size-aware error halves, from $1.05$ to $0.50$. They are still not as close as additivity, whose error is $0.29$.

## What it does not mean

- The own-control convention is not the correct one by decree. The two conventions answer different questions. The measured convention asks whether the model predicts the expression of the perturbed population. The own-control convention asks whether it predicts the change relative to its own baseline. A model whose control prediction misses the measured control is miscalibrated in absolute expression, and the measured convention correctly counts that against it.
- The result does not say what causes the offset. The audit points to two candidates. One is the gap between the log of an expected rate and the mean log of noisy counts. The other is a library size that counts all genes while the decoder models 5,000, so the likelihood is fit with means about 3.2 times the observed counts. Neither is tested here.
- Calibration and population quality (coverage, spread, $W_1$ and energy distance) are computed from sampled populations against held-out cells, not from a predicted change, so this round does not rescore them.

## Limits

- The primary was not blind.
- The scoring script was corrected twice before any result was printed: the tolerance amendment and the missing-value fix.
- The primary rests on one run of the frozen encoder and decoder, so its interval reflects the 20 pairs and not any seed variation.
- The comparisons involving seeded models average three seeds per pair first.

## Script and report

- [`run_round10_ceiling_pod.sh`](../../run_round10_ceiling_pod.sh) reruns the ten ceilings.
- [`19_score_interaction.py`](../../19_score_interaction.py) rescores each learned run.
- [`23_score_round10.py`](../../23_score_round10.py) holds the frozen protocol in its docstring and writes `output/round10_compare.json`.

---

*Up: [the results ledger](index.md). Previous: [round 9](round-09-sigreg-regularizer.md).*
