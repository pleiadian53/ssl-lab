# Chapter 4a — Reading the head-to-head: the scoreboard, the statistics, and the provenance

*A companion to [Chapter 4](04-results.md). That chapter tells the story of the flow-versus-VAE head-to-head; this one is the audit behind it. Every model in a single scoreboard, the statistics that turn a $0.020$ point gap into a tie, the seed noise that makes any single run untrustworthy, and a map from each number back to the experiment that produced it.*

> **Where this sits.** Read [Chapter 4](04-results.md) first for the narrative and the conclusion. This note is reference material for anyone who wants to check that conclusion rather than take it on trust: the numbers at full precision, the test that decides significance, and where each figure comes from. Nothing here changes the verdict. It shows the work behind it.

The metric throughout is the $\Delta$-correlation from [Chapter 4](04-results.md): for a perturbation, generate a predicted response population, form its differential expression $\Delta = \mathrm{mean}(\text{predicted}) - \mathrm{mean}(\text{control})$, and take the Pearson correlation $r$ between the predicted and true $\Delta$ over the perturbation's top-20 differentially-expressed genes. Every combination number below is the mean of that score over the **same twenty held-out two-gene combinations** (the `combo` split, 200 generated cells per perturbation, guidance $1.0$, gene-set condition).

## 1. The consolidated scoreboard

Effect size first, at full precision. The seed-averaged column is the mean over three training seeds; the single-seed A/B column is one fixed seed with the encoder, decoder, and seed all held constant so that only the flow's base distribution changes.

| model | JEPA? | seed-averaged mean $\Delta r$ | single-seed A/B |
|---|---|---|---|
| conditional NB-VAE (baseline) | no | **0.633** | — |
| transport flow (control to outcome) | yes | 0.613 | 0.621 (median 0.688) |
| Gaussian flow (noise to outcome) | yes | 0.584 | 0.580 (median 0.671) |
| transport flow with OT coupling | yes | 0.590 | — |

Calibration second, on the same twenty combinations and top-20 DE genes, with populations sampled as counts from the negative binomial rather than read as expected rates. Lower is better for the energy and Wasserstein distances; coverage should sit at its nominal $0.80$.

| model | joint energy $\downarrow$ | 1-Wasserstein $\downarrow$ | coverage (nominal 0.80) | spread $r$ |
|---|---|---|---|---|
| conditional NB-VAE | **0.0321** | **0.0084** | 1.00 | $-0.215$ |
| transport flow | 0.0383 | 0.0094 | 1.00 | $-0.317$ |
| Gaussian flow | 0.0450 | 0.0102 | 1.00 | $-0.363$ |

Read together, the two tables point the same way. The NB-VAE holds the top point estimate on effect size and the best number on every calibration metric. The best flow variant, transport, sits a shade below it on both axes. The Gaussian flow and the OT-coupled variant trail. Whether the VAE's lead on effect size is real is a question the point estimates cannot answer, which is the subject of the next two sections.

For orientation, the in-distribution reference number, held-out *cells* of *seen* perturbations on the `cells` split with the table condition, is a mean $\Delta r$ of $0.469$ across 216 perturbations. It is not comparable to the combination numbers: a different split, a different condition encoder, and an easier-effect subset of combinations all separate the two.

## 2. How a 0.020 gap becomes a tie

The scoreboard's ordering on effect size is VAE $0.633$, transport $0.613$, OT $0.590$, Gaussian $0.584$. Taken at face value that reads as "the VAE wins," but a point estimate over twenty combinations carries a wide error bar, and the honest question is whether each gap survives resampling.

The tool is a **paired bootstrap** over the twenty per-combination scores. Resample the twenty combinations with replacement ten thousand times; on each resample, recompute the mean of the per-combination *difference* between two models; then read the 2.5th and 97.5th percentiles of those ten thousand means as the 95% confidence interval. A confidence interval that excludes zero is a difference to trust. One that spans zero is within the noise.

The bootstrap is *paired*, comparing the two models combination by combination, for a specific reason. Combinations differ enormously in difficulty: some move many genes cleanly and score high for every model, others are weak and score low for every model. That shared, combination-driven variance is exactly what would swamp an unpaired comparison of two column means. Taking the per-combination difference first cancels it, so the interval reflects the disagreement between the models rather than the spread across combinations.

| comparison | difference | 95% CI | reading |
|---|---|---|---|
| transport $-$ Gaussian | $+0.028$ | $[-0.006, +0.064]$ | borderline; transport's edge is likely real |
| OT $-$ transport | $-0.023$ | $[-0.041, -0.007]$ | significant; OT coupling **hurts** |
| transport $-$ VAE | $-0.020$ | $[-0.077, +0.044]$ | not significant; a **tie** |
| VAE $-$ Gaussian | $+0.049$ | $[-0.021, +0.112]$ | not significant |

The decisive row is transport minus VAE: the VAE leads by $0.020$ on the mean, but the interval $[-0.077, +0.044]$ straddles zero, so "the VAE beats the flow" is not supported. It is a tie with the VAE nominally ahead. The only interval in the whole sweep that excludes zero is OT minus transport, so the single statistically clear finding is that optimal-transport coupling *hurts*. The point-estimate ranking is not the trustworthy conclusion; the intervals are.

## 3. Why single seeds mislead

The reason twenty combinations resolve so coarsely becomes concrete when the same configuration is retrained at three seeds.

| configuration | seed 0 | seed 1 | seed 2 | spread |
|---|---|---|---|---|
| Gaussian flow | 0.623 | 0.562 | 0.569 | ~0.06 |
| transport flow | 0.575 | 0.653 | 0.611 | ~0.08 |

The seed-to-seed spread, roughly $0.06$ to $0.08$, is three to four times the $0.020$ gap to the VAE. The gap the point estimates suggested sits well below the noise floor of a single run. The rankings flip accordingly: at seed 0 the Gaussian flow at $0.623$ *beats* transport at $0.575$, while at seed 1 transport at $0.653$ wins by a wide margin. Any conclusion drawn from one seed, including an earlier reading that had the VAE beating the flow by $0.053$, is an artifact of which seed was drawn.

This also resolves a possible confusion in reading Chapter 4, where two number-pairs appear for the same two models. The pair $0.621 / 0.580$ is the **single-seed A/B**: one fixed seed with everything held constant except the flow's base distribution, which is the cleanest isolation of the transport reformulation's effect. The pair $0.613 / 0.584$ is the **seed-averaged** version over seeds 0, 1, and 2. Both are real and mutually consistent; they differ only in aggregation, and the seed-averaged pair is the one the significance test uses.

## 4. Calibration, in the same audit frame

Effect size grades only the mean of the response, so calibration is where a flow's full predictive distribution could earn an edge the mean cannot show. It does not, and the structure of the calibration table says why.

The marginal metrics are decoder-dominated. Coverage sits at $1.00$ against a nominal $0.80$ for all three models, because the shared negative-binomial decoder is over-dispersed on the top-DE genes, and the per-gene spread correlation is negative for all three. Because the two flow variants share that decoder, they look nearly identical on the marginal metrics, which is the tell that those metrics are reading the decoder rather than the latent distribution.

The one metric built to see past the decoder is the joint energy distance, a multivariate two-sample distance sensitive to gene-gene correlation and multimodality, which is exactly the structure a rich latent flow could carry and a marginal cannot. Its ranking is VAE $0.0321$, transport $0.0383$, Gaussian $0.0450$. Transport beats the Gaussian flow, consistent with the effect-size axis, but does not beat the VAE, and the transport-minus-VAE difference is not significant. So the distributional axis agrees with the mean axis rather than rescuing the flow.

## 5. Provenance: every number, and how to reproduce it

Each figure above comes from a named evaluation run whose report is written under `output/<run>/reports/`, produced by one of the pipeline scripts in this example folder. The map below is the audit trail.

| claim | value | run / script |
|---|---|---|
| encoder effective rank (end of pretraining) | 176 / 256 | `norman_stage_a`, Stage-A training report |
| linear probe (test, chance $0.42\%$) | 5.2% | `norman_stage_a`, probe report |
| in-distribution cells, table condition | 0.469 / 216 perts | `norman_stage_a`, `06_eval_effect_size.py` |
| combo, transport flow (single-seed A/B) | 0.621 | `norman_flow_control`, `06_eval_effect_size.py` |
| combo, Gaussian flow (single-seed A/B) | 0.580 | `norman_flow_gaussian`, `06_eval_effect_size.py` |
| combo, NB-VAE baseline | 0.633 | `norman_combo`, `09_eval_cvae_baseline.py` |
| seed-averaged effect size (0.584 / 0.613 / 0.590) | 3 seeds × 3 configs | `norman_sweep_{gaussian,control,control_ot}_s{0,1,2}` |
| calibration (energy / Wasserstein / coverage) | see §1 | `10_eval_calibration.py --model {flow,vae}` |

Two aggregation subtleties are worth stating so the numbers reconcile exactly. The prose pair $0.621 / 0.580$ is the single-seed A/B, while the scoreboard's $0.613 / 0.584$ is the seed-averaged version; both are correct and differ only in how seeds are pooled. And the paired-bootstrap intervals in §2 are computed after the fact from the per-combination scores in the seed-sweep reports; they are a re-analysis of those saved scores rather than a separate saved report.

## 6. The verdict in one line

Across the two axes that can be measured on this test set, effect size and calibration, the JEPA plus conditional-flow stack does not beat a from-scratch conditional NB-VAE. It ties on effect size, with the VAE nominally ahead by $0.020$ and the difference not significant, and it is slightly behind on calibration. Within the flow family, transport is the right parameterization and OT coupling is the wrong one, the only clearly significant effect in the sweep. The compositional gene-set condition, shared by both models, is what drives combination generalization, so it is not the flow machinery that earns the generalization. The one axis not yet measured is data efficiency, where the premise of self-supervised pretraining could still pay off.

---

*Previous: [Chapter 4 — Results](04-results.md). Up: [the method series](index.md). Next: [Chapter 5 — Challenges and limitations](05-challenges-and-limitations.md).*
