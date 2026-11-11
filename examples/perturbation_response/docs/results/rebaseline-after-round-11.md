# The re-baseline after round 11: every published model under the corrected conventions

*Part of [the results ledger](index.md), which holds the standing verdict, the scoreboard and the settings every round shares. This page is not a round: it tests no hypothesis and states no verdict. It follows [round 11](round-11-library-size.md) and accompanies [round 12](round-12-coadaptation.md).*

**What changed.** [Round 11](round-11-library-size.md) found two conventions that had distorted every published number, and both are now the defaults.

- **The library size in the NB likelihood.** The count decoder's mean is $\mu = L \cdot \rho$, where $\rho$ is a rate profile over the 5,000 modeled genes that sums to one and $L$ is the cell's library size. Every published decoder set $L$ to the total count over all measured genes, about $3.2$ times the total over the modeled genes. That flattened $\rho$ and shifted every decoded profile. $L$ now counts the modeled genes.
- **The normalization of sampled counts.** The measured data was normalized by dividing each cell by its own total over the modeled genes. The benchmark's sampled populations were instead divided by the sampling depth, a fixed number, and sampled at the all-gene median depth of 13,921. Sampled populations are now normalized like the data, each sampled cell by its own total, at the median real total over the modeled genes, 4,291 across all cells.

This page reads every model on the published scoreboard under both corrections, so that later rounds and the method chapters can cite current numbers.

## Settings

| model | what was redone | runs, seeds 0, 1, 2 |
|---|---|---|
| Gaussian flow, transport flow, transport flow with OT coupling (round 1) | the trained flows, read through round 11's corrected decoder of the same seed; not retrained | `norman_rb_{gaussian,transport,ot}_s{s}` |
| action operator, deterministic and stochastic with residual (round 3) | the same | `norman_rb_{operator,operator_stoch}_s{s}` |
| conditioned NB-VAE (round 6) | round 11's retraining with the corrected likelihood | `norman_r11_vae_hvg_s{s}` |
| condition-blind NB-VAE (round 6) | retrained with the corrected likelihood, the round-6 recipe otherwise | `norman_rb_vae_blind_s{s}` |
| additive reference (round 7) | nothing: it uses neither correction | `norman_additive` |
| the ceiling's ladder | readout ceiling and mean-latent rung through each corrected decoder; the linear rung is round 10's | `norman_rb_ceiling_s{s}` |

- **What is not redone.** The flows and operators were trained on the frozen encoder's latents and never see the decoder's likelihood, so they are not retrained; only their readout changes. A flow trained with the corrected decoder in its loop would be a different experiment.
- **Prediction settings.** As every round: 200 generated cells per pair, generation seed 0, and for flows 100 Euler steps with guidance 1.
- **Population quality and calibration** ([`10_eval_calibration.py`](../../10_eval_calibration.py) at its new defaults). On each pair's top 20 genes:
  - coverage is the fraction of true (cell, gene) entries inside the gene's central 80% predicted interval, pooled over the 20 genes, with the nominal 0.80 as the target, not the maximum;
  - spread is the correlation of per-gene standard deviations across the 20 genes;
  - $W_1$ is per gene and averaged, so it scores marginals only;
  - energy distance is joint over the 20 genes, Euclidean, as a V-statistic.
- **Variance decomposition** ([`11_diagnose_variance.py`](../../11_diagnose_variance.py)) on seed 0, at the same depth and normalization.
- **Statistics.** Seeds averaged per pair, then plain percentile intervals from a paired bootstrap over the twenty pairs, 10,000 resamples, seed 0. No max-$t$ family, because nothing is tested.
- **Checks.** The scoring script stops unless each run's $\Delta r$, recomputed from its saved profiles, matches its own report to within $10^{-6}$, and unless every population report says it was normalized like the data.
- **Compute.** One A40 GPU, in the same job as round 12, after an on-pod smoke run.

## Effect size

| model | $\Delta r$ published | $\Delta r$ now | $e_p$ published | $e_p$ now | $\Delta r$, own control | $e_p$, own control |
|---|---|---|---|---|---|---|
| Gaussian flow | 0.612 | 0.808 | 1.300 | 0.625 | 0.777 | 0.702 |
| transport flow | 0.648 | 0.837 | 1.261 | 0.591 | 0.841 | 0.596 |
| transport flow, OT coupling | 0.627 | 0.831 | | 0.598 | 0.837 | 0.645 |
| action operator, deterministic | 0.645 | 0.848 | | 0.579 | 0.862 | 0.607 |
| action operator, stochastic + residual | 0.646 | 0.838 | | 0.598 | 0.821 | 0.654 |
| NB-VAE, encoder blind to the condition | 0.775 | 0.880 | 1.207 | 0.609 | 0.727 | 0.733 |
| NB-VAE, encoder reads the condition | 0.858 | 0.939 | 1.053 | 0.455 | 0.937 | 0.379 |
| additive reference | 0.961 | 0.961 | 0.289 | 0.289 | 0.961 | 0.289 |

"Published" is the measured-control convention as every round before round 10 reported it. "Now" is the same convention through the corrected likelihood. Every decoder-based model rises by $0.08$ to $0.20$ in $\Delta r$, and its size-aware error roughly halves.

Against additivity, under the measured control:

| model − additive | $\Delta r$ | $e_p$ |
|---|---|---|
| Gaussian flow | $-0.152$ $[-0.232, -0.089]$ | $+0.336$ $[+0.260, +0.413]$ |
| transport flow | $-0.124$ $[-0.201, -0.065]$ | $+0.302$ $[+0.231, +0.375]$ |
| transport flow, OT coupling | $-0.130$ $[-0.208, -0.071]$ | $+0.309$ $[+0.236, +0.382]$ |
| action operator, deterministic | $-0.113$ $[-0.174, -0.065]$ | $+0.290$ $[+0.220, +0.362]$ |
| action operator, stochastic + residual | $-0.123$ $[-0.194, -0.069]$ | $+0.309$ $[+0.241, +0.379]$ |
| NB-VAE, encoder blind | $-0.080$ $[-0.116, -0.051]$ | $+0.319$ $[+0.255, +0.385]$ |
| NB-VAE, encoder reads the condition | $-0.022$ $[-0.042, -0.007]$ | $+0.166$ $[+0.093, +0.245]$ |

Every interval excludes zero on both endpoints: under the corrected conventions, no learned model reaches additivity.

## Population quality and calibration

| model | coverage, published (depth 13,921, divided by depth) | coverage now (gap to 0.80) | spread | $W_1$ | energy |
|---|---|---|---|---|---|
| Gaussian flow | 0.357 | $0.755$ ($-0.045$) | 0.640 | 0.449 | 1.061 |
| transport flow | 0.375 | $0.771$ ($-0.029$) | 0.660 | 0.412 | 0.919 |
| transport flow, OT coupling | 0.365 | $0.766$ ($-0.034$) | 0.644 | 0.434 | 1.002 |
| action operator, deterministic | 0.382 | $0.790$ ($-0.010$) | 0.636 | 0.421 | 0.936 |
| action operator, stochastic + residual | 0.344 | $0.740$ ($-0.060$) | 0.658 | 0.411 | 0.947 |
| NB-VAE, encoder blind | 0.324 | $0.731$ ($-0.069$) | 0.733 | 0.463 | 1.028 |
| NB-VAE, encoder reads the condition | 0.340 | $0.776$ ($-0.024$) | 0.819 | 0.262 | 0.355 |

Published spread, $W_1$ and energy for comparison: spread $0.21$ to $0.23$ for the flows and operators, $0.52$ and $0.67$ for the two NB-VAEs; $W_1$ $0.98$ to $1.01$ and $0.96$ and $0.87$; energy $3.6$ to $3.8$ and $4.0$ and $3.2$.

- **Coverage sits near the nominal 0.80 for every model**, between $0.73$ and $0.79$, against $0.32$ to $0.38$ as published. The intervals of the deterministic operator, $[0.761, 0.817]$, the transport flow, $[0.739, 0.800]$, and the conditioned NB-VAE, $[0.746, 0.806]$, reach the nominal level. The published narrowness came mostly from dividing sampled counts by the sampling depth, as round 11 found.
- **Per-gene structure separates the models.** The conditioned NB-VAE has the best spread, $W_1$ and energy distance by a clear margin. The flow family is close together on every metric.

## The ceiling's ladder

| rung | $\Delta r$ published | $\Delta r$ now |
|---|---|---|
| identity, the truth | 1.000 | 1.000 |
| readout ceiling: real held-out cells → frozen encoder → decoder | 0.679 | 0.865 |
| linear readout of the frozen latents | 0.852 | 0.852 |
| mean real latent per pair → decoder | 0.706 (round 10) | 0.823 |
| transport flow → decoder | 0.648 | 0.837 |

The transition's headroom under the ceiling is still about $0.03$. What changes is the readout: through the corrected decoder, the ceiling now sits slightly above the linear readout, not $0.173$ below it.

## The variance decomposition

Per gene, the variance of a model's sampled cells splits into the count noise the decoder adds around each decoded mean, $\sigma^2_{\text{dec}}$, and the variance of the decoded means across the model's latents, $\sigma^2_{\text{bio}}$. The real cells' per-gene variance on the same genes is $0.824$, averaged over the top 20 genes of each pair.

| model, seed 0 | $\sigma^2_{\text{dec}}$ | $\sigma^2_{\text{bio}}$ | latent share | predicted / real |
|---|---|---|---|---|
| Gaussian flow | 0.504 | 0.195 | 0.27 | 0.89 |
| transport flow | 0.518 | 0.256 | 0.32 | 0.97 |
| transport flow, OT coupling | 0.507 | 0.215 | 0.29 | 0.91 |
| action operator, deterministic | 0.506 | 0.303 | 0.35 | 1.01 |
| action operator, stochastic + residual | 0.519 | 0.136 | 0.21 | 0.84 |
| NB-VAE, encoder blind | 0.419 | 0.212 | 0.33 | 0.82 |
| NB-VAE, encoder reads the condition | 0.424 | 0.203 | 0.32 | 0.80 |

As published, the transport flow's sampled variance was $0.678$, made of $0.538$ of count noise and $0.140$ from the latents, about $0.84$ of the real variance. Under the corrected conventions the total is $0.774$, $0.97$ of the real variance, and the latents' contribution has nearly doubled, from $0.140$ to $0.256$, while the count noise is about the same. The flow's populations are no longer short of variance in total. Count noise still makes up about two-thirds of it, as in every model here.

## What it means

The corrected conventions change the size of every published number and keep their order. The conditioned NB-VAE is still the strongest learned model and still behind additivity; the flow family is still clustered and below both NB-VAEs. What changes is the scale of the distances. The flows sit $0.10$ below the conditioned NB-VAE in $\Delta r$, not $0.21$, and the strongest learned model sits $0.022$ below additivity, not $0.103$. On populations, the story shifts from too narrow to the right width with the wrong per-gene structure.

## What it does not mean

- No contrast here is tested. The intervals describe the twenty pairs, and every comparison between models trained in different rounds is read through the references, as the ledger's rules ask.
- The flows were not retrained against the corrected decoder, so their numbers are a re-reading of the published models, not the best the flow family can do under the new conventions.

## Script and report

- [`run_rebaseline_pod.sh`](../../run_rebaseline_pod.sh) reads every model through the corrected decoder, retrains the condition-blind NB-VAE and reruns the ladder. [`run_rebaseline_round12_pod.sh`](../../run_rebaseline_round12_pod.sh) runs it on the pod with round 12.
- [`25_rebaseline.py`](../../25_rebaseline.py) tabulates it and writes `output/rebaseline_compare.json`.

---

*Up: [the results ledger](index.md). Previous: [round 11](round-11-library-size.md) · Next: [round 12](round-12-coadaptation.md).*
