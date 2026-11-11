# Round 11 — decoder calibration: does counting the library size over the modeled genes fix the decoder?

*Part of [the results ledger](index.md), which holds the standing verdict, the scoreboard and the settings every round shares. Previous: [round 10](round-10-control-reference.md) · Next: [the re-baseline](rebaseline-after-round-11.md), [round 12](round-12-coadaptation.md).*

**What changed.** One thing: the total that the negative-binomial count decoder multiplies its rate profile by. The decoder models the 5,000 highly variable genes with a softmax rate profile $\rho$ that sums to one, and its likelihood uses the mean $\mu = L \cdot \rho$. Every decoder before this round set $L$ to the total count over *all* measured genes, which is about $3.2$ times the total over the 5,000 genes it models. With $\rho$ forced to sum to one, the likelihood is then fitted with means about three times the observed counts. This round sets $L$ to the total over the modeled genes, the only scale on which a profile that sums to one can match the counts. Everything else is unchanged.

**Why this change.** [Round 10](round-10-control-reference.md) found that the decoder reads expression on a shifted, compressed scale that is the same for every condition. A diagnosis on control cells pointed at the rate profile itself; control cells are training data and are never scored. The profile was too flat: for highly expressed genes it was about $2.5$ times too small, with the mass spread onto low-expression genes. The diagnosis also ruled out two other suspects. The data's own gap between the log of a mean and the mean of logs runs the other way. And sampling counts instead of reading rates did not remove the offset. The mis-scaled library size explains the flattening: the likelihood would want the profile to sum to about $0.31$, the softmax forbids it, and the surplus is pushed to the genes where over-predicting costs least. Going into the round, this was the leading hypothesis, not a proven cause.

## Experimental settings

| arm | what it is | runs, seeds 0, 1, 2 |
|---|---|---|
| `dec_all` | decoder only on the frozen encoder, library size over all genes (within-round control) | `norman_r11_dec_all_s{s}` |
| `dec_hvg` | the same, library size over the modeled genes | `norman_r11_dec_hvg_s{s}` |
| `vae_all` | the conditioned NB-VAE of round 6, library size over all genes | `norman_vae_cond_s{s}` |
| `vae_hvg` | the same recipe, library size over the modeled genes | `norman_r11_vae_hvg_s{s}` |

- **Decoders.** `co-adaptation/02_coadapt_decoder.py --mode dec --library-size <all|hvg>` on the frozen encoder `norman_stage_a` (md5 `38f51067`): a fresh decoder of the default architecture, 30 epochs, batch 256, AdamW at learning rate $10^{-3}$, gradient clipping at 5.0, on the combo split's 96,322 training cells.
- **cVAE.** `08_train_cvae_baseline.py --split combo --epochs 60 --encoder-cond --library-size hvg`, the round-6 recipe otherwise.
- **Flows are not retrained.** The encoder does not change, so the round-1 transport flows `norman_sweep_control_s{s}` are read through each decoder arm's seed-$s$ decoder (`norman_r11_flow_{all,hvg}_s{s}`).
- **Prediction settings.** As every round: 200 generated cells per pair, generation seed 0, and for flows 100 Euler steps with guidance 1.
- **Population sampling.** Every sampled population, in every arm, is normalized the way the measured data was: each sampled cell by its own total over the modeled genes. The benchmark's published normalization divides by the sampling depth instead. That is not the same thing, because the old decoder's sampled totals vary far more from cell to cell than real cells do. Every arm is sampled at the same two depths: 4,832, the median real total over the modeled genes, and 13,921, the published depth. Arms are compared only within a depth. The published normalization at the published depth is kept as a continuity column.
- **Compute.** One A40 GPU, 30 minutes for the whole round: nine trainings three at a time, then the readout ceilings, the flow and cVAE grading, and the population sampling. The comparison ran on a laptop CPU.
- **Gates, all passed.**
  - The existing decoder's readout ceiling reproduced its stored $0.67916$ before training.
  - Every identity arm scores 1.000.
  - Every learned run's $\Delta r$ reproduces its own report.
  - Every new checkpoint records the library size its arm names.

## The pre-committed endpoint

The protocol is the docstring of [`24_score_round11.py`](../../24_score_round11.py), committed before any round-11 run. The primary endpoint is $\Delta r$ under the published convention, every predicted change taken against the measured control mean, on the 20 held-out pairs. One max-$t$ family holds two contrasts:

- the readout ceiling (real held-out cells, frozen encoder, decoder), `dec_hvg − dec_all`;
- the cVAE's benchmark prediction, `vae_hvg − vae_all`.

A contrast passes if its estimate is at least $+0.02$ and its simultaneous interval lies wholly above zero.

Whether the offset is actually fixed is judged separately, by magnitude and calibration diagnostics fixed in the same protocol. They are computed on 3,000 control cells, on the 500 genes with the highest measured mean, and each new arm is judged against its own all-gene baseline. The judgement rests on the sampled readout: counts sampled at each control cell's own total and normalized like the data, where a like-for-like decoder should show an offset near zero and a slope near one.

## Verdict: both contrasts pass, and the offset is fixed in both model families

| | all genes | modeled genes | contrast | simultaneous 95% interval |
|---|---|---|---|---|
| readout ceiling, $\Delta r$ | $0.718$ | $0.865$ | $\mathbf{+0.147}$ | $[+0.053, +0.241]$ |
| cVAE benchmark, $\Delta r$ | $0.858$ | $0.939$ | $\mathbf{+0.081}$ | $[+0.030, +0.133]$ |

The max-$t$ critical value is $2.684$. Both lower bounds lie above $0.02$, so these intervals establish gains larger than $0.02$, not only gains greater than zero.

**Was the offset fixed?** Yes, by the rule set in advance. On control cells:

| arm | sampled readout: offset (top 500) | slope | sampled per-cell total, coefficient of variation | judgement |
|---|---|---|---|---|
| decoder, all genes | $-0.680$ | $0.711$ | $1.56$ | |
| decoder, modeled genes | $\mathbf{-0.053}$ | $\mathbf{0.978}$ | $0.39$ | **fixed** |
| cVAE, all genes | $-0.590$ | $0.752$ | $2.22$ | |
| cVAE, modeled genes | $\mathbf{-0.038}$ | $\mathbf{0.983}$ | $0.39$ | **fixed** |

The measured cells' per-cell totals have a coefficient of variation of $0.39$, which both new arms now match exactly. The rate readout, reported only as context, moves from about $-0.73$ to $+0.16$, with a slope near $1.05$. That is close to the data's own population-level gap between the log of a mean and a mean of logs, $+0.198$, which is not an exact target.

## Secondary results, intervals only, no verdicts

- **The two conventions now agree.**

  | arm | $\Delta r$, measured control | $\Delta r$, own control | $e_p$, measured | $e_p$, own |
  |---|---|---|---|---|
  | readout ceiling, all genes | 0.718 | 0.859 | 1.226 | 0.634 |
  | readout ceiling, modeled genes | 0.865 | 0.885 | 0.491 | 0.531 |
  | cVAE, all genes | 0.858 | 0.931 | 1.053 | 0.498 |
  | cVAE, modeled genes | 0.939 | 0.937 | 0.455 | 0.379 |
  | transport flow through the decoder, all genes | 0.686 | 0.830 | 1.217 | 0.663 |
  | transport flow through the decoder, modeled genes | 0.837 | 0.841 | 0.591 | 0.596 |
  | linear readout (round 10) | 0.852 | 0.831 | 0.561 | 0.600 |
  | additive reference | 0.961 | 0.961 | 0.289 | 0.289 |

  The convention round 10 had to introduce, taking each change against the model's own control, is no longer needed to read these models fairly.

- **The readout gap is gone under both conventions.** Linear minus decoder roundtrip is $-0.013$ $[-0.065, +0.029]$ against the measured control. Against each readout's own control it is $-0.054$ $[-0.102, -0.021]$: the decoder now reads the frozen latents slightly better than the linear map does.
- **Every decoder-based model moves closer to the data.** The transport flow, read through the new decoder, goes from $0.686$ to $0.837$ in $\Delta r$ and from $1.217$ to $0.591$ in $e_p$ ($+0.151$ $[+0.071, +0.224]$; $-0.626$ $[-0.767, -0.489]$).
- **Additivity still leads, by less.** The cVAE with the new library size is $0.022$ $[-0.042, -0.007]$ below additivity in $\Delta r$. Its $e_p$ is $0.166$ $[+0.093, +0.245]$ above additivity's under the measured control, and $0.090$ $[+0.019, +0.160]$ above under its own control.
- **Population quality and calibration.** Every arm here is sampled at matched depth and normalized like the data. The definitions are in the protocol:
  - coverage is the fraction of true (cell, gene) entries inside each gene's central 80% predicted interval, pooled over the top 20 genes, with the nominal 0.80 as the target;
  - spread is the correlation of per-gene standard deviations;
  - $W_1$ is per gene, averaged;
  - energy distance is joint over the 20 genes.

  | depth 4,832 (real) | coverage (gap to 0.80) | spread | $W_1$ | energy |
  |---|---|---|---|---|
  | flow through the decoder, all genes | $0.797$ ($-0.003$) | $0.148$ | $0.745$ | $2.120$ |
  | flow through the decoder, modeled genes | $0.760$ ($-0.040$) | $0.657$ | $0.412$ | $0.920$ |
  | cVAE, all genes | $0.846$ ($+0.046$) | $0.500$ | $0.542$ | $1.141$ |
  | cVAE, modeled genes | $0.765$ ($-0.035$) | $0.820$ | $0.267$ | $0.366$ |

  With the library size over the modeled genes, spread correlation rises by $0.51$ for the flow and $0.32$ for the cVAE. $W_1$ falls by $0.33$ and $0.28$, and energy distance by $1.20$ and $0.78$; every one of these intervals excludes zero. Coverage moves slightly below nominal, $-0.036$ and $-0.080$ relative to each baseline. Coverage near nominal is not by itself a sign of good populations. The all-gene arms reach it with almost no correct per-gene structure (spread $0.148$ for the flow), because their sampled cells vary in total count about four to six times as much as real cells do. At the published depth of 13,921 the same pattern holds, with lower coverage throughout.
- **The published under-dispersion was mostly the normalization.** Read the published way, dividing sampled counts by the sampling depth, the flow through the all-gene decoder covers $0.378$, reproducing the published $0.33$ to $0.38$. Normalized like the data at the real depth, the same populations cover $0.797$. The narrowness every round reported was largely a property of how sampled counts were normalized, not only of the models.

## What it means

The decoder's calibration error came from a training convention, the library size in its likelihood. Changing that one thing fixed the offset by the protocol's own diagnostics. It raised every decoder-based model under the published convention, and it removed the readout gap that had motivated the co-adaptation program. The rate-profile flattening that the diagnosis predicted is what the change removed, so the library size is now supported as the cause of the offset, not just a candidate. The conditioned cVAE with the corrected likelihood is the strongest learned model in this project: $0.939$ in $\Delta r$ and an $e_p$ of $0.455$. It is still significantly behind additivity on both.

## What it does not mean

- It does not test any other cause of the remaining distance to additivity.
- It does not retrain the flows, whose latents are unchanged. A flow trained with the corrected decoder in its loop is a different experiment.
- Population quality and calibration are secondary here, read at matched depth and normalized like the data. These numbers replace the published ones only once a round makes them its primary.

## Limits

- The all-gene decoder retrained in this round scores $0.718$ in readout-ceiling $\Delta r$, against $0.679$ for the original seed-42 decoder. It was trained with gradient clipping at different seeds, and the within-round control is the right comparison.
- The all-gene cVAE arm is round 6's runs. They used the same recipe, trained earlier.
- Pairs share genes, so the twenty are not independent.

## Script and report

- [`run_round11_pod.sh`](../../run_round11_pod.sh) trains and reads every arm.
- [`24_score_round11.py`](../../24_score_round11.py) holds the frozen protocol and writes `output/round11_compare.json`.
- The option is `--library-size` in `co-adaptation/02_coadapt_decoder.py` and `08_train_cvae_baseline.py`, and the sampled-count normalization is `--normalize` in `10_eval_calibration.py`. Both live in `ssllab.generative.count_decoder`.

---

*Up: [the results ledger](index.md). Previous: [round 10](round-10-control-reference.md) · Next: [the re-baseline](rebaseline-after-round-11.md), [round 12](round-12-coadaptation.md).*
