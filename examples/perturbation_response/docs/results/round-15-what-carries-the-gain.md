# Round 15 — what carries Geneformer V2's gain? Pretraining, input breadth, list length and model size, changed one at a time

*Part of [the results ledger](index.md), which holds the standing verdict, the scoreboard and the settings every round shares. Previous: [round 14](round-14-foundation-encoder.md).*

> **Status.** Every result on this page is final. The protocol ran as committed, with no amendment.

**What changed.** [Round 14](round-14-foundation-encoder.md) found that Geneformer V2-104M's embedding, frozen, reads held-out Norman cells better than our frozen JEPA encoder: a readout ceiling of $0.908$ against $0.863$. V2-104M differs from our encoder in several ways at once. It was pretrained on about 104 million cells, it reads each cell's whole transcriptome rather than our 5,000-gene panel, it reads lists of up to 4,094 genes, and it has about 104 million parameters. This round builds four new versions of V2, each changing exactly one of those properties, and asks which change costs the gain. The [foundation-encoder series](../foundation-encoder/index.md) explains how Geneformer turns a cell into a ranked list of genes and reads it.

**Why this question.** Round 14 could not say why V2 helps, and the answer decides what is worth doing next. If pretraining carries the gain, then training Geneformer jointly with the decoder should start from the pretrained weights, and the published models are worth building on. If reading the whole transcriptome carries it, then our own encoder could gain the same by reading more genes, without any foundation model. If list length carries it, V1's tie in round 14 is explained by its 2,048-gene limit rather than by its size. If size carries it, the larger V2-316M is the model to use.

## Experimental settings

| arm | what differs from `v2_cell` | isolates | width | runs, seeds 0, 1, 2 |
|---|---|---|---|---|
| `v2_cell` | nothing: round 14's store, decoders and ceilings, re-read rather than retrained | the reference | 768 | round 14's `norman_gf_*_v2_cell_s{s}` |
| `v2rand_cell` | V2-104M's architecture with random weights, never trained | pretraining | 768 | `norman_gf_*_v2rand_cell_s{s}` |
| `v2panel_cell` | V2-104M reading only the genes of our 5,000-gene panel | input breadth | 768 | `norman_gf_*_v2panel_cell_s{s}` |
| `v2cut_cell` | V2-104M with every gene list cut after its first 2,048 genes | list length | 768 | `norman_gf_*_v2cut_cell_s{s}` |
| `v2big_cell` | V2-316M, 18 layers and 1,152 wide, in place of V2-104M's 12 layers and 768 | model size | 1,152 | `norman_gf_*_v2big_cell_s{s}` |
| `jepa` | our frozen JEPA encoder `norman_stage_a`, round 14's store and runs, re-read | context | 256 | round 14's `norman_gf_*_jepa_s{s}` |
| `<arm>_p` | each V2 store above projected onto its 256 leading principal components | the pipelines | 256 | `norman_gf_*_{arm}_p_s{s}` |

- **The embeddings.** Every arm is Geneformer's gene mean: the average, over a cell's gene tokens, of the second-to-last layer's hidden states, at Hugging Face revision `1f7fbae4`, for all 109,737 processed cells. [`10_embed_variants.py`](../../foundation-encoder/10_embed_variants.py) writes the four new stores, reusing round 14's tokenizing and embedding code unchanged.
  - *Random weights.* A V2-104M model is built from its published configuration with every weight drawn at random, seed 0, and saved before use, so its exact weights are on record. It reads the same token lists as `v2_cell`.
  - *Panel only.* Each cell's count vector is cut down to the 5,000 panel genes before tokenizing, of which 4,316 have V2 tokens. The total used to scale each cell's counts stays the cell's full total over every measured gene, so each panel gene's scaled value, and therefore its place relative to the other panel genes, is exactly what it was in `v2_cell`; only the genes outside the panel leave the list. A cell's list then holds a median of 1,100 genes, 634 to 1,538 between the 5th and 95th percentiles, and no list reaches the model's limit.
  - *Cut at 2,048.* Round 14's token lists for V2, with each one kept to its summary token, its first 2,048 genes and its end token. 101,584 of the 109,737 cells, 93%, have longer lists and are cut. V1 reads at most 2,048 genes, so this arm also separates V1's shorter input from its smaller model.
  - *V2-316M.* The larger published V2 model, read exactly as V2-104M is.
- **Projections.** [`05_project_store.py`](../../foundation-encoder/05_project_store.py) fits principal components on the 96,322 combo-split training cells only. The 256 components keep 97.2% of the variance of `v2_cell`, 91.6% of `v2rand_cell`, 95.2% of `v2panel_cell`, 96.3% of `v2cut_cell` and 94.9% of `v2big_cell`.
- **Decoders, readouts and pipelines** follow round 14 exactly. A fresh round-11 count decoder is trained on each store for 30 epochs ([`co-adaptation/02_coadapt_decoder.py`](../../co-adaptation/02_coadapt_decoder.py) `--mode dec --library-size hvg --latents <store>`). [`14_ceiling_analysis.py`](../../14_ceiling_analysis.py) reads it with the identity, roundtrip and linear arms. On each projected store, the round-1 transport flow, 256 wide with learning rate $10^{-3}$, is read out through the same seed's projected decoder. Predictions and populations use every round's defaults: 200 cells, generation seed 0, 100 Euler steps, guidance 1, sampled counts normalized like the data at the median real total over the modeled genes.
- **Compute.** One A40 GPU. Embedding every cell took 2.3 GPU-hours with random weights, 0.5 for the panel, 1.0 for the cut lists, whose shorter inputs run faster, and 5.2 for V2-316M, about 9 hours in all. The comparison, 24 decoders, 24 readout ceilings and 12 flows with an on-pod smoke run first, took about an hour and a half. It was interrupted twice, once by running too many readouts at the same time and once by a restart of the cloud machine; each rerun skipped only the steps whose final reports already existed, so every reported decoder, ceiling and flow comes from one uninterrupted run.
- **Gates, all passed before any arm was read.**
  - Every store covers all 109,737 processed cells, was computed on this cache and matches its manifest; no cell was dropped and no value is non-finite.
  - Every new decoder records the dec mode, the corrected library size, the combo split, no cell limit and 30 epochs, and the content hash of its arm's store.
  - Every identity arm scores 1.000, and every roundtrip $\Delta r$ recomputed from the saved profiles matches its stored report within $10^{-6}$.
  - Every pipeline reproduces its stored effect-size report within $10^{-6}$, and every flow records the 256-wide recipe and reads the same latent space as its decoder.
  - The re-read round-14 arms reproduce round 14 exactly: `v2_cell` $0.907653$ and `jepa` $0.862688$, equal to the published values to every stored digit.

## The pre-committed endpoint

The protocol is the docstring of [`11_score_round15.py`](../../foundation-encoder/11_score_round15.py), committed before any round-15 arm was trained. The primary endpoint is the readout ceiling's $\Delta r$ under the measured control. For each held-out pair, the measured change is the mean of its held-out cells minus the mean of the control cells, over the pair's top 20 genes, chosen from the measured data alone. The predicted change is the same difference computed from the decoder's outputs when the real held-out cells are looked up in the arm's store, with the measured control mean subtracted. $\Delta r$ is the Pearson correlation of the two over those 20 genes, averaged over the three seeds per pair and then over the 20 pairs.

One max-$t$ family holds four contrasts, each written so that a positive value means the property helps:

- pretraining: `v2_cell` minus `v2rand_cell`;
- input breadth: `v2_cell` minus `v2panel_cell`;
- list length: `v2_cell` minus `v2cut_cell`;
- model size: `v2big_cell` minus `v2_cell`.

The intervals come from a joint paired bootstrap over the 20 pairs, 10,000 resamples with seed 0, as in round 14. A contrast's property carries a measurable share of the gain if its estimate is at least $+0.02$ and its simultaneous interval lies wholly above zero. It costs readout if the interval lies wholly below zero. Otherwise it is not shown to matter, and the interval says how large a share can be ruled out. If no contrast passes, the round's reading is that no single property is shown to carry the V2 gain, which leaves open that several smaller shares add up.

## Verdict: no single property is shown to carry the gain

| readout ceiling, $\Delta r$ | arm | contrast | estimate | simultaneous 95% interval | verdict |
|---|---|---|---|---|---|
| `v2_cell`, the reference | $0.908$ | | | | |
| `v2rand_cell` | $0.708$ | pretraining | $+0.199$ | $[-0.097, +0.496]$ | not shown to matter |
| `v2panel_cell` | $0.921$ | input breadth | $-0.013$ | $[-0.053, +0.027]$ | not shown to matter |
| `v2cut_cell` | $0.907$ | list length | $+0.000$ | $[-0.019, +0.019]$ | not shown to matter |
| `v2big_cell` | $0.919$ | model size | $+0.012$ | $[-0.024, +0.048]$ | not shown to matter |
| `jepa`, context | $0.863$ | | | | |

The max-$t$ critical value is $5.972$, so each simultaneous interval is about six standard errors wide on each side of its estimate, against $3.387$ in round 14. None of the four contrasts passes, and none costs readout, so under the rule no single property is shown to carry the V2 gain.

For three of the four properties the intervals are informative on their own. Input breadth, list length and model size each move the readout ceiling by at most about $0.05$ in either direction, and list length by at most about $0.02$. Reading only the panel's genes, about a third as many genes per cell, does not lower the readout at all; its estimate is $0.013$ in the panel's favor. Cutting the lists at 2,048 genes changes it by $0.000$. Tripling the model's size raises it by $0.012$. The seeds agree: `v2_cell` scores $0.907$ to $0.908$, `v2panel_cell` $0.910$ to $0.931$, `v2cut_cell` $0.904$ to $0.909$, `v2big_cell` $0.914$ to $0.922$, and `v2rand_cell` $0.705$ to $0.710$.

For pretraining the interval is not informative. Random weights score $0.199$ below the pretrained model, the largest difference in the round, but the simultaneous interval runs from $-0.10$ to $+0.50$.

**Why the simultaneous intervals are so wide.** The max-$t$ procedure studentizes every bootstrap resample of every contrast by that resample's own standard error, and takes the 95th percentile of the largest result across the family as one critical value for all four. When one pair carries much of a contrast, the resamples that leave that pair out have a small standard error, their studentized value is large, and the critical value grows for every contrast in the family. Round 13 met the same effect. Here the pair is again OSR2+UBASH3B, the pair our encoder reads worst. V2 reads it at $0.573$, the panel arm at $0.701$ and V2-316M at $0.692$, so that one pair supplies $+0.128$ of the panel's per-pair differences, whose other 19 average $+0.007$, and $+0.119$ of model size's, whose other 19 average $+0.006$. Computed on one contrast at a time, the critical value would be $5.64$ for input breadth, $3.70$ for model size, $3.63$ for pretraining and $2.88$ for list length, so input breadth sets most of the family's $5.97$. That is the procedure fixed in advance, and the verdict stands on it.

## A closer look at pretraining, after the verdict

This section was not in the protocol. It was run after the verdict, to understand why the largest difference in the round does not pass, and its numbers carry no verdict.

**The difference is on every pair.** Random weights read every one of the 20 pairs worse than the pretrained model. The median difference per pair is $+0.123$, the smallest is $+0.029$, on SAMD1+ZBTB1, and the largest is $+1.035$, on CEBPE+RUNX1T1, which the pretrained model reads at $0.942$ and random weights at $-0.093$, a predicted change pointing the wrong way. The plain percentile interval, which ignores the other three contrasts, is $[+0.123, +0.304]$. The simultaneous interval reaches below zero only because the family's critical value, set mostly by input breadth, is applied to a contrast whose spread across pairs is large.

**Its size depends on how the embedding is read.** The random-weight embedding varies very little between cells. The standard deviation of its coordinates across cells is $0.0095$, against $0.035$ for the pretrained model's, so the decoder must tell cells apart from small differences. A plain linear map recovers more from it than the trained decoder does: the linear readout of `v2rand_cell` scores $0.849$, against its decoder's $0.708$. The pretraining difference is therefore smaller under the other readouts, though still positive on every pair:

| readout | pretrained | random weights | difference | percentile interval | pairs where pretrained is higher |
|---|---|---|---|---|---|
| decoder, full width (the primary endpoint) | $0.908$ | $0.708$ | $+0.199$ | $[+0.123, +0.304]$ | 20 of 20 |
| decoder, projected to 256 | $0.909$ | $0.797$ | $+0.112$ | $[+0.068, +0.165]$ | 20 of 20 |
| linear readout, full width | $0.913$ | $0.849$ | $+0.064$ | $[+0.036, +0.104]$ | 20 of 20 |
| linear readout, projected to 256 | $0.893$ | $0.837$ | $+0.057$ | $[+0.034, +0.086]$ | 20 of 20 |

Through the decoder, random weights also score below our JEPA encoder, by $0.155$ at the primary endpoint and by $0.066$ after projection, and the linear readouts tie with ours, $0.849$ and $0.837$ against $0.852$. So a randomly initialized V2 is not only worse than the pretrained one; it is no better than an encoder trained on this experiment's cells alone. The pretraining is what lifts V2 above our encoder. How much of the $0.20$ is information that pretraining puts into the embedding, and how much is the decoder's difficulty with an embedding that barely varies, this round does not separate. The linear readout bounds the first from below at about $0.06$.

## Secondary results, intervals only, no verdicts

- **Every readout of every arm.** Seed means over the 20 pairs. $e_p$ is round 8's size-aware error $\lVert \hat{\Delta}_p - \Delta_p \rVert_2 / \lVert \Delta_p \rVert_2$, where $\hat{\Delta}_p$ is the predicted and $\Delta_p$ the measured change of pair $p$ over its top 20 genes and $\lVert \cdot \rVert_2$ is the Euclidean length: predicting no change gives 1, and lower is better. "Own control" scores each prediction against the arm's own predicted control instead of the measured one ([round 10](round-10-control-reference.md)).

  | arm | roundtrip $\Delta r$ | $e_p$ | $\Delta r$, own control | $e_p$, own control | linear $\Delta r$ | linear $e_p$ |
  |---|---|---|---|---|---|---|
  | `jepa` | 0.863 | 0.490 | 0.886 | 0.524 | 0.852 | 0.561 |
  | `v2_cell` | 0.908 | 0.405 | 0.928 | 0.377 | 0.913 | 0.424 |
  | `v2rand_cell` | 0.708 | 0.695 | 0.634 | 0.852 | 0.849 | 0.676 |
  | `v2panel_cell` | 0.921 | 0.376 | 0.938 | 0.361 | 0.927 | 0.395 |
  | `v2cut_cell` | 0.907 | 0.405 | 0.927 | 0.395 | 0.912 | 0.443 |
  | `v2big_cell` | 0.919 | 0.374 | 0.934 | 0.338 | 0.927 | 0.371 |
  | `v2_cell_p` | 0.909 | 0.403 | 0.925 | 0.391 | 0.893 | 0.468 |
  | `v2rand_cell_p` | 0.797 | 0.603 | 0.861 | 0.701 | 0.837 | 0.699 |
  | `v2panel_cell_p` | 0.914 | 0.397 | 0.930 | 0.380 | 0.903 | 0.446 |
  | `v2cut_cell_p` | 0.904 | 0.419 | 0.924 | 0.411 | 0.888 | 0.483 |
  | `v2big_cell_p` | 0.919 | 0.383 | 0.928 | 0.360 | 0.910 | 0.417 |

- **The four contrasts on the other endpoints.** Percentile intervals; for $e_p$ a negative difference means the property helps.

  | contrast | roundtrip $e_p$ | $\Delta r$, own control | $e_p$, own control | linear $\Delta r$ |
  |---|---|---|---|---|
  | pretraining | $-0.290$ $[-0.349, -0.237]$ | $+0.293$ $[+0.154, +0.472]$ | $-0.475$ $[-0.550, -0.399]$ | $+0.064$ $[+0.036, +0.104]$ |
  | input breadth | $+0.028$ $[+0.018, +0.040]$ | $-0.010$ $[-0.024, -0.001]$ | $+0.016$ $[+0.008, +0.025]$ | $-0.014$ $[-0.023, -0.007]$ |
  | list length | $-0.001$ $[-0.011, +0.007]$ | $+0.000$ $[-0.007, +0.005]$ | $-0.019$ $[-0.029, -0.010]$ | $+0.001$ $[-0.007, +0.009]$ |
  | model size | $-0.031$ $[-0.051, -0.017]$ | $+0.006$ $[+0.001, +0.013]$ | $-0.039$ $[-0.062, -0.023]$ | $+0.014$ $[+0.006, +0.026]$ |

  Without the family's correction the small effects take a direction. The panel arm is slightly better than full-transcriptome V2 on every endpoint, $0.010$ to $0.014$ in $\Delta r$ and $0.016$ to $0.029$ lower in $e_p$, so reading the genes outside the panel, if anything, costs a little. V2-316M is slightly better than V2-104M, $0.006$ to $0.014$ in $\Delta r$ and $0.031$ to $0.039$ lower in $e_p$. List length leaves $\Delta r$ unchanged and lowers $e_p$ only under the own-control convention. Pretraining's effect on $e_p$ is large: random weights raise the size-aware error from $0.405$ to $0.695$, and to $0.852$ under the own-control convention, close to the $1$ of predicting no change at all.

- **Scored only on the genes Geneformer can see.** The scored genes are all panel genes, so every V2 arm sees the same ones: 368 of the 400 scored entries, the other 32 being seven non-coding RNAs ([Chapter 3 of the series](../foundation-encoder/03-the-scored-genes-and-the-blind-spot.md)). On those genes the four contrasts are $+0.201$ $[+0.125, +0.309]$ for pretraining, $-0.013$ $[-0.026, -0.003]$ for input breadth, $+0.002$ $[-0.004, +0.007]$ for list length and $+0.009$ $[+0.002, +0.020]$ for model size, the same as on all of them.

- **Every arm against `jepa`.** At the readout ceiling every pretrained V2 arm reads held-out cells better than our encoder, by $+0.045$ $[+0.025, +0.073]$ for the cut lists, $+0.057$ $[+0.032, +0.091]$ for V2-316M and $+0.058$ $[+0.033, +0.094]$ for the panel, with $e_p$ lower by $0.084$ to $0.116$. Random weights read them worse, $-0.155$ $[-0.257, -0.084]$. The panel comparison is the one the protocol singled out: `v2panel_cell` and `jepa` read the same genes, so their difference compares pretraining on many cells, together with Geneformer's ranked-list input and architecture, with our encoder at equal input breadth. That difference is the largest of all.

- **What projecting to 256 dimensions costs.** For the pretrained arms, almost nothing: projected minus full width is $+0.002$ $[-0.003, +0.006]$ for `v2_cell`, $-0.007$ $[-0.016, -0.000]$ for the panel, $-0.003$ for the cut lists and $-0.003$ for V2-316M. For random weights the projection helps, $+0.089$ $[+0.050, +0.149]$, and lowers $e_p$ by $0.092$, consistent with the decoder reading the projected random embedding more easily than the raw one.

- **The full pipelines, on the projections.** Each projected arm's transport flow generates latent cells for each held-out pair, and the same seed's projected decoder turns them into counts. Every new flow trained stably: no epoch's loss rose above the first epoch's, and the final losses were $1.46$ to $1.49$.

  | pipeline | $\Delta r$ | $e_p$ | $\Delta r$, own control | $e_p$, own control | minus `v2_cell_p`, $\Delta r$ | minus `v2_cell_p`, $e_p$ |
  |---|---|---|---|---|---|---|
  | `jepa` | 0.830 | 0.578 | 0.850 | 0.600 | $-0.036$ $[-0.059, -0.015]$ | $+0.039$ $[+0.005, +0.074]$ |
  | `v2_cell_p` | 0.866 | 0.539 | 0.882 | 0.560 | | |
  | `v2rand_cell_p` | 0.734 | 0.737 | 0.795 | 0.854 | $-0.132$ $[-0.206, -0.077]$ | $+0.198$ $[+0.141, +0.258]$ |
  | `v2panel_cell_p` | 0.869 | 0.536 | 0.882 | 0.546 | $+0.003$ $[-0.004, +0.009]$ | $-0.003$ $[-0.016, +0.010]$ |
  | `v2cut_cell_p` | 0.852 | 0.579 | 0.873 | 0.608 | $-0.014$ $[-0.022, -0.006]$ | $+0.040$ $[+0.023, +0.058]$ |
  | `v2big_cell_p` | 0.870 | 0.508 | 0.884 | 0.510 | $+0.004$ $[-0.002, +0.010]$ | $-0.031$ $[-0.056, -0.009]$ |
  | conditioned cVAE, round 11 | 0.939 | 0.455 | | | | |
  | additive reference | 0.961 | 0.289 | | | | |

  The pipelines repeat the ceiling's ordering with one exception. The panel and V2-316M pipelines match V2's in $\Delta r$, and V2-316M's has the lowest size-aware error of any pipeline on a 256-dimensional store, $0.508$. Random weights give a pipeline $0.132$ below V2's and $0.097$ $[0.046, 0.167]$ below ours. The exception is list length: cutting the lists costs nothing at the ceiling but $0.014$ in the pipeline, with $e_p$ $0.040$ higher, both intervals clear of zero. Against our encoder the panel, V2 and V2-316M pipelines gain $+0.036$ to $+0.040$, and the cut-list pipeline $+0.022$ $[-0.000, +0.046]$. Every pipeline remains below the conditioned cVAE and additivity.

- **Transition headroom**, readout ceiling minus pipeline: $0.033$ $[0.014, 0.054]$ for `jepa`, $0.043$ to $0.046$ for the projected V2, panel and V2-316M arms, $0.052$ $[0.029, 0.082]$ for the cut lists and $0.063$ $[0.037, 0.094]$ for random weights.

- **Population quality and calibration of the pipelines.** Coverage is the fraction of true (cell, gene) entries inside each gene's central 80% predicted interval, pooled over each pair's top 20 genes, with the nominal $0.80$ as the target; spread is the correlation of per-gene standard deviations across those genes; $W_1$ is the per-gene Wasserstein distance, averaged, so it compares marginals only; energy distance is joint over the 20 genes. Lower is better for $W_1$ and energy.

  | pipeline | coverage (gap to 0.80) | spread | $W_1$ | energy |
  |---|---|---|---|---|
  | `jepa` | $0.774$ ($-0.026$) | $0.673$ | $0.402$ | $0.895$ |
  | `v2_cell_p` | $0.761$ ($-0.039$) | $0.680$ | $0.370$ | $0.750$ |
  | `v2rand_cell_p` | $0.701$ ($-0.099$) | $0.530$ | $0.584$ | $1.593$ |
  | `v2panel_cell_p` | $0.767$ ($-0.033$) | $0.671$ | $0.361$ | $0.712$ |
  | `v2cut_cell_p` | $0.751$ ($-0.049$) | $0.654$ | $0.410$ | $0.896$ |
  | `v2big_cell_p` | $0.782$ ($-0.018$) | $0.656$ | $0.333$ | $0.609$ |

  V2-316M generates the closest populations of any flow on a 256-dimensional store: against V2-104M its coverage is $0.021$ $[0.010, 0.034]$ higher, its $W_1$ $0.037$ $[0.012, 0.064]$ lower and its energy distance $0.141$ $[0.044, 0.257]$ lower, and against our encoder its energy distance is $0.286$ $[0.117, 0.524]$ lower. The panel's populations match V2's. Cutting the lists makes them worse, energy $0.145$ $[0.076, 0.228]$ higher, back to our encoder's level, and random weights make them far worse on every measure.

- **The effective rank of each store**, on 2,000 training cells drawn with seed 0: `jepa` $211$ of 256 possible, `v2_cell` $407$, `v2rand_cell` $496$, `v2panel_cell` $450$ and `v2cut_cell` $415$ of 768, `v2big_cell` $574$ of 1,152; the projected stores $152$ to $222$ of 256. The random-weight store spreads over the most directions of the 768-wide stores and reads held-out cells worst, so this measure, again, is not a ranking of quality.

## What it means

Of the four properties that separate V2-104M from our encoder, three are shown not to matter much at the readout ceiling. Reading the whole transcriptome rather than our panel does not help; if anything the panel reads held-out cells slightly better, with about a third as many genes per cell. Reading lists longer than 2,048 genes does not help either, so V1's tie with our encoder in round 14 is not explained by its shorter input; with list length ruled out, V1's smaller model and its smaller, older pretraining corpus remain. Tripling the model's size gains at most a few hundredths. The intervals bound each of the three at about $0.05$ or less, against the $0.045$ by which V2 leads our encoder.

That leaves pretraining, and under the protocol it is not established. The family's critical value, inflated by one pair's weight in the breadth contrast, makes the pretraining interval reach below zero. Read outside the protocol, the evidence for pretraining is strong and consistent: random weights read all 20 pairs worse under each of four readouts, by $0.06$ to $0.20$ on average, and through the decoder read them worse than our own encoder as well. The size of the effect is less certain than its direction, because part of the $0.20$ at the primary endpoint reflects how poorly the decoder reads an embedding that barely varies between cells.

For the next step this has two consequences. Training Geneformer jointly with the decoder should start from the pretrained weights, since a random start begins far behind, and the random-weight twin remains the control that tells whether joint training keeps that head start, as round 13 found it did not for our own encoder. Cutting the lists at 2,048 genes, which makes each training step about three times cheaper, costs nothing at the ceiling, but it does cost a little in the generated cells, which this round measured only through the flow on the projections. The panel-only input is a cheaper option still: it needs about a third of the tokens per cell and loses nothing anywhere in this round.

## What it does not mean

- It does not show that pretraining is unimportant. The primary contrast did not pass, but its estimate is the largest of the round and its plain interval is wholly above zero; the verdict reflects how the family's interval is computed as much as the data.
- It does not show that the post-hoc pretraining analysis is a result. It was run after the verdict, chose its readouts knowing the outcome, and is reported to explain the verdict, not to replace it.
- It does not separate the parts of pretraining. The random-weight arm removes everything the pretraining learned at once: which genes go together, how ranks map to cell states, the learned position table. Which of those matters is not tested.
- It does not say that reading more genes cannot help a different encoder. The breadth contrast is about V2 reading its own ranked lists; our JEPA encoder reads expression values, and widening its input is a separate question.
- The readout ceiling is a diagnostic handed real held-out cells, not a model. The pipelines are the model, and they were measured only on the 256-dimensional projections, with one flow recipe.
- No learned model passes additivity. The best readout ceiling in the round, $0.921$, is below the additive reference's $0.961$.

## Limits

- The `v2_cell` and `jepa` arms are round 14's runs, re-read; the gate shows they reproduce round 14 exactly. Every new arm's decoder was trained once per seed, and every comparison is with those re-read runs.
- The random weights are one draw, seed 0, of the architecture's default initialization. A different initialization scale would give a different embedding scale and possibly a different readout.
- Each arm changes one property, but the properties are not fully independent: the panel arm's lists are also shorter, a median of 1,100 genes against about 3,100, so it changes list length as well as breadth, in a range the cut-list arm does not test.
- The intervals describe the twenty pairs, with seeds averaged per pair first. Pairs share genes, so the twenty are not independent.

## Scripts and reports

- [`foundation-encoder/10_embed_variants.py`](../../foundation-encoder/10_embed_variants.py) writes the four variant stores.
- [`foundation-encoder/run_gf15_embed_pod.sh`](../../foundation-encoder/run_gf15_embed_pod.sh) installs Geneformer and both V2 models at the pinned revision, writes the variant stores, projects them, checks that every store covers every cell, and then runs the comparison.
- [`foundation-encoder/run_gf_compare_pod.sh`](../../foundation-encoder/run_gf_compare_pod.sh) with `ROUND=15` runs the comparison: a smoke pass at tiny settings, then every new arm's decoders, ceilings and projected flows. `SKIP_DONE=1` skips the steps whose final reports exist.
- [`foundation-encoder/11_score_round15.py`](../../foundation-encoder/11_score_round15.py) holds the protocol and writes `output/gf15_compare.json`.
- The [foundation-encoder README](../../foundation-encoder/README.md) describes the whole sub-program and every script.

---

*Up: [the results ledger](index.md). Previous: [round 14](round-14-foundation-encoder.md).*
