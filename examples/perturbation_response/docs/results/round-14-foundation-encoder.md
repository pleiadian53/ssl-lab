# Round 14 — does an encoder pretrained on millions of cells read held-out cells better? Geneformer embeddings, frozen

*Part of [the results ledger](index.md), which holds the standing verdict, the scoreboard and the settings every round shares. Previous: [round 13](round-13-random-start.md).*

> **Status.** The primary endpoint and every secondary that reads only decoders and readout ceilings are final and reported below. The full-pipeline secondaries are being rerun under amendment 1 of the protocol, described in the full-pipeline section, and will be added to this page when they have been scored.

**What changed.** Every encoder in this ledger so far was trained on the Norman cells themselves. This round replaces the encoder with [Geneformer](https://huggingface.co/ctheodoris/Geneformer), a transformer pretrained on about 30 million (V1) or 104 million (V2) single-cell transcriptomes from many tissues and experiments, and keeps it frozen. Each Geneformer model turns every Norman cell into one vector, and the round-11 corrected count decoder and the round-1 transport flow are trained on those vectors exactly as they are on our JEPA encoder's. The [foundation-encoder series](../foundation-encoder/index.md) explains how Geneformer reads a cell, what its vectors can carry, and which scored genes it has no token for.

**Why this question.** Round 13 found that, once the encoder is trained jointly with the count decoder, starting from our JEPA pretraining does no better than starting from random weights. That JEPA was pretrained on the roughly 96,000 training cells of this one experiment, so it could not have learned anything the task's own data did not already contain. A foundation model is the opposite case: its pretraining saw far more cells, from far more conditions, than the benchmark has. This round asks whether that breadth gives the decoder a better starting point than our own encoder does. Published benchmarks gave a sober prior: Kedzierska et al. (Genome Biology, 2025) and Ahlmann-Eltze et al. (Nature Methods, 2025) found that foundation-model embeddings did not beat simple baselines on zero-shot and perturbation tasks, so a tie was the expected outcome.

## Experimental settings

| arm | where its cell vectors come from | width | flow hidden width | runs, seeds 0, 1, 2 |
|---|---|---|---|---|
| `jepa` | our frozen JEPA encoder `norman_stage_a` (md5 `38f51067`), exported to a store | 256 | 256 | `norman_gf_*_jepa_s{s}` |
| `v1_cell` | Geneformer V1-10M, mean over the cell's gene tokens | 256 | 256 | `norman_gf_*_v1_cell_s{s}` |
| `v2_cls` | Geneformer V2-104M, the `<cls>` summary token | 768 | 768 | `norman_gf_*_v2_cls_s{s}` |
| `v2_cell` | Geneformer V2-104M, mean over gene tokens | 768 | 768 | `norman_gf_*_v2_cell_s{s}` |
| `v2cl_cls` | Geneformer V2-104M_CLcancer, `<cls>` token | 768 | 768 | `norman_gf_*_v2cl_cls_s{s}` |
| `v2cl_cell` | Geneformer V2-104M_CLcancer, mean over gene tokens | 768 | 768 | `norman_gf_*_v2cl_cell_s{s}` |
| `v2_cls_p`, `v2_cell_p`, `v2cl_cls_p`, `v2cl_cell_p` | the four V2 stores above, projected onto their 256 leading principal components | 256 | 256 | `norman_gf_*_{arm}_s{s}` |

- **The models.** V1-10M has 6 layers and reads at most 2,048 genes per cell. V2-104M has 12 layers and reads up to 4,094 genes. V2-104M_CLcancer is the same V2 model after continued pretraining on about 14 million cancer transcriptomes, a relevant variant because Norman's K562 cells come from a leukemia line. Every vector is taken from the model's second-to-last layer, Geneformer's default, at Hugging Face revision `1f7fbae4`. A Norman cell expresses a median of about 3,120 genes that Geneformer has tokens for, so V1 cuts 93% of cells short and never reads 35% of their expressed genes, always the ones lowest in each cell's ranking. V2 cuts about 10% of cells and loses about 1% of the genes.
- **Embedding stores.** Each arm's vectors are written once for all 109,737 processed cells into an embedding store ([`ssllab.embeddings`](../../../../src/ssllab/embeddings/)): the vectors, the processed-cache row of each one, and a manifest recording the model, mode, layer, the fingerprint of the cache it belongs to and a hash of its contents. The decoder trainer, the flow trainer and the readout ceiling each read a store with `--latents` in place of the encoder, so every arm, including our own encoder, reaches them through the same code. Every decoder and flow records which store it read, and the pipeline refuses to pair a flow and a decoder from different latent spaces.
- **Our encoder's store.** [`04_export_jepa_latents.py`](../../foundation-encoder/04_export_jepa_latents.py) encodes every cell with `norman_stage_a` on a CPU and then re-encodes a sample of 2,000 cells on the fly, in a different batch layout; the largest difference was exactly 0. On a GPU the same two encodings differed by up to 0.0016, because the GPU's matrix kernels change their summation order with the batch shape, so the export runs on the CPU.
- **The projections.** [`05_project_store.py`](../../foundation-encoder/05_project_store.py) fits principal components on the 96,322 combo-split training cells only and projects every cell. The 256 components keep 93.2% of the variance of V2 `<cls>`, 97.2% of V2 gene mean, 95.4% of CLcancer `<cls>` and 97.1% of CLcancer gene mean. The projected arms are the comparison at equal width and equal flow capacity.
- **Decoders.** [`co-adaptation/02_coadapt_decoder.py`](../../co-adaptation/02_coadapt_decoder.py) `--mode dec --library-size hvg --latents <store>`, round 11's corrected recipe: a fresh count decoder trained on the frozen vectors for 30 epochs at batch 256 on the combo split's training cells, AdamW at learning rate $10^{-3}$, gradients clipped at 5.0, the library size counted over the modeled genes.
- **Readouts.** [`14_ceiling_analysis.py`](../../14_ceiling_analysis.py) `--latents <store>` with the identity, roundtrip and linear arms on each arm's own decoder (`norman_gf_ceiling_{arm}_s{s}`). The roundtrip looks up each real held-out cell's vector and decodes it. The linear arm fits a ridge map from the vector to expression on training cells, with penalty $\lambda = 1$.
- **Full pipeline.** The round-1 transport flow recipe (`04_train_cond_flow.py --split combo --cond-type geneset --flow-base control --coupling independent --epochs 60 --latents <store>`), read out through the same seed's decoder (`norman_gf_flow_{arm}_s{s}`). A 768-wide arm's flow has a 768-wide hidden layer: with the usual 256, its velocity could only move cells inside one fixed 256-dimensional subspace, a limit the 256-wide arms do not have. Predictions and populations at every round's defaults: 200 cells, generation seed 0, 100 Euler steps, guidance 1, sampled counts normalized like the data at the median real total over the modeled genes, 4,291.
- **Compute.** One A40 GPU. Embedding every cell took 0.26 GPU-hours for V1 and 2.2 to 2.3 hours for each V2 model and mode, about 9.2 hours in all, and exporting our encoder's store took 13 minutes on the pod's CPU. The comparison job, with its on-pod smoke run, 30 decoders, 30 readout ceilings and 30 flows, took about an hour and a half.
- **Gates, all passed before any arm was read.**
  - Every store covers all 109,737 processed cells, was computed on this cache, has finite values and matches its manifest. No cell was dropped by any Geneformer model.
  - Reading our encoder's store reproduces the published round-11 readout ceilings through the new code path: $0.866949$, $0.863373$ and $0.865014$ for seeds 0, 1 and 2, against the published $0.86695$, $0.86338$ and $0.86502$.
  - Every decoder's report records the dec mode, the corrected library size, the combo split, no cell limit and 30 epochs, and the content hash of its arm's store.
  - Every identity arm scores 1.000, and every readout ceiling's roundtrip $\Delta r$, recomputed from its saved profiles, matches its stored report to within $10^{-6}$.

## The pre-committed endpoint

The protocol is the docstring of [`07_score_comparison.py`](../../foundation-encoder/07_score_comparison.py), committed before any arm was trained. The primary endpoint is the readout ceiling's $\Delta r$: the Pearson correlation between the predicted and the measured change, over each held-out pair's top 20 genes, with real held-out cells looked up in the arm's store and decoded by the arm's decoder, under the measured control. One max-$t$ family holds five contrasts, each full-width Geneformer arm minus `jepa`, with seeds averaged per pair and a joint paired bootstrap over the 20 pairs (10,000 resamples, seed 0).

Each contrast passes if its estimate is at least $+0.02$ and its simultaneous interval lies wholly above zero. It is worse if the interval lies wholly below zero, and otherwise it does not pass. If any contrast passes, the round's reading is that a Geneformer embedding reads held-out cells better than the JEPA encoder. The protocol also fixed a trigger for a follow-up round: if the best Geneformer arm's ceiling is at least the `jepa` arm's minus 0.02, that model is trained jointly with the decoder in a separate round, against a random-start twin of the same architecture, as round 13 did.

## Verdict: every V2 embedding reads held-out cells better than our JEPA encoder; V1 ties

| readout ceiling, $\Delta r$ | estimate | minus `jepa` | simultaneous 95% interval | verdict |
|---|---|---|---|---|
| `jepa`, our frozen encoder | $0.863$ | | | |
| `v1_cell` | $0.862$ | $-0.001$ | $[-0.025, +0.024]$ | does not pass |
| `v2_cls` | $0.895$ | $+0.032$ | $[+0.002, +0.063]$ | passes |
| `v2_cell` | $0.908$ | $+0.045$ | $[+0.009, +0.081]$ | passes |
| `v2cl_cls` | $0.890$ | $+0.027$ | $[+0.006, +0.048]$ | passes |
| `v2cl_cell` | $0.901$ | $+0.038$ | $[+0.010, +0.066]$ | passes |

The max-$t$ critical value is $3.387$. Four of the five contrasts pass, so under the rule a Geneformer embedding reads held-out cells better than the JEPA encoder: V2-104M and V2-104M_CLcancer, in both modes. V1-10M ties with our encoder, with an interval that rules out a difference larger than about $0.025$ either way. The `jepa` arm's own ceiling, $0.863$, is within $0.002$ of the $0.865$ published in round 11, so the within-round control behaves as before. The joint-training trigger fires: the best Geneformer arm, `v2_cell`, is $0.045$ above the `jepa` arm.

The seeds agree. The three `jepa` seeds score $0.857$ to $0.867$, and the three `v2_cell` seeds $0.907$ to $0.908$. The narrowest pass, `v2_cls`, has seeds at $0.893$ to $0.899$.

**Where the gain comes from.** It is broad, not carried by one pair. `v2_cell` scores above `jepa` on all 20 pairs and `v2cl_cell` on all 20; the two `<cls>` arms on 18 of 20, where the two pairs they lose are lost by at most $0.005$. The median gain per pair is $+0.019$ to $+0.029$ across the four V2 arms. The largest gains are mostly on the pairs our encoder reads worst. OSR2+UBASH3B, which `jepa` reads at only $0.367$, gains between $+0.091$ and $+0.205$ across the four full-width V2 arms. The next two weakest, BCL2L11+TGFBR2 at $0.628$ and FOSB+UBASH3B at $0.679$, gain $+0.040$ to $+0.084$ and $+0.044$ to $+0.098$. One pair that `jepa` already reads well, TBX2+TBX3 at $0.870$, also gains $+0.066$ to $+0.101$. One pair carrying a large share of the difference is also why the simultaneous intervals are wider than the plain percentile intervals: the max-$t$ procedure studentizes each bootstrap resample, and resamples without that pair have small standard errors. The plain percentile intervals, which carry no verdict, are $[+0.018, +0.052]$ for `v2_cls`, $[+0.027, +0.068]$ for `v2_cell`, $[+0.016, +0.040]$ for `v2cl_cls` and $[+0.024, +0.055]$ for `v2cl_cell`.

## Secondary results, intervals only, no verdicts

- **Every readout of every arm.** Seed means over the 20 pairs. $e_p$ is round 8's size-aware error $\lVert \hat{\Delta}_p - \Delta_p \rVert_2 / \lVert \Delta_p \rVert_2$, where $\hat{\Delta}_p$ is the predicted and $\Delta_p$ the measured change of pair $p$ over its top 20 genes: predicting no change gives 1, and lower is better. "Own control" scores each prediction against the arm's own predicted control instead of the measured one ([round 10](round-10-control-reference.md)).

  | arm | roundtrip $\Delta r$ | $e_p$ | $\Delta r$, own control | $e_p$, own control | linear $\Delta r$ | linear $e_p$ |
  |---|---|---|---|---|---|---|
  | `jepa` | 0.863 | 0.490 | 0.886 | 0.524 | 0.852 | 0.561 |
  | `v1_cell` | 0.862 | 0.487 | 0.887 | 0.552 | 0.856 | 0.631 |
  | `v2_cls` | 0.895 | 0.416 | 0.918 | 0.395 | 0.904 | 0.440 |
  | `v2_cell` | 0.908 | 0.405 | 0.928 | 0.377 | 0.913 | 0.424 |
  | `v2cl_cls` | 0.890 | 0.418 | 0.917 | 0.396 | 0.912 | 0.437 |
  | `v2cl_cell` | 0.901 | 0.416 | 0.922 | 0.393 | 0.913 | 0.434 |
  | `v2_cls_p` | 0.905 | 0.408 | 0.922 | 0.387 | 0.889 | 0.468 |
  | `v2_cell_p` | 0.909 | 0.403 | 0.926 | 0.391 | 0.893 | 0.468 |
  | `v2cl_cls_p` | 0.907 | 0.403 | 0.924 | 0.385 | 0.899 | 0.467 |
  | `v2cl_cell_p` | 0.902 | 0.415 | 0.919 | 0.401 | 0.895 | 0.477 |

- **Every Geneformer arm against `jepa`, on the other endpoints.** Arm minus `jepa`; for $e_p$ a negative difference favors the Geneformer arm.

  | arm | roundtrip $e_p$ | $\Delta r$, own control | $e_p$, own control |
  |---|---|---|---|
  | `v1_cell` | $-0.003$ $[-0.023, +0.017]$ | $+0.001$ $[-0.014, +0.016]$ | $+0.028$ $[-0.004, +0.062]$ |
  | `v2_cls` | $-0.074$ $[-0.090, -0.060]$ | $+0.032$ $[+0.018, +0.047]$ | $-0.129$ $[-0.166, -0.099]$ |
  | `v2_cell` | $-0.085$ $[-0.109, -0.066]$ | $+0.041$ $[+0.026, +0.059]$ | $-0.147$ $[-0.185, -0.115]$ |
  | `v2cl_cls` | $-0.072$ $[-0.100, -0.050]$ | $+0.031$ $[+0.018, +0.045]$ | $-0.128$ $[-0.171, -0.093]$ |
  | `v2cl_cell` | $-0.074$ $[-0.090, -0.059]$ | $+0.036$ $[+0.023, +0.051]$ | $-0.131$ $[-0.171, -0.099]$ |
  | `v2_cls_p` | $-0.082$ $[-0.103, -0.064]$ | $+0.036$ $[+0.022, +0.051]$ | $-0.137$ $[-0.176, -0.103]$ |
  | `v2_cell_p` | $-0.087$ $[-0.109, -0.069]$ | $+0.039$ $[+0.025, +0.056]$ | $-0.133$ $[-0.167, -0.102]$ |
  | `v2cl_cls_p` | $-0.087$ $[-0.117, -0.064]$ | $+0.038$ $[+0.024, +0.054]$ | $-0.139$ $[-0.189, -0.100]$ |
  | `v2cl_cell_p` | $-0.075$ $[-0.092, -0.060]$ | $+0.033$ $[+0.021, +0.047]$ | $-0.123$ $[-0.159, -0.092]$ |

  Every V2 arm reproduces the size of the change better as well as its pattern: the size-aware error falls from $0.490$ to between $0.403$ and $0.418$, and under the own-control convention from $0.524$ to between $0.377$ and $0.401$. V1 matches our encoder on every endpoint.

- **Scored only on the genes Geneformer can see.** Of the 400 scored entries, 20 pairs times 20 genes, 368 have Geneformer tokens in both vocabularies; the 32 that do not are seven non-coding RNAs, MALAT1 and NEAT1 among them ([Chapter 3 of the series](../foundation-encoder/03-the-scored-genes-and-the-blind-spot.md)). Rescoring both the Geneformer arm and `jepa` on only each pair's visible genes, 17 to 20 per pair, leaves the picture unchanged. On those genes `jepa` scores $0.856$ and the V2 arms $0.886$ to $0.905$, differences of $+0.030$ to $+0.050$ with every interval above zero; `v1_cell` and `jepa` both score $0.855$, a difference of $-0.000$ $[-0.015, +0.013]$. The gap is about the same on the visible genes alone as on all of them, so the V2 advantage is not a matter of which genes are scored, and the genes Geneformer cannot see neither create the advantage nor erase it.

- **What projecting to 256 dimensions costs.** Nothing measurable, and slightly less than nothing for the `<cls>` arms. Projected minus full width, roundtrip $\Delta r$: `v2_cls` $+0.010$ $[+0.005, +0.016]$, `v2_cell` $+0.002$ $[-0.003, +0.006]$, `v2cl_cls` $+0.017$ $[+0.011, +0.026]$, `v2cl_cell` $+0.002$ $[-0.002, +0.005]$. On $e_p$ the differences are $-0.008$ to $-0.001$, with only `v2cl_cls` clear of zero at $-0.015$ $[-0.022, -0.009]$. The projected arms keep 93% to 97% of the variance, and the components they drop evidently carry little the decoder can use. The linear readout moves the other way, $0.013$ to $0.020$ lower after projection, so the dropped components are not empty, but the trained decoder does not need them.

- **The linear readout.** Descriptive only, because the ridge penalty $\lambda = 1$ is not scale-invariant across embeddings of different width and scale. It ranks the arms the same way as the roundtrip: the V2 arms lie $0.051$ to $0.061$ above `jepa` at full width and $0.037$ to $0.046$ above it projected, with $e_p$ $0.12$ to $0.14$ lower at full width; V1 ties on $\Delta r$ and is worse on $e_p$, $+0.070$ $[+0.029, +0.111]$.

- **The ranking hypothesis.** [Chapter 2 of the series](../foundation-encoder/02-expression-from-a-ranking.md) predicted that, because Geneformer reads a ranking rather than expression values, its embeddings would do relatively better on $\Delta r$, the pattern of a change, than on $e_p$, its size. The data do not show that weakness. The V2 arms reduce the size-aware error by $0.072$ to $0.087$, about a sixth of `jepa`'s $0.490$, and by about a quarter under the own-control convention; they raise $\Delta r$ by $0.027$ to $0.045$, a fifth to a third of the distance from `jepa`'s $0.863$ to 1. Both endpoints improve clearly, and neither the size nor the pattern of the change is where the V2 embeddings fall short of our encoder. The chapter named one way the prediction could fail: the decoder may read the size of a gene's change from the many small rank shifts it causes among correlated genes. That account fits the result, but this round does not test it, and the gene-level breakdown the chapter suggested, by direction and by the share of a change that is size rather than direction, has not been run.

- **How V2 uses the deep end of a cell's gene list.** Two descriptive checks from [`08_mechanism_diagnostics.py`](../../foundation-encoder/08_mechanism_diagnostics.py), on the 2,000 cells of the feasibility sample, 1,873 of which express more than 2,048 genes with tokens.
  - *Cutting the list at 2,048 genes.* Each cut cell's V2-104M vector moves by a median of $1.30$ times the median distance between two different cells for the `<cls>` token, and $1.35$ times for the gene mean, from $0.75$ to $1.8$ times between the 5th and 95th percentiles. Uncut cells do not move at all. The cosine similarity between the full and the cut vector stays high, $0.996$ and $0.947$, because all of a model's vectors point in nearly the same direction; the distance relative to the spread between cells is the informative scale. Reading only the first 2,048 genes, as V1 always does, therefore moves a cell's vector about as far as the gap between two typical cells.
  - *The learned position table.* V2-104M's position embeddings have a nearly constant norm along the slots, $0.40$ to $0.43$ in every band from slot 0 to 4,096, and each slot's embedding has a cosine similarity of $0.83$ to $0.86$ to its neighbor's throughout. The deep slots are learned as smoothly as the early ones.

  Together these make the deep end of the list a candidate for part of the difference between V1, which never reads it, and V2. They do not separate it from the other differences between the models: V2 has twice the layers, three times the width and more than three times the pretraining cells.

- **The full pipeline: pending, under amendment 1.** In the first full run, all twelve 768-wide flows, the four full-width V2 arms at three seeds, became unstable during training. Their epoch-mean flow-matching loss, about 1.5 after the first epoch, spiked to between $10^5$ and $4 \times 10^9$ before falling back, and three of them then generated non-finite cells. All eighteen 256-wide flows, including the four projected V2 arms, trained smoothly, ending at a loss of $0.83$ to $1.55$. A flow that diverged is a failed fit, not a measurement, so those twelve pipelines are not reported. Amendment 1 to the protocol, written into the scoring script's docstring before the rerun, retrains the twelve 768-wide flows with the gradient's total norm clipped at 1.0 before each optimizer step and changes nothing else. In a short test run on our encoder's store, a healthy flow's gradient norm was about 0.1 to 0.2, so the threshold should cap only spikes. Every flow now reports its per-epoch loss and gradient norms; a 768-wide flow whose loss ever exceeds twice its first epoch's will be reported as diverged, and its arm's pipeline as not trainable under this recipe. The amendment cannot touch the primary endpoint or any secondary above, which read only decoders and readout ceilings. The pipeline secondaries, the comparison with the conditioned cVAE and additivity, the headroom and the population quality and calibration of every arm will be scored together once the rerun is in.

## What it means

At the readout ceiling, with the encoder frozen and this decoder, Geneformer's V2 embeddings carry more of what the decoder needs to reproduce a held-out cell's change than our JEPA encoder does: about $0.03$ to $0.045$ more in $\Delta r$ and about $0.08$ less in size-aware error, on all or nearly all of the 20 pairs, most on the pairs our encoder reads worst. The gain survives projecting the embedding to our encoder's width, so it is not a matter of having more numbers per cell, and it survives scoring only the genes Geneformer can see. V1 ties with our encoder exactly. The CLcancer variant scores no higher than plain V2 in either mode, so the continued pretraining on cancer cells does not appear to help here, even though K562 is a cancer line; the protocol made no contrast between the two, so this is an observation, not a result.

This is the first round in which a frozen encoder other than our own JEPA has been placed in the pipeline, and the first in which a frozen representation reads held-out cells clearly better than `norman_stage_a`. The rounds that raised the readout ceiling further, 12 and 13, did it by training the encoder and the decoder together, to $0.936$ from the pretrained start and $0.948$ to $0.961$ from random starts. Those numbers come from other rounds and are not compared here, but they put the result in proportion: a frozen V2 embedding at $0.908$ gains about half of what joint training added to our frozen encoder's $0.865$, without touching the encoder at all. Whether joint training adds to a Geneformer start what it adds to ours is the question the fired trigger raises for a separate round.

## What it does not mean

- It does not show that Geneformer's pretraining is what helps. The Geneformer arms differ from ours in more than pretraining: they read each cell's whole transcriptome, about 19,000 measured genes of which about 3,100 per cell are expressed and have tokens, where our encoder reads the 5,000-gene panel; they encode ranks rather than values; and they are far larger. This round compares encoders as built, not the factors inside them. That V1, which also reads the whole transcriptome but only its top 2,048 genes, ties with our encoder narrows the candidates without settling them.
- It does not contradict the published benchmarks directly. Those compared foundation-model embeddings with simple baselines on other tasks. Here the comparison is with another learned encoder, through a decoder trained on the benchmark's own cells, and the endpoint is a readout of real held-out cells.
- The readout ceiling is a diagnostic handed real held-out cells, not a model. The full pipeline is the model, and its numbers are still to come.
- No learned model passes additivity. The readout ceiling of the best arm, $0.908$, is itself below the additive reference's $0.961$.
- It applies to these Geneformer models, at their default layer and modes, read by this decoder recipe at 30 epochs. Another layer, another model size or a different decoder could change it.

## Limits

- The `jepa` arm is our encoder read through the new store path within this round, not round 11's runs re-read; the interface gate shows the path reproduces round 11 exactly, and the within-round arm scores $0.863$ against round 11's $0.865$.
- Each arm's decoder was trained once per seed. The intervals describe the twenty pairs, with seeds averaged per pair first. Pairs share genes, so the twenty are not independent.
- The 768-wide flows needed a protocol amendment after their first run failed. The amendment was written before the rerun and changes only the flows' optimizer, but it is a deviation from the protocol as first committed, and the full-pipeline results will carry that note.
- The mechanism checks are descriptive and on a 2,000-cell sample of the 109,737.

## Scripts and reports

- [`foundation-encoder/run_gf_embed_pod.sh`](../../foundation-encoder/run_gf_embed_pod.sh) installs Geneformer at the pinned revision, writes every store, exports our encoder's store, projects the V2 stores and checks that every store covers every cell.
- [`foundation-encoder/run_gf_compare_pod.sh`](../../foundation-encoder/run_gf_compare_pod.sh) checks every input on the pod, runs the interface gate and a smoke pass at tiny settings, then trains every arm's decoders, ceilings and flows; `FLOWS_ONLY` reruns only the named arms' flows, as amendment 1 does.
- [`foundation-encoder/07_score_comparison.py`](../../foundation-encoder/07_score_comparison.py) holds the protocol, with amendment 1, and writes `output/gf_compare.json`.
- The [foundation-encoder README](../../foundation-encoder/README.md) describes the whole sub-program, its feasibility measurements and every script.

---

*Up: [the results ledger](index.md). Previous: [round 13](round-13-random-start.md).*
