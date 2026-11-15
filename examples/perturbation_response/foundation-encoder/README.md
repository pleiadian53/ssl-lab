# foundation-encoder: a pretrained single-cell model as the encoder

Every encoder in the perturbation-response benchmark so far was trained on the Norman cells themselves. The JEPA encoder was pretrained on the roughly 96,000 training cells, and [round 13](../docs/results/round-13-random-start.md) found that, once the encoder is trained jointly with the count decoder, starting from that pretraining does no better than starting from random weights. That result leaves one premise of pretraining untested. The JEPA encoder saw only the cells of this one experiment, so it could not have learned anything the task's own training data did not already contain. A foundation model is the opposite case: it was pretrained on tens of millions of cells from many tissues and experiments, far more unlabeled data than the benchmark has.

This sub-program puts such a model in the encoder's place and asks whether its cell embeddings give the rest of the pipeline a better starting point. The task, the data, the 20 held-out combinations and the references stay exactly as in the [results ledger](../docs/results/index.md). The first model is [Geneformer](https://huggingface.co/ctheodoris/Geneformer). The folder name leaves room for other pretrained encoders later. The narrative companion, [A foundation model as the encoder](../docs/foundation-encoder/index.md), explains how Geneformer reads a cell, what its embeddings can carry, and which scored genes it cannot see.

Published benchmarks give a sober prior. Kedzierska et al. (Genome Biology, 2025) and Ahlmann-Eltze et al. (Nature Methods, 2025) found that foundation-model embeddings did not beat simple baselines on zero-shot and perturbation tasks. A tie is therefore the expected outcome, and either answer is informative.

## How Geneformer reads a cell

Geneformer does not see expression values. It turns each cell into an ordered list of genes. The cell's raw counts are scaled to a common total, each gene's value is divided by that gene's typical expression across the pretraining corpus, and the expressed genes are sorted from the highest ratio down. Each gene in that list is one token, and the transformer reads the list the way a language model reads a sentence. A cell embedding is either the hidden state of a special summary token placed at the front (the `cls` mode, available in the newer V2 models) or the average hidden state over the cell's gene tokens (the `cell` mode).

Three practical consequences follow for this benchmark:

- Geneformer reads the whole transcriptome, about 19,000 measured genes per cell, not the 5,000-gene panel our models use. Its input therefore comes from the raw count file, and its embeddings are put back into the order of the processed cells afterwards.
- Only the genes in Geneformer's vocabulary have tokens: protein-coding and microRNA genes for the V2 models, and a somewhat larger set that includes some non-coding genes for V1. Long non-coding RNAs, such as MALAT1 and NEAT1, are invisible to both. In Norman, 4,316 of the 5,000 panel genes reach the V2 models (4,352 the V1 models), and for both series 368 of the 400 genes scored on the held-out pairs have tokens; every missing scored gene is a non-coding RNA. `06_visible_genes.py` writes the exact lists. The decoder still has to predict those genes, now from an embedding that never saw them.
- The perturbed gene itself is usually visible. Norman's perturbations switch genes on (CRISPR activation), so the targeted gene tends to rise in the cell's ranking. Of the 105 targeted genes, 104 have tokens.

## The scripts, in order

| script | what it does | where it runs | status |
|---|---|---|---|
| `01_build_cell_index.py` | maps each processed cell to its row in the raw file, with its label and split | locally, reads names only | **done** |
| `02_embed_geneformer.py` | builds Geneformer's input, tokenizes, embeds with each model, re-orders by cell, reports cost and embedding shape | pod, needs CUDA | **done** for the 2,000-cell sample |
| `run_geneformer_feasibility_pod.sh` | installs Geneformer at a pinned commit, runs a small smoke test, then embeds a 2,000-cell sample with every model | pod | **done** |
| `03_reference_ranks.py` | the same effective-rank measure on the same 2,000 cells for our own JEPA encoder, an untrained copy of it, and its input | locally, CPU, about 20 seconds | **done** |
| `04_export_jepa_latents.py` | writes our JEPA encoder's vectors for every cell as an embedding store, so both encoders reach the decoder and the flow by the same code path, and checks the stored vectors against on-the-fly encoding | locally or on the pod; about 5 minutes on a laptop CPU | ready |
| `05_project_store.py` | projects a 768-dimensional store onto its 256 leading principal components, fit on training cells only | where the stores are | ready |
| `06_visible_genes.py` | lists which panel genes, and which of each held-out pair's scored genes, the V1 and V2 vocabularies cover | locally, gene tables only | **done** |
| `07_score_comparison.py` | the pre-registered protocol (in its docstring) and the scoring: gates, the primary family, every secondary | on the pod after job 2, or locally after the fetch | ready |
| `08_mechanism_diagnostics.py` | how far V2 embeddings move when gene lists are cut at 2,048, and the learned position tables along the slots | pod (inside job 1) | ready |
| `run_gf_embed_pod.sh` | pod job 1: every embedding store, the projections, the coverage gate, the mechanism checks | pod | ready |
| `run_gf_compare_pod.sh` | pod job 2: the interface gate, an on-pod smoke run, then the comparison and its scoring | pod | ready |

**Embedding stores.** The decoder trainer (`co-adaptation/02_coadapt_decoder.py --mode dec`), the flow trainer (`04_train_cond_flow.py`) and the ceiling analysis (`14_ceiling_analysis.py`) each take `--latents <store>` in place of the encoder. A store is a directory with one vector per cell (`embeddings.npy`), the processed-cache row of each vector (`row_idx.npy`) and a manifest saying which model, mode and layer produced it and which version of the cache it belongs to; the package is [`ssllab.embeddings`](../../../src/ssllab/embeddings/). Every decoder and flow trained this way records which latent space it read, and the pipeline refuses to pair a flow and a decoder from different spaces, even when their widths agree. Reading our own encoder's exported store through this path reproduces the published round-11 readout ceiling: identical profiles to the on-the-fly encoder, and 0.86695 for seed 0.

The feasibility run compares four models: V1-10M (6 layers, 256-dimensional, pretrained on about 30 million cells), V2-104M (12 layers, 768-dimensional, about 104 million cells), V2-104M_CLcancer (the same model with continued pretraining on cancer cells, a relevant variant because Norman's K562 cells come from a leukemia line), and V2-316M (18 layers, 1,152-dimensional). It measures what the comparison will cost before anyone commits to it, not how well the embeddings predict.

## Feasibility: what embedding costs, and what the models see

A random sample of 2,000 cells (seed 0; 205 of them unperturbed controls) was embedded on one NVIDIA A40 GPU with every model and mode. No cell was dropped and no embedding contained a non-finite value. The projected time for all 109,737 cells is the measured forward-pass time per cell multiplied out, so it excludes the one-off model loading.

| model and mode | width | seconds per 1,000 cells | projected hours, all cells | peak GPU memory (GB) | effective rank on 2,000 cells |
|---|---|---|---|---|---|
| V1-10M, mean over genes | 256 | 7.6 | 0.23 | 3.7 | 151 |
| V2-104M, `<cls>` token | 768 | 85 | 2.6 | 8.6 | 452 |
| V2-104M, mean over genes | 768 | 84 | 2.6 | 8.8 | 404 |
| V2-104M_CLcancer, `<cls>` token | 768 | 83 | 2.5 | 8.6 | 423 |
| V2-104M_CLcancer, mean over genes | 768 | 84 | 2.6 | 8.8 | 406 |
| V2-316M, `<cls>` token | 1,152 | 189 | 5.8 | 12.6 | 657 |
| V2-316M, mean over genes | 1,152 | 191 | 5.8 | 12.9 | 572 |

Cost is not an obstacle. Every model and mode together comes to about 22 GPU-hours, and the two modes of one model share the same forward pass, so computing both at once would roughly halve that. Memory is far below the A40's 48 GB at a batch of 16 cells.

The input length matters more than the cost. A Norman cell expresses a median of about 3,120 genes that Geneformer has tokens for (5th to 95th percentile: about 1,940 to 4,360). The V1 model reads at most 2,048 of them, so it cuts 94% of cells short and never sees 36% of the expressed genes, always the ones lowest in each cell's ranking. The V2 models read up to 4,094 genes and cut 10% of cells, losing 1.1% of the genes.

The effective rank counts how many directions an embedding actually spreads over: it is the exponential of the entropy of the normalized singular values of the centered embedding matrix, computed by the same function as the JEPA diagnostics. Its maximum is the smaller of the width and the number of cells, here 2,000, so embeddings of different widths are best compared by the rank as a fraction of that maximum. To give the Geneformer numbers a reference, `03_reference_ranks.py` applies the same measure to the same 2,000 cells for three representations we already have:

| representation | width | effective rank | fraction of its maximum |
|---|---|---|---|
| the encoder's input, 5,000 log-normalized expression values | 5,000 | 1,729 | 0.86 |
| our pretrained JEPA encoder (the published pipeline's latent) | 256 | 210 | 0.82 |
| the same JEPA architecture, untrained (random weights) | 256 | 131 | 0.51 |
| Geneformer V1-10M, mean over genes | 256 | 151 | 0.59 |
| Geneformer V2 models, both modes | 768 to 1,152 | 404 to 657 | 0.50 to 0.59 |

Every Geneformer embedding spreads over hundreds of directions, so none is collapsed. Relative to their width, they spread less evenly than our pretrained JEPA latent and about as evenly as an untrained network of the JEPA's shape. At equal width, V1-10M's 151 sits between the untrained JEPA's 131 and the pretrained JEPA's 210. This is not a ranking of quality. The raw input has the most even spread of all, largely because sampling noise in sparse counts points in every direction, and an encoder that discards that noise in favor of a few strong biological programs would score lower on this measure while being more useful. The effective rank says nothing about whether the directions carry the perturbation response rather than technical variation such as sequencing depth. Only the comparison can say that.

## The comparison

The protocol is pre-registered in the docstring of `07_score_comparison.py`, committed before any arm was trained. In short:

- **Arms.** Our frozen JEPA encoder, exported to a store so that it reaches the decoder and the flow by exactly the same code as every other arm. Geneformer V1-10M (gene mean), V2-104M and V2-104M_CLcancer (summary token and gene mean). The four V2 embeddings again, projected onto their 256 leading principal components, fit on training cells only. Every arm gets the round-11 corrected count decoder and round 1's transport flow, three seeds each. A 768-wide flow gets a 768-wide hidden layer: with the usual 256, the flow could only move cells inside one fixed 256-dimensional subspace of the embedding.
- **Primary endpoint.** The readout ceiling's Δr, real held-out cells encoded and decoded, for each full-width Geneformer arm minus the JEPA arm: one family of five contrasts with simultaneous intervals, and the same pass rule as rounds 11 to 13 (at least +0.02, with the simultaneous interval wholly above 0).
- **Secondaries.** These have no verdicts: the size-aware error e_p; the score on only the scored genes Geneformer can see, for both encoders; what projecting to 256 costs; the full pipeline against the corrected cVAE and additivity; population quality and calibration; and whether Geneformer does relatively better on Δr than on e_p, as a rank encoding would predict.
- **What follows.** If the best frozen Geneformer arm comes within 0.02 of the JEPA arm, Geneformer is trained jointly with the decoder in a separate round, against a random-start twin of the same architecture, as in round 13.

Two pod jobs run it. `run_gf_embed_pod.sh` writes every store (about 11 GPU-hours on one A40) and checks that each covers every processed cell. `run_gf_compare_pod.sh` first checks that reading our encoder's exported store reproduces the published round-11 ceilings. It then runs the whole comparison at tiny settings, and only after that the full one, about 30 decoders, ceilings and flows, scored on the pod. `08_mechanism_diagnostics.py` runs inside the first job. It adds two descriptive checks of how V2 uses the deep end of a cell's gene list: how far an embedding moves when the list is cut at 2,048 genes, and how the learned position table looks along the slots.
