# foundation-encoder: a pretrained single-cell model as the encoder

Every encoder in the perturbation-response benchmark so far was trained on the Norman cells themselves. The JEPA encoder was pretrained on the roughly 96,000 training cells, and [round 13](../docs/results/round-13-random-start.md) found that, once the encoder is trained jointly with the count decoder, starting from that pretraining does no better than starting from random weights. That result leaves one premise of pretraining untested. The JEPA encoder saw only the cells of this one experiment, so it could not have learned anything the task's own training data did not already contain. A foundation model is the opposite case: it was pretrained on tens of millions of cells from many tissues and experiments, far more unlabeled data than the benchmark has.

This sub-program puts such a model in the encoder's place and asks whether its cell embeddings give the rest of the pipeline a better starting point. The task, the data, the 20 held-out combinations and the references stay exactly as in the [results ledger](../docs/results/index.md). The first model is [Geneformer](https://huggingface.co/ctheodoris/Geneformer). The folder name leaves room for other pretrained encoders later.

Published benchmarks give a sober prior. Kedzierska et al. (Genome Biology, 2025) and Ahlmann-Eltze et al. (Nature Methods, 2025) found that foundation-model embeddings did not beat simple baselines on zero-shot and perturbation tasks. A tie is therefore the expected outcome, and either answer is informative.

## How Geneformer reads a cell

Geneformer does not see expression values. It turns each cell into an ordered list of genes. The cell's raw counts are scaled to a common total, each gene's value is divided by that gene's typical expression across the pretraining corpus, and the expressed genes are sorted from the highest ratio down. Each gene in that list is one token, and the transformer reads the list the way a language model reads a sentence. A cell embedding is either the hidden state of a special summary token placed at the front (the `cls` mode, available in the newer V2 models) or the average hidden state over the cell's gene tokens (the `cell` mode).

Three practical consequences follow for this benchmark:

- Geneformer reads the whole transcriptome, about 19,000 measured genes per cell, not the 5,000-gene panel our models use. Its input therefore comes from the raw count file, and its embeddings are put back into the order of the processed cells afterwards.
- Only protein-coding and microRNA genes have tokens. Long non-coding RNAs, such as MALAT1 and NEAT1, are invisible to the model. In Norman, 4,308 of the 5,000 panel genes and 368 of the 400 genes scored on the held-out pairs have tokens; every missing scored gene is a non-coding RNA. The decoder still has to predict those genes, now from an embedding that never saw them.
- The perturbed gene itself is usually visible. Norman's perturbations switch genes on (CRISPR activation), so the targeted gene tends to rise in the cell's ranking. Of the 105 targeted genes, 104 have tokens.

## The scripts, in order

| script | what it does | where it runs | status |
|---|---|---|---|
| `01_build_cell_index.py` | maps each processed cell to its row in the raw file, with its label and split | locally, reads names only | **done** |
| `02_embed_geneformer.py` | builds Geneformer's input, tokenizes, embeds with each model, re-orders by cell, reports cost and embedding shape | pod, needs CUDA | **done** for the 2,000-cell sample |
| `run_geneformer_feasibility_pod.sh` | installs Geneformer at a pinned commit, runs a small smoke test, then embeds a 2,000-cell sample with every model | pod | **done** |

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

The effective rank counts how many directions an embedding actually spreads over: it is the exponential of the entropy of the normalized singular values of the centered embedding matrix, computed by the same function as the JEPA diagnostics. Its maximum is the smaller of the width and the number of cells, here 2,000. Every Geneformer embedding spreads over hundreds of directions. That says the embeddings are not collapsed, and nothing about whether those directions carry the perturbation response rather than technical variation such as sequencing depth. Only the comparison can say that.

## What the comparison will look like

The comparison itself needs a pre-registered protocol, written after the feasibility numbers are in. Its intended shape follows rounds 11 to 13. The Geneformer embedding enters frozen, read by the corrected count decoder, and is scored on the readout ceiling and the full pipeline against three arms: the frozen JEPA encoder (readout ceiling 0.865), the JEPA start trained jointly with the decoder (0.936), and a random start trained jointly (0.948 to 0.961). If the frozen arm comes close, Geneformer is also trained jointly with the decoder, against a random-start twin of the same architecture, as in round 13. Every heavy step, from installing the model to embedding all 109,737 cells, runs on the pod.
