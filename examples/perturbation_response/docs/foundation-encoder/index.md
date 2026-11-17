# A foundation model as the encoder

*A short series on putting a pretrained single-cell foundation model, Geneformer, in the place of our own JEPA encoder: how it reads a cell, what its embeddings can and cannot carry, what the benchmark's scored genes look like through its eyes, and how a sampleable generative model can be built on top of it.*

> **Why this series exists.** Every encoder in the perturbation-response benchmark so far was trained on the Norman cells themselves. [Round 13](../results/round-13-random-start.md) found that, once the encoder is trained jointly with the count decoder, starting from our own JEPA pretraining does no better than starting from random weights. That leaves one premise of pretraining untested. Our JEPA saw only the roughly 96,000 training cells of this one experiment, so it could not have learned anything the task's own data did not already contain. A foundation model pretrained on tens of millions of cells from many tissues is the opposite case, and Geneformer is the first one we test. Before running that comparison, it is worth being precise about what such a model actually sees, because several of its properties shape what the comparison can show.

The code for this work lives in [`foundation-encoder/`](../../foundation-encoder/README.md), whose README also holds the feasibility measurements: what it costs to embed every Norman cell, how much of each cell each model can read, and how evenly the embeddings spread compared with our own encoder.

## The chapters

1. **[How Geneformer reads a cell](01-how-geneformer-reads-a-cell.md)**: the input it needs, the rank-value encoding that turns a cell into an ordered list of genes, the learned token and position tables, the masked-gene pretraining task, the two ways a cell embedding is formed, and why the slots deep in a long list are trained on far fewer examples than the first ones. Each step is set beside the corresponding step of our own JEPA encoder.
2. **[Predicting expression from a ranking](02-expression-from-a-ranking.md)**: Geneformer never sees an expression value, yet its embeddings are used to predict expression. What a ranking keeps, what it discards exactly, which of the discarded parts our decoder supplies from elsewhere, and the one loss that remains. The chapter ends with a prediction the comparison can test.
3. **[The scored genes, and what Geneformer cannot see](03-the-scored-genes-and-the-blind-spot.md)**: why every held-out pair is scored on its top 20 differentially expressed genes, how many of those genes Geneformer has no token for, pair by pair, and how the comparison separates a blind spot from a weaker representation.
4. **[A generative model on top of a foundation model](04-a-generative-model-on-a-foundation-model.md)**: why a pretrained encoder plus a flow and a count decoder is a sampleable generative model, what has to hold for it to work, why one cannot simply sample from Geneformer itself, and how the comparison reads every encoder through one interface.

## Where this sits

The method series, [conditional flow + JEPA](../conditional-flow-jepa/index.md), describes the pipeline whose first stage this series replaces. The [results ledger](../results/index.md) holds every round on the shared benchmark, and the comparison of Geneformer against our own encoder is [round 14](../results/round-14-foundation-encoder.md), run under a protocol fixed before any arm was trained. [Round 15](../results/round-15-what-carries-the-gain.md) changes one property of V2 at a time, its pretraining, the breadth of its input, the length of its gene lists and its size, to see which one carries the gain. Published benchmarks give a sober prior: Kedzierska et al. (Genome Biology, 2025) and Ahlmann-Eltze et al. (Nature Methods, 2025) found that foundation-model embeddings did not beat simple baselines on zero-shot and perturbation tasks. A tie is therefore a plausible outcome, and either answer is informative.

---

*Up: [perturbation response documentation](../index.md).*
