# 02 · Predicting expression from a ranking

*Geneformer never sees an expression value. It sees which genes a cell expressed and in what order, and nothing else. Yet the comparison uses its embeddings to predict expression, gene by gene, in counts. This chapter explains why that is not a contradiction, what the ranking discards, which of the discarded parts our pipeline does not need from the encoder anyway, and the one real loss that remains.*

---

## 1. A ranking is a description of the levels

[Chapter 1](01-how-geneformer-reads-a-cell.md) described how a cell becomes a list: each expressed gene gets the ratio

$$r_g = \frac{10^4 \cdot c_g / N}{m_g},$$

where $c_g$ is the gene's count in the cell, $N$ the cell's total count over all genes, and $m_g$ the gene's median normalized value across the pretraining corpus. The genes are sorted by $r_g$, and only the order is kept.

That order carries two kinds of information. The first is **which genes appear at all**. A gene with a count of zero has no ratio and is simply absent from the list. In single-cell data, where most genes are undetected in most cells, whether a gene was detected is a large part of the signal. The second is **the order of the genes that do appear**. A typical Norman cell puts about 3,100 genes in order, and an ordering of that many items pins down a great deal. If you know a cell's ranking and each gene's typical level $m_g$, you can rebuild the shape of its profile fairly well. This is the same reason a rank correlation such as Spearman's is nearly as informative as Pearson's on most real data: most of what the values say is already in their order.

A sports league table is a useful picture. The standings tell you who is ahead of whom, and with twenty teams that says a lot about how the season went. They do not tell you the points, so you cannot see whether the leader is one point clear or twenty.

## 2. What the ranking discards exactly

The order of the $r_g$ is unchanged by any transformation that preserves order. Two such transformations matter here.

**The cell's total count.** If every count in the cell is multiplied by the same factor $a > 0$, the total $N$ is multiplied by $a$ as well, and every $r_g$ is unchanged. A cell sequenced twice as deeply produces exactly the same list. The ranking is blind to sequencing depth by construction.

**The overall scale and the spacing.** More generally, applying any strictly increasing function to all the $r_g$ of a cell leaves their order unchanged. So the ranking cannot tell whether a gene sits far above its neighbor in the list or only just above it. A gene whose ratio rises from 3 to 30 produces the same list as one that rises from 3 to 4, as long as it passes the same genes on the way.

Two smaller losses come on top. In the tail of the list, many genes have a count of one or two and tie, and among tied genes the order is decided by the corpus medians $m_g$ rather than by the cell. And for V1, everything past slot 2,048 is cut off.

## 3. Which of those losses the pipeline does not need from the encoder

The count decoder in our pipeline ([Chapter 8 of the method series](../conditional-flow-jepa/08-modeling-the-readout-count-decoder.md)) does not predict counts directly from the embedding. Writing $z$ for the cell's embedding, it predicts a rate profile

$$\rho = \mathrm{softmax}(\mathrm{net}(z)), \qquad \sum_{g=1}^{G} \rho_g = 1,$$

where $G$ is the number of modeled genes and $\rho_g$ is the share of the cell's transcripts that go to gene $g$. The mean of the negative-binomial count model is then $\mu = \ell \cdot \rho$, where $\ell$ is the cell's library size. Since [round 11](../results/round-11-library-size.md), $\ell$ is the total count over the 5,000 modeled genes. It is a given covariate, taken from the data for a measured cell and set by the sampling convention when cells are generated. It is never predicted from $z$.

So the first thing the ranking discards, the cell's depth, is exactly the thing the decoder takes from elsewhere. The embedding only has to describe the cell's *composition*, how its transcripts are shared among genes, and composition is what a ranking describes best. The overall scale is part of the same quantity and is handled the same way.

The decoder also learns the way back. It is trained on Norman's own training cells to map embeddings to rate profiles, so it learns, for these K562 cells, how a pattern of ranks corresponds to levels of expression. The embedding only has to keep cells with different expression apart; turning that into counts is a learned map supplied by the decoder. That is why the first measurement in the comparison is the **readout ceiling**: real held-out cells are encoded, decoded, and scored. It asks directly how much of each cell's measured profile survives the whole chain from ranking to embedding to decoded profile.

## 4. The loss that remains: the size of a change

What the pipeline cannot recover from elsewhere is the spacing. A change in a gene's expression is visible to Geneformer only insofar as it moves the gene past other genes in the list, and how far that is depends on where the gene sits.

- In the crowded middle of the list, many genes have similar ratios, so a modest change moves a gene across many slots and is plainly visible.
- At the top, few genes have very high ratios, so a gene that is already near the front can rise a great deal without passing anyone. A CRISPR-activated target that goes from strongly to very strongly expressed may barely move in the list.
- At the bottom, the deep slots are the least trained ([Chapter 1, §6](01-how-geneformer-reads-a-cell.md)), and a gene switched off completely does not sink, it disappears. The model sees that the gene is absent, not by how much it fell.

On top of that, the embedding compresses a list of about 3,100 tokens into 768 or 1,152 numbers, so some of what the ranking does keep may not survive into the vector.

## 5. A prediction the comparison can test

The benchmark scores each held-out pair $p$ on two endpoints, both computed over the pair's top 20 genes. Write $\Delta_p$ for the measured change, the mean of the pair's held-out cells minus the mean of the control cells, and $\hat{\Delta}_p$ for the predicted change. The first endpoint, $\Delta r$, is the Pearson correlation between $\hat{\Delta}_p$ and $\Delta_p$: it asks whether the *pattern* of the change is right, which genes go up and which go down, and it ignores overall size. The second, introduced in [round 8](../results/round-08-size-aware-endpoint.md), is the relative error

$$e_p = \frac{\lVert \hat{\Delta}_p - \Delta_p \rVert_2}{\lVert \Delta_p \rVert_2},$$

where $\lVert \cdot \rVert_2$ is the Euclidean length over the scored genes. Predicting no change at all gives $e_p = 1$, and lower is better. Unlike $\Delta r$, it counts the size of the change.

If the main thing a ranking loses is spacing, Geneformer's embeddings should do relatively better on $\Delta r$ than on $e_p$, compared with our JEPA, which reads expression values directly. The gap should show most on genes whose largest changes are in their size rather than their direction, and on genes that go down. This is a hypothesis for the comparison to report descriptively, not a result. It could fail for a simple reason: the decoder may learn to read size from the many small rank shifts a real change causes among a gene's correlated partners, even when the gene itself barely moves in the list.

---

*Up: [the foundation-encoder series](index.md). Previous: [How Geneformer reads a cell](01-how-geneformer-reads-a-cell.md). Next: [The scored genes, and what Geneformer cannot see](03-the-scored-genes-and-the-blind-spot.md).*
