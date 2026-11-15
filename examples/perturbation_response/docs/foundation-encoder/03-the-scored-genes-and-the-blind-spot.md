# 03 · The scored genes, and what Geneformer cannot see

*Every number on the benchmark is computed on a short list of genes per held-out pair, its top 20 differentially expressed genes. This chapter recalls why the list exists and how it is chosen, then looks at it through Geneformer's vocabulary: of the 400 scored entries, 32 are genes Geneformer has no token for, and they are concentrated in a few recurring genes rather than scattered.*

---

## 1. Why the benchmark scores a short list

A perturbation moves a handful of genes and leaves thousands alone. If the predicted change were correlated with the measured change across all 5,000 modeled genes, the correlation would be dominated by the untouched ones, where both changes are noise around zero. The score would mostly reward reproducing the unperturbed baseline, which every reasonable model already does well. Restricting the score to the genes the perturbation actually moved makes it a measurement of the response. [Reading Perturb-seq, Part 4](../reading-perturb-seq/04-success-metrics.md) develops this argument, and scoring the top 20 differentially expressed genes is a common convention in the perturbation-prediction literature, which makes results easier to set beside others.

## 2. How the 20 genes are chosen

For each perturbation, every gene's expression in the perturbed cells is compared with the control cells by a Wilcoxon rank-sum test, and genes are ranked by the size of its standardized statistic, $\lvert z \rvert$. A gene is eligible only if it is detected in at least 10% of the perturbation's cells and significant after multiple-testing correction. The first 20 eligible genes form the list. The data-preparation step stores the top 50 per perturbation; the scoring uses the first 20.

The more intuitive criterion, ranking by fold change, was tried and rejected. A fold change is a ratio, and a ratio explodes when its denominator is near zero, so ranking by it selects genes that are barely expressed at all. On Norman those genes were zero in every held-out cell, which made both the effect score and the calibration scores meaningless while they still returned plausible numbers. [Chapter 3e of the method series](../conditional-flow-jepa/03e-the-genes-the-metric-scores.md) tells that story in full, together with the guards that keep it from recurring.

## 3. Why 20, and what the choice costs

The number 20 is a convention rather than a derivation. It is large enough for a correlation to mean something and small enough to stay on genes the perturbation clearly moved. It has three costs worth keeping in view:

- Each pair's $\Delta r$, the Pearson correlation between predicted and measured change, is computed over just 20 numbers, so a single pair's score is noisy. The benchmark averages over the 20 held-out pairs and judges differences with a paired bootstrap over pairs for that reason.
- The rest of the transcriptome is never scored. A model could get the top genes right and the broader response wrong without the benchmark noticing. The population metrics, such as interval coverage and energy distance, are computed on the same 20 genes.
- The list is computed from the pair's own cells, including the held-out ones. That is the field's convention, since a held-out combination has no training cells of its own, but it means the benchmark tells every model where it will be graded.

## 4. The scored genes through Geneformer's eyes

Geneformer has tokens only for the genes in its vocabulary ([Chapter 1, §2](01-how-geneformer-reads-a-cell.md)), and its tokenizer matches genes by Ensembl ID after renaming retired IDs. Checking each held-out pair's 20 scored genes against the vocabulary gives the same answer for the V1 and the V2 models:

| scored genes Geneformer can see | number of held-out pairs |
|---|---|
| 20 of 20 | 3 |
| 19 of 20 | 6 |
| 18 of 20 | 7 |
| 17 of 20 | 4 |

That is 368 of the 400 scored entries. Every missing one is a non-coding RNA, and the 32 missing entries are only seven distinct genes:

| unseen gene | held-out pairs whose top 20 include it |
|---|---|
| MALAT1 | 11 |
| NEAT1 | 7 |
| RP11-301G19.1 | 6 |
| RP11-717F1.1 | 4 |
| AC079466.1 | 2 |
| RP11-620J15.3 | 1 |
| MIR4458HG | 1 |

A pair such as TBX2+TBX3 is scored on 20 genes, three of which, MALAT1, NEAT1 and RP11-717F1.1, never reach Geneformer. The count decoder still predicts them, because it predicts all 5,000 modeled genes, but it has to infer them from the genes Geneformer did see. If a non-coding gene moves together with a protein-coding program, that inference can work. If its variation is its own, the prediction for it will be poor however good the embedding is.

The recurrence is the interesting part. MALAT1 and NEAT1 are very abundant RNAs that stay in the nucleus, and they are among the top movers for many unrelated perturbations. That suggests they track an axis shared across perturbations rather than a response specific to each pair, a general stress or nuclear-RNA signal being plausible candidates. We have not tested which.

The blind spot also runs the other way. Our JEPA reads only the 5,000 most variable genes, while Geneformer reads roughly 14,360. Of the 5,000 modeled genes, 4,316 reach the V2 models and 4,352 the V1 models; the rest are invisible to them. But Geneformer sees nearly 10,000 genes that our encoder never reads. Each encoder has information the other lacks.

## 5. Separating a blind spot from a weaker representation

If Geneformer scores lower on a pair, part of the gap could be the unseen genes rather than a weaker embedding. The comparison is therefore planned to report two readings of every endpoint. The main reading uses all scored genes, exactly as every earlier round did, so the numbers stay comparable across the ledger. A second, descriptive reading scores *both* encoders, ours and Geneformer, on only the scored genes Geneformer can see, the same 17 to 20 genes per pair for both. A gap that persists on the visible genes is about the representation. A gap that closes is about the vocabulary. Because the unseen genes are a few recurring ones, the second reading can also show whether MALAT1 and NEAT1 in particular are driving a difference.

---

*Up: [the foundation-encoder series](index.md). Previous: [Predicting expression from a ranking](02-expression-from-a-ranking.md). Next: [A generative model on top of a foundation model](04-a-generative-model-on-a-foundation-model.md).*
