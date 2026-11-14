# 01 · How Geneformer reads a cell

*Geneformer and our JEPA encoder both turn a cell into a vector by running a transformer over tokens and averaging the result. They differ in what a token is, what the model was trained to predict, and how many cells it learned from. This chapter walks through Geneformer's side step by step and sets each step beside ours.*

---

## 1. The input: one list of counts per cell

Geneformer starts from the raw measurement. For every cell it needs the raw UMI count of every measured gene, not a selected subset and not a log-transformed value, with each gene named by its Ensembl ID, such as `ENSG00000102145` for GATA1. It also needs each cell's total count. On disk this is an ordinary cells-by-genes count matrix, stored sparsely because most entries are zero, so each cell is in effect a short list of (gene ID, count) pairs for the genes it expressed.

Our own encoder reads something different: the 5,000 most variable genes of the Norman data, normalized to counts per 10,000 and log-transformed. The two inputs come from the same experiment, though. The processed dataset was made from the raw Norman file by quality filtering, which removed 1,518 of its 111,255 cells, and by choosing the 5,000 genes. The raw file still holds every cell's counts over all 19,018 measured genes. A small script, [`01_build_cell_index.py`](../../foundation-encoder/01_build_cell_index.py), matches each processed cell to its row in the raw file by its barcode. Geneformer therefore embeds exactly the cells our pipeline trains and scores on, and each embedding is filed under the cell's position in our dataset, with its perturbation label and its split.

## 2. The rank-value encoding: from counts to an ordered list of genes

Geneformer does not feed counts to its transformer. It turns each cell into an ordered list of genes. Write $c_g$ for the count of gene $g$ in the cell and $N = \sum_g c_g$ for the cell's total count over all genes. The tokenizer computes, for every expressed gene,

$$r_g = \frac{10^4 \cdot c_g / N}{m_g},$$

where the numerator is the gene's count per 10,000, the usual library-size normalization, and $m_g$ is that gene's median normalized value across the pretraining corpus, taken over the cells in which the gene was detected. The expressed genes are then sorted by $r_g$ from highest to lowest, and each gene in the sorted list becomes one token.

Dividing by $m_g$ is what makes the order informative. Here is a toy cell with made-up numbers and a total of $N = 20{,}000$:

| gene | raw count $c_g$ | count per 10,000 | corpus median $m_g$ | ratio $r_g$ |
|---|---|---|---|---|
| ACTB | 400 | 200 | 150 | 1.3 |
| HBG1 | 200 | 100 | 10 | 10 |
| GATA1 | 30 | 15 | 2 | 7.5 |
| MYC | 6 | 3 | 3 | 1.0 |

The cell becomes the list HBG1, GATA1, ACTB, MYC. ACTB has the largest count but lands third, because it is high in nearly every cell. HBG1 and GATA1 rise to the front because they are far above their usual level. The order therefore says what is unusually high in this particular cell, which is closer to what distinguishes one cell state from another than raw abundance is. Once the order is fixed, the values themselves are discarded. Only the list goes into the model.

Two more details matter for the Norman cells. First, only protein-coding and microRNA genes have tokens. About 14,360 of the 19,018 measured genes do; the rest, mostly long non-coding RNAs such as MALAT1 and NEAT1, never reach the model. [Chapter 3](03-the-scored-genes-and-the-blind-spot.md) follows that blind spot into the scored genes. Second, the list has a maximum length. The first-generation model, V1, reads at most 2,048 genes. The second generation, V2, reads up to 4,096 slots, two of which hold the special tokens described in section 5, which leaves 4,094 for genes. A Norman cell expresses a median of about 3,120 genes that have tokens, so V1 cuts 94% of the cells short and never sees about a third of their expressed genes, always the ones at the bottom of the ranking. V2 cuts 10% of the cells and loses about 1% of the genes.

## 3. Two lookup tables: what a gene is, and where it sits

The transformer cannot read gene names, so each token is turned into a vector by looking it up in two tables and adding the results.

The **token embedding table** has one row per vocabulary entry. GATA1 always gets the same row, wherever it sits in the list. Its values start as small random numbers and are trained along with the rest of the model.

The **position embedding table** has one row per slot: one for the first gene in the list, one for the second, and so on up to the last slot. The model looks up the row for each slot and adds it to that slot's token vector, so "GATA1 in slot 2" and "GATA1 in slot 900" enter the network as different vectors. This is how relative expression level reaches a model that never sees a value: the slot a gene occupies is its rank in the cell.

Geneformer is a standard BERT model, and its position table is BERT's: learned, exactly like the token table, by backpropagation of the training loss. In V2-104M it has 4,096 rows of 768 numbers; in V1-10M, 2,048 rows of 256. It is worth knowing the alternatives, because they appear throughout the literature. The original transformer of 2017 used **sinusoidal** encodings: also a table with one vector per slot, but filled with sines and cosines at a range of frequencies and never trained. The model learns how to use them, and because they are a formula they extend to slots longer than any training sequence. A third family, **rotary** embeddings, used by most recent language models, adds nothing to the token vectors at all. It rotates the query and key vectors inside each attention step so that attention depends on the distance between two tokens rather than on their absolute slots. For sequences with a fixed maximum length, learned tables and sinusoids usually perform about equally well. A learned table is simpler and can fit whatever structure the data has, but it cannot go beyond its last trained row.

**Our JEPA has learned position tables too, and they play the opposite role.** Our encoder splits the 5,000 genes into 50 fixed groups of 100 genes each, chosen at random once and then never changed, so token 17 always holds the same 100 genes ([tokenization](../reading-perturb-seq/03-tokenization.md)). The 100 expression values of a group enter through one linear map shared by all 50 groups, which on its own cannot tell group 3 from group 17. The position table, a learned 50-by-256 matrix in [`vit.py`](../../../../src/ssllab/models/vit.py), is what gives each group its identity. In our model the slot says *which genes* a token carries and the content carries *how much* they are expressed. In Geneformer the token says which gene it is and the slot carries how high it ranks.

| | identity of the genes comes from | expression level comes from |
|---|---|---|
| our JEPA | the position table, one row per fixed gene group | the content: the values, through the shared linear map |
| Geneformer | the token table, one row per gene | the position table, since the slot is the rank in this cell |

Our model actually holds three position tables, though only two are trained by gradient. The encoder has one. The predictor has its own, which is how it knows which hidden slots it is asked to fill: every hidden slot starts as the same learned mask vector plus that slot's row of the predictor's table, so without the table every target would look identical. The third belongs to the slowly updated teacher encoder, a copy of the student whose weights, its position table included, move a small step toward the student's after every training step. It is a moving average of the encoder's table, not an independent one.

## 4. Pretraining: guess the hidden gene

During pretraining, a fraction of the genes in each cell's list is hidden, 15% in the original paper, and the model must guess which gene was in each hidden slot from the genes around it. To do that across tens of millions of cells, it has to learn which genes tend to rank high together, which programs exclude each other, and which genes mark a cell type. The vectors it builds for each gene inside the network are therefore *contextual*: GATA1's vector in an erythroid-like cell differs from its vector in a T cell, even though both start from the same row of the token table.

Our JEPA hides gene groups as well, but it does not ask which genes were hidden. It asks the predictor to produce the vector a teacher encoder assigns to each hidden group, a prediction in representation space rather than in the space of the input. The two models also learned from very different amounts of data. The V1 models were pretrained on about 30 million cells and the V2 models on about 104 million, from many tissues and experiments. Our JEPA saw about 96,000 cells from one experiment on one cell line.

| | our JEPA encoder | Geneformer |
|---|---|---|
| genes it reads | the 5,000 most variable genes | about 14,360 protein-coding and microRNA genes of the 19,018 measured |
| what a token is | a fixed group of 100 genes, carrying their log-normalized values | one gene, carrying only its identity and its slot in the ranking |
| tokens per cell | always 50 | as many genes as the cell expresses, up to 2,048 (V1) or 4,094 (V2) |
| pretraining task | predict the teacher's vectors for hidden gene groups | predict which gene sat in each hidden slot |
| trained on | about 96,000 Norman training cells | about 30 or 104 million cells |
| cell embedding | the average of the 50 output vectors, 256 numbers | the average over genes, or the summary token: 256, 768 or 1,152 numbers |

## 5. From gene vectors to one vector per cell

A cell embedding is built from the contextual gene vectors in one of two ways.

- **Gene average**, Geneformer's `cell` mode, takes the mean of the vectors of all the cell's genes. It is the only option for V1. Our JEPA's cell vector is built the same way, as the mean of its 50 output vectors.
- **Summary token**, the `cls` mode, exists only in V2. Every V2 list begins with a special `<cls>` token and ends with an `<eos>` token. Through attention, the `<cls>` token can read every gene in the list, and its output vector is used as the cell's summary. The pretraining loss is computed at the hidden gene slots, so no term of it scores the summary token's own output. It becomes a summary only because everything it can read is about the cell. Whether that makes it better or worse than the plain average is an empirical question, which is why the comparison extracts both.

In both modes the vectors are taken from the second-to-last layer of the network, Geneformer's default. The last layer is shaped most directly by the task of naming hidden genes, so the layer below it tends to describe the cell more generally.

Four models are relevant here, and the number in each name is roughly its parameter count. **V1-10M** has 6 layers and 256-number vectors, the same width as our JEPA. **V2-104M** has 12 layers and 768-number vectors; the "104" happens to match the 104-million-cell corpus as well, but in the name it counts parameters. **V2-104M_CLcancer** is V2-104M with continued pretraining on cancer cells, a relevant variant because Norman's K562 cells come from a leukemia line. **V2-316M** has 18 layers and 1,152-number vectors.

## 6. Why the deep slots are learned from fewer examples

A row of a learned position table is trained only when its slot holds a real gene. Cells in a training batch have lists of different lengths, so shorter lists are padded with a placeholder token to the length of the longest. Two mechanisms make the padded slots invisible to learning: an attention mask stops real genes from reading anything from padded slots, and the loss is computed only at real gene slots. Nothing the loss depends on passes through a padded slot, so the position row there receives no gradient from that cell.

Over the whole corpus, then, row 5 of the table is trained by essentially every cell, while row 3,500 is trained only by the cells that express at least 3,500 genes with tokens. How lopsided that is depends on the corpus. As an illustration, here is how often each slot is filled in a random sample of 2,000 Norman cells, counting the genes the V2 tokenizer keeps:

| slot | share of Norman cells that fill it |
|---|---|
| 1 to 1,000 | 100% |
| 2,048, V1's last slot | 94% |
| 2,500 | 82% |
| 3,000 | 59% |
| 3,500 | 28% |
| 4,000 | 12% |
| 4,094, V2's last gene slot | 10% |

If pretraining cells looked like these, row 3,500 would see about 28% as many training examples as row 5, and row 4,000 about 12%. The corpus's own length distribution is not something we have measured. Norman's cells are sequenced deeply, a median of about 14,000 UMIs per cell, and much atlas data is shallower, so the deep slots may well have been filled even less often during pretraining than in this table.

The deep slots are also less informative per example. The genes at the bottom of a list are the ones expressed at or below their usual level, most with a count of one or two. Many tie on their count, and among tied genes the order is decided mostly by the corpus medians $m_g$, not by anything specific to the cell. Guessing which of thousands of barely expressed genes sits in slot 3,500 is a weak training signal. And a learned table, unlike a sinusoidal one, builds in no relation between neighboring rows: rows 3,500 and 3,501 end up similar only if training pushes them there.

The likely result is a model that resolves the top of the list sharply and the deep tail coarsely, closer to "somewhere far down" than to a distinct position. That has a consequence for the comparison. V2's longer input looks like a large advantage over V1 when measured by how many genes each model is *given*, but if the deep rows are poorly trained, V2 may make only weak use of slots 2,049 to 4,094. This is a plausible mechanism, not a measurement. Two inexpensive checks can test it: reading V2's position table and plotting, against the slot number, how large each row is and how similar it is to its neighbor; and embedding the same cells with V2 twice, once with full lists and once cut at 2,048 genes, to see how far each cell's embedding moves.

---

*Up: [the foundation-encoder series](index.md). Next: [Predicting expression from a ranking](02-expression-from-a-ranking.md).*
