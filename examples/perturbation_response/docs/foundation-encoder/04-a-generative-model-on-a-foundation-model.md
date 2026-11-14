# 04 · A generative model on top of a foundation model

*The aim of this project is a generative model of perturbed cells: one that can be sampled to produce realistic populations of cells under a perturbation nobody measured. A foundation model like Geneformer is not such a model on its own. This chapter explains how one is built on top of it, what has to hold for that to work, and how the comparison is wired so that any encoder can be dropped in.*

---

## 1. The recipe: encode, model the vectors, decode

Our pipeline is already a generative model built on top of an encoder. It has three parts:

1. **An encoder** turns each cell into a vector, its embedding, and is kept frozen.
2. **A conditional flow** learns how the embeddings of control cells are transformed under each perturbation, and can be sampled to produce new embeddings for a perturbation, including combinations it has never seen together.
3. **A count decoder** turns each sampled embedding back into gene counts, drawing them from a negative-binomial distribution so that the generated cells carry realistic count noise.

This is the same design as latent diffusion for images, where a pretrained autoencoder is frozen, a generative model learns the distribution of its latent vectors, and the decoder turns samples back into pictures. Nothing in the recipe requires the encoder to have been trained by us. Putting Geneformer in the first slot gives a sampleable generative model whose representation was learned from about 104 million cells, while the flow and the decoder are trained on Norman.

## 2. What has to hold for it to work

Three things have to hold, and each corresponds to something the comparison measures.

**A way back to counts.** Geneformer has no decoder for expression. Its pretraining head only names which gene sat in a hidden slot, so it can produce a ranking but not a count. The count decoder is therefore trained on Geneformer's embeddings, and how much expression information those embeddings retain sets a ceiling on everything built above them. The readout ceiling, real held-out cells encoded and then decoded, measures exactly that. Because the rank encoding discards spacing ([Chapter 2](02-expression-from-a-ranking.md)), this ceiling may come out below our JEPA's, and it is the first thing the comparison will show.

**A distribution the flow can learn.** The flow has to learn, from the same Norman training cells, how control-cell embeddings shift under each perturbation. With Geneformer's V2 models that happens in 768 dimensions instead of our 256, a harder learning problem with the same amount of data. Whether to let the flow work in the full space or in a projection down to a smaller one is a design choice the protocol has to fix in advance. The full space keeps everything; a projection is easier to learn but discards information and adds a choice that could be tuned to the result.

**Real variation between cells.** A generative model has to reproduce how cells differ within one condition, not only the average response. In this pipeline the randomness comes from two places: the flow's sampling of embeddings and the decoder's count noise. The first only means something if the embedding keeps the differences between cells of the same condition, rather than mapping them all to nearly the same point. The population metrics on the benchmark, interval coverage and energy distance among them, check whether the generated cells spread like the real ones.

## 3. Why not sample from Geneformer itself

One might ask whether a model pretrained to fill in hidden genes could generate cells directly, by hiding every slot and filling them in one at a time. It can produce something, but it is not what is needed. Geneformer is a masked model, trained to fill a few gaps given everything else, not a model of complete cells with a probability for each. Filling a whole list gap by gap yields a ranking of genes, not counts, with no calibrated probabilities attached and no handle on sequencing depth. Building the generative model in its embedding space, with a decoder trained for counts, is the practical route.

## 4. Reading the outcome

The comparison can fail in two distinguishable ways, and each points somewhere different. A low readout ceiling means the embedding lost information the counts need, and no flow could recover it. A good ceiling with a poor full pipeline means the information is there but the flow struggles to model the geometry of the embedding space. If the frozen Geneformer arm comes close to our best models, a further arm trains Geneformer jointly with the decoder. That asks a different question: whether starting from Geneformer's weights helps once the encoder is retrained, which is round 13's question repeated for a model pretrained on far more cells.

## 5. One interface for any encoder

For the comparison to be fair, every encoder has to reach the decoder and the flow through the same code. The comparison is being built around a simple contract: an encoder writes one matrix with a row of numbers per cell, the list of cell positions those rows belong to, and a short description of where they came from, namely the source model, the embedding mode and layer, the width, and a fingerprint of the dataset the vectors were computed on. Our own JEPA's vectors are exported the same way, so the two arms differ only in the file they read. Before any Geneformer number is read, the JEPA's exported vectors must reproduce the published readout ceiling exactly through the new path; if they do not, the interface has changed something and every comparison built on it would be in doubt.

Only one of the three parts is fully general. The flow learns how a set of vectors shifts under a condition and does not care whether the vectors describe cells, images or proteins. The count decoder is specific to count data, and the description of a perturbation, as which genes were switched on, is specific to genetic perturbations. The reusable unit is therefore a conditional flow over any embedding space, paired with a decoder for the data type at hand.

The same contract could later apply to the condition side as well. Today the flow learns a vector for each perturbed gene from Norman alone, so it knows nothing about a gene that was never perturbed in training. Describing perturbations with pretrained gene vectors instead, such as Geneformer's own gene embeddings or a protein language model's vector for the target, could in principle let the model predict the effect of perturbing genes it has never seen perturbed. That is a separate question from whether better cell embeddings help, and it would be tested on its own.

---

*Up: [the foundation-encoder series](index.md). Previous: [The scored genes, and what Geneformer cannot see](03-the-scored-genes-and-the-blind-spot.md).*
