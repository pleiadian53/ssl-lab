# Perturbation response — data, method, and findings

This folder documents the perturbation-response work end to end: the single-cell data that feeds the model, the method built on top of it, and what we actually found when we ran it. It is organized as two series, one for the data and one for the method.

## The two series

### [Reading Perturb-seq](reading-perturb-seq/index.md)
A four-part primer on the data. What a single-cell perturbation experiment measures, what the intervention *is* biologically, how a cell becomes tokens the encoder reads, and how success is scored. Every number is real, drawn from a reproducible slice of Norman et al. 2019. Start here if the vocabulary of counts, dropout, and library size is new, or if you want to know exactly what the model is being asked to predict.

### [The conditional-flow + JEPA method](conditional-flow-jepa/index.md)
The method itself and its evaluation. The idea (a conditional flow prior over frozen JEPA latents, with a negative-binomial count decoder), the implementation, the training and pod workflow, the experimental results against a from-scratch baseline, the challenges we ran into, and the directions we think can push past the current limits. This series reports findings honestly, including the ones that did not go our way, because that is where the next idea comes from.

## How they relate

The [design-space survey](../../../docs/generative_jepa/index.md) decided *what* to build and *why*, at the level of theory. The **Reading Perturb-seq** series grounds that decision in the real dataset. The **conditional-flow + JEPA** series builds it, runs it, and measures it. Read the data series to understand the problem, the method series to understand the attempt and where it stands.

## The code

Both series describe code that lives in this folder and in [`src/ssllab`](../../../src/ssllab/). The numbered scripts `00`–`10` in [`examples/perturbation_response`](../) are the runnable pipeline; the generative modules are under [`src/ssllab/generative`](../../../src/ssllab/generative/) and the evaluation under [`src/ssllab/eval`](../../../src/ssllab/eval/).
