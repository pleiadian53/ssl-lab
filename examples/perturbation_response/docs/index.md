# Perturbation response — data, method, and findings

This folder documents the perturbation-response work end to end: the single-cell data that feeds the model, the method built on top of it, and what we actually found when we ran it. It is organized as two series, one for the data and one for the method.

## The two series

### [Reading Perturb-seq](reading-perturb-seq/index.md)
A four-part primer on the data. What a single-cell perturbation experiment measures, what the intervention *is* biologically, how a cell becomes tokens the encoder reads, and how success is scored. Every number is real, drawn from a reproducible slice of Norman et al. 2019. Start here if the vocabulary of counts, dropout, and library size is new, or if you want to know exactly what the model is being asked to predict.

### [The conditional-flow + JEPA method](conditional-flow-jepa/index.md)
The method itself and its evaluation. The idea (a conditional flow prior over frozen JEPA latents, with a negative-binomial count decoder), the implementation, the training and pod workflow, the experimental results against a from-scratch baseline, the challenges we ran into, and the directions we think can push past the current limits. This series reports findings honestly, including the ones that did not go our way, because that is where the next idea comes from.

## Running it yourself

### [Running the pipeline](running-the-pipeline.md)
The runnable companion to the method series: the exact sequence of commands from raw data to the effect-size scoreboard, what each script reads and writes, the GPU-pod workflow, a provenance map from every number in the results chapter back to its script and report, the failure modes and how to recover from them, and the pattern we use to run a lever experiment. Follow it top to bottom the first time, then keep it as a reference and a recovery guide.

### [Workflows: what the eighteen scripts are for](workflows/index.md)
The map, where the runbook above is the route. The scripts are numbered `00` to `17` in the order they were written, which is not an order anyone runs them in, and they are not one pipeline but **five kinds of workflow**: [Build](workflows/01-build.md) an artifact, [Score](workflows/02-score.md) a model, [Diagnose](workflows/03-diagnose.md) where the loss lives without training anything, [Decide](workflows/04-decide.md) whether a difference is real, and [Vary](workflows/05-vary.md) one lever without breaking comparability. Read this when you have a question rather than a recipe, when you want to add a variant, or when you are about to build something and have not yet checked whether the stage it improves has any room. It also covers `13` through `17`, which postdate the runbook, and a companion chapter on [preparing a dataset](workflows/01a-preparing-a-dataset.md): what `00` does to raw counts, how to change the gene panel or the token geometry or the source dataset, and what a new cache invalidates downstream.

### [Running an experiment you can trust](../../../docs/experimental-method/index.md)
The methodology behind every number in these series, written to transfer to any R&D project rather than to this one. What the metric is actually computed on (and the trap that had us scoring silent genes), seeds versus metric noise, ablations versus controls, and the joint bootstrap with simultaneous intervals that turns a difference into a verdict. This project is its worked example, including the mistakes.

## How they relate

The [design-space survey](../../../docs/generative_jepa/index.md) decided *what* to build and *why*, at the level of theory. The **Reading Perturb-seq** series grounds that decision in the real dataset. The **conditional-flow + JEPA** series builds it, runs it, and measures it. Read the data series to understand the problem, the method series to understand the attempt and where it stands.

## The code

Both series describe code that lives in this folder and in [`src/ssllab`](../../../src/ssllab/). The numbered scripts `00`–`10` in [`examples/perturbation_response`](../) are the runnable pipeline; the generative modules are under [`src/ssllab/generative`](../../../src/ssllab/generative/) and the evaluation under [`src/ssllab/eval`](../../../src/ssllab/eval/).
