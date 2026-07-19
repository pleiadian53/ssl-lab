# 2. Score: grading a model on the standing benchmark

*Four scripts, one metric function. The engineering here is almost entirely about making it impossible to grade two models on two slightly different metrics, because that failure is invisible from the outside and it invalidates every comparison you have.*

---

## The single-sourced harness

Every model in this project is graded by one function:

```python
run_effect_size_eval(predict_fn, *, hvg_X, pert_names, pert_id, is_test,
                     de_genes, control_mean, top_k, min_test_cells, ...)
```

A model participates by supplying a **`predict_fn(pid, name)`** that returns one perturbation's predicted per-gene expression. That is the entire interface. Everything else, the held-out truth, the gene support, the control baseline, the correlation, is fixed and shared.

Look at who enters through that one seam:

| caller | its `predict_fn` |
|---|---|
| `06` flow / operator / operator algebra | sample latents from Stage B, decode, average |
| `09` NB-VAE baseline | sample from the VAE, decode, average |
| `14` ceiling, `roundtrip` arm | encode *real* held-out cells, decode |
| `14` ceiling, `linear` arm | a ridge readout of real latents |
| `14` ceiling, `identity` arm | the true mean itself |

Five very different objects, one metric. This is why adding a new Stage B costs a loader and a flag rather than a new evaluator, and why an oracle can be dropped into the benchmark and compared to a model directly. It is also the reason the [ceiling analysis](03-diagnose.md) was cheap to write: it is not a new evaluation, it is four new `predict_fn`s.

> **If you add a model, do not add an evaluator.** Write a `predict_fn` and a loader. The moment two evaluators exist, the two numbers they produce are no longer known to be comparable, and nothing will tell you.

## The two axes

**Effect size (`06`) is the primary benchmark.** Per perturbation, correlate predicted against true $\Delta = \mathrm{mean}(\text{perturbed}) - \mathrm{mean}(\text{control})$ over that perturbation's top-DE genes.

```bash
python examples/perturbation_response/06_eval_effect_size.py --experiment norman_flow_control \
    --split combo --n 200 --top-k 20
```

Two flags carry the [Vary](05-vary.md) pattern. `--stage-b {flow,operator,operator_algebra}` picks which Stage B to grade, and `--decoder <path>` reuses an existing decoder rather than the experiment's own. A Stage-B lever does not change the readout, so reusing the decoder is what keeps the comparison honest.

**Calibration (`10`) is the distributional axis.** Effect size grades a *mean*; calibration asks whether the predicted *population* has the right shape, via interval coverage, per-gene spread correlation, Wasserstein distance, and joint energy distance.

```bash
python examples/perturbation_response/10_eval_calibration.py --experiment norman_flow_control \
    --model flow --split combo --n 200
```

The crucial difference from `06`: calibration **samples counts** from the decoder's NB, because a population of decoded rates is near-degenerate and no spread is measurable without it. Effect size does not sample, since it reads the mean rate profile directly.

**Sampling (`05`) is for looking, not scoring.** It generates a population for one perturbation and prints the top up-regulated genes.

```bash
python examples/perturbation_response/05_sample_perturbed.py --experiment norman_stage_a --pert CEBPE --n 500
```

Useful for sanity and for figures. It produces no metric and licenses no claim.

## The split is part of the question

`--split cells` holds out *cells of perturbations the model has seen*: an in-distribution test of whether the machinery recovers effect size at all. `--split combo` holds out *entire two-gene combinations*: a compositional generalization test, and the one the ledger scores.

They are different questions and they produce different numbers, so a result is meaningless without its split. Every report records it.

## What a Score result licenses, and what it does not

A Score result is a **measurement of one model**. It is not a claim about a difference.

The gap between those is the whole of the [Decide](04-decide.md) workflow. Two arms differing by $0.02$ on a twenty-perturbation test set is entirely consistent with noise, and eyeballing a scoreboard is how a project talks itself into a lever that does nothing. Report the number here; earn the comparison there.

Two failure modes specific to this workflow, both of which have bitten:

**A number that will not move.** If a metric returns the same value across models that share almost nothing, it is measuring something they have in common rather than measuring them. Coverage sat at exactly $1.00$ for months for this reason, and the cause was upstream in the gene list, not in any decoder. See [the scoring seam](../conditional-flow-jepa/3e-the-genes-the-metric-scores.md).

**Grading against a stale support.** Train on a pod, but **score locally**. A pod volume can hold an out-of-date `de_genes.json`, and a blanket results fetch once overwrote seven corrected reports with wrong ones, caught only because the same command returned a different number twice.

---

*Previous: [Build](01-build.md). Next: [Diagnose](03-diagnose.md). Up: [the workflow map](index.md).*
