# Reading the numbers

*Notes that go with the scripts in [`data-science/`](../README.md). Where a script prints an argument, a page here writes it down.*

The perturbation-response work reports a small number of figures over and over: Δr, `e_p`, an energy distance, an effective rank, a max-t critical value. Each compresses two populations of cells, or one matrix of embeddings, into a single number, and each throws away something specific in the process. These pages are about what gets thrown away, because that is usually where a disagreement between two rounds turns out to live.

## Pages

| | |
|---|---|
| [Reading the metrics](00-reading-the-metrics.md) | the four endpoint families side by side: what each is blind to, when they disagree, and which script prints which |

## The scripts these pages describe

Each script is a runnable version of an argument. They print their work, so the fastest way to use these pages is to read a section here and then run the matching section.

```bash
cd examples/perturbation_response/data-science
python 01_metric_energy_distance.py          # population comparison, and the two estimators
python 02_metric_effect_size_and_ep.py       # Δr and e_p, and the scale terms behind them
python 03_holdout_power.py                   # what is in the holdout, and what the mean hides
python 04_collapse_effective_rank.py         # effective rank, and what a random slice sees
```

Every script takes `--sections` so a single argument can be re-run on its own, and `--help` for the rest.

## What is settled here and what is not

These pages explain metrics the project has already committed to. They are not where a metric gets chosen or changed: an endpoint is fixed in a round's protocol before scoring, and the [results pages](../../docs/results/index.md) are the record of what each round pre-registered and found.

So if a page here and a round page disagree about a number, the round page is right and the page here needs fixing. The scripts carry the same asymmetry: they import the production implementation precisely so they cannot quietly define a second version of the metric.

---

*Up: [data-science](../README.md). The rounds: [results](../../docs/results/index.md). The pipeline: [the workflow map](../../docs/workflows/index.md).*
