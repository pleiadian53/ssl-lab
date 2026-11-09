# data-science — understanding the data, the metrics, and the methods

Runnable explainers. Each script answers one conceptual question with numbers from this repository, and prints its work. Nothing here trains a model, writes a checkpoint, or feeds a result into the [ledger](../docs/results/index.md).

The scripts exist because these answers were being derived in conversation and then lost. A question that takes ten minutes to answer and is asked three times should be a script.

## The scripts

| | question |
|---|---|
| [`01_metric_energy_distance.py`](01_metric_energy_distance.py) | how is energy distance computed, what does the joint view buy over per-gene marginals, and why are there two estimators in the codebase? |
| [`02_metric_effect_size_and_ep.py`](02_metric_effect_size_and_ep.py) | how do Δr and `e_p` turn two populations into one number, what do the scale terms `s_p` and `λ_p` mean, and how good can a perfect predictor look? |
| [`03_holdout_power.py`](03_holdout_power.py) | what is actually in the holdout, how much does each held-out combination contribute, and why did one perturbation push a max-t critical value to 4.63? |
| [`04_collapse_effective_rank.py`](04_collapse_effective_rank.py) | what does effective rank measure, and why does a random-slice isotropy penalty tolerate a rank-20 embedding in 256 dimensions? |

Each takes `--help`, defaults to the `norman2019` cache, and runs on a laptop in seconds. None requires a GPU. `02` and `03` need no checkpoint either: their predictors are the training-free references from rounds 7 and 8, so they are reproducible from the cache alone.

## What belongs here, and what does not

**Here:** open-ended inquiry. Exploratory data analysis, statistics, numerical demonstrations of a metric or an objective, sanity intuitions, "what would this look like if" calculations.

**Not here:** anything on the pipeline's dependency spine. Those are the numbered scripts in the parent directory, and the [workflow series](../docs/workflows/index.md) is their map:

- [`00a_probe_hvg_coverage.py`](../00a_probe_hvg_coverage.py) gates a cache rebuild, so it is Build-adjacent.
- [`00b_data_card.py`](../00b_data_card.py) is the canonical cell-count query. `--perturbation CEBPE+RUNX1T1` for one label, `--as-table --kind combo --in-split test` for the holdout as rows. If you want a count, it is there, not here.
- [`00c_show_architecture.py`](../00c_show_architecture.py) is the canonical model-anatomy query.

The dividing question: *is this about an artifact the pipeline produces, and would the runbook cite it?* If yes it is a numbered script. If it is inquiry that informs how we read a result, it is here.

## The house rule

**A script here never reimplements what it explains.** It imports the production implementation from `src/ssllab` and illustrates it. The reason is that an explainer which quietly diverges from the code is worse than no explainer: it teaches a metric the project does not actually compute.

Where a contrast genuinely needs a second implementation, the script states the relationship between the two and **asserts it**. `01` is the worked example: the production energy distance is a V-statistic, the probe's scoring estimator is a U-statistic, and the script asserts the exact identity relating them rather than leaving the reader to trust a claim about bias.

This is the discipline already used elsewhere in the package. [`00c_show_architecture.py`](../00c_show_architecture.py) replays a module's real forward and checks its own table against it. [`00a_probe_hvg_coverage.py`](../00a_probe_hvg_coverage.py) imports `differential_expression` from `00` rather than copying the ranking convention. Same idea, applied to concepts.

Three consequences worth stating, because they are what make the scripts trustworthy:

1. **Every number is traceable.** A script prints which function produced each figure, so a reader can go to the source rather than to the script's own arithmetic.
2. **They break when the code changes.** That is the point. If `pair_metrics` changes its definition of `e_p`, `02` should fail or print something different, which is a signal rather than a nuisance.
3. **They are not tests.** Tests live in [`tests/`](../../../tests/) and assert behavior. These assert only the relationships they explain, and otherwise print.

## Adding a script

Number it in sequence and keep one question per script. The module docstring is the explainer: state the question, the answer, and where in `src/ssllab` the implementation lives. Print sections with headers so the output reads top to bottom as an argument. If a section needs a figure, write it under `docs/assets/` and reference it from a page in [`docs/`](docs/index.md).

If the question turns out to be about an artifact the pipeline produces, it belongs in the parent directory instead.

---

*Up: [perturbation response](../README.md). The methods: [co-adaptation](../co-adaptation/README.md). What the rounds found: [results](../docs/results/index.md). Which script answers which question: [the workflow map](../docs/workflows/index.md).*
