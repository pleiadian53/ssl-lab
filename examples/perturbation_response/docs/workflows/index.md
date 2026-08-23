# Workflows: what the eighteen scripts are for

*The scripts in this example folder are numbered `00` through `17`, and the numbering is a fossil record of the order they were written rather than a sequence anybody runs. They are not one pipeline. They are **five different kinds of workflow**, each answering a different question, and knowing which kind you are in tells you what to run, what it costs, and what its answer is allowed to license.*

> **What this series is, and what it is not.** [Running the pipeline](../running-the-pipeline.md) is the **runbook**: the linear happy path from raw data to the scoreboard, with the exact commands in the exact order. Read that when you want to *reproduce* the result. This series is the **map**: which workflow answers which question, why the pieces are shaped the way they are, and which script to reach for when you have a question rather than a recipe. It also covers `13` through `17`, which the runbook predates and does not mention.
>
> It complements the [results ledger](../conditional-flow-jepa/results-ledger.md) the same way. The ledger records **what happened**, round by round. This series records **how you would ask a question like that** in the first place.

---

## The five workflows

| | workflow | the question it answers | cost | scripts |
|---|---|---|---|---|
| 1 | **[Build](01-build.md)** | *produce an artifact something else depends on* | GPU, minutes to hours | `00` `01` `03` `04` `08` `13` `16` |
| | ↳ **[Preparing a dataset](01a-preparing-a-dataset.md)** | *what `00` does to raw counts, and what a new cache invalidates* | CPU, once | `00` |
| 2 | **[Score](02-score.md)** | *how good is this model on the standing benchmark?* | CPU, minutes | `05` `06` `09` `10` |
| 3 | **[Diagnose](03-diagnose.md)** | *where is the loss, and is this stage even worth improving?* | CPU, minutes, **trains nothing** | `00a` `02` `11` `14` `15` |
| 4 | **[Decide](04-decide.md)** | *is this difference real, or am I reading noise?* | seconds | `12` `17` |
| 5 | **[Vary](05-vary.md)** | *add an arm without breaking comparability* | orchestration | `run_*.sh`, the reuse flags |

The distinction that matters most is between **Build** and everything else. Building is where the money and the time go, and it is the only workflow that creates state other steps depend on. The other four consume artifacts and produce numbers. A large fraction of the useful work in this project turned out to live in workflows 3 and 4, which train nothing at all.

## Start here: what are you actually trying to do?

- **"I want to reproduce the headline result."** You want the [runbook](../running-the-pipeline.md), not this series.
- **"I have an idea for making the model better."** Go to [Diagnose](03-diagnose.md) *first*, not [Build](01-build.md). Measure whether the stage you are about to improve has any headroom. Three rounds of this project were spent improving a stage that had $0.03$ of room, and one afternoon of diagnosis would have said so. This is the single most expensive lesson in the repository.
- **"I want to add a variant and compare it fairly."** [Vary](05-vary.md) for the mechanics, then [Score](02-score.md) for the grading, then [Decide](04-decide.md) for the verdict.
- **"I have two numbers and I want to know if the difference is real."** [Decide](04-decide.md).
- **"I want to use a different dataset, or more genes."** [Preparing a dataset](01a-preparing-a-dataset.md). Read the blast-radius section before you start: a new cache retrains the whole spine, not just the encoder.
- **"Something upstream might be broken."** [Diagnose](03-diagnose.md), and specifically the acceptance-gate idea: every stage should assert something about its own output before the next stage consumes it.

## The whole map, script by script

Grouped by workflow rather than by number, which is the point.

| script | workflow | what it produces | notes |
|---|---|---|---|
| `00_process_norman.py` | Build | the cache: HVG matrix, splits, `de_genes.json` | also owns the **scoring seam**; `--de-only` regenerates the gene list without touching splits. Knobs and blast radius: [1a](01a-preparing-a-dataset.md) |
| `01_pretrain_stage_a.py` | Build | `encoder.pt` (frozen thereafter) | the JEPA encoder; masked prediction, no dynamics or decoding objective |
| `03_train_count_decoder.py` | Build | `count_decoder.pt` | Stage C, trained on frozen latents |
| `04_train_cond_flow.py` | Build | `cond_flow.pt` | Stage B, the conditional flow prior |
| `08_train_cvae_baseline.py` | Build | `cvae_baseline.pt` | the from-scratch NB-VAE: no JEPA, no flow, trained end to end |
| `13_train_operator.py` | Build | `operator.pt` | Stage B variant: the action operator |
| `16_train_operator_algebra.py` | Build | `operator_algebra.pt` | Stage B variant: per-gene generators, group composition |
| `05_sample_perturbed.py` | Score | a generated population, for inspection | the qualitative look, not a metric |
| `06_eval_effect_size.py` | Score | `effect_size.json` | **the primary benchmark**; `--stage-b` selects which Stage B to grade |
| `09_eval_cvae_baseline.py` | Score | `effect_size.json` for the baseline | same metric, same harness, different model |
| `10_eval_calibration.py` | Score | `calibration_flow.json` | the distributional axis |
| `00a_probe_hvg_coverage.py` | Diagnose | `hvg_coverage.json` | is the gene panel missing the responding genes? run before any `--n-hvg` rebuild |
| `02_probe_cell_encoder.py` | Diagnose | `stage_a_probe.json` | linear probe: is the frozen encoder any good? |
| `11_diagnose_variance.py` | Diagnose | `variance.json` | splits predicted spread into decoder versus latent |
| `14_ceiling_analysis.py` | Diagnose | `ceiling.json` | **oracle substitution**: what is the best any Stage B could do? |
| `15_empirical_epistasis.py` | Diagnose | `epistasis.json` | model-free: how far each double departs from the sum of its singles |
| `12_compare_arms.py` | Decide | a verdict with simultaneous intervals | joint bootstrap over a contrast family |
| `17_eval_bracket_epistasis.py` | Decide | `bracket_epistasis.json` | a pre-committed structural endpoint with a permutation null |
| `run_*.sh` | Vary | arms on a pod | one lever per arm; reuse everything else |

`00a` is a companion to `00` rather than a new number in the sequence, for the same reason the docs use `1a`: it diagnoses the artifact `00` produces, and inserting it as a number would have implied it belongs in the build order. There is no `07`. It was never written, and the gap is left rather than renumbered, because renumbering would invalidate every command in every note and ledger entry that references a script by number.

## Three invariants that hold across all five

These are the load-bearing conventions. Break one and the comparisons across rounds stop meaning anything.

**The frozen-encoder invariant.** `encoder.pt` is trained once and never retrained by a downstream experiment. Every Stage-B and Stage-C variant is built *on top of the same frozen latents*, which is what makes arms from different rounds comparable at all. When an experiment retrains the encoder, it has left the method this ledger scores and become a different method. That is a legitimate thing to do and it is the ledger's named fork, but it must be declared rather than slipped in.

**The single-sourced harness.** Every model is graded by the same function, `run_effect_size_eval(predict_fn, ...)`, and a model participates by supplying a `predict_fn` that returns one perturbation's predicted per-gene expression. The flow, the operator, the operator algebra, the NB-VAE, and the oracle arms of the ceiling analysis all enter through that one seam. This is why a new Stage B costs a loader and a flag rather than a new evaluator, and why nobody can accidentally grade two models on two slightly different metrics.

**Grade locally, against the current gene selection.** Training happens on a pod; evaluation happens on your machine. The `de_genes.json` on a pod volume can be stale, and a blanket fetch once silently overwrote seven corrected reports with wrong ones. Fetch the checkpoint, score it here.

## What the numbering hides

Three things worth knowing before you read a script number as a step number.

**The numbers are chronological, not sequential.** `14` (the ceiling) should have been run before `04` (the flow), and running it late is exactly what cost three rounds. `15` produces a target that `17` consumes, so they are a pair. `02` gates `01`.

**The cheap scripts are the ones that changed the project's direction.** `14` trains nothing and runs in minutes, and it retired an entire lever family by showing the transition had $0.03$ of headroom. `11` trains nothing and it turned "calibration is bad" into a fork with two opposite consequences. The expensive scripts confirmed what the cheap ones had already implied.

**The workflow you are in constrains what your number licenses.** A Score result is a measurement of one model. A Decide result is a claim about a difference. A Diagnose result is a bound, and a bound is not a model: the ceiling's linear readout scores above the baseline and is not something you can ship. Mixing these up is how a skyline gets reported as a win.

---

*Up: [perturbation response](../index.md). The runbook: [Running the pipeline](../running-the-pipeline.md). What was found: [the results ledger](../conditional-flow-jepa/results-ledger.md). The general methodology, written to be reusable outside this project: [Running an experiment you can trust](../../../../docs/experimental-method/index.md).*
