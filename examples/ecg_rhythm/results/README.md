# ECG rhythm results

Committed results for the [ECG rhythm](../README.md) package. The package develops self-supervised representations of the electrocardiogram and tests them on cardiac rhythm classification. Methods are developed one controlled setting at a time, and each setting gets its own folder here.

Each setting asks one question. It fixes its protocol before any model is scored, and reports its results with their uncertainty and limits. Raw run outputs stay under `output/` and are not committed. What lives here is the interpretation, the figures and the curated metrics, regenerated from the frozen run outputs by a script, so numbers are never typed by hand.

## Settings

| # | setting | question | models | status | headline |
|---|---|---|---|---|---|
| 01 | [JEPA label efficiency](01-jepa-label-efficiency/README.md) | Does span-masked JEPA pretraining reduce the labelled ECGs a rhythm classifier needs? | same transformer encoder trained from scratch, JEPA-pretrained then fine-tuned, JEPA-pretrained with a frozen linear probe | complete (2026-10-12) | Fine-tuning beats scratch by +0.078 / +0.060 / +0.028 core macro AUROC at 5 / 10 / 25% of labelled patients, and ties with every patient labelled; the frozen probe is best at 5 to 10%. |

## What the settings share

Unless a setting says otherwise, the following hold.

- **Data.** The data is [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/): 21,799 ten-second, twelve-lead clinical ECGs from 18,869 patients.
- **Patient-level splits.** PTB-XL's folds keep each patient in one fold: folds 1–8 train, fold 9 validation, fold 10 test. Choices of every kind (learning rates, early stopping, thresholds) are made on validation.
- **The prediction task.** A multi-label prediction over PTB-XL's twelve rhythm statements. Scoring focuses on five core labels: sinus rhythm, atrial fibrillation, sinus tachycardia, sinus arrhythmia and sinus bradycardia. A statement counts as present whatever its recorded likelihood; a recording with no rhythm statement is left out of the rhythm labels rather than counted as negative.
- **Uncertainty.** Uncertainty is reported two ways and the two are never merged. A paired patient-cluster bootstrap covers which test patients happened to be sampled. Repeat-to-repeat spread covers training randomness.

A caution for later settings: setting 01 has examined test fold 10. A new setting should make every choice on validation alone. A claim of improvement over an earlier setting is best confirmed on data that neither setting has seen.

## Folder layout

```
results/
  README.md                      this index
  01-jepa-label-efficiency/
    README.md                    question, task, design, results, discussion, next steps
    figures/                     figures shown on the page
    metrics/                     curated numbers (summary.json) and per-run tables (CSV)
```

A new setting takes the next number and a short name for its question, `NN-<approach>-<question>`, and follows the same layout. Each setting's README is self-contained: it restates the task and design it used, so it can be read without the others.
