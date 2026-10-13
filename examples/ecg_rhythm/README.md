# ECG rhythm — self-supervised representations of the electrocardiogram

The first runnable instance of the [Time-Series JEPA](../../docs/time_series_jepa/index.md) series on a real physiological signal. The question is a controlled one: does span-masked JEPA pretraining make a fixed encoder more useful for **cardiac rhythm classification** when only a limited number of patients carry labels?

The comparison is built so that its answer can be trusted. The same encoder is trained from scratch and from pretraining. Labelled patient subsets are identical across methods. Test patients stay fixed. Variation across training repeats is reported separately from patient-bootstrap uncertainty. Whether pretraining helps, ties or hurts is an open question here, not an assumption.

**Dataset:** [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) (Wagner et al. 2020, *Scientific Data*; PhysioNet, CC BY 4.0). It holds 21,799 ten-second, 12-lead clinical ECGs from 18,869 patients, with 500 Hz signals and an official 100 Hz copy. Patient-confined stratified folds ship with the data: folds 1–8 train, 9 validation, 10 test.

## Result (study v1)

Span-masked JEPA pretraining helps most when labels are scarce. With labels for 5% of training patients, the fine-tuned pretrained encoder beats the same encoder trained from scratch by +0.078 core macro AUROC (95% interval over test patients +0.064 to +0.091; all five repeats agree), and the advantage shrinks to a tie when every patient is labelled. A frozen pretrained encoder with only a linear readout does best of all at 5 to 10% but stalls as labels grow. Full figures, tables, uncertainty and caveats: [**results/**](results/README.md).

## Status

| Stage | Script | State |
|---|---|---|
| Data layer | `00_process_ptbxl.py` | done |
| Data card | `01_data_card.py` | done |
| Reading real records | `01a_plot_records.py` | done |
| Saved labelled-patient subsets | `02_make_subsets.py` | done |
| Span-masked JEPA pretraining | `03_pretrain_jepa.py` | done (5 repeats) |
| Downstream arms (scratch / fine-tune / probe) | `04_train_downstream.py` | done (60 runs) |
| Learning-rate selection on validation | `05_select_lr.py` | done |
| Paired patient-cluster bootstrap comparison | `06_compare_arms.py` | done |
| Pilot report and grid estimate | `07_pilot_report.py` | done |
| Results page | `08_make_results.py` → [`results/`](results/README.md) | done |
| Calibration, robustness, external shift, attribution | later | planned |

## 0 · Acquire and process PTB-XL

The data is open access; no credentials are needed. The AWS open-data mirror is much faster than the PhysioNet zip:

```bash
export SSLLAB_DATA_ROOT=~/work/data
aws s3 sync --no-sign-request s3://physionet-open/ptb-xl/1.0.3/ $SSLLAB_DATA_ROOT/ecg/ptbxl/raw/1.0.3/
# PhysioNet's SHA256SUMS.txt uses one space between hash and name; shasum wants two
(cd $SSLLAB_DATA_ROOT/ecg/ptbxl/raw/1.0.3 && sed -E 's/^([0-9a-f]{64}) /\1  /' SHA256SUMS.txt | shasum -a 256 -c --quiet)
```

Build the 100 Hz cache in the lake and link it into `data/`:

```bash
python examples/ecg_rhythm/00_process_ptbxl.py --fs 100 --out-dir $SSLLAB_DATA_ROOT/ecg/ptbxl/cache100
python -c "from ops.datasets import link_dataset; link_dataset('ecg/ptbxl/cache100', link_name='ptbxl100')"

# no network, seconds: a synthetic PhysioNet-shaped tree through the same code path
python examples/ecg_rhythm/00_process_ptbxl.py --smoke
```

No extra dependencies are needed. WFDB signals are read by a small numpy reader ([`ssllab.data.wfdb16`](../../src/ssllab/data/wfdb16.py)), and it verifies every record's per-lead checksum.

### Cache contents (`data/ptbxl100/`)

| File | Contents |
|---|---|
| `signals.npy` | `int16 (21799, 12, 1000)`, microvolts, memory-mapped |
| `records.json` | per record: `ecg_id`, `patient_id`, `strat_fold`, age, sex, quality flags, `scp_codes` |
| `manifest.json` | provenance, leads, sampling rate, rhythm vocabulary, label rules |

## 1 · Data card

```bash
python examples/ecg_rhythm/01_data_card.py        # -> output/ecg_rhythm/reports/data_card.{json,md}
python examples/ecg_rhythm/01a_plot_records.py    # -> output/ecg_rhythm/figures/
```

The data card covers four things:

- the records and patients in each split;
- label semantics: likelihood values, records with no rhythm statement, co-occurring rhythm codes;
- **patient-level** class support in training and validation;
- the same support inside nested label budgets of 1, 5, 10 and 25% of training patients, over several seeded draws.

The label set and the smallest budget for the main comparison are fixed from these counts **before any model is evaluated**, and never revised from model performance.

## Label rules

These are fixed in [`ssllab.data.ptbxl`](../../src/ssllab/data/ptbxl.py):

- **Present** means the code appears in `scp_codes`, whatever its likelihood. PTB-XL stores likelihood 0 for many rhythm statements, where it means "not graded".
- A record with **no rhythm statement** is unknown. It is excluded from rhythm tasks, never counted as negative.
- **AF pilot, primary rule.** Exclude unknown records and any record carrying AFLT (atrial flutter), AFIB+AFLT co-labels included. Among the rest, AFIB decides positivity.
- **AF pilot, sensitivity rule.** Flutter-only records count as negatives, and any AFIB record stays positive.
- **"Non-AF" is a clinical population**, not a healthy-control group. It includes other arrhythmias, paced rhythms, and sinus records with diagnoses such as MI or conduction disease.
- **Label budgets** are patient-level and nested. One seeded permutation of all training patients decides who carries labels, so subsets are paired across methods and tasks.

## Consuming the loader

```python
from ssllab.data.ptbxl import get_ptbxl_dataloaders
from ssllab.data.signal import SignalGeometry

geom = SignalGeometry(fs=100, duration_s=10, patch_s=0.25, span_s=1.0)   # physical time
train, val, test = get_ptbxl_dataloaders(task="af_primary", leads="all", geometry=geom,
                                         label_frac=0.10, budget_seed=0)
batch = next(iter(train))
batch["tokens"].shape   # (B, 40, 300): 40 tokens of 250 ms x 12 leads x 25 samples
train.meta              # geometry, leads, task, label names, budget, record counts
```

`task` is one of the following:

- `af_primary` or `af_sensitivity`: the binary AF pilot under the primary or sensitivity rule;
- `rhythm`: one sigmoid per rhythm statement;
- `none`: no labels, every record (for pretraining).

`leads="II"` gives the single-lead input, with tokens of shape `(B, 40, 25)`. A geometry whose sampling rate does not match the cache is refused: rates are compared by retraining, never by feeding a 100 Hz model 500 Hz input.

## 2 · Study v1: does pretraining help at a fixed label budget?

The study was frozen in [`configs/study_v1.json`](configs/study_v1.json) before any model was evaluated, and that file stays byte-identical. Two **validation-informed amendments**, made after the pilot, widen the shared learning-rate grid for all arms: first to {1e-4, 3e-4, 1e-3, 3e-3} ([`configs/study_v1_amend1.json`](configs/study_v1_amend1.json)), then, because the scratch baseline selected the top rate, to the final {1e-4, 3e-4, 1e-3, 3e-3, 1e-2} ([`configs/study_v1_amend2.json`](configs/study_v1_amend2.json)), with a pre-declared rule that the grid is not extended again. Each names its base by hash, may change only allow-listed fields, and pins every reused artifact (the saved patient subsets and the repeat-0 pretraining checkpoint) by SHA-256.

Every artifact records hashes of the settings it depends on. Scripts reuse an artifact only when nothing relevant to it changed, record why, and stop on any unexplained incompatibility. Pretraining is not resumable (the checkpoint holds the student encoder and predictor only), so an interrupted run restarts from scratch; finished runs are skipped.

| | |
|---|---|
| Core labels | SR, AFIB, STACH, SARRH, SBRAD (fixed across methods, budgets and repeats) |
| Budgets | 5, 10, 25, 100% of all 15,023 training patients |
| Model | one 12-output sigmoid head on a mean-pooled `JEPAEncoder` (40 tokens of 250 ms) |
| Primary comparison | the same encoder trained from scratch vs. JEPA-pretrained and fully fine-tuned |
| Secondary readout | frozen pretrained encoder with a trained linear head |
| Repeats | five: each changes the labelled-patient subset (nested within the repeat), the downstream seed, and the JEPA pretraining run |
| Headline metric | core macro AUROC at recording level; method differences with a paired patient-cluster bootstrap; repeat-to-repeat spread reported separately |
| AF operating point | AFIB output on the primary AF cohort; threshold = highest with validation sensitivity ≥ 0.90, applied unchanged to test |

Pretraining is inductive: it sees training-fold recordings only. A pilot (repeat 0, validation only; `run_ecg_pilot_pod.sh`, kept as the record) checked the pipeline and measured runtime. The amended search and the full grid then run as:

```bash
python examples/ops/ops_stage_data.py data/ptbxl100          # once, onto the pod's volume
ssh <c> 'cd /runpod-volume/ssl-lab/data && mv cache100 ptbxl100'   # staging names the copy after the symlink target
python examples/ops/ops_run_pipeline.py --execute --cluster <c> -- bash examples/ecg_rhythm/run_ecg_lrsearch_pod.sh   # amended grid search
python examples/ops/ops_run_pipeline.py --execute --cluster <c> -- bash examples/ecg_rhythm/run_ecg_grid_pod.sh       # full grid
python examples/ecg_rhythm/06_compare_arms.py --tag main --split test                                              # bootstrap, locally
```
