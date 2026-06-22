# Reading Perturb-seq — the Norman 2019 dataset, end to end

*A four-part tutorial on the data that feeds the generative model: what a single-cell perturbation experiment measures, what the "intervention" actually is biologically, how a cell becomes tokens the JEPA encoder can read, and how we know whether a model got the answer right.*

> **Why this series exists.** Before training anything, you should be able to look at the raw dataset and say *what every number means, what the model is being asked to predict, and how success is scored.* The [generative-JEPA design-space series](../../../docs/generative_jepa/index.md) decided **what** to build (a conditional flow prior over JEPA latents, with a count decoder) and **why** ([the verdict](../../../dev/generative_jepa/QA/verdict-which-route-to-build-first.md)). This series is the companion that grounds that decision in the actual data — Norman et al. 2019, the benchmark the whole perturbation-biology field is scored on.

Every number in these chapters is real. They come from a small, reproducible slice of Norman 2019 that we processed with the pipeline in this folder — so you can run the same commands and see the same things.

## The worked-example slice

The full Norman 2019 dataset is ~111,255 cells across 287 perturbations — multi-gigabyte. To make the tutorial runnable and concrete, we subsampled it to a **10-perturbation slice** (`control` + 6 single-gene perturbations + 3 two-gene combinations), capped at ~400 cells each → **3,945 cells after quality control**, with a **2,000-gene** highly-variable panel. The slice is chosen so every combination's two constituent singles are *also* present — which is what makes the headline "predict an unseen combination" experiment possible.

Reproduce it (needs `pip install -e ".[perturb]"`):

```bash
python examples/perturbation_response/00_process_norman.py \
  --source pertpy --artifact norman_tutorial --n-hvg 2000 \
  --subsample-perts "control,CEBPE,RUNX1T1,TBX2,TBX3,CNN1,ETS2,CEBPE+RUNX1T1,TBX2+TBX3,CNN1+ETS2" \
  --max-cells-per-pert 400 --val-frac 0.34 --test-frac 0.34
```

The production run drops the subsample flags and uses `--n-hvg 5000` on all 287 perturbations.

## The four chapters

1. **[What Perturb-seq measures, and what the intervention *is*](01-what-is-perturb-seq.md)** — the experiment, CRISPR activation, why single + combination perturbations, and the biological meaning of the condition the model is steered by.
2. **[Reading the dataset as model input](02-reading-the-dataset.md)** — counts, library size, dropout, controls, and the train/validation/test split that defines the prediction task. The numbers, and why each one is shaped the way it is.
3. **[From a cell to tokens](03-tokenization.md)** — how a 2,000-gene expression vector becomes the `(n_tokens, token_dim)` tensor the JEPA encoder consumes, the gene-group partition, and why this is the gene-space analogue of image patches.
4. **[Did the model get it right? — success metrics](04-success-metrics.md)** — effect size (the field's benchmark), differential expression, calibration, and data efficiency — what each measures and why effect size, not state reconstruction, is the one that matters.

> **A note on scope.** This series reads and interprets the *data*. It does not train the model — that is the [generative-JEPA series](../../../docs/generative_jepa/index.md) and the upcoming Stage A encoder. If single-cell vocabulary (counts, dropout, library size) is brand new, the [data-modalities primer](../../../docs/generative_jepa/appendix-data-modalities.md) is a gentler two-minute on-ramp; this series goes deeper and grounds everything in the real slice.
