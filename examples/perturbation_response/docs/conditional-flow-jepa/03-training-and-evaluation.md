# Chapter 3 — Training and evaluation

*How the three stages are trained end to end, the Norman 2019 data and its combination-holdout split, the GPU-pod workflow that keeps the training environment clean, and the evaluation harness that grades effect size and calibration.*

> **Where this sits.** [Chapter 1](01-the-approach.md) gave the idea: a JEPA encoder, a negative-binomial count decoder, and a conditional flow prior, one mechanism for each of the two gaps. [Chapter 2](02-implementation.md) walked the code that realizes each piece. This chapter is the workflow: what actually runs, in what order, on what data, and how we score the result. The numbers themselves belong to [Chapter 4](04-results.md). Here we build the machine that produces them.

## 1. The three stages, and the scripts that run them

The method has three trainable components, and each has its own runnable script in the example folder. We introduce them in the *conceptual* order from [Chapter 1](01-the-approach.md): the encoder first (Stage A), then the decoder that closes the latent-to-data gap (Stage C), then the flow that closes the point-to-distribution gap (Stage B). The stage letters were assigned earlier and are not alphabetical in this order, and the script numbers follow their own file-ordering convention, so neither the letters nor the numbers line up one to one with the order of presentation. Read the letters as the idea and the numbers as the files.

The through-line across all three is the same: the encoder is trained once and then *frozen*, and everything downstream is fit against its fixed latents. That is what makes the stages independent, and it is why the decoder and the flow can each be trained by a plain, single-objective regression.

### Stage A — pretrain the JEPA cell encoder

Script: [`01_pretrain_stage_a.py`](../../01_pretrain_stage_a.py), with the encoder-quality probe in [`02_probe_cell_encoder.py`](../../02_probe_cell_encoder.py).

Stage A is self-supervised and uses no perturbation labels. A cell arrives already tokenized as $50$ gene-group tokens. The objective masks a subset of those tokens and predicts their embeddings from the visible ones, the same I-JEPA objective the project's MNIST encoder uses. This is the payoff of a modality-agnostic design: the masking is a one-dimensional random token permutation, so nothing about the masking code had to change to move from image patches to gene groups. The cell adapter hands the encoder a $(B, n_{\text{tokens}}, \text{token\_dim})$ tensor, and the existing JEPA trains on it unaltered.

A self-supervised predictor can cheat by collapsing every cell to the same latent, which drives the prediction loss down while destroying all information. To guard against that, a VICReg-style regularizer is added to the loss, weighted by `--reg-coef`. The default weight is $0.04$; setting it to zero disables the guard. [Chapter 3c](3c-the-vicreg-collapse-guard.md) unpacks what that regularizer is and how its two terms work. After each epoch the script logs two collapse diagnostics on a held-out batch, the *effective rank* of the latent covariance and the per-dimension *feature standard deviation*, so a collapsing run is visible early rather than discovered at evaluation time.

The default geometry is an embedding dimension of $256$ and an encoder depth of $6$ transformer blocks, trained for roughly $50$ epochs. The number of masked target tokens per cell defaults to $12$ of the $50$. One split choice matters here and is deliberate: Stage A pretrains on the `combo` split, one of the two data splits (`combo` and `cells`) that [§2 below](#2-the-data-and-the-holdout-split) defines in full. In brief, the `combo` split's *training* side excludes the held-out two-gene combinations entirely. Pretraining on `combo` guarantees that no held-out test combination ever leaks into the representation, so the encoder cannot have quietly memorized an outcome it will later be asked to generalize to. The output is a single checkpoint, `encoder.pt`, carrying the JEPA config and weights, which every downstream stage loads and freezes.

The probe script is a quality gate, not part of the benchmark. It extracts one pooled latent $z$ per cell and asks two questions. First, does a logistic-regression probe recover the perturbation label from $z$ alone, well above the chance rate of $1 / n_{\text{perturbations}}$? Second, has the representation collapsed, judged again by effective rank and feature standard deviation? The probe runs on the `cells` split rather than `combo`, because a random per-cell hold-out keeps the same perturbation vocabulary in train and test and so makes the multiclass probe well-posed. Pretraining on `combo` and probing on `cells` are two independent choices, each matched to its own purpose.

### Stage C — the negative-binomial count decoder

Script: [`03_train_count_decoder.py`](../../03_train_count_decoder.py).

Stage C closes the gap between a latent and actual data, where effect size lives. It loads the frozen encoder, encodes each cell to its latent under `torch.no_grad`, and trains a `CountDecoder` to map that latent back to gene counts. The training signal is the negative-binomial log-likelihood of the *real* integer counts of each cell, evaluated against the decoder's predicted per-gene rates and dispersion, with the cell's library size passed in so the rates are on the right scale. A `--zinb` flag switches to a zero-inflated variant that adds an explicit dropout gate for the extra zeros single-cell counts often carry.

Because the encoder is frozen, the decoder is the only thing learning, and it learns by one clean likelihood objective on a fixed target. The default is roughly $30$ epochs. The output is `count_decoder.pt`. The decoder trains on the `cells` split, matching the in-distribution effect-size test the pipeline runs by default.

### Stage B — the conditional flow

Script: [`04_train_cond_flow.py`](../../04_train_cond_flow.py).

Stage B closes the point-to-distribution gap by learning the conditional velocity field $v_\eta(z, t, c)$ over cell latents. The pipeline is deliberate. Freeze the encoder. Precompute every cell's latent exactly once, so the encoder never runs again during flow training. Standardize those latents to zero mean and unit variance per dimension, a stabilizing step that the sampler later inverts. Then train the velocity field by flow matching on the standardized latents. Each training step draws the baseline latent $z_b$ from the pool of control-cell latents, because Perturb-seq is destructive and gives no paired before-cell for a perturbed one, so the baseline can only be the control *population*. The default is roughly $60$ epochs. The saved checkpoint, `cond_flow.pt`, bundles the velocity field, the condition encoder, the standardization statistics, and the standardized control pool the sampler needs.

Stage B carries the method's main design dials as flags, and each one is a lever we exercise in [Chapter 4](04-results.md):

- `--cond-type {table, geneset}` chooses how the intervention is embedded. The `table` option is a learned lookup vector per perturbation, which works in distribution but cannot embed a combination it never saw. The `geneset` option builds the embedding compositionally from per-gene parts, so an unseen two-gene combination still has a vector assembled from its single-gene pieces.
- `--flow-base {gaussian, control}` chooses what the flow transports from. Under `gaussian`, the source is a noise sample and the condition fuses both parts, $c = (z_b, z_p)$, so the field maps noise to the outcome latent. Under `control`, the source *is* a control latent and the condition is the intervention $z_p$ alone, so the field transports a baseline to an outcome and models the displacement, the effect, directly.
- `--compose {additive, deepsets}` sets how the gene-set embedding combines its per-gene parts, either a plain additive sum where the embedding of $A + B$ is the sum of the embeddings of $A$ and $B$, or a DeepSets refinement on top of that sum.
- `--coupling {independent, ot}` applies only to the `control` base. It decides how source baselines are paired to target outcomes within a minibatch, either an independent random control per target, or a minibatch optimal-transport pairing that straightens the transport paths and lowers the variance of the flow-matching target.
- `--p-drop` sets the condition-dropout probability for classifier-free guidance. During training the condition is randomly dropped with this probability so the same network learns both a conditional and an unconditional velocity field, which sampling then blends to sharpen the conditional response.

### Sampling and evaluation

Script: [`05_sample_perturbed.py`](../../05_sample_perturbed.py) for a single perturbation, [`06_eval_effect_size.py`](../../06_eval_effect_size.py) for the benchmark.

Once all three components exist, generating a response is the full story read left to right. Pick a perturbation. Draw a population of outcome latents from the conditional flow, integrating under that perturbation's condition with the baseline drawn from the control pool. Decode each latent to a gene-count profile with the NB decoder. A thousand draws simulate a thousand responding cells. The sampler exposes the number of integration steps and the classifier-free guidance weight as knobs, and it inverts the Stage B standardization so decoded counts are on the real data scale. The evaluation script wraps this into the effect-size benchmark, covered in Section 4.

## 2. The data and the holdout split

The data are from Norman et al. 2019, a CRISPR-activation Perturb-seq screen. The [Reading Perturb-seq series](../reading-perturb-seq/index.md) develops the biology of this dataset in full, including why the negative binomial is the right count model. Here we need only the shape of the processed cache the pipeline trains on.

After quality control the cache holds $109{,}737$ cells on a highly-variable-gene panel of $5{,}000$ genes. There are $237$ perturbations: one non-targeting control, $105$ single-gene activations, and $131$ two-gene combinations. Tokenization splits each cell's $5{,}000$-gene vector into $50$ gene-group tokens of $100$ features each, using a deterministic random partition of gene indices. That partition is the gene-space analogue of image patchify, and its seed is recorded in the cache so processing and training always agree on the same grouping.

Two splits are precomputed and stored per cell, and choosing between them is choosing what question you are asking.

The `combo` split is the generalization test, and it is the Norman headline. It holds out $20$ two-gene combinations, chosen under one deliberate rule: for each held-out pair `A+B`, both single genes `A` and `B` still appear *on their own* as single-gene perturbations in the training set. So at test time the model has met each gene individually, many times, but has never seen the two applied together.

That rule is what makes the split a clean test of *compositional* generalization — predicting the joint effect of `A+B` from having learned `A` and `B` separately — and a hard one, because two perturbations applied together usually interact: their combined effect is rarely just the sum of the two single effects, so the model has to capture the interaction, not merely add. The rule also makes the test well-posed for the gene-set embedding, which builds a combination's representation additively from its single-gene parts, $z_p(A{+}B) = e(A) + e(B)$: both parts must have been trained for the model to have *any* basis on which to compose the unseen pair. And because the model already holds every ingredient — each single gene — the only thing new at test time is the conjunction, so a success or failure here is attributable to composition specifically rather than to missing information. The training side of `combo` spans $197$ perturbations covering $105$ distinct target genes.

The `cells` split is the in-distribution control. It holds out random cells of *seen* perturbations, so the same perturbation vocabulary appears in both train and test. This split measures whether the method recovers effect size at all when generalization to novel combinations is not being asked of it, and it is the default for the decoder, the flow, and the standard effect-size run.

## 3. The GPU-pod workflow

Training runs on a rented A40 GPU, provisioned through SkyPilot on RunPod. The pattern is worth documenting because a real environment conflict shapes it, and the fix generalizes well beyond this project.

The conflict is this. Processing Norman 2019 from raw data needs the heavy single-cell stack, `pertpy` and `scanpy` and their dependencies. Installing that stack upgrades `numpy`, `scipy`, and `scikit-learn` to versions from a newer era than the pod image's PyTorch and scikit-learn were built against. The upgrade breaks both. PyTorch, compiled against the older NumPy, fails at `torch.from_numpy` with the message that NumPy is not available, and scikit-learn breaks against the upgraded SciPy. So the very install that prepares the data poisons the environment that trains on it.

The fix separates the two environments cleanly. Preprocess the dataset exactly *once* into a compact, torch-native cache. That cache is small and portable: a single `tokens_meta.npz` holding the normalized expression, the raw counts, library sizes, perturbation labels, control flags, and both split assignments, plus a few small JSON files for the perturbation-level splits, the top differentially-expressed genes per perturbation, and the provenance manifest. The data adapter that training imports reads only that cache and depends on nothing heavier than torch, NumPy, and JSON. The heavy processing lives entirely in the one-time processing script and is never imported at train time.

With the cache in hand, the training pod never installs `pertpy` at all. The cache is staged onto the pod through SkyPilot's file mounts, and every training script sees it already present and runs in the clean base image. The pod scripts keep an on-pod processing path only as a fallback for a fresh pod with no cache staged, and even that path captures the base versions of NumPy, SciPy, and scikit-learn before the heavy install and restores them immediately after processing, so training still starts from a clean stack. The reproducible lesson is the shape, not the shell script: pin the environment-polluting preprocessing behind a single artifact, stage that artifact, and keep the training environment untouched.

One device detail rounds this out. The code resolves its device with an `"auto"` default that prefers CUDA on the pod and falls back to CPU locally. It never auto-selects Apple's MPS backend, which has recurring operator gaps that silently break training, so MPS is only ever used when a run requests it explicitly.

## 4. The evaluation harness

Two axes grade the method, and they are complementary. Effect size asks whether the *mean* shift is right. Calibration asks whether the *distribution* around that mean is right. A model can win on one and lose on the other, and the whole reason to build a full flow rather than a point predictor is the second axis, so both are measured.

### Effect size — the field's benchmark

Code: [`effect_size.py`](../../../../src/ssllab/eval/effect_size.py), the functions `delta_correlation` and `run_effect_size_eval`.

For a perturbation, the effect is its differential expression, the vector

$$
\Delta = \operatorname{mean}(\text{predicted}) - \operatorname{mean}(\text{control}),
$$

where $\operatorname{mean}(\text{predicted})$ is the per-gene mean over a generated response population and $\operatorname{mean}(\text{control})$ is the per-gene mean over control cells. The score is the Pearson correlation $r$ between the predicted $\Delta$ and the true $\Delta$, computed only on the perturbation's top differentially-expressed genes, the genes the intervention actually moved. Each entry of these vectors is one gene's mean normalized expression, on the log1p-CP10K scale. This is the metric scGen, CPA, and scPPDM report, and the number the method must reproduce.

The harness is single-sourced on purpose. `run_effect_size_eval` takes a `predict_fn(pid, name)` that returns one perturbation's predicted per-gene expression, and for every evaluable perturbation it computes the held-out truth as the mean over that perturbation's *test* cells, then scores `delta_correlation` on the top-DE genes. The exact same loop grades the flow, the from-scratch NB-VAE baseline, and any future predictor, so no model can win by being scored on a friendlier harness. Perturbations with too few held-out cells are skipped, and the per-perturbation correlations are aggregated to a mean and a median. The generalization number in [Chapter 4](04-results.md) comes from running this harness on the held-out cells of the `combo` split.

### Calibration — the distributional axis

Code: [`calibration.py`](../../../../src/ssllab/eval/calibration.py).

Effect size is blind to spread. Two models with an identical mean $\Delta$ can disagree completely on how much a perturbed population varies, and on which genes vary. That is precisely where a flow, which represents a full distribution over latents, should beat a cruder generator, so calibration is the axis that decides whether the generative machinery earns its keep once the means tie. Four reads, all computed on a perturbation's top-DE genes by comparing a generated population against the held-out real cells, cover it:

- **Per-gene spread correlation** is the Pearson correlation between the predicted and true per-gene standard deviation. It asks whether the model knows *which* genes vary in the response.
- **Central-interval coverage** is the fraction of true cells that fall inside the model's central predicted interval per gene. The ideal value is the nominal width of that interval. Below nominal means the predicted population is too tight and the model is over-confident; above means too diffuse and under-confident, and the sign of the gap says which way.
- **Mean 1-Wasserstein** is the average earth-mover distance between the predicted and true per-gene distributions, one holistic number folding mean, spread, and shape together, where lower is better.
- **Energy distance** is a multivariate two-sample distance on the *joint* top-DE gene space. Unlike the per-gene reads it sees gene-gene correlations and multimodality, the joint structure a rich latent flow can capture but marginals cannot, and it is zero exactly when the two populations are identically distributed.

One measurement subtlety governs all four and is easy to get wrong. Measuring spread requires *sampling counts* from the decoder's negative binomial, not reading its predicted rates. A population of decoded rates has almost no spread of its own, because the flow's latent variation compresses through the decoder into nearly the same rate vector, so scoring rates would make every model look absurdly over-confident. The spread that calibration measures is the count-level stochasticity of the NB likelihood, and it only appears once counts are actually drawn. What each metric *reports* on the real data is [Chapter 4](04-results.md)'s job. Here the point is only what each one measures, and the one sampling step you must not skip to measure it honestly.

---

*Previous: [Chapter 2 — Implementation](02-implementation.md). Up: [the method series](index.md). Next: [Chapter 4 — Results](04-results.md).*
