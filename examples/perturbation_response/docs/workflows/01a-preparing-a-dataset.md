# 1a. Preparing a dataset: what `00` does, and what changes when you change it

*A companion to [Build](01-build.md), which introduces `00_process_norman.py` in a paragraph and moves on. This chapter is the paragraph expanded: the eight transformations a raw count matrix goes through before the encoder ever sees it, the three knobs worth turning, and the blast radius of turning them. The last part is the one that costs money if you get it wrong.*

---

## Why this step is separate from everything else

`00` is the only script in the repository that imports `scanpy`, `anndata`, or `pertpy`. That is deliberate and it is enforced by structure rather than by discipline: the runtime loader [`ssllab.data.perturbseq`](../../../../src/ssllab/data/perturbseq.py) depends on nothing but torch, numpy, and json, so `pytest` and every training script stay light and fast. All the heavy single-cell machinery is quarantined in this one script, which runs once, on CPU, and writes a cache everything downstream reads.

The consequence is a hard seam. `00` speaks biology and produces an artifact; everything after it speaks tensors and consumes one. The artifact is the contract, and the manifest is where the contract is written down.

Because the single-cell stack is an optional extra, install it before running anything here:

```bash
pip install -e ".[perturb]"          # anndata, scanpy, scikit-misc, pertpy
# or the whole environment:  mamba env create -f environment.yml
```

`scikit-misc` is the non-obvious one. It supplies the loess fit that `flavor="seurat_v3"` needs, and without it the failure happens deep inside HVG selection rather than at import.

## The transformation chain

Eight steps, in the order [`process()`](../../00_process_norman.py) runs them. The right-hand column is the real Norman 2019 run, which is the worked example throughout this chapter.

| # | step | what it does | Norman 2019 |
|---|---|---|---|
| 0 | **counts to `.X`** | if a `counts` layer exists, promote it to `.X` | the download's `.X` is already log-normalized, so this matters |
| 1 | **QC** | drop genes in $<3$ cells, cells with $<500$ genes, cells over $10\%$ mitochondrial reads | $111{,}255 \times 19{,}018 \rightarrow 109{,}737 \times 19{,}018$ |
| 2 | **library size** | total UMI per cell, over the **full** gene set | $(109{,}737,)$ float32, a covariate the decoder needs |
| 3 | **HVG selection** | `seurat_v3` on raw counts, keep the top $n$ | $19{,}018 \rightarrow 5{,}000$ genes |
| 4 | **normalize** | CP10K then $\log(1+x)$, raw counts preserved in a layer | two matrices, same shape |
| 5 | **perturbation coding** | canonical labels, integer ids, control mask | $237$ perturbations |
| 6 | **differential expression** | Wilcoxon vs control, top-$k$ genes per perturbation | $50$ genes each, the scoring seam |
| 7 | **splits** | train/val/test at the *perturbation* level, and a per-cell sanity split | combo: $197 / 20 / 20$ perturbations |
| 8 | **densify and write** | dense float32 features, int32 counts, the manifest | $4.1$ GB npz, $2.0$ GB h5ad |

Four of these are worth more than a table row.

**Step 1 removes cells, not genes, on this dataset.** All $19{,}018$ genes are already detected in at least three cells, because the pertpy artifact was filtered upstream. The $1.4\%$ cell loss is the mitochondrial and complexity filters. Do not assume this holds for a new source.

**Step 2 runs before step 3, and that ordering is load-bearing.** Library size is the total UMI count of a cell across *all* genes, and it is what the negative-binomial decoder uses to convert a predicted rate into a predicted count. Computing it after the HVG subset would give the total over $5{,}000$ genes instead of $19{,}018$, silently rescaling every prediction the decoder makes.

**Step 3 selects on raw counts, and step 4 normalizes afterward.** `seurat_v3` fits a mean-variance trend on counts and ranks genes by variance-stabilized variance, so feeding it normalized data would be selecting on data that has already had the trend removed. This is also why step 0 exists: the raw download's `.X` is log-normalized, and using it directly would corrupt selection. What HVGs are and why the mean-relative comparison is the right one is covered in the docstring of [`make_gene_partition`](../../../../src/ssllab/data/perturbseq.py) and in [Reading Perturb-seq, chapter 2](../reading-perturb-seq/02-reading-the-dataset.md).

**Step 6 is not a training input.** The DE gene list is the *scoring seam*: the per-perturbation genes the effect-size metric is computed on. No stage trains on it. That separation is what makes `--de-only` possible, and it has its own chapter, [3e](../conditional-flow-jepa/3e-the-genes-the-metric-scores.md), because a bad DE selection once made the whole benchmark score models on a matrix of zeros.

Note what the chain does **not** contain: tokenization. `00` only writes the token *geometry* into the manifest and asserts the partition is well-formed. The actual gather from a $(B, 5000)$ gene vector into $(B, 50, 100)$ tokens happens at train time, in the loader's collate function, from a seed. That is why changing `--n-tokens` alone does not require rewriting the multi-gigabyte cache.

## What comes out

```
data/<artifact>/
  tokens_meta.npz     4.1 GB   hvg_X, counts, libsize, pert_id, is_control,
                               ctrl_group, split_combo, split_cells, gene_ids, pert_names
  splits.json          5 KB    which perturbations are train / val / test
  de_genes.json      1.5 MB    the scoring seam
  manifest.json        600 B   every knob that produced the above
  processed.h5ad     2.0 GB    the biology-native artifact, NOT read at train time
```

Two of these deserve attention. `manifest.json` is how geometry propagates: the loader reads `n_tokens` and `partition_seed` from it, derives `token_dim` from the panel width, and hands both to the model builders. Nothing downstream hardcodes a shape. And `processed.h5ad` is never touched at train time, existing so that biology-side questions can be asked in the native format and so `--de-only` has something to recompute from.

## Knob 1: a wider gene panel

The question this answers is "is the $5{,}000$-gene panel discarding perturbation signal?"

```bash
python examples/perturbation_response/00_process_norman.py \
  --source h5ad --h5ad data/norman_2019.h5ad \
  --n-hvg 10000 --n-tokens 50 \
  --artifact norman2019_hvg10k
```

**`--artifact` is not optional here.** The default is `norman2019`, so omitting it overwrites the existing cache in place and every checkpoint trained against it is now scored on a panel its encoder never saw. Name the artifact for the thing that changed.

**Keeping `--n-tokens 50` is what makes this a one-variable experiment.** `token_dim` is derived, not configured: $\lceil 10000 / 50 \rceil = 200$, so tokens become $(B, 50, 200)$ and `PatchEmbed` becomes `Linear(200, 256)`, $+26$k parameters. The transformer trunk is untouched because the sequence is still $50$ long, the masking ratio is still $12/50$, and the JEPA task is therefore identical. Only the input resolution per token changed.

Three costs to know before starting.

**Memory is the likely blocker.** [`load_cache`](../../../../src/ssllab/data/perturbseq.py) uses plain `np.load` with no `mmap_mode`, so `hvg_X` (float32) and `counts` (int32) both land fully in RAM: $109{,}737 \times 10{,}000 \times 8 = 8.8$ GB, up from $4.4$ GB. Processing peak is worse, since step 8 densifies both matrices while the AnnData is still live. Budget around $20$ GB transient, and stage the result under `SSLLAB_DATA_ROOT` rather than in the repo tree.

**Effect-size numbers stop being comparable to every existing result.** DE is computed *within* the panel, so a $10{,}000$-gene panel produces a different scored gene set per perturbation. A $10$k effect size of $0.65$ and a $5$k effect size of $0.65$ are not the same measurement. The oracle ceiling of $0.679$, the flow-versus-VAE table, the $0.852$ linear readout: none of them transfer. A wider-panel run needs its own re-derived baselines, which roughly doubles the experiment rather than adding one arm to it.

**Rank $10{,}000$ of $19{,}018$ is over half the transcriptome.** At that depth the ranking is largely separating noise from noise, so "more genes" and "more signal" are not the same thing. The negative-binomial decoder also doubles its output width, and the decoder is already the identified effect-size bottleneck, so this widens the constrained end of the pipeline.

There is a cheaper way to ask the same question first, and it has its own script:

```bash
python examples/perturbation_response/00a_probe_hvg_coverage.py --artifact norman2019
```

It computes DE on the full $19{,}018$ genes from the raw file and measures what fraction of each perturbation's top-DE genes already fall inside the current $5{,}000$-gene panel. High coverage means the panel is not the problem and the rebuild will teach you nothing. Low coverage gives you a quantified reason to rebuild, and the report's union figure tells you whether a *targeted* panel, the HVG set unioned with the DE set, beats a larger unsupervised one. Details in [Diagnose](03-diagnose.md).

## Knob 2: token geometry

`--n-tokens` sets the sequence length the transformer sees, and `token_dim` follows from it. The panel width $H$ and the token count $T$ fix the group size at $\lceil H/T \rceil$; there is no third degree of freedom.

| `--n-tokens` | token_dim at $H = 5000$ | sequence length | attention cost | `--n-target` for a $24\%$ mask |
|---|---|---|---|---|
| $25$ | $200$ | $25$ | $0.25\times$ | $6$ |
| $50$ | $100$ | $50$ | $1\times$ | $12$ |
| $100$ | $50$ | $100$ | $4\times$ | $24$ |

Attention is quadratic in sequence length, so token count is the expensive axis and token width is the cheap one. Widening tokens costs only the `PatchEmbed` projection.

The deeper trade-off is what a token *is*. More tokens with fewer genes each gives the encoder finer-grained units to relate but makes each one a noisier observation, since a group of $50$ genes in a sparse count matrix is mostly zeros. Fewer, wider tokens give each unit a more stable signal but less for attention to work with. Since the partition is random, a token has no biological meaning at any size, which is a design choice the method deliberately makes and [6a](../conditional-flow-jepa/06a-the-tokenization-design-space.md) examines at length.

One practical note: `--n-tokens` is recorded in the manifest and the partition is regenerated from `(n_hvg, n_tokens, seed)` at load time, so changing it does not require rewriting `tokens_meta.npz`. It does require a new artifact directory anyway, because the manifest is what the loader reads, and it does require retraining the encoder.

## Knob 3: a different dataset

`--source h5ad --h5ad path/to/other.h5ad` is the entry point, and the script tries to meet a new dataset partway:

- **Perturbation column** is auto-detected from `perturbation`, `perturbation_name`, `condition`, `guide_identity`, `gene`, and `--pert-col` overrides.
- **Labels are canonicalized** by [`normalize_label`](../../00_process_norman.py): non-targeting aliases such as `ctrl`, `nt`, `non-targeting`, `neg` all become `control`, and combos are alphabetized and `+`-joined, so `KLF1_CEBPA` and `CEBPA+KLF1` land on the same label.
- **Combo separator** is auto-detected among `+`, `_`, `|`, and `--combo-sep` overrides.

Four assumptions are *not* negotiable without editing the script, and they are the real limits of "point it at another dataset."

**Raw integer counts must be reachable**, in `.X` or in a `counts` layer. The whole pipeline is built on a count likelihood; there is no path for a dataset distributed only as normalized values.

**A control population must exist.** Step 5 raises if no cell normalizes to `control`. Everything downstream depends on it: the decoder's baseline rate profile, the transport flow's source distribution, the DE contrast, and the effect-size metric are all defined relative to control.

**One batch is assumed.** `ctrl_group` is hardcoded to zeros with the comment "Norman: single batch, one pool." For a multi-batch or multi-cell-line dataset this is wrong, and control pairing would draw baselines from the wrong batch. Supporting that means populating `ctrl_group` from a real covariate column, which is a small change with correctness consequences worth thinking through.

**Gene symbols are used for mitochondrial detection** via the `MT-` prefix, which is human convention. A mouse dataset using `mt-` survives, since the check upper-cases, but Ensembl IDs as `var_names` would silently detect zero mitochondrial genes and skip that filter.

Beyond those, the perturbation type matters for interpretation rather than for the code. Norman is CRISPRa, meaning genes are activated. A CRISPRi or knockout dataset runs through the same chain and produces a valid cache, but the `geneset` condition encoder's compositional assumption, that a combination is built from its single-gene parts, is a claim about the biology that should be re-examined rather than inherited.

## What a new cache invalidates

This is the part worth being precise about, because the intuitive answer is wrong in an expensive direction.

The intuition says: a new gene panel changes the encoder's input, so retrain Stage A; the flow and the decoder work on latents, so they should survive. The first half is right. The second half is backwards for the decoder and only half right for the flow.

```
00 cache ──→ 01 encoder.pt ──┬──→ 03 count_decoder.pt   (G2)
   │            (FROZEN)     ├──→ 04 cond_flow.pt       (G1)
   │                         ├──→ 13 operator.pt
   │                         └──→ 16 operator_algebra.pt
   └──→ 08 cvae_baseline.pt  (no encoder, end to end)
```

Everything hangs off `encoder.pt`. That is the [frozen-encoder invariant](01-build.md) and it is what makes arms comparable across rounds, but it also means a new cache invalidates the entire spine. Separate the two reasons.

| stage | what its shape depends on | shape changes when | must retrain when |
|---|---|---|---|
| Encoder, Stage A | `token_dim` $= \lceil H/T \rceil$ | $H$ or $T$ changes | any cache change |
| Decoder G2, `03` | `n_genes` $= H$, the **output** width | $H$ changes | any cache change |
| Flow G1, `04` | `embed_dim` $= 256$; plus `n_perts` for `table`, gene vocabulary for `geneset` | the perturbation set changes | any cache change |
| Baseline, `08` | `n_genes` $= H$ | $H$ changes | any cache change |
| Metric, `06` / `10` | `de_genes.json` | any cache change | rescore always |

**The decoder is the most affected stage, not the least.** Its final layer is literally $H$ wide, at [`03_train_count_decoder.py`](../../03_train_count_decoder.py): `n_genes = train_loader.meta["n_hvg"]`. Going to $10$k doubles its output. And even at an unchanged panel *width*, a re-selected panel means output unit $j$ now names a different gene, so the weights are meaningless regardless of whether the shape happens to match.

**The flow escapes the shape change but not the retraining.** `VelocityMLP` operates on $z \in \mathbb{R}^{256}$, and $256$ is `embed_dim`, which has nothing to do with $H$. So a wider panel leaves the flow's architecture untouched. It still must be retrained, for two reasons that are easy to miss. The flow is fit to the latent distribution of one specific encoder, and its checkpoint stores the normalization statistics of that distribution; a new encoder produces a differently-shaped cloud and the stored statistics are wrong. Separately, the *condition* encoder does change shape when the dataset changes: `table` is `nn.Embedding(n_perts, cond_dim)`, and `geneset` builds its vocabulary from the target-gene names, so a new perturbation set resizes both.

So the honest summary is that the flow is the only stage whose *core architecture* survives a panel change, and no stage survives without retraining. A new cache is a new pipeline, not a new input to an existing one.

**One gotcha that will bite.** All sixteen downstream scripts default `--artifact` to `norman2019`. A new cache means passing `--artifact norman2019_hvg10k` to *every* one of them, and forgetting it on a single evaluation script scores a new model against the old panel's DE genes. The failure is silent, since the shapes all match. Set it in a `run_*.sh` wrapper rather than typing it sixteen times.

## The one change that invalidates nothing

If only the *scoring* gene selection needs to change, and not the panel, the splits, or the tokens:

```bash
python examples/perturbation_response/00_process_norman.py --de-only --artifact norman2019
```

This recomputes `de_genes.json` from the cached `processed.h5ad`, backs up the previous selection to `de_genes.prev.json`, and leaves `splits.json` and `tokens_meta.npz` untouched. **Every trained checkpoint stays valid.** Only the evaluators need re-running.

This is the payoff of keeping the scoring seam in its own file, out of the training path. It is the difference between a correction that costs minutes and one that costs a full rebuild plus every downstream retrain.

## Checklist before a rebuild

1. **Name the artifact for what changed.** `norman2019_hvg10k`, not `norman2019_v2`. Six months later the name is the only documentation you will actually read.
2. **Check whether `--de-only` is enough.** If only the scoring genes are in question, stop here.
3. **Diagnose before building.** Run [`00a_probe_hvg_coverage.py`](../../00a_probe_hvg_coverage.py). Coverage of full-transcriptome DE genes by the current panel answers the wider-panel question in one CPU run instead of a few days. This is the same lesson [Build](01-build.md) opens with and the one this project learned three times.
4. **Budget the memory.** Roughly $H \times N \times 8$ bytes resident, and about twice that transient during processing.
5. **Stage it in the data lake.** `SSLLAB_DATA_ROOT`, symlinked in with `ops.datasets.link_dataset`, not committed into the repo tree.
6. **Smoke test the chain first.** `--smoke --data-dir /tmp/smoke --artifact smoke_test` runs the entire pipeline on synthetic data in about a minute and catches an environment problem before a multi-hour run does.
7. **Declare the fork.** A new cache means new baselines. Record in the [ledger](../conditional-flow-jepa/results-ledger.md) that the scoreboard restarted, because numbers from two panels sitting in one table are not a comparison.

---

*Up: [the workflow map](index.md). Companion to: [Build](01-build.md). What the data is: [Reading Perturb-seq](../reading-perturb-seq/index.md). Why tokens look like this: [the tokenization design space](../conditional-flow-jepa/06a-the-tokenization-design-space.md). What the metric scores: [3e](../conditional-flow-jepa/3e-the-genes-the-metric-scores.md).*
