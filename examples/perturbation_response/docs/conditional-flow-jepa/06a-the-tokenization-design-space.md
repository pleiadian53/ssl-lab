# Chapter 6a — The tokenization design space: random groups, biology, and bias

*A companion to [Chapter 6](06-beyond-the-current-limit.md), in the same spirit as [3d](3d-the-perturbation-vocabulary.md) for Chapter 3. [Chapter 3 §2](03-training-and-evaluation.md) and [Reading Perturb-seq, Part 3](../reading-perturb-seq/03-tokenization.md) already introduce tokenization: a fixed random partition of highly-variable genes into $50$ tokens, the gene-space analogue of I-JEPA's patchify. This note is the discussion those chapters defer. Why is that default defensible? What alternatives exist? How could each help or hurt? And if you swap schemes, how do you test it without breaking the combo-split discipline?*

> **Where this sits.** Read [Reading Perturb-seq, Part 3 §3](../reading-perturb-seq/03-tokenization.md) first for the mechanics (`make_gene_partition`, `tokenize_cells`, the `(B, n_tokens, token_dim)` contract). [Chapter 6](06-beyond-the-current-limit.md) lists alternative tokenization as one lever among many. This note is the full argument behind that one line.

## 1. Recap: where tokenization sits, and why the choice matters

By the time Stage A runs, each cell has already been turned into a token sequence. The order in the pipeline is:

```
raw counts → HVG panel → log1p-CP10K features → gene partition → tokens → JEPA encoder (Stage A) → latents → flow / decoder
```

Tokenization is **not** part of the flow and not part of the decoder. It is the adapter that turns a flat gene vector into the sequence shape the JEPA transformer already expects from MNIST (the same move as `patchify` for images). Stage A's masked-prediction objective operates on those tokens. Everything downstream (the linear probe, the flow, the NB decoder) inherits whatever representation Stage A learned **given that partition**.

That last phrase is the key. Changing the grouping is not a hyperparameter tweak on the flow. It is a **Stage A design choice**: new partition, retrain the encoder, new latents, then re-run Stages B and C. That is why the full tradeoff discussion lives here, in the forward-looking chapter, while [Chapter 3 §2](03-training-and-evaluation.md) need only name the default and point forward.

## 2. A toy example: what "a token" means in practice

Before comparing schemes, it helps to see one cell at toy scale. Suppose a panel has only six highly-variable genes and we want three tokens (so $d_{\text{tok}} = 2$ genes per token).

**Random partition (seed 42).** Shuffle gene indices once and deal them into groups:

| Token | Genes in this token |
|---|---|
| 0 | `GAPDH`, `MYC` |
| 1 | `CEBPE`, `KLF1` |
| 2 | `TBX2`, `AHR` |

A cell's expression vector `[g₁, g₂, …, g₆]` becomes three small vectors, one per row. The encoder never sees a flat six-vector. It sees three tokens and runs self-attention over them.

**Pathway partition (hypothetical).** Group by a curated module instead:

| Token | Genes in this token |
|---|---|
| 0 | `CEBPE`, `KLF1` (same transcription-factor program) |
| 1 | `MYC`, `TBX2` (cell-cycle related, say) |
| 2 | `GAPDH`, `AHR` (catch-all / housekeeping mix) |

Same six genes, same cell, different tokens. Stage A would see a different input geometry and, after retraining, would produce a different latent $z$. In production the numbers are larger ($5{,}000$ genes, $50$ tokens, $100$ features each), but the logic is identical: **the partition defines what "local structure" means before attention even runs.**

## 3. What the default actually does in code

The default partition is built once, deterministically, from a seed:

```python
def make_gene_partition(n_hvg: int, n_tokens: int, seed: int) -> torch.Tensor:
    group_size = token_dim_for(n_hvg, n_tokens)
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n_hvg, generator=g)
    ...
    return perm.view(n_tokens, group_size)
```

[`tokenize_cells`](../../../../src/ssllab/data/perturbseq.py) then gathers each cell's normalized gene values into those groups. That step is a pure index operation. There is no randomness at runtime.

The seed is written into the cache manifest. The loader **refuses to run** if training's seed disagrees with the cache's, so a trained encoder always sees the same grouping it was trained on.

**Takeaway for a fixed, trained model:** same partition, same features, same frozen encoder gives the same $z$ every time. A different partition seed requires rebuilding the cache and retraining Stage A. That is effectively a different model, not a relabeling of the same one.

## 4. Is random grouping a good default? A qualified yes

Two architectural facts make random grouping defensible.

**Global self-attention.** The JEPA encoder's transformer trunk attends over all tokens. A CNN on a shuffled image is genuinely hurt by a bad patch layout, because early layers only mix nearby pixels. A transformer can relate any gene group to any other in one layer. An arbitrary shuffle does not block the architecture the way it would block a locality-biased one.

**No imported prior.** Random grouping imposes no claim about which genes "belong together." Pathway databases and co-expression modules carry assumptions that may be wrong, stale, or simply not true for K562. Neutral grouping lets the data speak, at the cost of learning all structure from scratch.

The honest cost: each random token is a bag of $40$–$100$ *unrelated* genes. There is no local coherence inside a token for the model to exploit before attention runs. Every useful relationship must be discovered across tokens. Image patches work well for ViT partly because nearby pixels *are* correlated. Random gene bags do not get that head start. So the default is **safe, but plausibly suboptimal** for sample efficiency on a $5{,}000$-gene panel with finite training data.

## 5. What alternatives exist, and four ways each could bias you

[Reading Perturb-seq, Part 3](../reading-perturb-seq/03-tokenization.md) names two families. Here is what each would actually buy or cost, in plain terms.

### Pathway / gene-program groups

Group genes by curated modules (KEGG, Reactome, GO).

**Upside.** Each token could mean something like "how active is this pathway." That is coherent internal structure, closer to the image-patch analogy where a patch already contains correlated pixels.

**Downside (annotation bias).** Databases reflect decades of *uneven research attention*. Well-studied genes get rich, coherent tokens. Poorly annotated genes (often the novel ones) land in vague catch-alls. The model would systematically learn more from well-described genome regions, not because those genes matter more biologically, but because the database described them better.

### Co-expression groups (data-driven)

Cluster genes by how they co-vary across cells in this dataset.

**Upside.** This reflects K562's actual behavior, not someone else's annotation.

**Downside (context and leakage).** Co-expression is state- and cell-type-dependent. Modules fit on the *whole* dataset would include held-out `combo` combinations and leak information into the tokenization before Stage A begins. The pipeline already guards against that class of mistake: Stage A pretrains on the `combo` split's training side only ([Chapter 3](03-training-and-evaluation.md)). Any data-driven grouping must follow the same rule, i.e. **computed strictly from training-split cells**, or it quietly undermines the generalization test.

### The subtlest risk: shortcut learning

If tokens are biologically coherent, the masked-prediction objective becomes partly solvable by learning "recover this known module's activity" without ever relating *different* modules to each other.

Random tokens offer no such shortcut. Predicting a masked token from visible ones *requires* cross-token structure learned via attention. Paradoxically, the "worse" random default may force more genuinely cross-cutting biology to be learned. A pathway-groomed tokenization could make the encoder good at recovering what curators already knew and less good at anything novel or cross-pathway.

## 6. If you run the experiment: the defensible version and what to measure

Given the mechanisms above, the weakest candidate is an external pathway database (annotation bias plus wrong cell line). The strongest is **co-expression modules from `combo`-split training cells only**. That version is biologically grounded, leakage-safe, and implementable as a drop-in replacement for `make_gene_partition` with `"scheme": "coexpression"` in the manifest.

But shortcut learning means the direction of the effect is **not obvious a priori**. The honest test compares random vs. co-expression partitions on the diagnostics the series already uses to judge Stage A:

| Diagnostic | What a tokenization swap would reveal |
|---|---|
| **Effective rank** | Does coherent grouping encourage collapse (fewer independent latent directions)? |
| **Linear probe accuracy** | Is perturbation-relevant information easier or harder to read off linearly? |
| **Combo-generalization $\Delta r$** | Does encoder tokenization change combination generalization at all? Likely minimal: [Chapter 4](04-results.md) shows combo generalization is driven by the gene-set *condition* encoder, not JEPA latents. Still worth checking. |

**The protocol, step by step:**

1. Build two caches with the same cells and splits but different `make_gene_partition` schemes (random vs. leakage-safe co-expression).
2. Train Stage A on each cache (same hyperparameters, same `combo`-split training cells).
3. Freeze each encoder. Re-run Stages B and C from scratch for each.
4. Compare effective rank, probe accuracy, and combo $\Delta r$ on the same held-out combinations.

One invalid shortcut: keep the same trained encoder and swap the partition only at inference. The partition is baked into the weights. That test tells you nothing useful.

## 7. What to carry forward

Tokenization belongs with patchify in the I-JEPA story. It is the preprocessing adapter that defines what "a token" means before self-supervised pretraining begins.

The random default is a reasonable, reproducible choice. Global attention removes the worst architectural penalty, and neutrality avoids importing a possibly wrong biological prior. It is not claimed to be optimal.

Alternatives sit on a spectrum from curated pathways to leakage-safe co-expression modules. Each carries distinct bias risks rather than a single vague "baking in a prior." Because a partition change ripples through the entire stack, the full tradeoff belongs in the discussion chapter ([Chapter 6](06-beyond-the-current-limit.md)), while the training walkthrough need only name the default and point here.

The experiment is cheap relative to the operator route or joint training: swap `make_gene_partition`, retrain Stage A, measure. The outcome is informative either way. If a biologically informed scheme wins on probe accuracy and effective rank, Stage A was leaving signal on the table. If random wins or ties, the neutral default was doing more work than it looked. Either result narrows the design space for making JEPA generative, which is the point of running the comparison at all.

---

*Previous: [Chapter 6 — Beyond the current limit](06-beyond-the-current-limit.md). Up: [the method series](index.md). Related: [Reading Perturb-seq, Part 3](../reading-perturb-seq/03-tokenization.md) · [Chapter 3 §2 — tokenization](03-training-and-evaluation.md) · [Chapter 3c — VICReg](3c-the-vicreg-collapse-guard.md).*
