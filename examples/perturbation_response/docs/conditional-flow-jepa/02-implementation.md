# Chapter 2 — Implementation

*The method as running code: a frozen JEPA encoder, a negative-binomial count decoder, a conditional velocity field with flow matching, two condition encoders, classifier-free guidance, and the sampling path that ties them together.*

Chapter 1 laid out the shape of the method. A frozen representation gives us a latent per cell. We then need two pieces the representation alone does not provide: a way to *sample* new latents conditioned on a perturbation, and a way to turn a latent back into gene counts. Chapter 1 called these G1 (a prior over latents) and G2 (a decoder to data). This chapter walks through the actual modules that implement them, mapping each idea to the code that carries it. Everything below lives under [`src/ssllab/generative`](../../../../src/ssllab/generative/).

## The JEPA encoder, upstream of everything

Before any of the generative machinery runs, a cell is turned into a latent by the Stage-A encoder. This is a joint-embedding predictive architecture (JEPA): it is trained by masking part of the input and predicting the representation of the masked part from the visible part, so the masking is modality-agnostic and the encoder learns structure without ever reconstructing raw counts. For our purposes the encoder is *frozen*. It exposes an `embed` call that maps one cell to a $256$-dimensional latent $z \in \mathbb{R}^{256}$, and every module in this chapter consumes those latents rather than the raw expression matrix. The JEPA is upstream of the whole pipeline: it is trained once, then held fixed while the flow and the decoder are fit on top of its latents.

Two design consequences follow from the encoder being frozen. First, the flow and the decoder both operate purely on vectors, which keeps them simple and modality-agnostic. Second, the latent space is fixed, so the flow's job is to learn a distribution *over that space* rather than to co-adapt a representation with a generator.

## G2: the count decoder

The decoder is the piece that recovers *effect size*: given a latent, it produces a distribution over the gene counts of a cell. It lives in [`count_decoder.py`](../../../../src/ssllab/generative/count_decoder.py).

### Why a count likelihood, not MSE

Single-cell RNA-seq counts are non-negative integers, heavily over-dispersed, and dominated by dropout zeros. A Gaussian decoder trained by mean-squared error is the wrong measurement model for that data. Instead the decoder emits the *parameters* of a count distribution and is trained by the likelihood of the real counts under those parameters. The default distribution is a negative binomial (NB); with `zinb=True` the decoder additionally emits a per-gene dropout gate, giving a zero-inflated NB (ZINB).

### The mean assembly

`CountDecoder.forward(z, library_size)` runs the latent through an MLP trunk and a `rate_head` linear layer, then applies a softmax across genes:

$$\rho = \mathrm{softmax}(\mathrm{net}(z)), \qquad \sum_g \rho_g = 1.$$

Here $\rho \in \mathbb{R}^{G}$ is a *relative gene-rate profile*: a probability distribution over the $G$ genes that says what fraction of a cell's transcripts each gene should get. It carries no notion of how deeply the cell was sequenced. The NB mean is then

$$\mu = \ell \cdot \rho,$$

where $\ell$ is the cell's library size, a *given* covariate rather than something the decoder predicts. Separating the shape $\rho$ from the depth $\ell$ is what keeps the decoder's output comparable across cells sequenced to different depths, and it is why the mean-expression readout later can drop $\ell$ entirely and work from $\rho$ alone.

### Per-gene learned dispersion

The NB needs a dispersion parameter. The decoder holds one learned scalar per gene, stored as `log_kappa`, an `nn.Parameter` of shape $(G,)$, and maps it to a positive value:

$$\kappa = \mathrm{softplus}(\texttt{log\_kappa}) + 10^{-4}.$$

Under this NB parameterization the variance is $\mu + \mu^2 / \kappa$, so a small $\kappa$ means heavy over-dispersion and a large $\kappa$ approaches Poisson. Making $\kappa$ per-gene and *not* a function of $z$ is the common scVI choice: dispersion is treated as a property of each gene's measurement, not of the individual cell.

### The likelihood and its stable form

`nb_nll(x, mu, kappa)` returns the negative binomial negative log-likelihood, averaged over cells. It computes the standard per-gene log-likelihood

$$\log p(x \mid \mu, \kappa) = \log\Gamma(x + \kappa) - \log\Gamma(\kappa) - \log\Gamma(x + 1) + \kappa \log\frac{\kappa}{\kappa + \mu} + x \log\frac{\mu}{\kappa + \mu},$$

rearranged into a numerically stable scVI-style form that shares the $\log(\kappa + \mu)$ term across the two logarithmic pieces. This function is verified against `scipy.stats.nbinom` in the test suite, so the loss really is the NB likelihood and not an approximation of it. `zinb_nll` extends this to the zero-inflated case: a zero can now arise either from the dropout gate firing or from the NB itself producing a zero, and the two routes are combined through a softplus so the mixture stays stable. `CountDecoder.nll` is the training entry point; it decodes $z$ and returns `nb_nll` or `zinb_nll` on the real counts.

### Sampling counts

To generate a cell rather than score one, `sample_counts` draws integer counts from the decoded NB by exploiting the Gamma-Poisson mixture representation of the negative binomial:

$$\lambda \sim \mathrm{Gamma}(\kappa, \kappa/\mu), \qquad x \sim \mathrm{Poisson}(\lambda).$$

The Gamma draw supplies the over-dispersion and the Poisson draw supplies the count noise. With ZINB, each sampled gene is then zeroed with probability $\mathrm{sigmoid}(\texttt{pi\_logits})$. There is also `expected_counts`, which returns the mean $\mu$ directly, scaled by the non-dropout probability under ZINB, for when you want the expectation without sampling.

## G1: the conditional flow

The flow is the piece that makes the frozen latent *sampleable*: it learns a distribution over latents and lets us draw new ones. It lives in [`flow.py`](../../../../src/ssllab/generative/flow.py) and is a rectified flow specialized to flat vector latents.

### The velocity field

The core object is `VelocityMLP`, a network $v_\eta(z, t, c)$ that predicts a velocity in latent space at a point $z$, a time $t \in [0, 1]$, and an optional condition $c$. Time enters through a sinusoidal embedding, which an MLP turns into a scale and a shift that FiLM-modulate the hidden state: $h \leftarrow h \cdot (1 + \mathrm{scale}) + \mathrm{shift}$. When `cond_dim > 0` the condition $c$ is injected the same way through a second FiLM, giving the network a per-condition modulation on top of the per-time one. With `cond_dim = 0` the module is exactly the original unconditional field and $c$ must be `None`; the forward pass raises if the presence of $c$ disagrees with `cond_dim`. The body is a stack of residual MLP blocks, and `out_proj` maps back to the latent dimension so the output is a velocity of the same shape as $z$.

The conditional field also holds a learned `null_cond` parameter, a single condition-shaped token that stands for "no condition." It is inert during ordinary conditional sampling and becomes load-bearing only for classifier-free guidance, described below.

### The rectified-flow path

`linear_interpolant(z0, z1, t)` defines the straight-line path the flow is trained to follow. Given a source point $z_0$ and a target point $z_1$, it returns the interpolated point and the target velocity:

$$z_t = (1 - t) \cdot z_0 + t \cdot z_1, \qquad u_t = z_1 - z_0.$$

The target velocity $u_t = z_1 - z_0$ is constant along the path: on a straight line the direction from source to target never changes. This is the defining simplification of rectified flow, and it makes the regression target a plain displacement vector.

### The flow-matching loss

`cfm_loss(model, z1, c, p_drop, z0)` is the conditional flow-matching objective. For a batch of target latents $z_1$ it samples a time $t$ per row, forms $(z_t, u_t)$ from the interpolant, and regresses the network's prediction onto the target velocity:

$$\mathcal{L} = \big\| v_\eta(z_t, t, c) - u_t \big\|^2.$$

It is a mean-squared regression and nothing more exotic. Two arguments matter for what follows. The source `z0` defaults to `None`, in which case it is drawn as standard Gaussian noise $z_0 \sim \mathcal{N}(0, I)$, giving the usual noise-to-data prior; passing real samples instead makes the loss learn a *distribution-to-distribution* transport from that source population to $z_1$. And `p_drop` randomly replaces a fraction of the conditions with the `null_cond` token, which is what trains the guidance behavior.

### Sampling the ODE

`euler_sample(model, n, dim, n_steps, c, guidance, z0)` integrates the learned velocity field forward in time. It starts at the source ($t = 0$) and takes `n_steps` Euler steps of size $\mathrm{d}t = 1/\texttt{n\_steps}$ up to the data ($t = 1$):

$$z \leftarrow z + \mathrm{d}t \cdot v_\eta(z, t, c).$$

Like the loss, the source defaults to Gaussian noise and can instead be a supplied `z0`. The `guidance` scale controls classifier-free guidance and is discussed next; at `guidance = 1` the sampler just follows the conditional velocity.

### Optimal-transport coupling

`ot_couple(z0, z1)` reorders a batch of source points so that each source is paired with the target it is cheapest to reach, solving the exact minibatch assignment that minimizes $\sum_i \| z_0[i] - z_1[i] \|^2$ via `scipy.optimize.linear_sum_assignment`. Pairing this way straightens the source-to-target paths within a batch, so flow matching regresses a lower-variance target field. It is only meaningful when the source is real data rather than fresh noise, so it belongs to the transport setting.

## The condition: turning a perturbation into $c$

The velocity field neither knows nor cares what $c$ means; it just consumes a vector. The job of building that vector from a perturbation lives in [`condition.py`](../../../../src/ssllab/generative/condition.py), kept separate precisely because it is the perturbation-specific part. Chapter 1 framed the condition as a pair: a baseline state $z_b$, the latent of a control cell, and an intervention identity $z_p$, an embedding of the perturbation. This module offers two ways to build $z_p$, and they differ in exactly one respect that turns out to decide whether the model can generalize to unseen combinations.

### The table encoder

`ConditionEncoder` is the straightforward version. It embeds the integer perturbation label through a learned `nn.Embedding` table to get $z_p$, concatenates it with the baseline latent $z_b$, and passes the pair through a small MLP to produce the condition:

$$c = \mathrm{MLP}([z_b, z_p]), \qquad z_p = \texttt{pert\_emb}(\texttt{pert\_id}).$$

One row of the embedding table per perturbation, learned end to end. This is the natural first choice and it works for perturbations seen during training.

### The gene-set encoder

The compositional alternative replaces that per-perturbation table with a per-*gene* table, so that a perturbation's embedding is built from the genes it targets. Three pieces implement it.

`build_pert_gene_matrix(pert_names)` parses the perturbation vocabulary into a multi-hot membership matrix. Each perturbation name is a `+`-joined set of target genes, so `"CEBPE+RUNX1T1"` targets two genes and `"control"` targets none. The function returns a matrix $M$ of shape $(\texttt{n\_perts}, \texttt{n\_genes})$ with $M_{p,g} = 1$ exactly when perturbation $p$ targets gene $g$, along with the sorted list of distinct target genes that indexes the columns.

`GeneSetEmbedding` holds that matrix as a fixed buffer and a learned per-gene embedding table, and composes $z_p$ by pooling the embeddings of the genes a perturbation targets. The default `compose="additive"` pooling is a plain sum:

$$z_p = \sum_{g \in \text{targets}(p)} e(g),$$

which makes composition exact: $z_p(A{+}B) = e(A) + e(B)$, and `control`, targeting no genes, maps to the zero vector. This is the maximal compositional inductive bias. The `compose="deepsets"` variant instead computes $z_p = \phi\big(\sum_g \psi(e(g))\big)$, a permutation-invariant DeepSets refinement that can capture interactions the pure sum cannot; it is kept as an ablation.

`GeneSetConditionEncoder` is a drop-in for `ConditionEncoder` with the identical `forward(z_b, pert_id) -> c` signature. The only change is that $z_p$ now comes from a `GeneSetEmbedding` instead of a per-perturbation table; the fusion MLP on top is the same.

### Why the gene-set encoder generalizes and the table encoder cannot

This is the crux of the design, so it is worth stating plainly. Suppose the two-gene combination $A{+}B$ was never seen in training, but the single perturbations $A$ and $B$ were.

The table encoder has one embedding row per perturbation. The combination $A{+}B$ is its own label, so it has its own row, and that row was never touched by a gradient. At test time the encoder can only return an untrained, essentially random embedding for it. There is no path from having trained on $A$ and $B$ to knowing anything about $A{+}B$, because the table indexes *combinations*, not their parts.

The gene-set encoder indexes *genes*. The embedding of $A{+}B$ is $e(A) + e(B)$, and both $e(A)$ and $e(B)$ were trained whenever $A$ or $B$ appeared in any perturbation. The unseen combination reuses already-trained parts: its multi-hot row references the same gene columns as the singles it is composed of. Pooling is always over gene slots and never over a per-combination slot, so a held-out combination is embedded from trained material rather than from an untouched row. That single change, from a table indexed by combination to a table indexed by target gene, is what lets the model place an unseen combination sensibly in condition space. Chapter 4 quantifies how much of the observed combination generalization this drives.

## Classifier-free guidance

Classifier-free guidance (CFG) lets one trained conditional flow trade fidelity against diversity at sampling time, without a separate classifier. It rests on the `null_cond` token introduced above and touches training and sampling in one place each.

At training time, `cfm_loss` calls the internal helper that replaces each sample's condition with `null_cond` with probability `p_drop`, exposed on the training script as `--p-drop`. The same network therefore learns both the conditional velocity, when it sees a real $c$, and the unconditional velocity, when it sees the null token.

At sampling time, `euler_sample` with `guidance` not equal to $1$ blends the two velocities at every step:

$$v = v_\eta(z, t, \varnothing) + w \cdot \big(v_\eta(z, t, c) - v_\eta(z, t, \varnothing)\big),$$

where $\varnothing$ denotes the `null_cond` token and $w$ is the `--guidance` scale. This extrapolates away from the unconditional velocity to sharpen the effect of the condition. Setting $w = 1$ recovers plain conditional sampling and leaves the null token inert. The full derivation and the design-space reasoning behind CFG are in the companion note [09b — classifier-free guidance](../../../../docs/generative_jepa/09b-classifier-free-guidance.md).

## Two axes that parameterize the flow

The checkpoint saves two orthogonal choices that together decide how the flow is set up, and every loader and sampler below branches on them.

`flow_base` selects the *source* of the flow. Under `gaussian` the flow starts from noise and generates the outcome latent, with the condition being the fused pair $c = \mathrm{cond}(z_b, \texttt{pert\_id})$; this is the noise-to-data prior. Under `control` the flow instead transports a real control-cell latent toward the perturbed outcome, so the source $z_0$ is a baseline latent and the condition is the perturbation embedding *alone*, $c = \mathrm{cond}(\texttt{pert\_id})$. The transport base exists because anchoring the sample to a real baseline gives the flow a structural advantage the Gaussian base does not; the full reasoning for why transport helps is deferred to Chapters 4 and 5, and here we only describe what the code does.

`cond_type` selects the condition encoder: `table` for `ConditionEncoder` and its per-perturbation embedding, `geneset` for the gene-set composition. The two axes are independent, giving four combinations the loader must be able to rebuild.

## Sampling and loading

[`perturb.py`](../../../../src/ssllab/generative/perturb.py) is the sampling side of the pipeline, and it is where the two axes become concrete control flow.

`load_cond_flow(path)` reads a Stage-B checkpoint and returns a ready bundle: the `VelocityMLP`, the condition module, the standardization statistics `mean` and `std`, the control pool of baseline latents, and the saved `flow_base` and `cond_type`. Rebuilding the condition module is delegated to a helper that walks all four cases. Under the `control` base the condition is the perturbation embedding by itself, so it rebuilds either a bare `nn.Embedding` (table) or a `GeneSetEmbedding` (gene-set). Under the `gaussian` base the condition fuses $z_b$ and the perturbation, so it rebuilds either a `ConditionEncoder` or a `GeneSetConditionEncoder`. The right class is chosen from the checkpoint fields, its weights are loaded, and it is set to eval.

`sample_perturbed_latents(bundle, pert_id, n, ...)` draws $n$ outcome latents for one perturbation. It first samples $n$ baseline latents $z_b$ from the control pool. Then it branches on `flow_base`. For the `control` base it calls `euler_sample` with the baselines as the source `z0` and the condition set to $\mathrm{cond}(\texttt{pert\_id})$, transporting each real baseline toward its conditioned outcome. For the `gaussian` base it starts from noise and conditions on the fused $\mathrm{cond}(z_b, \texttt{pert\_id})$. Either way, the flow works in a standardized space, so the final step de-standardizes with the saved statistics, $z = z_{\text{std}} \cdot \texttt{std} + \texttt{mean}$, returning latents back in the encoder's own space.

Two readouts sit on top of the sampled latents, and they differ deliberately in whether they sample counts.

`predicted_expression(bundle, decoder, pert_id, n, ...)` returns the *mean* predicted expression over a generated population, as a single $G$-vector. Because the decoder's rate profile $\rho$ sums to one and is free of library size, the mean can be read straight from $\rho$ with no count sampling: it returns $\mathrm{log1p}(10^4 \cdot \rho)$ averaged over the $n$ cells, which is directly comparable to the normalized log1p-CP10K expression in the data cache. This is the clean estimate of the mean response and drives the effect-size metric.

`predicted_population(bundle, decoder, pert_id, n, library_size, ...)` instead returns a per-cell population of shape $(n, G)$, and here it *does* sample counts through `decoder.sample_counts`. The reason is calibration. A population of decoded rates alone is near-degenerate, because the only variation is the latent-to-latent spread; the real technical count noise that dominates scRNA-seq is missing. Sampling counts injects that noise, so the per-cell spread of the generated population is measurable against the real one. The sampled counts are renormalized to log1p-CP10K using the supplied `library_size`.

The split between the two readouts mirrors the two things Chapter 4 measures: effect size, which is a statement about the mean, and calibration, which is a statement about the spread. The decoder is the same in both; only the presence of count sampling changes.

## Where this leaves us

We now have every module in hand: a frozen JEPA encoder producing latents, a conditional velocity field that samples latents given a condition, two ways to build that condition, classifier-free guidance layered on top, and a count decoder that turns latents into gene counts with a proper NB likelihood. The next chapter puts them in motion: the three training stages, the data and the combination-holdout split that tests generalization, and the evaluation harness that turns generated populations into the effect-size and calibration numbers.

*Previous: [Chapter 1 — The approach](01-the-approach.md). Up: [the method series](index.md). Next: [Chapter 3 — Training and evaluation](03-training-and-evaluation.md).*
