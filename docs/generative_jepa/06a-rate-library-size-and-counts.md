# Part 6a — From rate to counts: composition, library size, and the NB mean, worked through

*A companion to [Part 6 §2](06-route-a-latent-decoder-head.md) for readers who want the sentence "a relative expression profile $\rho$ that sums to one, then scaled by the library size $\ell$" turned into arithmetic. We run one small example — a handful of genes — all the way from the decoder's output to a sampled count vector, and use it to show why the rate-times-depth split is the right way to model counts.*

> **Why this chapter exists.** [Part 6 §2](06-route-a-latent-decoder-head.md) introduces the count decoder's core trick in one line: emit a **rate** $\rho$ over genes (a softmax, so it sums to one), then multiply by the cell's **library size** $\ell$ to get the negative-binomial mean $\mu = \ell \rho$. That line packs in a real modeling decision — *separate the biological composition from the technical sequencing depth* — and it is far easier to feel with numbers than with symbols. This chapter assumes only [Part 6 §2](06-route-a-latent-decoder-head.md)'s vocabulary (the NB mean $\mu$, dispersion $\kappa$, library size $\ell$, rate $\rho$) and works one example end to end.

The plan: first the two ingredients ($\rho$ and $\ell$) on a six-gene toy cell; then assembling the mean $\mu = \ell \rho$; then the two scenarios that justify the split — *same biology at different depths*, and *a perturbation that reshapes the profile* (where effect size lives); and finally the step from the mean to an actual noisy count draw. Real single-cell numbers are bigger but the mechanics are identical, and we scale up once at the end.

---

## 1. The two ingredients — a composition and a depth

Take a deliberately tiny cell with just **six genes**, call them $A$ through $F$. (Real data has thousands; six is enough to see everything.)

**The rate $\rho$ — what fraction of the cell's transcripts each gene accounts for.** The decoder reads the latent $z$ and emits one real number ("logit") per gene, then a **softmax** turns those logits into a probability vector: every entry positive, all entries summing to one. Concretely, if the decoder's logits for three genes were $(2.0, 1.0, 0.0)$, the softmax exponentiates and normalizes —

$$
\rho = \frac{(e^{2.0},\ e^{1.0},\ e^{0.0})}{e^{2.0} + e^{1.0} + e^{0.0}} = \frac{(7.39,\ 2.72,\ 1.00)}{11.11} = (0.665,\ 0.245,\ 0.090),
$$

— and those three shares sum to one. For our six-gene cell, suppose the softmax produces

$$
\rho = (\underbrace{0.40}_{A},\ \underbrace{0.25}_{B},\ \underbrace{0.15}_{C},\ \underbrace{0.10}_{D},\ \underbrace{0.07}_{E},\ \underbrace{0.03}_{F}), \qquad \textstyle\sum_g \rho_g = 1.
$$

Read this as **composition**: gene $A$ makes up 40% of this cell's transcripts, $B$ a quarter, down to $F$ at 3%. It is a *relative* statement — it says nothing yet about *how many* molecules were actually counted, only their proportions. This is the part the decoder predicts, and it is where the biology lives.

**The library size $\ell$ — how many transcripts this cell yielded in total.** Sequencing does not read every molecule in a cell; it captures a sample of them, and that sample's total size varies a lot from cell to cell — driven by technical factors (capture efficiency, sequencing depth) far more than biology. We call that per-cell total the **library size** $\ell$. For our example cell, say the sequencer captured

$$
\ell = 10{,}000 \text{ total counts}.
$$

The crucial point [Part 6 §2](06-route-a-latent-decoder-head.md) makes: $\ell$ is a **given covariate, not a prediction.** You read it straight off the observed cell (sum its counts); the decoder never has to guess it. The decoder's whole job is the *shape* $\rho$; the *scale* $\ell$ is handed in.

---

## 2. Assembling the mean — $\mu = \ell \rho$

Now the one-line assembly. Multiply each gene's share by the total depth to get its **expected count**:

$$
\mu_g = \ell \cdot \rho_g.
$$

For our cell, gene by gene:

| gene | rate $\rho_g$ | $\times\ \ell = 10{,}000$ | mean count $\mu_g$ |
|---|---|---|---|
| $A$ | 0.40 | | **4000** |
| $B$ | 0.25 | | **2500** |
| $C$ | 0.15 | | **1500** |
| $D$ | 0.10 | | **1000** |
| $E$ | 0.07 | | **700** |
| $F$ | 0.03 | | **300** |
| **sum** | **1.00** | | **10,000** |

So $\mu = (4000, 2500, 1500, 1000, 700, 300)$ — the number of transcripts we *expect* to count for each gene in this cell. Notice the bottom row: because the rates sum to one, the expected counts sum to exactly the library size, $\sum_g \mu_g = \ell \sum_g \rho_g = \ell \cdot 1 = \ell$. That is the whole elegance of the parameterization — **$\rho$ distributes the budget $\ell$ across genes**, and the budget is always spent exactly. (The mean is not the decoder's direct output; it is *assembled* from the predicted shape $\rho$ and the given depth $\ell$, exactly as [Part 6 §2](06-route-a-latent-decoder-head.md) stressed.)

---

## 3. Why split it this way — depth is a nuisance, composition is the signal

The decomposition earns its keep the moment you compare cells. Here are the two scenarios that make it click.

### 3.1 Same biology, different depth

Take **two cells with identical biology** — same cell type, same state, so the *same* composition $\rho$ — but sequenced to different depths: cell 1 at $\ell_1 = 10{,}000$, cell 2 at $\ell_2 = 4{,}000$. The means scale with depth, term by term:

| gene | $\rho_g$ | cell 1: $\mu_g = 10{,}000 \cdot \rho_g$ | cell 2: $\mu_g = 4{,}000 \cdot \rho_g$ |
|---|---|---|---|
| $A$ | 0.40 | 4000 | 1600 |
| $B$ | 0.25 | 2500 | 1000 |
| $C$ | 0.15 | 1500 | 600 |
| $D$ | 0.10 | 1000 | 400 |
| $E$ | 0.07 | 700 | 280 |
| $F$ | 0.03 | 300 | 120 |

The **raw counts look completely different** — cell 1's gene $A$ reads ~4000, cell 2's ~1600 — yet *nothing biological differs between them.* The entire gap is sequencing depth, a technical artifact. If you fed a model the raw counts and asked it to "learn the difference," it would waste capacity modeling a difference that is pure nuisance.

The rate-times-depth split dissolves this. Both cells share the *same* $\rho$; the model only ever has to predict that one depth-invariant profile, and the per-cell $\ell$ — read off the data — absorbs the technical scale. Equivalently: divide any observed count vector by its total and you recover $\rho$, which is why $\rho$ is the *normalized*, depth-free description of the cell. **The decoder predicts the biology ($\rho$); the library size carries the technical depth ($\ell$); they are kept apart on purpose.**

### 3.2 A perturbation reshapes the profile — and that is effect size

Now the case the whole series is about. A perturbation does not change a cell's *depth* — it changes its *composition*. Suppose a drug strongly **up-regulates gene $C$** (and, because shares must still sum to one, pulls the others down to compensate). Hold the depth fixed at $\ell = 10{,}000$ to isolate the biological change:

| gene | control $\rho^{\text{ctrl}}_g$ | $\mu^{\text{ctrl}}_g$ | perturbed $\rho^{\text{pert}}_g$ | $\mu^{\text{pert}}_g$ | change |
|---|---|---|---|---|---|
| $A$ | 0.40 | 4000 | 0.32 | 3200 | −800 |
| $B$ | 0.25 | 2500 | 0.20 | 2000 | −500 |
| $C$ | 0.15 | 1500 | 0.30 | 3000 | **+1500 (2×)** |
| $D$ | 0.10 | 1000 | 0.08 | 800 | −200 |
| $E$ | 0.07 | 700 | 0.06 | 600 | −100 |
| $F$ | 0.03 | 300 | 0.04 | 400 | +100 |

Gene $C$'s expected count **doubles**, $1500 \to 3000$; the others dip slightly. That vector of changes — $(-800, -500, +1500, -200, -100, +100)$ — *is* the perturbation's effect, and the **effect size** from [Part 5 §3](05-two-gaps-four-routes.md) is how well a model reproduces it (the correlation between predicted and true change on the most-affected genes). A model that predicts the reshaped $\rho^{\text{pert}}$ accurately recovers this; a model that stops at a latent and never decodes to counts has no place to express "+1500 on gene $C$" at all. This is the concrete mechanism behind [Part 6 §2](06-route-a-latent-decoder-head.md)'s claim that the count decoder is *where effect size is recovered.*

> **An honest wrinkle — composition is relative.** Because $\rho$ sums to one, a share *cannot* rise without others falling: the table's increases and decreases must balance. So the decoder models **relative** expression, and a perturbation that truly raised *every* gene's absolute transcript count (with depth held fixed) is not representable as a change in $\rho$ alone — it would show up only through $\ell$. This is the well-known compositional nature of sequencing counts, and it is one reason library-size handling and normalization are treated so carefully in practice. For most differential-expression questions ("which genes shift *relative to the rest*?") the relative profile is exactly what you want; just know the constraint is there.

---

## 4. From the mean to an actual count — the NB draw

The mean $\mu_g$ is only the *expected* count; the observed count is noisy around it, and the **dispersion $\kappa$** sets how noisy. Recall from [Part 6 §2](06-route-a-latent-decoder-head.md) that the NB variance is $\mu + \mu^2/\kappa$. Take gene $A$ with $\mu_A = 4000$ and see how the spread depends on $\kappa$:

| dispersion $\kappa_A$ | variance $= \mu + \mu^2/\kappa$ | std. dev. | reading |
|---|---|---|---|
| $2$ (high overdispersion) | $4000 + 8{,}000{,}000 = 8{,}004{,}000$ | $\approx 2829$ | counts swing wildly, ~1000–7000 |
| $100$ (mild) | $4000 + 160{,}000 = 164{,}000$ | $\approx 405$ | counts cluster ~3600–4400 |
| $\kappa \to \infty$ (Poisson) | $4000$ | $\approx 63$ | tight, ~3940–4060 |

So a *single* predicted state ($\mu_A = 4000$) can produce an observed count anywhere from ~1000 to ~7000 when the gene is heavily overdispersed ($\kappa = 2$). That is the **measurement spread** [Part 6 §3](06-route-a-latent-decoder-head.md) warns is *not* the same as outcome heterogeneity: even with the cell's true state pinned, the readout is noisy, and $\kappa$ is what lets the NB model that noise honestly instead of pretending (as a Poisson or a fixed-variance Gaussian would) that counts cluster tightly around their mean.

Putting the pipeline together for our cell: the decoder emits $\rho$ (and a $\kappa_g$ per gene); we assemble $\mu = \ell \rho$; then each gene's observed count is a draw $x_g \sim \mathrm{NB}(\mu_g, \kappa_g)$. One plausible sampled count vector for the control cell (with, say, moderate $\kappa$) might read $(3870, 2560, 1410, 980, 720, 280)$ — close to $\mu$ but jittered, and summing to roughly but not exactly $\ell$. Draw again and you get a different vector: that is one cell's measurement noise, on top of whatever outcome variation the latent already carried.

> **At generation time, where does $\ell$ come from?** During training $\ell$ is read off each real cell. When you *generate* a new cell, there is no observed total to read, so you **supply** a target depth — sample $\ell$ from the empirical library-size distribution of the dataset, or fix it to a reference depth so generated cells are comparable. The decoder still only predicts the shape $\rho$ and dispersion $\kappa$; you choose the scale.

---

## 5. Scaling up — the same arithmetic at real size

Real single-cell data only changes the numbers, not the mechanics. A typical experiment has a few thousand genes (often ~2000 after selecting the highly variable ones) and per-cell library sizes with a **median around 10,000–15,000 counts** — so $\rho$ is a ~2000-entry probability vector and $\ell$ is a five-figure total. The same three steps run unchanged: softmax to a rate $\rho$ that sums to one, scale by the cell's $\ell$ to get $\mu = \ell \rho$, draw counts from $\mathrm{NB}(\mu, \kappa)$. The only practical difference is sparsity: with the budget $\ell$ spread across thousands of genes, most $\mu_g$ are small (well under 1), and many genes read exactly zero in any given cell — which is the dropout that [Part 6 §2](06-route-a-latent-decoder-head.md)'s **ZINB** refinement adds an explicit zero-inflation term for.

---

## 6. Recap — the rate-times-depth picture in one breath

- **$\rho$ is composition** — a softmax over genes, summing to one, the *depth-invariant* biology the decoder predicts.
- **$\ell$ is depth** — the cell's total captured counts, a *given* technical covariate read off the data, not predicted.
- **$\mu = \ell \rho$ assembles the mean** — $\rho$ distributes the budget $\ell$ across genes, so $\sum_g \mu_g = \ell$ exactly.
- **The split is the point** — same biology at different depths shares one $\rho$ (depth is nuisance); a perturbation reshapes $\rho$ (composition is signal), and that reshaping *is* the effect size the count decoder exists to recover.
- **$\kappa$ then sets the measurement noise** of the actual count draw $x_g \sim \mathrm{NB}(\mu_g, \kappa_g)$ around that mean.

With the assembly concrete, the [Part 6 §2](06-route-a-latent-decoder-head.md) line "a rate $\rho$ that sums to one, scaled by the library size $\ell$" should now read as a small, sturdy piece of machinery rather than a notational flourish — and the [count likelihood written out](06-route-a-latent-decoder-head.md) in the rest of §2 is just this same $\mu = \ell\rho$ fed into the negative-binomial probability, gene by gene.

---

*Companion to [Part 6 — Route A: a decoder on the latent](06-route-a-latent-decoder-head.md). New to single-cell counts? The [data-modalities primer](appendix-data-modalities.md). Symbols: the [notation reference](notation.md).*
