# Chapter 3c — The VICReg collapse guard: how Stage A avoids learning nothing

*A companion to [Chapter 3](03-training-and-evaluation.md), alongside [3a](3a-the-models-in-the-head-to-head.md) and [3b](3b-reading-the-calibration-metrics.md). Chapter 3 mentions in one line that Stage A adds "a VICReg-style regularizer, weighted by `--reg-coef`" as a collapse guard. This note unpacks that line: why a predict-embeddings objective can cheat, what VICReg is, what the two terms in our version actually do, and how they mirror the two collapse diagnostics the training loop logs.*

> **Where this sits.** Read [Chapter 3 §Stage A](03-training-and-evaluation.md) first for the self-supervised objective and the collapse diagnostics. This note is the mechanism behind the guard, implemented in [`src/ssllab/objectives/jepa_loss.py`](../../../../src/ssllab/objectives/jepa_loss.py). Nothing here is a result; it is the reason Stage A produces a representation worth generating in rather than a constant.

## 1. The cheat a predict-embeddings objective invites

Stage A trains the encoder to **predict embeddings from embeddings**: mask some of a cell's gene-group tokens, encode the visible ones, and predict the masked tokens' embeddings as produced by an EMA "teacher" copy of the encoder. That objective has a trivial degenerate solution. If the encoder maps *every* cell to the *same* latent vector, the prediction is always perfect — you are forever predicting the same constant — and the loss collapses to near zero while the representation encodes nothing. This is **representation collapse**, and every self-supervised method that scores predictions in its own latent space is exposed to it.

Two flavors are worth naming, because the guard attacks each with its own term:

- **Dimensional collapse** — some latent directions shrink to near-constants and stop varying across cells, so they carry no information. In the limit every dimension collapses and all cells map to one point.
- **Informational collapse** — the dimensions vary but are redundant: several encode the same thing (they move together), so the *effective* number of independent directions is far below the nominal width.

## 2. VICReg in one breath

VICReg — **V**ariance–**I**nvariance–**C**ovariance **Reg**ularization (Bardes, Ponce & LeCun, 2022) — prevents both flavors with three terms on a batch of embeddings:

1. **Invariance** — embeddings of two views of the same input should agree. This is the term that *learns*, and the one that would collapse everything to a constant if left unchecked.
2. **Variance** — each embedding dimension must keep a minimum standard deviation across the batch. This forbids dimensional collapse.
3. **Covariance** — different dimensions should be decorrelated across the batch. This forbids informational collapse.

The point is that you do not need contrastive negatives to avoid collapse: two cheap statistics on the batch's embeddings suffice.

## 3. What our version actually adds

Stage A does *not* re-implement all three terms, and seeing why is the key to reading the code. The **invariance term is already there** — it is the JEPA prediction loss itself, which pulls the student's predictions toward the EMA teacher. And JEPA carries a collapse defense a plain Siamese network lacks: the teacher is a slow-moving average of the student, so predicting a *moving* target makes the constant solution harder to settle into.

So the project treats variance and covariance as **cheap insurance and an explicit knob** on top of an objective that already resists collapse:

$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{pred}}}_{\text{invariance (JEPA)}} \;+\; \texttt{reg\_coef}\cdot\big(\underbrace{\texttt{var\_coef}\cdot \mathcal{L}_{\text{var}} + \texttt{cov\_coef}\cdot \mathcal{L}_{\text{cov}}}_{\text{VICReg insurance}}\big).$$

The regularizer runs on the **student context embeddings** (the online encoder's output on the visible tokens) and is added only when `reg_coef > 0`. Set `--reg-coef 0` and Stage A is pure JEPA, leaning entirely on the EMA target.

## 4. The variance term — keep every dimension alive

```python
# Variance: hinge each dimension's std up to 1.
std = torch.sqrt(z.var(dim=0) + eps)
var_term = F.relu(1.0 - std).mean()
```

`z` is a batch of embeddings, `(M, D)`. `z.var(dim=0)` is each dimension's variance *across the batch*; the square root makes it a per-dimension standard deviation. Then `relu(1 - std)` is a **hinge**: if a dimension's std is at or above $1$, the term is zero and the dimension is left alone; if it drops below $1$, the penalty grows linearly as the std shrinks toward $0$. Average over dimensions for one scalar.

The effect is a one-sided floor — "every dimension must keep a spread of at least $1$; below that I push back, above it I do not care." A dimension drifting toward constant gets an increasing gradient that inflates its spread again, which is exactly what stops dimensional collapse. The hinge is what makes it play nicely with the prediction loss: a naive "maximize variance" term would have no stopping point and would fight the objective forever, whereas the hinge goes silent once a dimension is healthy.

## 5. The covariance term — keep dimensions distinct

```python
# Covariance: penalize off-diagonal entries of the covariance matrix.
zc = z - z.mean(dim=0, keepdim=True)
cov = (zc.T @ zc) / max(m - 1, 1)
off_diag = cov - torch.diag(torch.diag(cov))
cov_term = off_diag.pow(2).sum() / d
```

Center the embeddings and form the $D \times D$ covariance matrix across the batch. The diagonal is per-dimension variance (the variance term's business, ignored here); the **off-diagonal** entries measure how much pairs of dimensions move together. The term is the sum of squared off-diagonals, normalized by `d`.

Driving it down pushes every pair of dimensions toward being **uncorrelated**, eliminating redundancy: if dimensions 3 and 7 always rise and fall together they are one dimension wearing two hats, and the representation is smaller than it looks. Decorrelation forces each dimension to carry its own slice of information, which prevents informational collapse and pushes the encoder to *use* the full width of its latent space.

The division of labor is the thing to remember: the variance term keeps each dimension individually alive, the covariance term keeps them collectively distinct, and neither alone suffices — you can have lively dimensions that all measure the same thing, or decorrelated dimensions several of which are nearly constant.

## 6. The knob and its defaults

| symbol | default | role |
|---|---|---|
| `var_coef` | $1.0$ | weight on the per-dimension spread term |
| `cov_coef` | $0.04$ | weight on the decorrelation term |
| `reg_coef` | $0.04$ in Stage A (`--reg-coef`); $0$ in the loss default | overall weight on the whole regularizer; $0$ disables it |

The internal split leans hard on variance ($1.0$) over covariance ($0.04$): keeping dimensions alive is the first-order defense, decorrelation the finer correction. The *outer* `reg_coef`, the flag exposed on the command line, defaults to $0.04$ in Stage A — low strength, because the JEPA prediction loss and EMA teacher do the heavy lifting and the VICReg penalty is a safety net. `--reg-coef 0` is a clean ablation: if the run still trains without collapsing, the EMA target was carrying it alone.

## 7. Why it mirrors the two diagnostics

Stage A logs two collapse diagnostics each epoch — **effective rank** and **per-dimension feature standard deviation** (see [Chapter 3 §Stage A](03-training-and-evaluation.md)). They line up with the two regularizer terms almost one to one, which is no accident:

- **Feature standard deviation** is exactly what the **variance term** defends. If the logged std stays healthy (near or above $1$), the floor is doing its job — or was never needed.
- **Effective rank** is what the **covariance term** defends. Effective rank falls when dimensions become redundant, the same informational collapse the covariance term penalizes; a representation of decorrelated, individually-alive dimensions has a high effective rank.

So the loss *prevents* the two failure modes and the diagnostics *detect* whether prevention worked — two views of the same two problems. It is why Chapter 3 can promise a collapsing run is "visible early rather than discovered at evaluation time," and why Chapter 4 can open by reporting an effective rank of $176/256$: the guard held.

## 8. What to carry into the results

Stage A's job is to produce a representation the flow can generate in, and the VICReg guard is what makes that non-trivial. Full VICReg is invariance + variance + covariance; here the JEPA prediction loss *is* the invariance term and the EMA teacher already resists collapse, so the code adds only the two anti-collapse terms — a variance hinge that floors each dimension's spread at $1$, and a covariance penalty that decorrelates dimensions — at low strength via `--reg-coef`. Those two terms mirror the two logged diagnostics (variance ↔ feature std, covariance ↔ effective rank), so when [Chapter 4](04-results.md) reports a high effective rank and a probe well above chance, it is reporting that this guard worked.

---

*Previous: [Chapter 3b — Reading the calibration metrics](3b-reading-the-calibration-metrics.md). Up: [the method series](index.md). Next: [Chapter 4 — Results](04-results.md).*
