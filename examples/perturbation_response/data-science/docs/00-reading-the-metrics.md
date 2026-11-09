# Reading the metrics

*Four endpoint families, what each one is blind to, and why two of them have disagreed about this project's results for three rounds. Every number below is printed by one of the scripts in [`data-science/`](../README.md) and traced to the function in `src/ssllab` that computes it.*

---

## The holdout, first

Everything is a mean over the same 20 held-out two-gene combinations, so it is worth knowing what they are before reading any score.

| | |
|---|---|
| held-out cells | **7,375** across 20 combinations |
| cells per combination | min 81, median 259, max 1,198, a **14x range** |
| each combination's weight in every reported mean | $1/20$, regardless of its size |
| constituent singles | all 31 target genes have a measured single in training |

The unweighted mean is a deliberate choice, because the estimand is how well a method does on *a perturbation* rather than on *a cell*. The cost is that the noisiest pairs count as much as the best-measured ones, and `03_holdout_power.py` puts a number on how noisy: an oracle predictor still scores $e_p \approx 0.24$ on the 81-cell pair and $0.03$ on the 1,198-cell pair, against an additive reference that averages $0.289$ overall.

## The four families

| family | what it compares | scale-invariant? | sees the joint? | blind to |
|---|---|---|---|---|
| **Δr** | mean change, gene by gene | yes | no | magnitude |
| **e_p** | mean change, as a vector | no | no | distribution shape |
| **energy / W1 / coverage** | full populations | no | energy only | nothing in principle, much in practice |
| **effective rank, feature_std** | the embedding matrix | no | yes | anything downstream |

### Δr: the pattern, not the size

$$\hat\Delta = \text{predicted pair mean} - \text{control mean}, \qquad \Delta = \text{true pair mean} - \text{control mean}$$

$$\Delta r = \mathrm{corr}(\hat\Delta, \Delta) \quad \text{over the pair's top-20 DE genes}$$

Being a correlation it is invariant to scale, and that is the whole problem. `02_metric_effect_size_and_ep.py` makes it concrete on five hand-sized predictions:

| prediction | Δr | e_p |
|---|---|---|
| perfect | 1.000 | 0.000 |
| right pattern, 60% of the size | **1.000** | 0.400 |
| right pattern, 10% of the size | **1.000** | 0.900 |
| no change at all | undefined | 1.000 |
| sign flipped | −1.000 | 2.000 |

Three predictions share $\Delta r = 1$ and are not remotely equally good. A model that knows which genes move but commits to a tenth of the movement is graded as perfect.

### e_p: the size, and what the shortfall is made of

$$e_p = \frac{\lVert \hat\Delta - \Delta \rVert}{\lVert \Delta \rVert}$$

Not scale-invariant, so $e_p = 1$ is exactly the score of predicting no change, and $e_p > 1$ means a prediction is further from the truth than silence. Projecting onto the additive direction $s = \Delta_A + \Delta_B$ says *why* a model's $e_p$ is large:

$$s_p = \frac{\langle \hat\Delta, s\rangle}{\langle s, s\rangle}, \qquad \lambda_p = \frac{\langle \Delta, s\rangle}{\langle s, s\rangle}$$

so $s_p/\lambda_p$ is the fraction of the real change a model actually commits to. Round 8 measured the learned models near $0.6$: right direction, short distance.

One identity worth carrying, which `02` asserts rather than claims. For the additive reference $\hat\Delta = s$, so $\hat\Delta - \Delta = -I_p$ where $I_p = \Delta - s$ is the interaction. Therefore

$$e_p(\text{additive}) = \frac{\lVert I_p \rVert}{\lVert \Delta \rVert} = \text{rel\_GI}$$

exactly. The additive reference's score *is* the interaction it ignores, which makes `rel_GI` simultaneously its score and the floor it cannot beat. On this holdout that is $0.289$.

### Why the two disagree about this project

| | Δr | e_p |
|---|---|---|
| additive reference | 0.961 | 0.289 |
| conditioned NB-VAE | 0.858 | 1.053 |
| transport flow | 0.648 | 1.261 |

On Δr the learned models look close to a reference that beats them by $0.10$. On $e_p$ they are $3.6$ to $4.4$ times its error, and both sit **above 1.0**, the score of predicting no change. The scripts reproduce the reference column exactly from the cache ($\Delta r$ 0.9606, $e_p$ 0.2891, oracle floor 0.075), which is the check that these are the same quantities the round pages report.

That gap is not a paradox. It is two questions: *does the model know which genes respond* (yes) and *does it know how much* (no).

### Population metrics: energy distance and friends

Δr and $e_p$ both compare *means*. A model with the right mean and no spread scores perfectly on both. The population axis is what notices, and energy distance is the member of that family that sees the joint distribution:

$$E(P, Q) = 2\,\overline{\lVert p - q\rVert} - \overline{\lVert p - p'\rVert} - \overline{\lVert q - q'\rVert}$$

Zero exactly when the two clouds share a distribution. No pairing is needed, which is the point: generated cell $i$ is not real cell $i$, and the two clouds need not be the same size.

`01_metric_energy_distance.py` shows what the joint view buys. Two 2-gene clouds with *identical* per-gene marginals differing only in correlation ($0$ versus $0.95$) score $0.036$ on mean per-gene 1-Wasserstein and $0.092$ on energy distance, $2.6$ times more. Gene by gene they are the same distribution; jointly they are not.

Scale, from the same script: control cells against held-out perturbed cells run $E \approx 0.73$ to $4.44$ depending on the pair, and the split-half floor is essentially zero even at 81 cells. The reported transport flow sits at $3.62$ and the conditioned VAE at $4.09$, both near or above the distance from the *untouched control cloud* to the truth.

**Two estimators, deliberately.** The production `energy_distance` is a V-statistic: `cdist(x, x).mean()` averages over all $n^2$ within-cloud pairs including the $n$ zeros on the diagonal. The Koopman probe carries a U-statistic that divides by $n(n-1)$. They differ by exactly

$$E_V - E_U = \frac{\text{within}_V(P)}{n-1} + \frac{\text{within}_V(Q)}{m-1}$$

which `01` asserts to $4 \times 10^{-7}$ rather than asking you to trust the word "biased". The consequence is practical: $E_V$ is inflated by about $\text{within}/(n-1)$, a bias falling like $1/n$, so two clouds of *different sizes* are not on the same scale. The probe's finite-sample floor is computed on half-size clouds, and under the V-statistic that floor came out larger than the signal it was a floor for, sending a reduction ratio to 216. The V-statistic stays as the training loss, where a constant offset cannot move the argmin; scoring uses the U-statistic.

At the sizes the calibration evaluation actually uses (200 generated cells against up to 1,198 real) the bias is about $+0.03$, under 1% of a figure of 3.6. It would dominate any reported energy of order $0.01$ to $0.05$.

### Embedding diagnostics: effective rank

$$\text{effective rank} = \exp\big(H(\text{normalized singular values of centered } z)\big) \in [1, D]$$

$\exp$ of an entropy is an *effective count*, the same construction as perplexity or a Hill number: variance spread evenly over $k$ directions gives $k$, variance in one direction gives $1$ however large it is.

It exists because `feature_std` cannot see the second collapse mode. From `04_collapse_effective_rank.py`, holding total variance fixed while collapsing the geometry:

| rank | feature_std | effective rank |
|---|---|---|
| 256 | 1.001 | 251.7 |
| 64 | 0.995 | 63.7 |
| 20 | 0.989 | 20.0 |
| 1 | 0.819 | 1.0 |

`feature_std` barely moves. A monitor watching only the standard deviation would report a healthy encoder at rank 1. For reference, VICReg Stage A logs effective rank 176 of 256 at `feature_std` 0.755; the same encoder logs 213 on the probe's 8,000-cell subset, because the measure is batch and split dependent and is only meaningful compared against itself.

## A worked case: why a random-slice penalty permitted rank 20

Round 9 trained encoders with SIGReg, which tests whether the projection onto each of many random unit directions looks like $N(0,1)$, averaged over directions. The arms came out with roughly 20 dimensions of variance out of 256. How?

The tempting answer, that a marginal test cannot see rank at all, is **wrong**, and `04` measures the real one. Every one-dimensional projection of a Gaussian is Gaussian whatever the rank, so the slice's *shape* carries no rank information. What leaks is the slice's *variance*: for a rank-$r$ embedding with matched total variance, a random direction picks up a fluctuating share of the active subspace, with relative spread about $\sqrt{2/r}$. Measured against predicted, at $r = 20$: $0.31$ against $0.316$.

So the penalty does see low rank. The question is how strongly, compared with everything else it measures:

| synthetic embedding | effective rank | slice variance | SIGReg |
|---|---|---|---|
| full rank, $N(0, I)$ | 251.6 | 0.998 | **1.02** |
| rank 20, total variance matched | 20.0 | 1.002 | **22.4** |
| full rank, variance 4 | 251.6 | 3.992 | 483 |
| full rank, shifted by 1.5 | 251.6 | 0.998 | 1459 |

Low rank costs about $22\times$ the target's statistic; a global scale error costs $480\times$ and a mean shift $1460\times$. In a convex objective $(1-\lambda)\cdot\text{prediction} + \lambda\cdot\text{SIGReg}$, the gradient is dominated by getting the mean and the scale right, and low rank is a comparatively faint signal a prediction term can afford to trade against.

Those figures are an isolation of the statistic on synthetic Gaussians, not a reproduction of the run, which reports $3.85$ against $1.07$ on real embeddings at a different $N$ and slice count. And the scope is narrow: it says a random-slice $N(0,I)$ test is weak against low rank, not that LeJEPA's full method is.

## A caution about simultaneous intervals

The project grades families of contrasts with a max-t bootstrap, whose critical value is the 95th percentile of the largest studentized statistic across the family. Two things follow, and `03_holdout_power.py` demonstrates both on training-free arms.

**Family size widens everything.** Two contrasts that each carry a critical value of $2.17$ alone carry $2.52$ together. Adding an arm to a family costs the arms already in it.

**Leverage belongs to the (arm, contrast) pair, not to a perturbation.** It is tempting to read "CEBPE+RUNX1T1 inflated the critical value to 4.63 in Round 9" as a standing fact about the holdout. It is not. With the training-free references, dropping that same pair moves the critical value by $-0.007$, essentially nothing, and the pairs with real leverage are different ones. Round 9's figure came from how the SIGReg arms happened to differ on that pair.

The transferable habit: when an interval spans zero, check whether the critical value is unusually large before concluding the arms are close, because "these arms are similar" and "this family has a wide critical value" are different findings. Re-run the leave-one-out against the arms in question. It is not a licence to drop a pair, since the family and the 20 pairs are fixed in the protocol before scoring precisely so that leverage cannot be discovered and then removed.

---

*Up: [reading the numbers](index.md). The scripts: [data-science](../README.md). What each round pre-registered and found: [results](../../docs/results/index.md).*
