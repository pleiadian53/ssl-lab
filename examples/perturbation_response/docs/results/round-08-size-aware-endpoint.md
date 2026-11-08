# Round 8 — an endpoint that sees the size of the change: does any learned model beat additivity?

*Part of [the results ledger](index.md), which holds the standing verdict, the scoreboard and the settings every round shares. Previous: [round 7b](round-07b-composition-deepsets.md) · Next: [round 9](round-09-sigreg-regularizer.md).*

**What changed.** Nothing was trained. Round 7 left a question its own metric could not answer: $\Delta$-correlation ignores scale, so a prediction with the right shape and the wrong size scores perfectly, and the additive reference already has nearly the right shape. Round 8 rescored existing checkpoints on an endpoint that counts size as well as shape, proposed in this ledger before any learned model's value on it existed. For a held-out pair $p$, with the measured change $\Delta_p$ (held-out pair cells minus control, in log1p-CP10K) and an arm's predicted change $\hat{\Delta}_p$, both restricted to the pair's top 20 genes,

$$
e_p = \frac{\lVert \hat{\Delta}_p - \Delta_p \rVert_2}{\lVert \Delta_p \rVert_2},
$$

lower being better. Predicting no change at all, $\hat{\Delta}_p = 0$, gives $e_p = 1$. For the additive reference $e_p$ is exactly round 4's relative interaction $\mathrm{rel}_{\mathrm{GI}}$, the size of the measured departure from additivity as a fraction of the whole change.

**What was known beforehand.** The protocol was frozen in the scoring script before scoring, and it says what had already been seen. The additive reference's value was known from round 4. A composition ablation scored two days earlier, [round 7b](round-07b-composition-deepsets.md), had reported this endpoint, descriptively, for both primary learned arms, on the same checkpoints and predictions. So the primary verdict below was not blind. What the pre-registration protected was the endpoint and the rule: both were written in this ledger before those numbers existed, and neither was changed. The exploratory arms, the scale and all-gene secondaries and the noise floor had not been seen, except one exploratory value printed by a smoke test and disclosed in the protocol.

**The pre-committed endpoint.** Mean $e_p$ over the twenty pairs, seed-averaged per pair for the learned arms. Two contrasts in one max-$t$ family over a joint paired bootstrap: the conditioned NB-VAE minus additive, and the transport flow minus additive. A learned model improves on additivity in magnitude only if its simultaneous interval lies wholly below zero; wholly above zero means additivity is better.

**Verdict: on the size of the change, additivity is better than both learned models, by a wide margin, on every pair.**

| arm | mean $e_p$ | contrast with additive | simultaneous 95% interval | pairs below additive | verdict |
|---|---|---|---|---|---|
| additive reference | $0.289$ | | | | |
| **conditioned NB-VAE** | $1.053$ | $\mathbf{+0.764}$ | $[+0.602, +0.926]$ | $0/20$ | **additivity better** |
| **transport flow** | $1.261$ | $\mathbf{+0.972}$ | $[+0.771, +1.173]$ | $0/20$ | **additivity better** |

As geometric means of the per-pair ratio, the conditioned VAE's error is $3.6$ times additivity's and the transport flow's $4.3$ times. Both learned models have $e_p$ near or above $1$: the conditioned VAE's error is about as large as the change it is predicting, and the transport flow's is larger, so on this endpoint it does worse than predicting no change at all.

**The noise floor.** The truth is itself estimated from a finite number of held-out cells. Splitting each pair's cells in half and comparing the halves gives the error a perfect predictor would still show: a mean of $0.075$. Additivity's $0.289$ therefore leaves about $0.21$ that only a model of the interaction could recover, and the learned models are nowhere near it.

**Secondary results, intervals only, no verdicts.**

- *Scale.* Round 4's additive scale $\lambda_p = \langle \Delta_p, a_p \rangle / \lVert a_p \rVert^2$, with $a_p = \Delta_A + \Delta_B$ the additive prediction, says how much a pair does along the additive direction compared with its singles combined; its mean is $1.02$. The same projection of each model's prediction, $s_p = \langle \hat{\Delta}_p, a_p \rangle / \lVert a_p \rVert^2$, averages $0.62$ for the conditioned VAE and $0.57$ for the transport flow. The learned models predict roughly half to two-thirds of the change along the direction the singles point. Their mean $\lvert s_p - \lambda_p \rvert$ is $0.50$ and $0.59$, against additivity's $0.16$. Across pairs, $s_p$ correlates only weakly with $\lambda_p$ ($0.26$ and $0.33$, both intervals including zero), so the models do not reliably know which pairs are super- or sub-additive. This shortfall along the additive direction accounts for only part of the error; the rest lies in directions the singles do not point.
- *Interaction recovery.* The correlation between a model's predicted departure from additivity and the measured one is $0.32$ for the conditioned VAE and $0.22$ for the transport flow, computed against the measured singles. Against each model's *own* predicted singles, the only form that isolates what the model adds when it composes two genes, it is $-0.11$ and $-0.04$: the models do not compose their singles into the measured interaction.
- *All 5,000 genes.* On every measured gene, the learned models' $e_p$ is about $2.7$ against additivity's $0.59$, so the top-20 selection does not drive the result; if anything it flatters the models.
- *$\Delta$-correlation*, recomputed from the same predictions, reproduces rounds 6 and 7 exactly ($0.858$, $0.648$, $0.961$).

**Exploratory arms, no verdicts.** The condition-blind NB-VAE of round 6 scores $e_p = 1.207$ and the Gaussian flow $1.300$, both above additivity on all twenty pairs. Round 7's pair-agnostic reference, which predicts the same average training profile for every pair, scores $0.803$: lower than every learned model on average, and lower than the conditioned VAE on 13 of the 20 pairs. On the ten most strongly interacting pairs the picture is the same: additivity $0.35$, conditioned VAE $1.19$, transport flow $1.36$.

**What it means.** Round 7's question, asked again with an endpoint that sees size, gets the same answer more sharply. On correlation the learned models looked respectable against additivity, $0.10$ to $0.31$ below it. On size their errors are three to four times larger, on every pair, and they lose even to a reference that ignores which pair it is asked about. The two failures the secondaries name are a shortfall in magnitude, about half the change along the additive direction, and error in directions the singles do not point. This is consistent with the ceiling's finding that the effect is present in the frozen representation and lost in the readout, though round 8 itself does not localize where the magnitude is lost.

**What it does not mean.** It says nothing about population quality: spread, coverage and distributional distance are the separate question, still scored by the calibration metrics, on which the conditioned NB-VAE remains the strongest model tested. It does not say the combinations are additive: additivity's own error is four times the noise floor. And it is scoped to the twenty held-out pairs, their top-20 genes, and additivity in log1p-CP10K.

**Limits.** The primary verdict was not blind, as stated above. One endpoint, twenty pairs, three seeds; pairs share genes, so they are not independent. Not included, and listed as future work: a Systema-style reference that measures change against the average perturbed cell instead of control, which defines a different additive reference; cross-validation over all 131 pairs and held-out single genes, both of which need retraining.

**Script and report.** [`19_score_interaction.py`](../../19_score_interaction.py) scores each run and stops unless its $\Delta$-correlation reproduces the run's stored report exactly; [`21_score_round8.py`](../../21_score_round8.py) holds the frozen protocol in its docstring and writes `output/round8_compare.json`. The metrics live in `ssllab.eval.interaction`, the bootstrap in `ssllab.eval.bootstrap`.

---

*Up: [the results ledger](index.md). Previous: [round 7b](round-07b-composition-deepsets.md) · Next: [round 9](round-09-sigreg-regularizer.md).*
