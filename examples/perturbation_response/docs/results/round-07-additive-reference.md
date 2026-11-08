# Round 7 — the additive reference: does any learned model beat assuming no interaction?

*Part of [the results ledger](index.md), which holds the standing verdict, the scoreboard and the settings every round shares. Previous: [round 6](round-06-conditioned-encoder.md) · Next: [round 7b](round-07b-composition-deepsets.md).*

**What changed.** Nothing was trained. This round scored, for the first time, the answer a biologist would write down without any model. The **additive model** of genetic interaction assumes the two genes do not interact, so a combination's effect is the sum of its singles' effects:

$$
\hat{\Delta}(A{+}B) = \Delta(A) + \Delta(B), \qquad \Delta(g) = \bar{x}_g - \bar{x}_{\mathrm{ctrl}} ,
$$

where $\bar{x}_g$ is the mean normalized expression (log1p-CP10K) of the cells perturbed by gene $g$ alone, $\bar{x}_{\mathrm{ctrl}}$ the mean of the control cells, and $\hat{\Delta}$ the prediction. Each $\Delta(g)$ comes from the training side of the `combo` split. Every single is on that side and only the combinations are held out, so the reference never reads a held-out cell. It is graded by the same harness as every model, on the same twenty pairs and the same top-20 genes. Published work finds this reference hard to beat on combination benchmarks of this kind (Ahlmann-Eltze, Huber and Anders, "Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines", *Nature Methods*, 2025). Round 4 had already computed the same prediction as an epistasis diagnostic, from the same single-gene cells, but it was never set beside the models.

**The pre-committed endpoint.** $\Delta r$ on the twenty held-out combinations, with two contrasts in one max-$t$ family: additive minus the conditioned VAE (the round-6 bar), and additive minus the transport flow. The decision rule, fixed before any number: if the additive model is significantly above the bar, it becomes the bar. There are no secondary endpoints, because the additive model predicts a mean profile, not a population, and calibration does not apply to it.

**Verdict: on mean-effect prediction, the additive reference beats every learned model tested, and it becomes the reference on $\Delta$-correlation at $0.961$.**

| contrast | $\Delta$ | simultaneous 95% interval | verdict |
|---|---|---|---|
| **additive − conditioned NB-VAE** | $\mathbf{+0.103}$ | $[+0.024, +0.182]$ | **significant** |
| **additive − transport flow** | $\mathbf{+0.313}$ | $[+0.205, +0.421]$ | **significant** |

The additive model scores a mean of $0.961$ and a median of $0.971$. Pair by pair, the conditioned VAE beats it on none of the twenty and ties it on one, CBL+CNN1, at $0.956$ against $0.955$. Its weakest pair is PLK4+STIL at $0.81$; every other pair is at $0.94$ or above.

**How the rule applies.** The rule fixed in advance made the additive model the bar if it won, and it won. A reference that predicts only a mean can be a bar only for the question a mean can answer, so the additive model is the reference for mean-effect prediction on $\Delta r$. The conditioned VAE stays the strongest learned baseline, and the reference for every population metric.

**Exploratory references, added after the result and reported without verdicts.** They ask how much of the score needs the pair at all.

| reference, scored by the same harness | mean $\Delta r$ |
|---|---|
| the additive model (the result above) | 0.961 |
| one single alone, the first gene of the pair | 0.828 |
| one single alone, the second gene | 0.765 |
| one average training effect, predicted identically for every pair | 0.711 |
| two randomly chosen singles added together, 20 draws | 0.398 (0.26 to 0.53) |

**What it means.** Three readings follow, in decreasing order of certainty.

- *No learned model in this program has yet beaten the reference that assumes no interaction.* The conditioned VAE comes closest, $0.103$ below. The transport flow sits $0.313$ below, and below even a prediction that ignores which pair it is asked about ($0.711$). The ceiling sharpens this: a perfect Stage B through the current decoder reaches only $0.679$, and the linear-readout diagnostic $0.852$, so neither the transition nor the representation as read out today could reach the additive reference.
- *The metric rewards getting the singles right more than getting the interaction right.* Pearson correlation ignores scale, so a pair that does more or less than its singles combined costs the additive model almost nothing. Round 4 measured exactly that kind of interaction on these pairs, an additive scale $\lambda$ from $0.655$ to $1.266$, and $\Delta r$ cannot see it. About $70\%$ of each pair's top-20 genes are also among its singles' top-20, so the genes being scored are mostly genes the singles already move.
- *The scoreboard therefore has little room above the reference.* At $0.961$ there is $0.04$ left before a perfect score. A model that captured epistasis well could struggle to show it on $\Delta r$, which is a property of the metric, not evidence about any model.

**What it does not mean.** It does not mean the pairs do not interact: round 4 measured interactions of $17\%$ to $54\%$ of the effect on these same pairs. It does not mean predicting combinations is solved, because the additive reference needs both singles measured, and its additivity is assumed in log1p-CP10K, where adding effects corresponds to multiplying them on the count scale. And it does not touch population generation: the additive reference produces no cells, so whether a learned model generates useful, calibrated populations is a separate question, still open, on which the conditioned VAE remains the strongest model tested. The conclusion is scoped to the twenty held-out combinations and to correlation on each pair's top 20 genes.

**Checks.** The reference reproduces round 4's additive correlation pair by pair to within $2 \times 10^{-7}$, and a test confirms the prediction does not change when the held-out cells do. The additive model is deterministic, so the interval reflects the twenty test perturbations alone; the learned arms are seed-averaged first, as in every round.

**Limits.** The rule held for this endpoint, but the result says as much about $\Delta r$ as about the models. A metric that scores the size of the change as well as its direction could rank the arms differently. That is a different endpoint and needs its own pre-registered round, proposed as open item 6 and run as [round 8](round-08-size-aware-endpoint.md).

**Script and report.** [`18_eval_additive_baseline.py`](../../18_eval_additive_baseline.py) builds and scores the reference, including the exploratory block, and writes `output/norman_additive/reports/effect_size_additive.json`. The contrasts come from `12_compare_arms.py` with the arm kind `additive`, which writes `output/additive_baseline_compare.json`.

---

*Up: [the results ledger](index.md). Previous: [round 6](round-06-conditioned-encoder.md) · Next: [round 7b](round-07b-composition-deepsets.md).*
