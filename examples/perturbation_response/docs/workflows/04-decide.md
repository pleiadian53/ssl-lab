# 4. Decide: turning numbers into verdicts

*Two scripts, and a discipline that is mostly about what you commit to before you look. Score produces measurements; this workflow decides whether a difference between them is real. On a twenty-perturbation test set, that gap is wide enough to have produced one retraction in this project already.*

---

## `12` — comparing arms

```bash
python examples/perturbation_response/12_compare_arms.py \
    --arm flow=norman_flow_control --arm vae=norman_combo --arm operator=norman_operator_s0 \
    --contrast flow-vae --contrast operator-flow --primary delta_r
```

It reads the per-perturbation vectors out of existing reports and does three things that a difference of means does not.

**It pairs.** The same twenty perturbations are scored by every arm, and some perturbations are simply harder than others. Comparing per-perturbation differences removes that shared difficulty, which on a small test set is most of the noise.

**It resamples the contrast family *jointly*.** Contrasts that share an arm are correlated: if `flow-vae` and `operator-flow` both involve the flow, an unlucky bootstrap draw for the flow moves both. Resampling them together preserves that correlation instead of pretending the comparisons are independent.

**It reports max-$t$ simultaneous intervals.** Testing a family of contrasts at $95\%$ each does not give $95\%$ confidence in the family. The max-$t$ adjustment widens the intervals to cover the whole family at once, which here moved the critical value from $1.96$ to roughly $2.95$. That is a large penalty and it is the honest price of asking several questions of one dataset.

## `17` — a pre-committed structural endpoint

```bash
python examples/perturbation_response/17_eval_bracket_epistasis.py --experiment norman_operator_algebra_s0
```

A different shape of decision: not "is arm A better than arm B" but "does this model have the structural property it was designed to have." It correlates a model-side quantity against a model-free target from [`15`](03-diagnose.md), and reports a **permutation null** rather than a parametric p-value, because the null distribution of a rank correlation on $20$ points is not something to assume.

It also splits the test deliberately:

- **in-sample** (the $91$ training combinations) carries the statistical power. A null *here* is fatal, because it means the property is absent even where the model was fit on it.
- **held-out** (the $20$) is the real generalization claim, and it is underpowered by construction.

Reporting both is what lets a null be read correctly. A single underpowered number is ambiguous; a null in the powered arm is not.

## The discipline

**One primary endpoint, pre-committed, before any number exists.** Significance is claimed on exactly one metric. Everything else is reported as an interval with no verdict. A difference found on a secondary endpoint *after the fact* is a hypothesis, not a result, and it does not become one until it is confirmed on data that did not suggest it.

This project has a retraction that makes the point concretely: a flow "win" on joint energy distance was announced on a single seed, and on three seeds the interval was $[-0.960, +0.172]$, crossing zero. Post-hoc, secondary endpoint, discovered after changing the metric. Every ingredient of a garden-of-forking-paths error.

**Choose the endpoint to match what can move.** The operator-algebra round deliberately did *not* use $\Delta r$ as its endpoint, because the [ceiling](03-diagnose.md) had already shown the transition was saturated at $0.03$ of headroom. Grading it on $\Delta r$ would have measured the one quantity already known to be pinned. The endpoint was a structural claim instead, benchmark-independent and needing no comparison to the baseline.

**Write the decision rule down first.** For that round the rule was: *a null on the in-sample arm kills the idea and we skip the three-seed confirmation.* When the null arrived, the rule had already been written, so following it was bookkeeping rather than a judgement call made while staring at a disappointing number.

**Say the seed count.** One seed is directional. A claim needs seed averaging and a simultaneous interval that excludes zero.

## Reading a null honestly

Two distinctions that decide whether a negative result means anything.

**"The hypothesis failed" versus "this run could not test it."** The first operator-algebra run returned a null endpoint, and that null was *uninformative*: the generators had drifted far from the identity, into a regime where the quantity being correlated no longer measured what it was supposed to. Reporting it as a refutation would have been wrong. Only a diagnostic on the run's own premise distinguished the two cases, and the trainer now gates itself on that premise so the confusion cannot recur.

**A failed rescue strengthens a negative.** After the endpoint failed, two post-hoc variants were tried, both aimed at plausible weaknesses in the original test. Both also failed. That makes the negative *stronger*, not weaker, and it is worth reporting the attempts precisely because they were genuine attempts to save the hypothesis.

Both of these belong in the write-up. A negative result with a named cause is far more useful than one without.

---

*Previous: [Diagnose](03-diagnose.md). Next: [Vary](05-vary.md). Up: [the workflow map](index.md). The statistics in full: [From a difference to a verdict](../../../../docs/experimental-method/05-from-difference-to-verdict.md).*
