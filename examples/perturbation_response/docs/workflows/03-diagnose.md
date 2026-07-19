# 3. Diagnose: measuring without training

*The cheapest workflow and the one that changed this project's direction most. Four scripts, none of which trains anything, all of which run on a laptop in minutes. They answer structural questions that no amount of Score results can: where the loss lives, whether a stage is worth improving, and whether a number you are about to trust means what you think.*

---

## The defining property

A Diagnose step **fits nothing, or fits something in closed form**. No optimizer, no epochs, no seed, no early stopping. That is not a limitation, it is what makes the answer trustworthy: with no training loop there is no training loop to blame, and the result is a property of the artifacts rather than of a run.

It is also what makes these steps *free enough to run first*, which is the recurring lesson: **the expensive scripts in this repository confirmed what the cheap ones had already implied.**

## `14` — the ceiling: is this stage worth improving?

The most consequential script here. Stage B's job is to produce the perturbed latents, so hand the pipeline the **real** ones and score the result through the same decoder and the same metric. That is a Stage B which is perfect by construction, and whatever it scores is the ceiling.

```bash
python examples/perturbation_response/14_ceiling_analysis.py --split combo
```

Four arms, each removing one suspect:

| arm | what it substitutes | what it tells you |
|---|---|---|
| `identity` | the true mean itself | **an acceptance gate.** Must score $1.000$, or the harness is misaligned and every number below is meaningless |
| `roundtrip` | real held-out latents → trained decoder | **the ceiling.** The best any Stage B could do |
| `latent_mean` | the mean real latent, no spread | does latent *spread* matter for this metric? |
| `linear` | real latents → a ridge readout | **attributes** the loss between encoder and decoder |

What it found here, and why the project changed course: the oracle scores $0.679$ while the actual flow scores $0.648$, so **the transition is at $96\%$ of its ceiling** and at most $0.03$ was ever available there. Meanwhile a plain linear readout of the same frozen latents scores $0.852$, above the from-scratch baseline. The representation is information-rich; the decoder is where the effect is lost.

Three rounds of work went into the stage with $0.03$ of headroom. **Run this before you build a lever, not after.** The general form of the technique, and how to build the equivalent rung for a retrieval pipeline, an agent, or an RL policy, is [Chapter 7 of the methodology series](../../../../docs/experimental-method/07-the-ceiling.md).

## `02` — the probe: is the frozen encoder any good?

A logistic regression from a frozen latent to the perturbation label, $237$ classes.

```bash
python examples/perturbation_response/02_probe_cell_encoder.py --experiment norman_stage_a --split cells
```

Run it immediately after Stage A, as a gate. It is the same instrument as the ceiling's `linear` arm wearing a classification head instead of a regression head: freeze the representation, fit the simplest possible map, read the result.

**Always read a probe against a floor.** $5.2\%$ looks like failure until you notice chance is $0.42\%$, which makes it roughly twelve times chance on a $237$-way problem. Absolute probe numbers carry no information; only the ratio to a floor does. The reasoning behind weak instruments, and the rest of the diagnostic family, is [Chapter 7a](../../../../docs/experimental-method/07a-probes-and-weak-instruments.md).

## `11` — the variance decomposition: whose spread is wrong?

When a model's predicted population is too narrow, the law of total variance says *which part* is too narrow:

$$\mathrm{Var}[x_g] = \underbrace{\mathbb{E}_z[\mathrm{Var}(x_g \mid z)]}_{\sigma^2_{\text{dec}}, \text{ the readout's noise}} + \underbrace{\mathrm{Var}_z[\mathbb{E}(x_g \mid z)]}_{\sigma^2_{\text{bio}}, \text{ the generator doing its job}}$$

```bash
python examples/perturbation_response/11_diagnose_variance.py --experiment norman_flow_control --model flow --split combo
```

This turned "calibration is bad" into a fork with **opposite consequences**: coverage can be fixed by growing either term, but growing $\sigma^2_{\text{dec}}$ makes the generator *less* visible while growing $\sigma^2_{\text{bio}}$ makes it more so. A single coverage number could never have said that, and the fork is now the spine of two method chapters.

## `15` — empirical epistasis: a model-free target

Computes, straight from the cache with no model involved, how far each double perturbation departs from the sum of its singles:

$$\mathrm{GI}(A,B) = \big\lVert \Delta(A{+}B) - (\Delta(A) + \Delta(B)) \big\rVert \quad \text{on the pair's top-DE genes.}$$

```bash
python examples/perturbation_response/15_empirical_epistasis.py
```

Because $\mathrm{GI}$ is a **norm**, it is unsigned: it says the additive model is wrong but never whether the pair did *more* than its singles predict or *less*. So the script also fits the least-squares scale $\lambda$ that best stretches the additive prediction onto the truth and splits the interaction orthogonally into a magnitude part and a direction part. That split matters because the two flavors are different phenomena, and reporting only $\mathrm{GI}$ conflates them. It also caught a wrong claim: a pair described as synergy from the unsigned number turned out to be the most strongly *sub-additive* one in the set.

This script exists to answer a question before an expensive round starts: **is there any signal here to fit?** If every pair were near-additive, a model predicting epistasis would have nothing to predict.

## The pattern: a diagnostic must be unable to be right for the wrong reason

Every script here has a structural guarantee, and the guarantee is what makes the number worth anything:

- an **oracle** cannot score better than perfect, so its number is a real bound
- a **linear probe** cannot invent structure that is not linearly present, so a good score is real evidence
- an **identity arm** cannot pass unless the harness is aligned
- a **model-free target** cannot be contaminated by the model it will be used to test

Reaching for a more powerful diagnostic feels like rigor and is usually the opposite, because power is what lets an instrument return the answer you were hoping for regardless of the truth.

## What a Diagnose result licenses, and what it does not

**A bound is not a model.** The ceiling's `linear` arm scores $0.852$, above the baseline's $0.766$, and there is no sense in which that is a model you can ship: it is handed the real held-out latents, which a deployed system cannot produce, and it emits a mean rather than a distribution, so it cannot be scored on calibration at all. The honest sentence keeps the condition attached: *if the transition were perfect and the readout were linear, the score would be $0.852$.* That is a statement about headroom.

**A ceiling is specific to what you held fixed.** The $0.679$ oracle is the ceiling *for this decoder*, not a universal bound on the task. Change the readout and it moves, which is exactly what the `linear` arm demonstrates.

---

*Previous: [Score](02-score.md). Next: [Decide](04-decide.md). Up: [the workflow map](index.md).*
