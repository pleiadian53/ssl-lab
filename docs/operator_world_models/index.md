# Operator World Models

*From a model that watches the world to a model that can act on it.*

A JEPA encoder learns what data *means* by predicting it in latent space. That gives you a **map** of the data — but only a map. This series is about adding a **way to move on that map**: latent dynamics you can run forward, condition on real-world interventions, and use to ask "what would happen if…". The unifying object is the **action operator** — a transformation of state — and the claim is that turning JEPA into a world model that carries such operators is a natural extension, not a rebuild.

> **Read the foundations first.** This series assumes two things. From the [Action Operators](../action_operator/00-from-actions-to-operators.md) foundation: what an action operator is, *why* JEPA benefits from integrating one (the [bridge note](../action_operator/01-jepa-action-operators.md)), and a [gallery](../action_operator/02-operator-gallery.md) of concrete operators. From the [Time-Series JEPA](../time_series_jepa/index.md) series: the temporal predictor this work conditions, and the precise blind spot it leaves open. Read both before continuing here — the ideas land far better with the foundations established than by jumping in.

---

## The arc in three moves

The series tells one story in three movements, from passive prediction to actionable dynamics:

1. **JEPA watches.** Standard JEPA learns by observation: hide part of the data, predict its embedding from context. Passive, label-free, and an excellent fit for domains drowning in unlabeled signal.
2. **JEPA as a world model.** Reinterpret "predict the masked region" as "predict the next timestep." Now the predictor is a *dynamics* operator on a latent trajectory $z_t \to z_{t+1}$, and the gap between prediction and reality becomes a **surprise** signal — the substrate for detecting change.
3. **JEPA you can act on.** Vanilla temporal prediction knows only *when* (how far ahead) — never *what acted* in between. Condition the operator on the intervention $c_t$ and you get $f_{\theta(c_t)}$: an operator that says *what happened*. That single change unlocks counterfactual rollout, composable interventions, and a surprise signal cleaned of known causes.

---

## Two application domains, two poles of one dial

The series is anchored by two domains that look unrelated but are the *same construction* with one dial — the **expressiveness ↔ structure** dial — set to opposite extremes. The payoff is that one piece of operator machinery serves both.

### Digital phenotyping — the learned pole

Continuous, passive, multimodal monitoring of a person's behavioral and cognitive state, from wearable and phone signals: the natural home for "a world model of a person." Here the operator's structure is **learned** — you do not know in advance what "a week of more sleep" does to someone's latent state, so the data teaches you. Equivariance is learned, the operators are *associational* dynamics (counterfactual validity must be earned separately), and the payoff is rolling the latent forward under a chosen intervention and flagging when a person departs from their own baseline.

This domain is the regime self-supervised learning was built for: an ocean of unlabeled sensor stream against a trickle of sparse, noisy labels.

### Protein ML — the given pole

3D structure prediction and dynamics, where the relevant transformations are rigid motions in space. Here the operator's structure is **given by physics**: the SE(3) group of rotations and translations. Equivariance is *demanded* as a hard symmetry rather than learned, and correctness is guaranteed by construction. The action-operator framing is strongest on the *dynamic and generative* tasks — placing residue frames one SE(3) operator at a time, modeling conformational change, predicting the effect of an in-silico mutation — as opposed to merely representing a static structure equivariantly.

> **The shared machinery.** Both domains use the same latent operator $f_\theta(z) = \exp(M_\theta) z + b_\theta$ and the same conditioned-prediction objective. What differs is only the **generator basis** that builds $M_\theta$: free and learned for behavior, fixed by the SE(3) algebra for proteins. The range between those poles is the point of the whole framework.

---

## Reading order

**Prerequisites — read first:** the [Time-Series JEPA](../time_series_jepa/index.md) series and the [Action Operators](../action_operator/00-from-actions-to-operators.md) foundation. This series then continues:

| Part | Topic | What you get |
|---|---|---|
| **[0 — What is a world model?](00-what-is-a-world-model.md)** | map vs. world model; rollout + action-conditioning; "not only Time-Series JEPA" | the concept the series rests on — **start here** |
| **[1 — State and latent operators](01-state-and-latent-operators.md)** | $\hat O_\theta$ vs $f_\theta$, the commuting square, Koopman | the keystone: why an action lives in two spaces and how the encoder bridges them |
| **[2 — JEPA as a temporal world model](02-jepa-as-a-temporal-world-model.md)** | the temporal forward pass; predictor as a query operator; rollout | how masked-region prediction becomes latent dynamics |
| **[3 — Conditioning JEPA on actions](03-conditioning-jepa-on-actions.md)** | $f_{\theta(c_t)}$; counterfactual rollout; the surprise signal | the single edit that makes the world model actionable |
| **[4 — Generator bases and the operator in code](04-generator-bases-and-the-operator-in-code.md)** | the basis $\{B_i\}$; the runnable module | the two poles, realized as one swappable piece of code |
| **[Worked example — a personal world model for diabetes](05-worked-example-diabetes.md)** | the whole series threaded through one person's CGM, insulin, carbs, exercise, and medical-code data | every symbol earned by something real — read alongside the parts |

New to the symbols? The [notation reference](notation.md) defines every one. To see them all at work in one story, read the [worked example](05-worked-example-diabetes.md).

---

*Read the [Time-Series JEPA](../time_series_jepa/index.md) and [Action Operators](../action_operator/00-from-actions-to-operators.md) foundations first, then start here with [Part 0 — What is a world model?](00-what-is-a-world-model.md).*
