# JEPA as a Temporal World Model

*Read Time-Series JEPA's predictor as a latent operator — then roll it forward.*

> **Prerequisite.** [Part 0 — What is a world model?](00-what-is-a-world-model.md), [Part 1 — State and latent operators](01-state-and-latent-operators.md), and the [Time-Series JEPA](../time_series_jepa/index.md) series (whose predictor we re-read here). New to the symbols ($E_\xi$, $g_\phi$, $z_t$, $f_\theta$, $M_\theta$)? Keep the [notation reference](notation.md) open.

We have, by now, three things in hand. [Part 0](00-what-is-a-world-model.md) said a world model is a model of dynamics you can *roll forward* and *condition on actions*. [Part 1](01-state-and-latent-operators.md) gave the keystone: the intractable real-world operator $\hat O_\theta$ has a tractable latent shadow $f_\theta$, and JEPA only ever computes the shadow. And the [Time-Series JEPA](../time_series_jepa/index.md) series built a predictor that advances a latent one step through time.

This chapter connects them. It makes one recognition — that Time-Series JEPA's predictor *is* the latent operator $f_\theta$ — and then exercises the first world-model property, **rollout**, by composing that operator with itself. Along the way two issues surface that the rollout alone cannot fix; naming them is what sets up Part 3 (the second property, action-conditioning). The running thread stays the person's behavioral rhythm from the Time-Series JEPA series.

---

## 1. Time-Series JEPA, in one breath — from a new angle

You built this machine already, so we will not rebuild it; we will *re-read* it. The Time-Series JEPA series framed the method as *"image JEPA with masked-region replaced by future-timestep."* Here is the same machine from the angle this series cares about — **as a one-step latent transition**:

> Encode the recent past into a latent, $z_t = E_\xi(x_{\le t})$. A predictor $g_\phi(z_t, q)$ proposes the *next* latent, $\hat z_{t+1}$, where the query $q$ carries the time offset. Train it to match a slow EMA target encoder's reading of what actually came next, all in latent space:
> $$\mathcal{L} = \big\lVert g_\phi(z_t, q) - \mathrm{sg}(E_{\bar\xi}(x_{t+1})) \big\rVert^2.$$

(For how and *why* each piece works — the online/target encoders, the EMA goalpost, the masking — see the [Time-Series JEPA](../time_series_jepa/index.md) series. Here we need only the shape: *a latent now, a map, a latent next*.)

Read that way, $g_\phi(z_t, \cdot)$ is not "an in-filler for missing timesteps." It is a map that takes *where the system is* to *where it goes next* — a step of dynamics in latent space. Which is exactly the type signature of a latent operator. That re-reading is the whole hinge of this chapter.

---

## 2. The predictor *is* the latent operator

Here is the recognition: **the predictor and the latent operator are the same object.** The [bridge](../action_operator/01-jepa-action-operators.md) said so in its dictionary (predictor $\leftrightarrow f_\theta$); in the temporal setting it is literal — "advance the latent under query $q$" is what $f_\theta$ does. The query is a primitive stand-in for the operator parameter $\theta$, and $g_\phi(z_t, \cdot)$ is $f_\theta$ waiting to be named.

So the action operator is not a *different* thing bolted onto JEPA. It is the operator JEPA already runs — seen clearly — plus two upgrades, one on each axis where the bare predictor falls short of the full operator from the foundation:

**Axis 1 — the form of the map.** The predictor is an unstructured network: whatever map from $z_t$ to $\hat z_{t+1}$ its layers happen to learn, with no committed algebraic shape. The operator from the [gallery](../action_operator/02-operator-gallery.md), $f_\theta(z) = \exp(M_\theta) z + b_\theta$, *imposes* structure — invertibility, near-identity initialization, clean composability — that the bare predictor does not guarantee. Same role, different constraint. (Section 3 shows why that structure pays off.)

**Axis 2 — what the query carries.** The query carries only the offset $\Delta t$: *when* to predict, nothing more. It is set by a fixed schedule, not chosen, and says nothing about *what acted* between $t$ and $t+1$. Upgrading it — a policy that *chooses* the query, carrying an *action* — is Part 3. (Section 5 is where this gap bites.)

| action operator | Time-Series JEPA counterpart | status in the vanilla model |
|---|---|---|
| latent operator $f_\theta$ | predictor $g_\phi(z_t, \cdot)$ | present, but **unstructured** |
| operator parameter $\theta$ | query / offset $q_{\Delta t}$ | present, but carries only *when* |
| policy choosing $\theta$ | the offset schedule | **fixed, not learned** |
| predicted next state $z' = f_\theta(z)$ | $\hat z_{t+1}$ | present |

Read down the last column: the temporal model already has the operator and the next-state prediction; what it lacks is *structure on the map* and *meaning in the query*.

---

## 3. Rolling the operator forward — with numbers

A one-step operator becomes a world model the moment you **iterate** it: predict the next latent, feed that prediction back in as the new state, and predict again.

$$
z_t \longrightarrow \hat z_{t+1} \longrightarrow \hat z_{t+2} \longrightarrow \cdots
$$

Read as an operator $f$ applied repeatedly, a $k$-step rollout *is* the composition $f^k$ — and this is where Axis 1's structure earns its keep. For the structured operator, composition is not an opaque stack of $k$ network calls; it is one matrix:

$$
f^k(z) = \exp(M_\theta)^k z = \exp(k M_\theta) z.
$$

**Make it concrete with the rhythm example.** Recall from the Time-Series JEPA series that, for this person, one day's dynamics is approximately a rotation of the weekly-cycle phase by $360^\circ / 7 \approx 51.4^\circ$ — the gallery's rotation operator, generated by some $M$. Now ask the world model to look a *week* ahead. Iterating the bare predictor means seven successive network calls, each feeding the last. As a structured operator it is a single computation:

$$
f^{7}(z_t) = \exp(7 M) z_t, \qquad 7 \times 51.4^\circ = 360^\circ.
$$

A full turn — so the predicted phase a week out equals today's phase (same weekday, next week), which is exactly what the weekly rhythm should say. From a phase of $100^\circ$ today, the rollout lands back at $100^\circ$ in seven days, in one clean matrix multiply rather than seven chained guesses.

> **Key takeaway.** A world model is a one-step latent operator *iterated*. Putting structure on the operator (Part 1's $\exp(M)$) is what makes a multi-step rollout a single clean, interpretable map instead of an unravelling stack of predictions — which is the rollout half of Part 0's definition, delivered.

---

## 4. The anchoring caveat

There is a seam where rollout can quietly go wrong, and it shapes everything downstream.

During *training*, every prediction is checked against a real encoded observation — the EMA target $E_{\bar\xi}(x_{t+1})$ of an *actually-observed* next step. The target keeps the predictor honest: each step is nudged toward something the world really produced. Predicting tomorrow, in training, is grounded.

During a genuine *rollout* at inference — projecting a week of futures you will never observe — that anchor is gone. You feed the operator's own output back as the new state, with nothing to correct you mid-trajectory, and small errors compound step over step. This is the difference between *predicting an observed-but-masked step* (grounded, what the model trains on) and *predicting an unobserved future* (ungrounded, what a world model does when it imagines).

> **Why this matters next.** With no target to correct it mid-rollout, the model needs an *internal* signal for which predictions to trust — and the natural one is the gap between prediction and reality, the **surprise** residual. That residual is the substrate for detecting change. The trouble, as the next section recalls, is that in the vanilla model the residual is contaminated — the second reason Part 3's conditioning is needed.

---

## 5. The blind spot, recalled — the seam to Part 3

The [Time-Series JEPA](../time_series_jepa/index.md) series ended on this exact limit, so we only recall it here, from the operator's vantage. The predictor's query says *when*, never *what acted*. Between $t$ and $t+1$ the person slept badly, took medication, had a stressful call — none of it reaches the operator. So it can only fit the **average** over those unseen actions, with two costs we now recognize as Axis-2 symptoms:

- **Wrong on each instance, even when right on average** — one operator, forced to cover the bad-night future and the good-night future at once, lands between them.
- **A contaminated surprise signal** — a residual spike could be a genuine destabilization *or* an ordinary unobserved intervention, and the model cannot tell which.

> **The seam.** Both costs trace to the same missing ingredient: the operator has no slot for the action. Give it one — let $\theta$ be *configured by the intervention* rather than fixed by a schedule — and the operator explains the known causes, the rollout becomes controllable (Part 0's second property, action-conditioning, finally delivered), and the surprise residual measures only the *unexplained* part. That single edit is **Part 3**.

---

## Where to go next

- **Next: [Part 3 — Conditioning JEPA on actions](03-conditioning-jepa-on-actions.md)** — the single edit $g_\phi(z_t, q) \to f_{\theta(c_t)}(z_t)$: counterfactual rollout, composable interventions, and a surprise signal cleaned of known causes.
- **The concept:** [Part 0 — What is a world model?](00-what-is-a-world-model.md) — rollout (this chapter) and action-conditioning (next).
- **Concrete operators:** [A Gallery of Operators](../action_operator/02-operator-gallery.md) — what the $\exp(M)$ rollouts of Section 3 actually do.
- **In a real scenario:** [the worked example](05-worked-example-diabetes.md) — rollout and the anchoring caveat applied to a two-week glucose forecast for one person.
