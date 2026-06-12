# From Actions to Operators

*What it means to treat an action as a transformation of state — and why that small shift opens up a lot.*

This series builds toward world models you can *act on*. Before any of that, we need to be precise about one word: **action**. Most readers meet "action" through reinforcement learning, where it means one thing; this note introduces a more general meaning that the rest of the series depends on. No prior background is assumed — if you know what a function is, you have enough.

> **Where the full story lives.** The action-operator framework is developed in depth in the sibling **GRL** project: <https://github.com/pleiadian53/GRL> (see the series under `docs/action_operator/`). This note is the short, self-contained on-ramp; GRL is the deep reference for the formalism, the operator families, and the learning algorithms.

---

## 1. The familiar picture: an action is a label

In classical reinforcement learning, an **action** is a symbol chosen from a fixed menu — *left*, *right*, *jump*. The agent picks one; the **environment** then interprets that symbol and returns the next situation. Write the situation as a **state** $s$ (everything the agent currently is or knows), and the menu choice as $a$. The environment owns a hidden rule, the *transition*, that turns the pair into a next state $s'$.

The crucial thing to notice is *where the meaning lives*. The action $a$ is just a name. All the substance — what *left* actually does to the world — is locked inside the environment's transition rule, which the agent cannot see or change. The agent chooses a label and waits to find out what it meant.

---

## 2. The shift: an action is a transformation

The action-operator framework makes one move, and everything follows from it:

> An action is not a label the agent *selects*. It is a **function that transforms the state** — an operator the agent *constructs and applies*.

We write such an operator as $\hat O$ (read "O-hat"; the hat marks it as an operator — a function — rather than a number). Its job is to turn a state into its successor:

$$
s' = \hat O(s).
$$

The operator has a type signature, $\hat O : \mathcal{S} \to \mathcal{S}$, meaning it takes a state from the **state space** $\mathcal{S}$ (the set of all possible states) and returns another state in the same space. The meaning that used to hide inside the environment now lives *in the operator itself* — and the agent is the one holding it.

This is less exotic than it sounds. Everyday transformations are already operators on a state:

- **A thermostat nudge.** State = the current temperature; operator = "raise it by two degrees." Applying the operator gives the next temperature.
- **Turning an object in your hands.** State = the object's orientation; operator = "rotate thirty degrees about this axis." Applying it gives the new orientation.
- **Sliding a piece on a board.** State = the board layout; operator = "move this piece one square forward." Applying it gives the new layout.

The framing is closer to the *action principle* in physics — where nature is described by how configurations transform — than to a game controller with a fixed set of buttons.

---

## 3. Two properties that make operators worth the trouble

Treating actions as operators buys two structural properties that plain labels do not have.

**Operators have a size.** Each operator $\hat O$ carries an **energy** $E(\hat O) \ge 0$ — a single number measuring *how large* a transformation it is. A tiny nudge has low energy; a violent rearrangement has high energy. This lets a model *prefer the smallest action that achieves something* — the principle of least action, imported into learning. (Among several operators that reach the goal, take the gentlest one.)

**Operators compose.** Doing one operator and then another is itself an operator. Write $\hat O_2 \circ \hat O_1$ for "apply $\hat O_1$ first, then $\hat O_2$" (read "$\hat O_2$ after $\hat O_1$"). Because actions chain this way, a **skill** is just a sequence of operators glued together. The order generally matters — meds-then-stress need not equal stress-then-meds — which turns out to be a feature, not a nuisance, later in the series.

Both properties have a fuller algebraic treatment (energy bounds, group and Lie structure) in the GRL reference. For this series, the intuitions above — *size* and *composition* — are all you need to carry forward.

---

## 4. A family of operators, indexed by a parameter

We rarely want a single fixed operator; we want a *family* and a way to pick from it. We index the family by a **parameter** $\theta$ (Greek *theta*) and write $\hat O_\theta$ for the specific operator that $\theta$ selects. "Rotate by angle $\theta$" is a one-parameter family; "raise the temperature by $\theta$ degrees" is another. The parameter lives in a **parameter space** $\Theta$, and a **policy** can learn to choose a good $\theta$ for the current state.

This restores something the menu version had — a finite, choosable set of moves — but without giving up the operator's two virtues. Instead of a handful of labels, the agent has a *continuous dial* of transformations, each one a genuine function on the state, each one with an energy and the ability to compose.

> **Key takeaway.** A label points at a transformation the environment hides; an operator $\hat O_\theta$ *is* the transformation, held by the agent — sized by its energy, chainable by composition, and selected by a parameter $\theta$.

---

## 5. The question this raises — where does an operator live?

Here is the tension that powers the rest of the series. The operator $\hat O_\theta$ is defined on the *raw* state $s$ — a person's full physiological state, a protein's every atomic coordinate. That raw state is often enormous, messy, and partly unobservable, and the true operator on it is hopelessly complicated to write down.

But machine-learning models rarely work on raw states. They work on **latents** — compact learned encodings $z$ that capture a state's *meaning* rather than its surface detail. So a natural question appears: can the *same* transformation be described not on the unwieldy raw state, but on its tidy latent encoding? And if so, can the latent version be made far simpler than the original?

That question — one transformation, two spaces, and the bridge between them — is exactly the subject of [Part 1](01-state-and-latent-operators.md). It is where the **state operator** $\hat O_\theta$ (on raw states) meets the **latent operator** $f_\theta$ (on encodings), and where JEPA enters as the machinery that makes the latent version tractable.

---

## 6. The two domains we will follow

To keep the ideas concrete, the series carries two real application domains. You do not need expertise in either; one plain sentence of each is enough for now.

- **Digital phenotyping** — continuously estimating a person's mental and cognitive state from everyday behavioral signals (phone usage patterns, typing, movement, sleep and heart-rate from a wearable), passively and in context. Here an **action** is a real-life intervention: a week of more sleep, starting a medication, a stressful event.
- **Protein ML** — predicting and modeling the 3D shape of a protein. Here an **action** is a geometric move: a rigid rotation-and-translation of the structure, placing a building block into position, or an *in-silico mutation* (changing one component and asking how the shape responds).

These two are deliberately chosen to be opposites. In one, we will have to *learn* what each action does; in the other, physics *tells* us exactly what the action is. The same operator machinery will cover both — which is the payoff the series is built to demonstrate.

---

## Where to go next

- **Next in this series:** [Part 1 — State and latent operators](01-state-and-latent-operators.md).
- **The JEPA connection (companion):** [JEPA as an Action-Operator World Model](../action_operator/01-jepa-action-operators.md) — how predicting masked data and predicting an action's effect are the same learning problem.
- **The full formalism (deep reference):** the GRL action-operator series, <https://github.com/pleiadian53/GRL>.
- **Symbols:** the [notation reference](notation.md) defines every symbol used across the series.

*Series home: [Operator World Models](index.md).*
