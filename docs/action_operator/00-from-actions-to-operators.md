# From Actions to Operators

*What it means to treat an action as a transformation of state, and why that small shift opens up a lot.*

This material builds toward world models you can *act on*. Before any of that, we need to be precise about one word: **action**. Most readers meet "action" through reinforcement learning, where it means one thing; this note introduces a more general meaning that everything afterward depends on. No prior background is assumed. If you know what a function is, you have enough.

> **Where the full story lives.** The action-operator framework is developed in depth in the sibling **GRL** project: <https://github.com/pleiadian53/GRL> (see the series under `docs/action_operator/`). This note is the short, self-contained on-ramp; GRL is the deep reference for the formalism, the operator families, and the learning algorithms.

---

## 1. The familiar picture: an action is a label

In classical reinforcement learning, an **action** is a symbol chosen from a fixed menu: *left*, *right*, *jump*. The agent picks one; the **environment** then interprets that symbol and returns the next situation. Write the situation as a **state** $s$ (everything the agent currently is or knows), and the menu choice as $a$. The environment owns a hidden rule, the *transition*, that turns the pair into a next state $s'$.

The crucial thing to notice is *where the meaning lives*. The action $a$ is just a name. All the substance is locked inside the environment's transition rule. What *left* actually does to the world, the agent cannot see or change. The agent chooses a label and waits to find out what it meant.

---

## 2. The shift: an action is a transformation

The action-operator framework makes one move, and everything follows from it:

> An action is not a label the agent *selects*. It is a **function that transforms the state**, an operator the agent *constructs and applies*.

We write such an operator as $\hat O$ (read "O-hat"; the hat marks it as an operator, a function, rather than a number). Its job is to turn a state into its successor:

$$
s' = \hat O(s).
$$

The operator has a type signature, $\hat O : \mathcal{S} \to \mathcal{S}$, meaning it takes a state from the **state space** $\mathcal{S}$ (the set of all possible states) and returns another state in the same space. The meaning that used to hide inside the environment now lives *in the operator itself*, and the agent is the one holding it.

This is less exotic than it sounds. Everyday transformations are already operators on a state:

- **A thermostat nudge.** State = the current temperature; operator = "raise it by two degrees." Applying the operator gives the next temperature.
- **Turning an object in your hands.** State = the object's orientation; operator = "rotate thirty degrees about this axis." Applying it gives the new orientation.
- **Taking a medication.** State = a person's physiological and cognitive condition; operator = "a week-long course of this drug." Applying it may lead to an updated health state. (This is exactly the kind of action we will model in *digital phenotyping*, one of the two domains in Section 6.)

The framing is closer to the *action principle* in physics, where nature is described by how configurations transform, than to a game controller with a fixed set of buttons.

---

## 3. Three properties that make operators worth the trouble

Treating actions as operators buys three structural properties that plain labels do not have. The first is the deepest, and it is the one the framework inherited from its precursor, *parametric actions*.

**Operators carry a description, so actions gain *semantics*.** A menu label is atomic. *Left*, *right*, and *jump* are three distinct symbols, and there is no meaningful sense in which any two are "closer" than the third. A label has no inside. An operator does. It can be **described by parameters** $\theta$ (*theta*) that specify *how* it transforms the state: how far to rotate, how many degrees to raise, how strong a dose. Write $\hat O_\theta$ for the operator that the description $\theta$ picks out.

That description changes everything, because it makes actions *comparable*. Two operators whose parameters $\theta$ lie close together transform the state in nearly the same way; they are *similar actions*. A large gap in $\theta$ means a genuinely different effect. A bare menu of names offers only "same or different"; a parametric description offers distances, neighborhoods, and interpolation. The flat list of labels becomes a *space of actions with semantics*, and a model can now learn *over* that space, generalizing from one action to its neighbors instead of treating every action as unrelated. (Section 4 picks up this thread, zooming out from one operator's description to the whole family.)

**Operators have a size.** Each operator $\hat O$ carries an **energy** $E(\hat O) \ge 0$, a single number measuring *how large* a transformation it is. A tiny nudge has low energy; a violent rearrangement has high energy. This lets a model *prefer the smallest action that achieves something*, the principle of least action imported into learning. (Among several operators that reach the goal, take the gentlest one.)

**Operators compose.** Doing one operator and then another is itself an operator. Write $\hat O_2 \circ \hat O_1$ for "apply $\hat O_1$ first, then $\hat O_2$" (read "$\hat O_2$ after $\hat O_1$"). Because actions chain this way, a **skill** is just a sequence of operators glued together. The order generally matters (meds-then-stress need not equal stress-then-meds), which turns out to be a feature, not a nuisance, later on.

All three have a fuller treatment (the parametric-action lineage, energy bounds, group and Lie structure) in the GRL reference. For what follows, three intuitions are all you need to carry forward: *description*, *size*, and *composition*.

---

## 4. A family of operators, configured by a parameter

Section 3 gave a single operator a description $\theta$. Step back and look at the whole **family** it belongs to: let $\theta$ range over all its allowed values and you get $\{\hat O_\theta : \theta \in \Theta\}$, where $\Theta$ (capital *theta*) is the **parameter space**, the set of all allowed descriptions. "Rotate by angle $\theta$" is a one-parameter family; "raise the temperature by $\theta$ degrees" is another; the linear latent operators introduced later are a richer family with $\theta$ a short vector.

It is tempting to say $\theta$ *indexes* the family, and in a precise mathematical sense it does. But "index" quietly imports the wrong picture: a **lookup from a shelf**, as if finished operators $\hat O_1, \hat O_2, \dots$ already sat on a rack and $\theta$ were a shelf number. That picture is only literally true for a finite menu. In the case we care about, where $\theta$ is *continuous*, nothing pre-exists: the operator is **synthesized from $\theta$ on demand by a formula**. So the honest verb is **configure**. $\theta$ is the setting of the knobs that *builds* the operator and fixes how it acts on the state. It is a recipe followed, not an address retrieved.

> **Configure, not look up.** For a continuous parameter, there is no shelf of ready-made operators. $\theta$ is a *complete description*: hand the formula a $\theta$ and it constructs the operator $\hat O_\theta$ from scratch. Configuring *is* selecting, but selecting from a continuum you generate, not a table you browse.

### From a number to a changed state: two worked examples

"Synthesized from $\theta$ by a formula" is easiest to believe once you watch it happen. Two small cases, each showing the full chain: pick a $\theta$, the formula builds the operator, the operator changes the state.

**A shift on the number line.** Take the one-parameter family with formula

$$
\hat O_\theta(s) = s + \theta, \qquad \theta \in \mathbb{R}.
$$

Configure it with $\theta = 3$ and the formula *builds* the operator "add 3." Apply it to the state $s = 7$:

$$
\hat O_{3}(7) = 7 + 3 = 10.
$$

Now configure the *same formula* with $\theta = -2$ and it builds a different operator, "subtract 2," giving $\hat O_{-2}(7) = 5$. Neither operator sat on a shelf. The single formula manufactured each one from the number you handed it. That is "configure" made literal.

**A rotation in the plane.** Take $\theta = \omega$, an angle, with the formula that builds a rotation matrix and applies it to a 2D point $s$:

$$
\hat O_\omega(s) = R(\omega) s, \qquad R(\omega) = \begin{pmatrix} \cos\omega & -\sin\omega \\ \sin\omega & \cos\omega \end{pmatrix}.
$$

Configure with $\omega = 90^\circ$: the formula builds a quarter-turn, and applying it to the point $s = (1, 0)$ swings it to $s' = (0, 1)$. This example also makes the **similarity** property from Section 3 concrete: configure a *nearby* angle, $\omega = 85^\circ$, and you get a *nearby* operator. It sends $(1, 0)$ to about $(0.09, 1.00)$, almost the same place. Close $\theta$, close action. A menu of labels could never say that $85^\circ$ and $90^\circ$ are "almost the same move"; a parametric description says it automatically.

(The first example has the flavor of a phenotyping intervention, *how much* of something to apply. The second has the flavor of a protein geometry move. Both are the same idea: a formula turning $\theta$ into a transformation of the state.)

This restores what the menu had, a choosable set of moves, without giving up any of the operator's virtues. Instead of a handful of unrelated labels, the agent holds a *continuous dial* of transformations: each a genuine function on the state, each sized by its energy, each able to compose, and each *near* its neighbors in $\Theta$. A **policy** then learns to turn that dial, choosing a good $\theta$ for the current state. Because $\theta$ is a complete description, emitting a $\theta$ is the same as committing to a full transformation, which is precisely what lets a policy *act*.

> **Key takeaway.** A label points at a transformation the environment hides; an operator $\hat O_\theta$ *is* the transformation, held by the agent: described by $\theta$, sized by its energy, chainable by composition, and *configured* (built on demand, not looked up) by setting $\theta$.

---

## 5. The question this raises: where does an operator live?

Here is the tension that powers everything that follows. The operator $\hat O_\theta$ is defined on the *raw* state $s$, a person's full physiological state, a protein's every atomic coordinate. That raw state is often enormous, messy, and partly unobservable, and the true operator on it is hopelessly complicated to write down.

But machine-learning models rarely work on raw states. They work on **latents**: compact learned encodings $z$ that capture a state's *meaning* rather than its surface detail. So a natural question appears: can the *same* transformation be described not on the unwieldy raw state, but on its tidy latent encoding? And if so, can the latent version be made far simpler than the original?

That question, whether a transformation can live in a tidy *latent* instead of the unwieldy raw state, is where **JEPA** enters. The [next note](01-jepa-action-operators.md) shows why JEPA's latent space is exactly where the operator becomes practical, and which problems make integrating an action operator essential. The full state-vs-latent mechanics are then developed in the [Operator World Models](../operator_world_models/index.md) series.

---

## 6. The two domains we will follow

To keep the ideas concrete, these notes carry two real application domains. You do not need expertise in either; one plain sentence of each is enough for now.

- **Digital phenotyping.** Continuously estimating a person's mental and cognitive state from everyday behavioral signals (phone usage patterns, typing, movement, sleep and heart-rate from a wearable), passively and in context. Here an **action** is a real-life intervention: a week of more sleep, starting a medication, a stressful event.
- **Protein ML.** Predicting and modeling the 3D shape of a protein. Here an **action** is a geometric move: a rigid rotation-and-translation of the structure, placing a building block into position, or an *in-silico mutation* (changing one component and asking how the shape responds).

These two are deliberately chosen to be opposites. In one, we will have to *learn* what each action does; in the other, physics *tells* us exactly what the action is. The same operator machinery will cover both, which is the payoff this material is built to demonstrate.

---

## Where to go next

- **Next:** [Augmenting JEPA with Action Operators](01-jepa-action-operators.md). Why a passive predictor needs an action, and how the action comes to live in the latent.
- **Operators by example (worked gallery):** [A Gallery of Operators: What θ Does to a State](02-operator-gallery.md). Concrete generators and what each does to a state (a cycle, a perturbation relaxing to baseline, a damped oscillation, an instability flag).
- **Then the world-model series:** [Operator World Models](../operator_world_models/index.md). The full JEPA + operator formalism, which assumes this foundation.
- **The full formalism (deep reference):** the GRL action-operator series, <https://github.com/pleiadian53/GRL>, including more operator *families* (affine, field, potential, kernel).
- **Symbols:** the [notation reference](notation.md) defines every symbol used in this foundation.
