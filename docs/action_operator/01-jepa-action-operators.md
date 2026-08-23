# Augmenting JEPA with Action Operators

*Why a passive predictor needs an action, and how the action comes to live in the latent.*

> **Builds on** [From actions to operators](00-from-actions-to-operators.md), which introduces the action operator $\hat O_\theta$ (an action read as a function that transforms the state, configured by a parameter $\theta$). This note assumes that idea and asks what happens when you bring it to **JEPA**. New to the symbols ($E$, $\hat O_\theta$, $f_\theta$, $\Theta$)? Keep the [notation reference](notation.md) open alongside this page.

JEPA is an excellent way to learn what data *means*. But for a whole class of problems, the ones that ask *what an action does to a state*, JEPA on its own has a structural blind spot. This note pins down that blind spot precisely, then shows that the action operator from the previous note is exactly what fills it, and that the natural place for the operator to act is JEPA's own latent space.

---

## 1. What JEPA already does well

Most self-supervised learning invents a *pretext task*: corrupt the data in a known way, then ask a model to undo the corruption. Because you applied the corruption, you know the answer for free, with no human labels required.

**JEPA** (Joint-Embedding Predictive Architecture) makes one elegant choice: instead of predicting raw pixels or nucleotides, it predicts the **embedding** of a hidden region from the embedding of the visible context. An encoder $E$, a neural network, maps an input $x$ to a latent vector $z = E(x)$ that captures *meaning* rather than surface form. (This encoder is written $\varphi_\psi$ in the GRL literature; $E$ and $\varphi_\psi$ are the same object.) JEPA masks part of $x$, encodes the rest, and trains a **predictor** to guess the latent of the masked part, comparing against a slow exponential-moving-average copy of the encoder, the *target encoder* $E_{\text{target}}$.

This sidesteps a problem that plagues other methods: predicting exact pixels wastes capacity on unpredictable detail, and designing "two augmented views" of a data point is genuinely hard outside of images. JEPA needs none of that: just *mask and predict in latent space*. For learning a representation, this is complete on its own: it gives you a faithful **map** of the data.

> **The limitation, stated plainly.** JEPA learns by *watching*. It never gets to **act**, to ask "what if I changed this part?" and see what happens. For many problems that is fine. For one important class, it is the whole difficulty.

---

## 2. A class of problems JEPA cannot answer alone

Some questions are not about *what the data is*, but about *what an action does to it*:

- **Digital phenotyping.** How does an intervention (a week of more sleep, starting a medication) change a patient's behavioral and cognitive state? Which intervention helps most?
- **Functional genomics.** How does a gene **knockout** (KO, removing the gene) or **knockdown** (KD, reducing its activity) perturb the cell's expression profile?
- **Splicing.** How does a **mutation** alter splicing? (Splicing is how a gene's pre-mRNA is edited into mature mRNA: stretches called introns are removed and the remaining exons joined, and *which* cuts are made shapes the final protein. A single mutation can change which splice sites are used, so this is a question about what an *edit* does.)

Every one of these has the same shape: *apply this action, predict the effect.* And vanilla JEPA has **no slot for the action**. It can encode the before and the after, but it has no place to put "what was done in between."

---

## 3. Two blind spots, made precise

JEPA's predictor takes two inputs: the context embedding, and a **query** that says which target to predict. Write it $\mathrm{Pred}(E(\text{context}), q)$. The query $q$ carries only *where* or *when*: which masked region (in image JEPA), or how far ahead (in a temporal version). Two distinct deficits follow.

**Blind spot 1: the query is blind (the *where/when* axis).** Which region to predict is sampled *at random*. The model has no notion that some targets are more informative or more valuable than others; it cannot *choose where to look*. There is no active perception.

**Blind spot 2: the prediction is unconditioned (the *what-acted* axis).** Even once the target is fixed, the predictor never sees *what intervened* between context and target. So it is forced to fit the **average** over every unobserved action in the data. Between a patient's two observations they may have slept seven hours, taken medication, or had a stressful call; between two cell measurements a gene may have been knocked out; between two sequences a base may have been mutated, and all of it collapses into one blurred prediction.

> **The gap, in one line.** JEPA's query says *where* and *when*, never *why* or *under what action*. All the causal, exogenous structure (slept-7h, took-meds, gene-KO, base-mutated) is invisible, and the predictor can only average over it.

---

## 4. The fix: give JEPA an action in the latent

The action operator from the [previous note](00-from-actions-to-operators.md) is exactly the missing piece. Replace the bare query with an operator the model configures: the predictor becomes $f_\theta$, a transformation applied to the latent $z$, with the action parameter $\theta$ as its configuration. This closes both blind spots at once:

- a **policy** chooses $\theta$, so the model now selects an informative or valuable target rather than a random one (closing blind spot 1);
- $\theta$ **conditions** the prediction: the operator says *what acted*, so the prediction is no longer an average over unknown interventions (closing blind spot 2).

The crucial design choice is *where* the operator acts. The action operator on the **raw** state, $\hat O_\theta$, is the physically meaningful object, but it is generally intractable, and for problems like phenotyping you have no explicit access to it at all. Its image on the **latent**, $f_\theta$, is the object you can actually compute with: a clean, tractable map such as $f_\theta(z) = \exp(M_\theta) z$. JEPA already encodes everything into a latent; that latent is exactly the right home for the operator.

> **The move.** The intractable real-world action stays implicit in the data; the operator the model *builds and applies* lives in JEPA's latent space, where it can be simple. (The [operator gallery](02-operator-gallery.md) shows concrete $f_\theta$ and what each does to a state; the full state-vs-latent mechanics, namely the commuting square and the Koopman linearization that justify a *linear* $f_\theta$, are developed in the [Operator World Models](../operator_world_models/index.md) series.)

---

## 5. Why this integration is natural: one loss, two readings

The integration is not a bolt-on; JEPA's own training loss is *already* an operator-consistency loss with a fixed, blind operator. Line up the two objectives:

$$
\underbrace{\big\lVert E(\hat O_\theta(s)) - f_\theta(E(s)) \big\rVert^2}_{\textbf{(A) operator equivariance loss}}
\qquad\Longleftrightarrow\qquad
\underbrace{\big\lVert E_{\text{target}}(\text{target}) - \mathrm{Pred}\big(E(\text{context}), q\big) \big\rVert^2}_{\textbf{(B) JEPA prediction loss}}
$$

They are the **same equation** under a simple dictionary:

| JEPA (right) | Operator framework (left) |
|---|---|
| query / position $q$ | action parameter $\theta$ |
| predictor $\mathrm{Pred}(\cdot, q)$ | latent operator $f_\theta$ |
| "reveal region $q$", a *fixed* rule | action operator $\hat O_\theta$, *configured by a policy* |

So **JEPA is the special case where the action is a fixed, passive "look here" that the agent never gets to choose.** Going beyond it means letting a policy configure the operator, which is precisely the two-blind-spot fix from Section 4.

### Reading the two losses in words

**Equation (B): "predict the meaning of what you can't see."** Hide a region, keep the rest as context. Encode the context with $E$ and ask the predictor for the embedding of the hidden region. Compare against the true embedding from the slow target encoder $E_{\text{target}}$. The squared distance is *how far off the guess was*; minimizing it forces representations in which missing parts are predictable from context.

**Equation (A): "predict the effect of an action."** Instead of hiding a region, *transform* the state with $\hat O_\theta$. The left term $E(\hat O_\theta(s))$ is the embedding of the **actually-transformed** state; the right term $f_\theta(E(s))$ is the model's **prediction** of where that transformation lands in latent space, computed without redoing it. Minimizing the gap teaches the model to **anticipate the consequences of its own actions** directly in the latent.

**Why they are the same.** Both have the shape *(embedding of the true outcome) − (a prediction of that outcome)*, squared. JEPA's outcome is an unseen *view*; the operator's outcome is a transformed *state*. JEPA conditions on *which region* ($q$); the operator conditions on *which action* ($\theta$). Swap one for the other and the losses coincide: **predicting masked content and predicting an action's effect are the same learning problem, conditioned on different things.**

---

## 6. Two senses of action, and the blind spot each one closes

The integration splits cleanly into two layers, one per blind spot.

**Sensing: "where to look" (closes blind spot 1).** The model chooses *which* region to examine next, to learn as fast as possible. This is **active perception**: no transformation of the world, only a choice of what to observe, rewarded by how much the choice *teaches*. Informative regions get visited; bland or random ones get ignored. A blind query becomes an active one.

**Perturbing: "what an action means" (closes blind spot 2).** The model applies a real transformation (an intervention, a knockout, an in-silico edit) and predicts the consequence. The **meaning of the action is its effect**: the change it induces in the latent,

$$
\Delta z = E(\hat O_\theta(x)) - E(x).
$$

If a particular edit sharply moves the relevant latent, that edit *matters*. The operator's energy penalty $E(\hat O)$ prefers the *smallest* action that produces the effect, which, for an intervention or a mutation, is a bias toward the true causal change. An unconditioned average becomes a conditioned, interpretable prediction.

The working loop: **sense** to find a candidate, **perturb** to probe its meaning, then **justify** the finding against independent evidence.

---

## 7. Justification: the part that requires care

The "justify" step is where a model can fool itself, so it deserves a clear rule. A predicted effect is only *credible* if it agrees with a **genuinely independent view** of the same phenomenon, one the model cannot influence. This is the heart of multi-view reasoning: if two *independent* windows onto the same hidden truth agree, the truth is probably real; if they share their inputs, agreement may be **shared bias**.

> Grounding a model against another predictor that reads the **same input** is **distillation**, not validation: the model can at best *match* that predictor, never surpass it. Genuine validation needs a view from a *different source*, an actual **measurement** (a perturbation experiment), not another model of the same input.

A frozen predictor is still useful: it gives a fast, non-gameable training signal to get a system off the ground. Just do not mistake "the model agrees with it" for "the model is right."

---

## 8. What this buys, and what is still hard

**Why it is worth doing.** A JEPA encoder gives you a *map* of the data; action operators give you a *way to move on that map*: to run counterfactuals ("what would this patient's state be under more sleep?"), localize what matters, and generate hypotheses. A *stochastic* policy that samples actions rather than committing to one even doubles as a **generative model**: sample an operator, apply it, and you produce a new plausible data point. Prediction, exploration, and generation become facets of one system.

**What is genuinely hard.** Three honest limits:

1. **Associational, not automatically causal.** An operator learned from observational data captures *dynamics conditioned on* an action, not necessarily its *causal* effect. Genuine counterfactual validity needs interventional data or assumptions you must defend separately. Treat $f_\theta$ as "dynamics given the action" until you have earned more.
2. **This is active inference, not classical RL.** A counterfactual edit to a fixed dataset has no observable "next state" to learn from unless you have a simulator or real perturbation data. The signal is *self-consistency plus independent validation*, not an environment's reward.
3. **Sequences of actions are the frontier.** Crediting *which* step in a multi-action skill earned the outcome is the classic hard problem of RL, and it returns the moment operators compose. A research frontier, not the first milestone.

---

## Where to go next

- **The foundation you just built on:** [From actions to operators](00-from-actions-to-operators.md).
- **Operators by example:** [A Gallery of Operators: What θ Does to a State](02-operator-gallery.md).
- **The full formalism:** the [Operator World Models](../operator_world_models/index.md) series. Temporal prediction, conditioning on interventions, and the runnable operator code, all assuming this foundation.
- **The action-operator formalism (deep reference):** the GRL project, <https://github.com/pleiadian53/GRL> → `docs/action_operator/` (*The GRL0 Gap* → *Action-Operator Formalization* → *From Fixed to Learned Kernels*).
- **Symbols:** the [notation reference](notation.md).
