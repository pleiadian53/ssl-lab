# JEPA as an Action-Operator World Model

*Learning not just to predict the world, but to act on it.*

> **Companion documents**
> - **Notation reference** (every symbol, with a "read as" column): [notation.md](notation.md)
> - **The action-operator formalism** lives in a separate project, **GRL**:
>   <https://github.com/pleiadian53/GRL> — see the three-part series under `docs/action_operator/`
>   (*The GRL0 Gap* → *Action-Operator Formalization* → *From Fixed to Learned Kernels*).

This note connects two ideas that are usually discussed apart: **JEPA**, a self-supervised way to learn what data *means*, and **action operators**, a way to formalize what it means to *act*. The claim is that the second happens to be a natural extension of the first — and that the extension is what turns a passive predictor into something that can explore, hypothesize, and test.

New to the symbols ($\varphi_\psi$, $\hat O_\theta$, $\Delta(\Theta)$, …)? Keep [notation.md](notation.md) open alongside this page.

---

## 1. Where we start: JEPA learns meaning by predicting

Most self-supervised learning invents a *pretext task*: corrupt the data in a known way, then ask a model to undo the corruption. Because you applied the corruption, you know the answer for free — no human labels required.

**JEPA** (Joint-Embedding Predictive Architecture) makes one elegant choice: instead of predicting raw pixels or nucleotides, it predicts the **embedding** of a hidden region from the embedding of the visible context. Concretely, an encoder $\varphi_\psi$ — a neural network with weights $\psi$ — maps an input $x$ to a latent vector $z = \varphi_\psi(x)$ that captures *meaning* rather than surface form. JEPA masks part of $x$, encodes the rest, and trains a **predictor** to guess the latent of the masked part.

This sidesteps a problem that plagues other methods: predicting exact pixels wastes capacity on unpredictable detail, and designing "two augmented views" of a data point is genuinely hard outside of images (what is a sensible augmentation of an RNA sequence?). JEPA needs none of that — just *mask and predict in latent space*.

But notice what JEPA does **not** do. It learns by *watching*. It never gets to **act** — to ask "what if I changed this part?" and see what happens. A baby learns the world not only by observing it but by poking it. That missing half is where action operators come in.

---

## 2. The other idea: an action is an *operator*

In classical reinforcement learning, an "action" is a symbol picked from a menu: *left, right, jump*. The **action-operator** framework (developed in the [GRL project](https://github.com/pleiadian53/GRL)) replaces that menu with something far more general:

> An action is a **function that transforms the state** — an operator $\hat{O}:\mathcal{S}\to\mathcal{S}$
> the agent *constructs and applies*, rather than a label it selects.

This is closer to the *action principle* in physics than to a game controller. An operator carries an **energy** $E(\hat{O})$ measuring how big a transformation it is, which lets you prefer *small, parsimonious* actions — the least-action principle, imported into learning. Operators also **compose** ($\hat O_2 \circ \hat O_1$), so a *skill* is just a sequence of operators. (The full formalism — generators, operator families, algebraic structure — is the GRL series linked above.)

---

## 3. The bridge: JEPA's "mask" is a frozen action operator

Here is the connection. Line up the two training objectives:

$$
\underbrace{\big\lVert \varphi_\psi(\hat{O}_\theta(s)) - f_\theta(\varphi_\psi(s)) \big\rVert^2}_{\textbf{(A) operator equivariance loss (GRL)}}
\qquad\Longleftrightarrow\qquad
\underbrace{\big\lVert \varphi_{\bar\psi}(\text{target}) - \mathrm{Pred}\big(\varphi_\psi(\text{context}), p\big) \big\rVert^2}_{\textbf{(B) latent prediction loss (JEPA)}}
$$

They are the **same equation** under a simple dictionary:

| JEPA (right) | Operator framework (left) |
|---|---|
| position/mask token $p$ | action parameter $\theta$ |
| predictor $\mathrm{Pred}(\cdot, p)$ | feature-space operator $f_\theta$ |
| "reveal region $p$" — a *fixed* rule | action operator $\hat O_\theta$ — *chosen* by a policy |

So **JEPA is the special case where the action is a fixed, passive "look here" mask that the agent never gets to choose.** Going *beyond* JEPA means letting a **policy** $\pi$ choose the operator, and rewarding that choice by how much it teaches.

### Reading the two equations (in words)

If the math above is dense, here is what each side is *saying*.

**Equation (B), JEPA's loss — "predict the meaning of what you can't see."** Take a data point, hide a region, and keep the rest as context. Encode the context with $\varphi_\psi$ and ask the predictor: *what is the embedding of the hidden region?* Compare that guess to the true embedding, computed by a slow "target" copy of the encoder, $\varphi_{\bar\psi}$. The squared-distance $\lVert\cdot\rVert^2$ is simply *how far off the guess was*. Minimizing it forces the encoder to build representations in which missing parts are predictable from context — i.e. representations that have captured the data's structure.

**Equation (A), the operator loss — "predict the effect of an action."** Now instead of hiding a region, *transform* the state with an operator $\hat O_\theta$ (for example, an in-silico edit to a sequence). The left term, $\varphi_\psi(\hat O_\theta(s))$, is the embedding of the **actually-transformed** state. The right term, $f_\theta(\varphi_\psi(s))$, is the model's **prediction** of where that transformation *should* land in latent space, computed without redoing the transformation from scratch. Minimizing the gap teaches the model to **anticipate the consequences of its own actions** directly in latent space.

**Why they are "the same."** Both losses have the identical shape: *(embedding of the true outcome) − (a prediction of that outcome)*, squared. JEPA's "outcome" is an unseen *view*; the operator's "outcome" is a transformed *state*. JEPA's "prediction" is conditioned on *which region* ($p$); the operator's is conditioned on *which action* ($\theta$). Swap "which region" for "which action" and one becomes the other. That is the whole bridge: **predicting masked content and predicting action effects are the same learning problem, conditioned on different things.**

**One honest caveat.** This clean correspondence holds for *transformation* actions — edits that genuinely produce a new state. It does **not** apply to pure *sensing* actions ("look over there"), which change what you've observed but don't transform the world. Those are active *perception*, and they map onto JEPA even more directly. The next section keeps the two straight.

---

## 4. Two kinds of action: sensing and perturbing

The extension splits naturally into two layers. A good test question — *"where are the splice sites in this RNA, and what do they mean?"* — needs both.

**Layer A — Sensing ("where to look").** The agent chooses *which* region to examine next, to learn as fast as possible. This is **active perception**: there is no transformation of the world, only a choice of what to observe. Reward the agent for choosing regions that *teach it the most*. Informative regions — like the boundaries that mark a splice site — get visited; bland or purely random regions get ignored. This answers *where*.

**Layer B — Perturbing ("what does it mean").** The agent applies a real transformation — an in-silico edit, like changing one nucleotide — and predicts the consequence. The **meaning of the edit is its effect**: the change it induces in the latent, $\Delta z = \varphi_\psi(\hat O_\theta(x)) - \varphi_\psi(x)$. If editing a particular base sharply changes the predicted splicing latent, that base *matters*. The energy penalty $E(\hat O)$ prefers the *smallest* edit that produces the effect — which, biologically, is a bias toward the true causal change. This answers *what it means*.

The loop: **sense** to find a candidate site, **perturb** to probe its meaning, then **justify** the finding against independent evidence.

---

## 5. Justification: the part that requires care

The "justify" step is where a model can fool itself, so it deserves a clear rule.

A perturbation's predicted effect is only *credible* if it agrees with a **genuinely independent view** of the same phenomenon — one the model cannot influence. This is the heart of multi-view / co-training reasoning: if two *independent* windows onto the same hidden truth agree, the truth is probably real; if they share their inputs, agreement might just be **shared bias**.

This has a sharp practical consequence worth stating plainly:

> Grounding a model against another predictor that reads the **same input** (e.g. another
> sequence-based splice predictor) is **distillation**, not validation — the model can at best *match*
> that predictor, never surpass it. Genuine validation needs a view from a *different source* — for
> biology, an actual **measurement** (a perturbation experiment), not another sequence model.

A frozen predictor is still useful: it gives a fast, non-gameable training signal to get a working system off the ground. Just don't mistake "the model agrees with it" for "the model is right."

---

## 6. What this buys, and what's still open

**Why it's worth doing.** A JEPA encoder gives you a *map* of the data. Action operators give you a *way to move on that map* — to run counterfactuals, localize what matters, and generate new hypotheses. A *stochastic* policy (one that samples actions rather than committing to one) even doubles as a **generative model**: sampling an operator and applying it produces a new, plausible data point. Prediction, exploration, and generation become facets of one system.

**What's genuinely hard.** Two honest limits:

1. **This is active inference, not classical RL.** A counterfactual edit to a fixed dataset has no observable "next state" to learn from unless you have a simulator or real perturbation data. The learning signal is *self-consistency plus independent validation*, not a reward from an environment.
2. **Sequences of actions are the hard part.** Crediting *which* step in a multi-action skill deserves the reward is the classic hard problem of RL, and it re-enters the moment actions compose. It should be the research frontier — not the first milestone.

---

## 7. Where to go next

- **Symbols & notation:** [notation.md](notation.md)
- **The action-operator formalism (GRL project):** <https://github.com/pleiadian53/GRL> → `docs/action_operator/`
  1. *The GRL0 Gap — From Parametric Actions to the Operator Imperative*
  2. *Action-Operator Formalization — Definitions, Families, Implementation*
  3. *From Fixed to Learned Kernels — the Feature-Map Extension* (the equivariance / Koopman link that makes the bridge above precise)
- **Background on JEPA and the SSL families it grew from:** the self-supervised-learning tutorial in this project.

---

*This is an evolving research note. Feedback and pull requests welcome.*
