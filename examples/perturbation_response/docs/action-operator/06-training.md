# 6. Training an operator

*Chapter 1 gave the operator its meaning through the equivariance loss, and then noted that the loss cannot be computed when the data comes unpaired. This chapter is about what you do instead, and the answer is not a workaround. Matching distributions rather than pairs is the more general objective, of which the paired loss is a special case, and seeing it that way is what keeps the cell setting and the temporal setting the same construction rather than two different ones.*

> **Where this sits.** [Chapter 1](01-what-an-operator-is.md) established the equivariance loss and the crack that opens when cells come unpaired. This chapter closes that crack. It also states the objective-side origin of the identifiability failure that [Chapter 4](04-basis-policy-least-action.md) diagnosed from the parameterization side, and that [Chapter 7](07-the-application.md) then reports as a measured result.

---

## 1. The paired case, where the loss computes directly

When you have a system before and after, training is exactly Chapter 1's bridge. For a transition $s \to s'$ under intervention $\theta$, encode both ends and penalize the gap between the operator's prediction and the encoded truth:

$$\mathcal{L}_{\text{equiv}} = \big\lVert f_\theta(E(s)) - \mathrm{sg}\big(E(s')\big) \big\rVert^2 ,$$

where $\mathrm{sg}(\cdot)$ is the stop-gradient of Chapter 1, holding the target still so the encoder cannot move the goalpost. This is a per-pair squared error, and it computes with no further machinery, because every training example carries its own answer: the same system, one step later.

A temporal world model lives here for free. The state $s$ is a person at time $t$, $s'$ is *that same person* at $t+1$, and you observed the whole trajectory, so every consecutive pair $(s_t, s_{t+1})$ is a labelled example and the sum runs over all of them. Nothing in this section is hard, which is exactly why it is worth being clear that the setting this series is about does not get to use it.

## 2. The crack, restated as a question about coupling

Sequencing a cell's transcriptome destroys the cell. You measure a control cell *or* a perturbed cell, never the same cell before and after, so the pair $(s, s')$ does not exist and $E(s')$ in the loss above has no referent. The per-pair error cannot be formed at all.

The temptation is to treat this as a defect to patch. It is better to ask what precisely is missing, because the answer is narrow. You still have a population of control cells and a population of perturbed cells. What you have lost is only the **coupling**: the knowledge of *which* perturbed cell corresponds to *which* control cell.

> **The reframe.** The object we actually care about is a **distribution**, and pairing is merely *how much we happen to know about it*. With pairs, we know the coupling exactly. Without pairs, we know the two marginals and nothing about the coupling. The operator's job is unchanged in both cases: carry the control distribution onto the perturbed one.

## 3. Matching distributions, and why it generalizes rather than patches

Define the **pushforward** of the operator. Take the whole population of control latents $\{z_b^{(j)}\}_{j=1}^{n}$, apply the operator to each, and collect the results:

$$\big\{ A_p z_b^{(j)} \big\}_{j=1}^{n} \quad = \quad \text{the predicted perturbed cloud.}$$

What you observe is the *real* perturbed cloud $\{z'^{(k)}\}_{k=1}^{m}$, from actual perturbed cells, with $m$ generally different from $n$ and no correspondence to the controls. The goal is to make the predicted cloud look like the observed one, as *shapes*, without ever asking which point matches which.

Here is why this is the general objective and the paired loss is a corner of it. The only quantity that changes between the paired and unpaired settings is whether the coupling is known:

- **Coupling known** (temporal): match the clouds *given* the correspondence, which is exactly the per-pair squared error of §1.
- **Coupling absent** (cells): match the clouds' *marginals*, with a distance that needs no correspondence.

So per-pair MSE is not a different objective from distribution matching. It is distribution matching in the special case where the coupling is handed to you. Build the cell operator against a marginal-matching loss and the temporal paired case is recovered automatically the moment pairs become available. That is what makes the two settings one construction with a dial turned, rather than two methods that happen to share notation.

## 4. The energy distance

A distance between two point clouds that needs no pairing is the **energy distance**:

$$\mathcal{E}(X, Y) = 2 \mathbb{E}\lVert X - Y \rVert - \mathbb{E}\lVert X - X' \rVert - \mathbb{E}\lVert Y - Y' \rVert ,$$

where $X, X'$ are independent draws from the predicted cloud, $Y, Y'$ independent draws from the observed cloud, and $\lVert \cdot \rVert$ is Euclidean distance. In words: twice the average distance *between* the two clouds, minus the average spread *within* each. It is zero exactly when the two clouds share a distribution and positive otherwise, and it is estimated from samples as a handful of pairwise-distance means. Gradients flow into the predicted cloud, which is where the operator is, so minimizing it trains the operator. Its kernel cousin, maximum mean discrepancy, has the same no-pairing property and is a drop-in alternative.

Two things anchor this objective to the rest of the corpus.

**It contains the metric you already score.** The crudest way to match two clouds is to match only their **means**, ignoring shape. Matching cloud means in expression space is precisely the per-gene $\Delta$ that the whole effect-size benchmark is built on. Energy distance and MMD are strictly *richer* matches that also align spread and higher structure. So the operator's training objective is a generalization of the quantity it is graded on, not a different thing bolted alongside it.

**Do not fabricate a coupling.** It is tempting to invent pairs with optimal transport, coupling each control to its nearest perturbed cell, and then train against per-pair error as if the pairing were real. Round 1 of this project found that this *hurts*: lowering the transport-coupled training loss lowered the downstream score, because a forced coupling collapses the genuine spread of responses into a single invented correspondence. For destructive assays the honest default is marginal matching, and a better training loss that is not a better model is a trap this project sprang more than once.

## 5. What the surrogate costs

The move from §1 to §4 is a real generalization, and it is also a real loss, and both halves have to be said.

The equivariance loss of Chapter 1 was what *gave the operator its meaning*: it forced the commuting square, so that $f_\theta$ became the faithful latent image of the real transformation rather than an arbitrary matrix. Marginal matching asks for something weaker. It constrains the operator to carry the control **marginal** onto the perturbed **marginal**, and that is all. It says nothing about which control cell should land where, because by construction it cannot.

The consequence is that **many different operators satisfy it equally well.** Any two operators that push the control cloud to the same shape are indistinguishable to the energy distance, even if they route individual points completely differently. The objective under-determines the map.

> **The premise, and its silent failure.** Distribution matching identifies the operator's *effect on a marginal*, not the operator. When the operator carries more structure than the marginal constrains, that extra structure is fixed by initialization, regularization, and chance rather than by data. Nothing in the training loss reports this, because the loss is measuring exactly the thing it can see, and doing it well.

This is the objective-side origin of a failure this corpus meets twice more. [Chapter 4](04-basis-policy-least-action.md) gave the parameterization-side origin: a dense $65{,}536$-parameter generator per intervention has vastly more freedom than one marginal can pin. The two compound. A richly parameterized operator trained against a marginal-matching loss is under-identified from both directions at once, and quantities read off the surplus structure, the commutator most of all, are then reading directions the data never touched. [Chapter 7](07-the-application.md) reports what that did.

There are honest ways to add identifying constraints back. A low-rank basis (Chapter 4) shrinks the surplus structure until the marginal can pin it. Matching *many* marginals, one per intervention, with generators shared across them, couples the constraints so that a generator is pinned by every combination it participates in rather than by a single cloud. Neither recovers the full information of a true pairing, but both narrow the gap between "the operator's effect is identified" and "the operator is identified," which is the gap this section is about.

## 6. A note on the moving target

One piece of §1's machinery looks idle in the unpaired, frozen-encoder setting and becomes load-bearing again the moment the encoder is unfrozen.

With a frozen encoder the observed cloud $\{z'^{(k)}\}$ is *fixed*: the real perturbed cells encode to the same latents every step, so the energy distance has a static target and the stop-gradient of §1 has nothing to do. This is the regime every round in this project ran.

Unfreeze the encoder, as [Chapter 8](08-what-comes-next.md) considers, and the target starts moving: the observed cloud is now $\{E(\text{perturbed cell})\}$ with $E$ itself under training, so the operator is chasing a distribution that shifts as the encoder learns. The failure this invites is collapse. The encoder can make the objective trivial by mapping every cell to one point, at which point the two clouds coincide, the energy distance is zero, and nothing has been learned. The devices that prevent it are the ones §1 already named and the frozen setting let us ignore: a stop-gradient or an exponential-moving-average target encoder for the observed cloud, so the goalpost cannot chase the operator, and an anchor that keeps the latent distribution from collapsing. They are standard, and they are the price of end-to-end training.

---

*Previous: [Composition](05-composition.md). Next: [the application, and what it taught](07-the-application.md). Up: [the series index](index.md).*
