# Modeling the Transition: Action Operators for Perturbation Response

*The effect of a perturbation is how far its operator departs from doing nothing — and the operator turns out to be your transport flow with its velocity field held on a short leash.*

> **Recap — where this sits.** The earlier chapters built a flow-based generator on a frozen JEPA encoder and then, honestly, took it apart. [Challenges and limitations](05-challenges-and-limitations.md) traced the core failure to **baseline dominance**: the model reconstructs a cell's *absolute* state well but estimates the *effect* of a perturbation poorly, because the effect is a small shift on a large, intervention-independent baseline. Two structural culprits followed — the conditioning latent $z_b$ turned out to be **inert** for the mean effect (so the conditional flow bought no edge over a plain conditional VAE), and the decoder was flagged as a shared bottleneck that carries most of the predicted spread. [Beyond the current limit](06-beyond-the-current-limit.md) nominated two levers for closing the gap to a from-scratch NB-VAE, which beats the stack by $0.118$: modeling the **transition** (this chapter) and modeling the **readout** ([Modeling the readout](08-modeling-the-readout-count-decoder.md)).
>
> This chapter opens the transition thread with the **action operator**, developed *cell-first* (a single control-to-perturbed transition) but with every construct kept general, so it is the $T=1$ corner of an operator world model for temporal dynamics. One framing correction up front, because it colors everything below: the operator is **not a rival to the flow**. It is the flow with a structurally restricted velocity field — a claim we earn in §6. The two levers, transition and readout, are complementary: this one models the change; the decoder chapter makes the measurement trustworthy so the change can reach the scoreboard.

> **Prerequisites and notation.** We reuse the JEPA encoder $E$ from the earlier chapters and the per-gene effect vector $\Delta$ and calibration metrics from [Results](04-results.md). Every symbol is defined on first use.

---

## 1. The idea in one line

The metric grades the **change**, not the destination. A perturbation's score is the agreement between predicted and true $\Delta = \operatorname{mean}(\text{perturbed}) - \operatorname{mean}(\text{control})$, per gene, on the genes that actually moved. A method that models the *destination* — "what does a perturbed cell look like" — and subtracts the baseline afterward is spending capacity in the wrong place, because the baseline dominates the destination and washes out of the difference.

An **action operator** flips this. Instead of modeling where the cell lands, it models the *transformation* that carries control to perturbed — and the effect falls straight out as the operator's departure from doing nothing. Modeling the transformation *is* modeling the change.

---

## 2. An action lives in two spaces

Start concrete, with one perturbation from your own data: **activate gene X in a cell.**

What physically happens is a tangled biological cascade. The cell's full expression state — call it $s$, the vector of all $\sim\!20{,}000$ gene counts — becomes a new state $s'$, the perturbed cell. Nobody can write that transformation as a formula; it is the sum of every downstream regulatory consequence of switching on gene X. Call this real, meaningful, *intractable* transformation the **state operator** $\hat O_\theta$:

$$
s' = \hat O_\theta(s).
$$

Reading the symbols: $s$ is the starting (control) state, $s'$ the transformed (perturbed) state, and $\hat O_\theta$ is the operator — a *function that acts on the whole state space*. The hat marks it as an operator rather than a number; the subscript $\theta$ names *which* perturbation it is ("activate gene X" is one $\theta$, "activate gene Y" another). You never hold this object; it lives implicitly in the before-and-after cells it produced.

Now encode. The JEPA encoder $E$ maps a state to a compact latent, $z = E(s)$, a vector (here $z \in \mathbb{R}^{256}$) that captures the cell's *meaning* rather than its raw counts. The **same** perturbation, viewed in latent space, is a second object — a map sending the control latent to the perturbed latent:

$$
z' = f_\theta(z), \qquad z = E(s), \quad z' = E(s').
$$

This is the **latent operator** $f_\theta$. And here is the entire reason for moving to latent space: you get to *choose* $f_\theta$ to be something simple you can actually compute with. The default choice is an affine map,

$$
f_\theta(z) = A_\theta\, z + b_\theta,
$$

read as: *take the latent $z$, multiply it by a matrix $A_\theta$, then add an offset $b_\theta$*. Symbol by symbol: $A_\theta \in \mathbb{R}^{256\times 256}$ is the **operator matrix** — the operator itself; $b_\theta \in \mathbb{R}^{256}$ is a shift vector; and the subscript $\theta$ says both depend on which perturbation was applied.

> **Key distinction.** $\hat O_\theta$ is the operator you *mean* — real, meaningful, intractable. $f_\theta$ is the operator you *build* — a matrix you chose to be simple. They are the same transformation seen in two spaces, and the encoder $E$ is the bridge between them.

### A worked example, before we generalize

Shrink to $D = 2$ so the matrix is legible. Say a control cell lands at $z_b = (2,\,1)$, and the perturbation's operator is

$$
A_\theta = \begin{pmatrix} 1.2 & 0 \\ 0 & 0.8 \end{pmatrix}, \qquad b_\theta = 0.
$$

Then

$$
z' = A_\theta\, z_b = (1.2 \cdot 2,\; 0.8 \cdot 1) = (2.4,\; 0.8).
$$

This operator *amplified* the first latent direction and *damped* the second. The perturbation's **effect** is the displacement

$$
z' - z_b = (0.4,\; -0.2).
$$

Now the anchor to hold for the rest of the series. Suppose $A_\theta = I$, the identity matrix (ones on the diagonal, zeros off it), with $b_\theta = 0$. Then $z' = z_b$: the operator does nothing, the perturbation has no effect.

> **The anchor.** A perturbation's effect is exactly *how far its operator departs from the identity*. Modeling the operator is modeling the change — which is the differential $\Delta$ the metric scores.

Operators need not be diagonal. A matrix like $A_\theta = \begin{pmatrix} \cos\phi & -\sin\phi \\ \sin\phi & \cos\phi\end{pmatrix}$ *rotates* the latent by angle $\phi$, mixing the two coordinates; scaled and combined, such matrices let one perturbation reshape the *directions* of variation, not merely stretch along axes. (The classic additive-shift model of a perturbation — CPA's "add a learned vector" — is the special case $A_\theta = I$, $b_\theta \neq 0$: a pure translation, no reshaping. The operator is the strict generalization that lets $A_\theta$ differ from $I$.)

---

## 3. What makes the built operator mean anything: the bridge

There is a hole in Step 2. On its own, $f_\theta(z) = A_\theta z + b_\theta$ is *just some matrix on vectors*. Pick any $A_\theta$ and you get *a* map — it simply would not correspond to "activate gene X." Something has to force $f_\theta$ to be the faithful latent image of the real $\hat O_\theta$. That something is a training signal routed through the encoder, cleanest to draw as a **commuting square**:

$$
\begin{array}{ccc}
s & \xrightarrow{\ \hat O_\theta\ } & s' \\[4pt]
{\scriptstyle E}\big\downarrow & & \big\downarrow{\scriptstyle E} \\[4pt]
z & \xrightarrow{\ f_\theta\ } & z'
\end{array}
$$

Read it as two routes from the top-left state $s$ to the bottom-right latent $z'$:

- **Transform, then encode** (top then right-down): apply the real operator, $s \to s'$, then encode, $s' \to z'$. In symbols, $E(\hat O_\theta(s))$.
- **Encode, then transform** (down then right): encode first, $s \to z$, then apply your built operator, $z \to z'$. In symbols, $f_\theta(E(s))$.

If those two routes land on the same $z'$, the square *commutes*, and $f_\theta$ is a faithful shadow of $\hat O_\theta$. You do not get this for free — you *make* it hold by penalizing the gap between the two routes:

$$
\mathcal{L}_{\text{equiv}}
= \big\lVert\, \underbrace{E\big(\hat O_\theta(s)\big)}_{\text{encode what truly happened}} \;-\; \underbrace{f_\theta\big(E(s)\big)}_{\text{the operator's prediction}} \,\big\rVert^{2}.
$$

Spelling it out: the left term encodes the *actually transformed* state $s' = \hat O_\theta(s)$ — the ground truth of where the cell went. The right term is the operator's *prediction* of that landing point, computed in latent space without redoing the biology. The squared norm $\lVert\cdot\rVert^2$ is the sum of squared coordinate differences between the two $256$-vectors. Driving this to zero forces $A_\theta$ (and $b_\theta$) to become whatever matrix *is* "activate gene X" in latent coordinates.

In practice the left term is a **stop-gradient** target — written $\operatorname{sg}(\cdot)$ — so no gradient flows back through it. This is the same anti-collapse device as the EMA target encoder in the JEPA pretraining: the goalpost must not chase the predictor. Your first-draft `context_operator.py` implements exactly this loss (`conditioned_jepa_loss`, with `z_target.detach()`).

> **Key insight.** Equivariance is not decoration. It is the *precise condition* that makes $f_\theta$ mean anything as a stand-in for the real transformation. Without the commuting square, the operator is an arbitrary matrix.

---

## 4. The crack: cells come unpaired

Look hard at what $\mathcal{L}_{\text{equiv}}$ *requires*. For a single training example you need **both** $s$ and its own $s'$ — the same system, before and after — so that "encode what truly happened" ($E(s')$) is defined.

In a temporal world model this is free. The state $s$ is a person at time $t$; $s'$ is *that same person* at $t+1$; you observe the whole trajectory, so every consecutive pair $(s_t, s_{t+1})$ is a training example and the loss computes directly. This is the **paired** regime the draft operator was written for.

Now the single-cell reality. Reading a cell's transcriptome by sequencing **destroys** it. You measure a control cell *or* a perturbed cell — never the same cell before and after. So the pair $(s, s')$ does not exist. There is no $s'$ that belongs to a given $s$.

> **The crack.** For destructive single-cell assays, the equivariance loss $\mathcal{L}_{\text{equiv}}$ *cannot be computed at all*, because its basic ingredient — the before/after pair — is missing. The bridge that gives the operator its meaning does not build in the paired form.

This is the same destructive-measurement fact behind the $z_b$-inertness result from [Challenges and limitations](05-challenges-and-limitations.md), now wearing a different face: there it made the conditioning latent uninformative; here it removes the very target of the training objective.

---

## 5. The fix *is* the generalization: match distributions, not pairs

Do not patch this with an ad-hoc trick. The right repair is exactly the generalization principle we want — it makes the cell case a corner of a broader construction rather than a special exception.

The object we actually care about is a **distribution**, and "pairing" is merely *how much we know about it*.

Define the **pushforward** of the operator. Take your whole population of control latents $\{z_b^{(j)}\}_{j=1}^{n}$, apply the operator to each, and collect the results:

$$
\big\{\, O_p\big(z_b^{(j)}\big) \,\big\}_{j=1}^{n} \quad=\quad \text{the predicted perturbed cloud.}
$$

Here $O_p$ is the operator for perturbation $p$, and the braces denote the *set* of transformed points — the model's guess for the cloud of perturbed cells. What you actually observe is the **real** perturbed cloud $\{z'^{(k)}\}_{k=1}^{m}$, from actual perturbed cells (with $m$ generally $\neq n$, and no one-to-one tie to the controls). The goal is simply:

$$
\text{make the predicted cloud} \;\{O_p(z_b^{(j)})\}\; \text{look like the observed cloud} \;\{z'^{(k)}\}.
$$

The only thing that varies between settings is whether you know **which predicted point should match which observed point** — a *coupling*. That single question spans both worlds:

- **Coupling known (temporal).** Consecutive frames hand you the identity coupling: predicted $O(z_t)$ should match *its own* $z_{t+1}$. With the coupling in hand, "match the clouds" collapses to the per-pair squared error — exactly $\mathcal{L}_{\text{equiv}}$ from §3.
- **Coupling absent (cells).** No pairing exists, so you match the **marginals** — the clouds as *shapes* — with a sample-based distance that needs no correspondence.

So per-pair MSE is not a *different* objective from distribution matching; it is the *special case where the coupling is given*. Build the cell operator against a marginal-matching loss, and the temporal paired case is recovered automatically the moment pairs become available.

### The marginal-matching objective, concretely

A distance between two point clouds that needs no pairing is the **energy distance**:

$$
\mathcal{E}\big(X, Y\big) = 2\,\mathbb{E}\lVert X - Y\rVert \;-\; \mathbb{E}\lVert X - X'\rVert \;-\; \mathbb{E}\lVert Y - Y'\rVert,
$$

where $X, X'$ are independent draws from the predicted cloud, $Y, Y'$ independent draws from the observed cloud, and $\lVert\cdot\rVert$ is Euclidean distance in latent space. In words: *twice the average distance between the two clouds, minus the average spread within each cloud.* It is zero exactly when the two clouds share a distribution (in the population limit) and positive otherwise. Estimated from samples, it is a handful of pairwise-distance means:

```python
import torch
from torch import Tensor


def energy_distance(x: Tensor, y: Tensor) -> Tensor:
    """Sample energy distance between two latent point clouds (no pairing needed).

    Args:
        x: (n, D) the PREDICTED perturbed latents, O_p(z_b). Gradients flow here,
           so minimizing this trains the operator.
        y: (m, D) the OBSERVED perturbed latents z'. Treated as a fixed target.

    Returns:
        A scalar >= 0; zero (in the population limit) iff x and y share a
        distribution. Uses  E = 2*mean||x-y|| - mean||x-x'|| - mean||y-y'||.
    """
    d_xy = torch.cdist(x, y).mean()   # average distance BETWEEN the clouds
    d_xx = torch.cdist(x, x).mean()   # average spread WITHIN the predicted cloud
    d_yy = torch.cdist(y, y).mean()   # average spread WITHIN the observed cloud
    return 2.0 * d_xy - d_xx - d_yy
```

What to expect from it (a quick smoke test on $D=8$ Gaussians): two samples of the *same* distribution score $\approx 0.01$ (near zero, a small positive bias because the within-cloud means include the zero self-distances); a cloud shifted by $0.5$, $1.0$, $2.0$ scores $\approx 0.50$, $1.81$, $5.95$ — rising monotonically with the shift; and the loss passes a nonzero gradient into `x`, the predicted cloud, which is precisely the signal that updates the operator. (For a lower-bias estimator, exclude the diagonal by averaging over the $n(n-1)$ off-diagonal pairs; for a loss, the simple form above is standard and fine.) The kernel cousin of this, **MMD** (maximum mean discrepancy — the distance between the two clouds' mean embeddings in a kernel feature space), is a drop-in alternative with the same "no pairing" property.

### The tie back to $\Delta$, and a guardrail

There is a satisfying continuity with the earlier chapters. The *crudest* way to match two clouds is to match only their **means** — ignore shape, align centers. Matching cloud means in expression space is precisely the per-gene $\Delta$ we built the whole metric around. Energy distance and MMD are just *richer* matches that also align spread and higher structure, not only the center. So the operator's training objective is a strict generalization of the quantity you already score.

And one guardrail from your own results: do **not** try to *fabricate* a coupling with naive optimal transport to force pairs into existence. [Challenges and limitations](05-challenges-and-limitations.md) found that lowering the transport-coupled training loss actually *hurt* the downstream metric — a better training objective is not the same as a better score — because forced couplings collapse the response diversity. For cells the honest default is marginal matching, not invented pairs.

There is, however, a catch this objective *cannot* fix on its own — a limit not of the loss but of the operator's expressiveness — and seeing it requires the view in §6.

---

## 6. The operator is a restricted transport flow

Here is the reframe that dissolves the false "operator vs. flow" choice, and it is the strongest form of the whole argument. A rectified flow transports a source distribution to a target by integrating an ordinary differential equation,

$$
\frac{dz}{dt} = v_\eta(z, t), \qquad z(0)\sim\text{source},\ \ z(1)\sim\text{target},
$$

where $v_\eta$ is the learned **velocity field** — at each point and time, the direction and speed the sample moves. Now *restrict* that velocity to be linear in the state and constant in time, $v(z) = M z$ for a fixed matrix $M$. The ODE becomes $\dot z = M z$, a linear autonomous system whose solution is the matrix exponential:

$$
z(t) = e^{Mt}\, z(0) \quad\Longrightarrow\quad z(1) = e^{M}\, z(0) = A\, z(0).
$$

Symbol by symbol: $M$ is the **flow generator** (a matrix living in a flat vector space), $e^{Mt}$ is the matrix exponential evaluated at time $t$, and at $t=1$ it *is* the operator $A = \exp(M)$. (Add a constant forcing, $v(z) = Mz + c$, and you integrate to $z(1) = e^M z(0) + M^{-1}(e^M - I)c$ — the affine operator $Az + b$, bias and all.)

So the action operator is not an alternative to your `--flow-base control` transport. It is *that flow with its velocity field constrained*: affine (linear in $z$), identity-anchored (near-$I$ init $\Leftrightarrow M \approx 0 \Leftrightarrow v \approx 0$), low-rank ($M = \sum_{k\le 16}\alpha_k B_k$ over a small basis), and Frobenius-penalized (least action on $\lVert M\rVert$).

> **Key insight.** The operator *bakes the priors into the map's structure* rather than into a penalty the model may ignore. Baseline dominance (start at identity, effects are small departures) and combo behavior live in the parameterization itself. A free flow has the expressiveness to represent this map but no reason to *find* the structured one; a VAE, which maps noise$\to$outcome with no transport structure at all, cannot express a control-conditioned transport. That is the mechanism by which the operator could beat *both*.

This also has a practical payoff: you implement the operator by *restricting* $v_\eta$ inside the flow you already have, not by building a separate module.

### What the restriction costs: modality

Now the catch §5 promised. $A = \exp(M)$ is always **invertible**, and an invertible smooth map is a diffeomorphism, which *preserves the number of modes* of a density — peaks map to peaks, one-to-one. So a deterministic operator maps a unimodal control cloud to a unimodal cloud; it can rotate, scale, and shear the cloud, but it **cannot split it**. Multimodal fates — some cells to fate 1, others to fate 2 — are exactly "more peaks downstream than upstream," and are therefore unreachable by a deterministic operator. Making the coefficients stochastic (a distribution over $M$) gives a *mixture* of operators that can broaden the cloud, but it only separates it into clean modes if that distribution is itself multimodal and its branches push to well-separated regions — fragile, not something to rely on.

This is **Gap G1**, the multimodal-response gap that motivated the flow in the first place, returning as a hard limit on what a pure operator can represent. It does not hurt effect size, which is a *mean*, but it forfeits the distributional promise.

### The fork: replace, or become the drift

Because the operator is a restricted velocity field, the resolution is to simply *not restrict all of it*. Decompose the velocity into a structured part and a free residual:

$$
v_\eta(z, t, c) = \underbrace{M(c)\,z}_{\text{structured drift = the operator}} \;+\; \underbrace{w_\eta(z, t, c)}_{\text{free residual velocity}}.
$$

Here $M(c)z$ is the operator part — linear, identity-anchored, carrying the mean effect with the right inductive bias — and $w_\eta$ is a small free field the integrator can use to *curve* trajectories apart and manufacture the extra modes. Set $w_\eta \equiv 0$ and the operator **replaces** the flow (clean effect size, G1 abandoned); let $w_\eta$ be a small learned field, regularized toward zero by a least-action penalty (mirroring the Frobenius penalty on $M$), and the operator **becomes the drift** while the residual restores G1 — at modest extra cost.

> **The load-bearing decision.** Whether the operator replaces the flow or becomes its drift determines whether the model improves effect size while quietly abandoning the distributional promise, or keeps both. The recommended default is the drift-plus-residual form with a *dialable* residual penalty: start with $w_\eta$ heavily penalized (almost pure operator, clean effect size) and relax it, watching whether calibration improves without effect size degrading. Build the decomposition so the metric-only-versus-distribution choice never has to be made irreversibly up front.

---

## 7. The generalization map

Because we built each construct as a dial rather than a fixed choice, the cell model and the temporal world model share the same core and differ only in a few settings:

| construct | cell setting (this chapter) | temporal world model (extension) |
|---|---|---|
| velocity restriction $v(z)=M(c)z \Rightarrow A=\exp(M)$, basis $\{B_i\}$, policy $\pi(z,c)\!\to\!\alpha$ | **identical** | **identical** |
| number of applications $T$ | $1$ (control $\to$ perturbed) | many ($z_0 \to z_1 \to \cdots$) |
| context $c$ feeding the policy | perturbation set embedding $e(p)$ | action $c_t$ at each timestep |
| multi-intervention combination | coefficients from the set embedding, $\alpha\!\big(e(\{A,B\})\big)$ — epistasis lives in $\alpha$, not the algebra | sequential product $\exp(M_{c_1})\exp(M_{c_0})$ (roll forward) |
| distributional spread (G1) | residual velocity $w_\eta$ on top of the operator drift | same residual, applied per step |
| training coupling | none → match **marginals** (energy / MMD) | full → per-pair **MSE** (equivariance) |

The top row is the point: the restricted velocity field, its generator basis, and the policy that emits its coefficients are *the same machinery* in both worlds. The cell case sets $T=1$, feeds a perturbation embedding as context, and matches marginals — each the simplest setting of a dial the temporal model turns up.

---

## 8. What we deferred, and where this goes next

This chapter answered *what* the operator is, *why it is a restricted flow*, and *how it learns when cells are unpaired*. Four things were named but not opened, each a natural next step:

- **The form of $A$ (next in the operator thread).** Why the latent operator should be linear at all — the **Koopman** argument, that you choose the encoder so nonlinear state dynamics become linear latent dynamics — and why $A=\exp(M)$ specifically. §6 already gave the spine: $\exp(M)$ is the time-1 map of the *linear* velocity field, so the Koopman story and the flow-restriction story are one story. Invertibility, the near-identity start, and clean composition all follow.
- **Where $A$ comes from.** The generator basis $M = \sum_i \alpha_i B_i$ and the policy $\pi(z, c)\to\alpha$ that emits the coefficients — read against the real `context_operator.py`, and judged for the cell task (a free basis with a policy on the perturbation embedding, versus one named generator per gene).
- **Multi-gene composition and epistasis.** Two senses of "additive" must be kept apart. *Additive basis expansion* — $M = \sum_k \alpha_k B_k$, the generator is linear in its coefficients — is worth keeping; it is what makes the operator a clean low-rank object. *Additive composition* — forcing $M_{AB} = M_A + M_B$, the double equals the sum of the singles — must be dropped, because it hard-codes *no epistasis*, and epistasis is exactly what makes held-out combinations hard. The fix keeps additivity in the algebra but routes the coefficients through a nonlinear set embedding, $\alpha = \alpha\big(e(p)\big)$: then $e(\{A,B\})$ need not equal $e(\{A\}) + e(\{B\})$, so $M_{AB}$ is free to be non-additive and the interaction lives in $\alpha(e(p))$. The honest trade: algebraic additivity generalizes to unseen combos for free but only in a world without epistasis; the embedding route must *learn* combo structure from some observed pairs but *can* represent it.
- **The readout handoff — see [Modeling the readout](08-modeling-the-readout-count-decoder.md).** An earlier draft of this chapter listed "state-dependent dispersion, scoring $\Delta$ on the expected rate" as prerequisites for seeing an operator improvement. That was imprecise, and the decoder chapter corrects it: scoring $\Delta$ on the expected rate is *already done* (effect size reads the mean rate profile with no count sampling), so the decoder's dispersion touches only **calibration**, not effect size. On effect size the relevant decoder component is the softmax **mean head**, whose simplex *attenuates* — does not gate — the low-abundance moved genes; an identity-anchored mean head un-attenuates it. So the operator's effect-size advantage does not wait on the decoder, while its *distributional* advantage (the residual $w_\eta$) does need the decoder's dispersion anchored first, or the calibration metrics cannot see it.

> **Throughline.** The perturbation's effect is the operator's departure from identity; the equivariance loss is what makes the operator *mean* that departure; when cells come unpaired, matching clouds rather than pairs recovers the objective; and the operator is best understood as a transport flow on a short leash — restricted to carry the mean cleanly, with a dialable residual that hands back the multimodal spread when you want it.

---

*Previous: [Beyond the current limit](06-beyond-the-current-limit.md). Next: [Modeling the readout — count decoders](08-modeling-the-readout-count-decoder.md). The operator form, generator basis, and composition continue the transition thread after the readout chapter. Deeper foundation: the operator world-models note* State Operators and Latent Operators.
