# 8. What comes next

*A theory series usually ends at the theory. This one ends at a list of experiments, because the diagnosis in [Chapter 7](07-the-application.md) was specific enough to name them, and because the most valuable thing a careful negative result produces is a short list of things worth trying that you would not have known to try otherwise. Everything here is grounded in a measured finding from the preceding chapters. None of it has been run.*

> **Where this sits.** The forward-looking close. It turns the diagnoses of Chapters 4 through 7 into concrete next steps, in rough order of how directly each attacks a measured bottleneck, and it is honest about which are cheap tests and which are new methods.

---

## 1. The one sentence the whole series earned

Masked-prediction pretraining preserves a great deal of information, which is why the [ceiling analysis](../../../../docs/experimental-method/07-the-ceiling.md) found the frozen latents support a linear readout above the baseline. But **nothing in that objective shapes the geometry of the latent**. It had no reason to make an intervention a small motion, no reason to make the dynamics close linearly, no reason to co-adapt with the decoder that consumes the latent. Every open direction below is a different answer to the same question: *if the encoder will not supply the geometry the downstream stages need, what should we do about it?*

There are two families of answer. Change the encoder, or change what is asked of it downstream. The first is where the leverage is, and it is where most of this chapter goes.

## 2. Unfreeze the encoder

The frozen encoder is the assumption every round held fixed, and three separate findings now converge on relaxing it: the ceiling says the representation was never shaped for the decoder that consumes it, the baseline wins with an architecturally *identical* decoder so the difference is co-adaptation, and [Chapter 2](02-why-linear-koopman.md) says the operator's linearity is unearned without an encoder trained for it. Unfreezing is not one experiment. It is three, and they differ in which downstream objective is allowed to reshape the encoder.

### (a) Co-adapt the encoder with the decoder

Keep the JEPA pretraining as an initialization, then train the encoder and the negative-binomial decoder *jointly* by reconstruction, and train the transition on the co-adapted latents afterward.

This is the cheapest unfreeze and the one that attacks the largest measured loss. The ceiling budget put the decoder's own shortfall at about $0.17$ of effect-size correlation, six times the transition's, and the mechanism was that a frozen latent shaped for masked prediction is not a latent the decoder can read cleanly. Co-adapting the two is the direct test of that. Pre-register it as one hypothesis: *does letting the encoder and decoder shape each other recover the decoder's loss?* Because the transition stays frozen, the result is still attributable, which the fuller unfreezes below give up.

### (b) Co-adapt the encoder with the operator

Train the encoder jointly with the transition, under an objective that rewards a latent in which the intervention's dynamics close linearly. This is [Chapter 2](02-why-linear-koopman.md) §6 made operational: equivariance and Koopman invariance are the same equation, so training toward the equivariance loss *is* training toward Koopman coordinates. It is the direct fix for the geometry failure of [Chapter 7](07-the-application.md), the one that made the fitted operator a large rotation instead of a small motion.

It is also the most delicate, for the reason [Chapter 6](06-training.md) §6 named: once the encoder trains, the transition's target distribution moves, and the trivial escape is collapse. It needs a stop-gradient or exponential-moving-average target encoder and an anti-collapse anchor, which is machinery JEPA already carries. This is the highest-value direction conceptually and the one most likely to require real engineering care.

### (c) Full joint training

Train encoder, transition, and decoder together under one combined objective. This is the most end-to-end and the most CVAE-like, and it is worth being clear about what it becomes: a conditional variational autoencoder with a **conditional flow prior instead of a Gaussian one**, and a **pretrained encoder instead of a random one**. That is a legitimate and interesting model. But at full joint training the method's only surviving distinctions from the baseline are the flow prior and the initialization, so it is no longer testing "does a frozen self-supervised representation help." It is testing whether a flow prior beats a Gaussian one and whether pretraining beats random initialization, which are different and also worthwhile questions.

## 3. The cost every unfreeze shares: attribution

The frozen encoder is the only reason the [ledger](../conditional-flow-jepa/results-ledger.md) can attribute anything. Every clean result in it, the flow tying the baseline, the operator tying the flow, the ceiling saturating the transition, depends on holding the encoder fixed so that a difference between two arms is a difference in the thing that changed. Unfreeze, and a gain could come from a better representation, a better decoder, better co-adaptation, or the transition, with no way to tell which.

This is not a reason to avoid unfreezing. It is a reason to unfreeze *one stage at a time* and to say so. Variant (a) keeps the transition frozen and stays attributable. Variant (c) gives that up entirely. The progression from (a) to (b) to (c) is a progression from an attributable test of one hypothesis to a performance result that cannot be decomposed, and knowing which you are running is the difference between an experiment and a number.

## 4. Fixes that do not require unfreezing

Two directions attack the [Chapter 7](07-the-application.md) failures without touching the encoder, and they are cheaper.

**A low-rank shared basis.** [Chapter 4](04-basis-policy-least-action.md) and [Chapter 6](06-training.md) located the operator's identifiability failure in two compounding places: a dense per-gene generator has far more freedom than one marginal can pin, and marginal matching identifies only the operator's effect, not the operator. A shared low-rank basis, $M_g = \sum_i \beta_{g,i} B_i$, shrinks the surplus structure and, by [Chapter 5](05-composition.md) §5, confines every bracket to $\mathrm{span}\{[B_i, B_j]\}$ rather than letting it point anywhere in $D^2$ dimensions. It is a named mechanism with a clear prediction, and it should be pre-registered like any other round rather than run as a rescue of the refuted claim.

**A named basis where the interventions are loggable.** The gauge argument of [Chapter 4](04-basis-policy-least-action.md) §4 says a free basis cannot support any per-generator interpretation, because its elements are defined only up to a change of basis. Where the interventions can be named and logged, a named basis breaks that gauge and makes the generators identified objects. This is less relevant to a gene screen, where the "basis" is the gene vocabulary, and more relevant to the phenotyping setting, where interventions are sleep, exercise, and medication, and reading what each generator does is a large part of the point.

## 5. The premise that was never tested at all

Every number in this project is a full-data number, and the self-supervised premise the whole approach rests on is not a full-data claim. It says a representation learned on abundant *unlabeled* cells pays off when *labeled* examples are scarce. Nothing here has tested that, because nothing here has varied the amount of labeled data.

The experiment is a subsampling ladder: shrink the labeled cells per perturbation, retrain the transition and the baseline at each rung, and plot effect-size correlation against the amount of data. If the pretrained stack degrades more gracefully than the from-scratch baseline as labels vanish, the method has a real and practically important niche even at full-data parity, and the frozen-encoder design is vindicated exactly where it was meant to matter. This is independent of everything above, it needs only a subsampling flag, and it is the one experiment that puts the founding premise on trial rather than a downstream lever.

## 6. Where the framework belongs

The deepest next step is not on this dataset at all, and [Chapter 7](07-the-application.md) §6 said why. Perturbation response amputated three of the framework's assumptions: it is unpaired, so the objective is a surrogate; it is simultaneous, so half the algebra is unobservable; and its encoder is frozen, so the linearity premise is unearned. A temporal world model repairs all three at once. It is paired, so the equivariance loss computes and the operator recovers its meaning. It is sequential, so the swap-odd part of the algebra becomes observable and the order of a plan is a real quantity rather than a symmetrized shadow. And it trains the encoder jointly, so the Koopman coordinates are learned rather than hoped for.

The [operator world-models](../../../../docs/operator_world_models/index.md) series is where that setting is developed, and the [GRL project](https://github.com/pleiadian53/GRL) is where the reinforcement-learning objective it was built for lives. The honest summary of this whole series is that the operator was tested in the corner where its distinctive machinery is invisible, learned exactly what that corner could teach, and pointed clearly at the setting where the rest of it could finally be seen.

## 7. The list, in order

- **First, and cheapest to decide:** the data-efficiency ladder (§5). It tests the founding premise and needs no new model.
- **Highest measured leverage:** co-adapt the encoder with the decoder (§2a). It attacks the $0.17$ the ceiling attributed to the readout, and stays attributable.
- **Cheap structural test:** the low-rank shared basis (§4). It attacks the identifiability failure without unfreezing anything.
- **Highest conceptual value:** co-adapt the encoder with the operator under a Koopman objective (§2b). It is the direct fix for the geometry failure and the bridge to the temporal setting.
- **The real destination:** the operator in a temporal world model (§6), where paired data, sequential composition, and joint training are all available and the framework can finally be tested whole.

---

*Previous: [The application, and what it taught](07-the-application.md). Up: [the series index](index.md). Where the temporal setting is developed: [operator world models](../../../../docs/operator_world_models/index.md). The record of what has been run: [the results ledger](../conditional-flow-jepa/results-ledger.md).*
