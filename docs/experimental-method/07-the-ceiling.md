# Chapter 7 — The ceiling: how good could this possibly get?

*Every previous chapter was about trusting a number once you have it. This one is about a question you should ask before you compute any number at all: can the thing you are about to improve even matter? There is a cheap experiment that answers it, and running it can save you months of improving a component that was never the bottleneck.*

---

## 1. Three rounds of the wrong question

The flow-based generative JEPA we developed for capturing cell perturbation responses (see the [conditional-flow + JEPA](../../examples/perturbation_response/docs/conditional-flow-jepa/index.md) method series) had a pipeline in three stages. An encoder turns a cell into a latent vector. A middle stage, call it the transition, turns a perturbation into a cloud of predicted latents. A decoder turns each latent into predicted gene counts. The benchmark scores the transition: given a perturbation the model has never seen, how well does its predicted response correlate with the real one.

The transition was improved three times. First a generator that mapped noise to outcome, scoring $0.612$. Then a reformulation that transported a real control cell to its perturbed state, $0.648$. Then an action operator with a carefully chosen inductive bias, $0.645$. Each round compared the new transition to the previous one and to a from-scratch baseline that scored $0.766$, and each round asked the same question: is this transition better than the last transition?

Not one round asked the question that turns out to decide everything:

> **Given this encoder and this decoder, what is the best score that *any* transition could achieve?**

That number is a ceiling, and it is measurable without training anything. When we finally measured it, it was $0.679$. The best transition physically possible, on this encoder and this decoder, scores $0.679$. The transition we already had scored $0.648$. Every round of the previous three months had been fighting over the last three percent of a budget that was capped far below the goal, and no amount of cleverness in that stage was ever going to reach the baseline, because the stage was not where the loss was.

This chapter is about the experiment that produces that number, why it is almost free, and how to read it.

## 2. The idea: replace the component with an oracle

Take any stage of a pipeline. It consumes something, it produces something, and the thing it produces flows downstream to a metric. You want to know whether improving this stage can move the metric. The obstacle is that the stage is imperfect, so its output is contaminated by its own errors, and you cannot see past them to ask what a perfect version would achieve.

So build the perfect version. Not by training a better one, which is the expensive thing you are trying to decide whether to attempt. Build it by **substitution**: replace the stage's output with the ground truth it was trying to produce, and run the rest of the pipeline on that.

This is called an **oracle** or a **skyline**, the upper-bound twin of a baseline. A baseline tells you the floor, what you get with the least effort. A skyline tells you the ceiling, what you would get if this one stage were solved perfectly and everything else stayed as it is. The gap between the skyline and where you are now is the **headroom** in that stage. The gap between the skyline and your goal is the verdict on whether the stage can get you there at all.

The construction is specific to each stage but the recipe is always the same. Ask what the stage is *supposed* to produce, obtain the real version of that thing from held-out data, and inject it. In our case the transition is supposed to produce the perturbed latents. The real perturbed latents exist: they are what you get by encoding the actual held-out perturbed cells. So encode them and hand them straight to the decoder, skipping the transition entirely. That is a transition which is perfect by construction, and whatever it scores is the ceiling.

## 3. A ladder of oracles, each removing one suspect

A single skyline gives you one number. A **ladder** of them, each substituting a different amount of ground truth, decomposes the loss and tells you *which* stage is spending it. In our project the ladder had four rungs.

**The identity rung: feed the metric the truth itself.** Before trusting any oracle, check that the metric scores a perfect answer as perfect. Set the prediction equal to the true held-out mean, the exact quantity the metric compares against, and confirm the score is $1.000$. If it is not, the harness is misaligned and every other rung is meaningless. This is the same acceptance-gate habit as [Chapter 2](02-the-scoring-seam.md): a metric that cannot score the truth as perfect cannot be trusted to score anything. Ours returned $1.000$, so the ladder was safe to climb.

**The roundtrip rung: the real latents, through the real decoder.** Encode the actual held-out perturbed cells and decode them. The transition is perfect here; only the encoder and decoder are in the loop. This is the ceiling. It scored $0.679$.

**The latent-mean rung: the real latents, collapsed to their mean.** Feed the decoder a single averaged latent per perturbation instead of the full cloud, to ask whether the *spread* of the latents matters for this metric. It scored $0.706$, slightly *higher* than roundtrip, which is a finding in itself and is taken up in §5.

**The linear rung: the real latents, through a decoder that isn't the real one.** Fit a plain linear map from latents to expression on the training cells, and apply it to the real held-out latents, bypassing the trained decoder entirely. This asks a different question from the others: not "how good is our decoder" but "how much of the answer is present in the latent at all, for *any* readout." It scored $0.852$.

Read as a budget, from a perfect $1.000$ down to what the real transition achieves:

| step | from | to | lost | what the loss is |
|---|---|---|---|---|
| encoder ceiling | $1.000$ | $0.852$ | $0.148$ | what a linear readout of the latent cannot recover: information the encoder did not preserve |
| decoder | $0.852$ | $0.679$ | $0.173$ | what the trained decoder loses relative to a plain linear readout of the same latent |
| transition | $0.679$ | $0.648$ | $0.031$ | what the real transition loses relative to the perfect one |

The stage everyone had been improving was responsible for $0.031$ of the loss. The decoder, which no one had touched during those three rounds, was responsible for $0.173$, almost six times as much. The ladder did not just measure a ceiling. It named the bottleneck, and the bottleneck was not the stage under active development.

## 4. The reading that changes the project

Two conclusions follow directly, and a third one is a surprise the ladder was not built to find.

**The transition is saturated.** It scores $0.648$ against a ceiling of $0.679$. There is at most $0.031$ of headroom in that stage, no matter how it is modeled. The three rounds of work were real and the reformulation genuinely helped, but the stage is now within a whisker of the best it can do given what feeds it and what consumes it. A fourth round aimed at the transition is a round aimed at three percent. This is the single most useful thing to know when deciding what to build next, and it was available for the cost of one script that trains nothing.

**The bottleneck is the decoder.** The largest single loss in the budget, $0.173$, is the trained decoder underperforming a plain linear map on the very same latents. The chapter on the decoder had predicted this in the abstract, that its output parameterization would attenuate the signal, but the ceiling turned the prediction into a measured quantity and showed it was the dominant term. Effort aimed at the decoder has room to move the metric by an order of magnitude more than effort aimed at the transition.

**The representation is better than the whole project's headline said.** The linear rung scored $0.852$. The from-scratch baseline that had been beating the method scored $0.766$. A linear readout of the frozen representation, on held-out perturbations, is well *above* the baseline. The information the method needed was in the latent the whole time. It was being lost downstream, at the decoder, not upstream at the representation. The headline result of the project up to that point had been that the elaborate representation did not earn its keep. The ceiling says the representation was fine and the readout was the problem, which is a completely different project with a completely different next step.

That third conclusion is exactly why the experiment is worth making a habit. It is not only a go/no-go on the current stage. It relocates the problem.

## 5. Two subtleties that are easy to get wrong

**A ceiling is not a result.** The linear rung scored $0.852$, above the baseline's $0.766$, and it is tempting to write down "a linear readout beats the baseline." It does not, in any sense you can ship. The linear rung was handed the *real* held-out latents, which a deployed model does not have; producing them is the transition's job, and the transition is imperfect. The linear readout is also not a generative model. It emits a mean, not a distribution, so it cannot be scored on calibration at all. An oracle rung is a **diagnostic, not a model**. It tells you where the loss lives; it is not a system you can run. Reporting a skyline as if it were an achieved result is its own failure mode, and it is a tempting one precisely because the number is good. Keep the ceiling in the sentence that names it: "if the transition were perfect *and* the readout were linear, the score would be $0.852$," which is a statement about headroom, not about a model that exists.

**A ceiling is specific to what you held fixed.** The roundtrip ceiling of $0.679$ is the ceiling *for this decoder*. It is not a universal upper bound on the task. The baseline scores $0.766$ using its own, differently trained decoder, which is why the baseline can sit above our decoder's ceiling without contradiction. State a ceiling with its conditions attached, always: this is the best the transition can do *given this encoder and this decoder*, and changing either moves the ceiling. The linear rung is what makes this concrete. It changed the readout and the ceiling jumped from $0.679$ to $0.852$, which is the whole point, that the ceiling was a property of the decoder and not of the task.

There is also a small genuine discovery hiding in the latent-mean rung. It scored $0.706$, above the full-cloud roundtrip's $0.679$. Feeding the decoder the averaged latent beat feeding it the real spread of latents. The reason is that the metric compares *means*, the decoder is nonlinear, and the mean of a nonlinear function over a spread-out cloud is not the function of the mean. For a mean-valued metric, latent spread is not just unnecessary, it actively costs a little through the curvature of the decoder. That is a real design fact, that a deterministic transition is not merely sufficient for this metric but slightly preferable, and it fell out of a rung that was included only for completeness. Oracle ladders tend to do this. Because they isolate one factor at a time, they surface effects that a full model blends together.

## 6. Why this is a control, and why it is cheap

[Chapter 4](04-ablations-and-controls.md) drew the line between an ablation, which removes one component to measure its contribution, and a control, which changes the whole setup to answer a different question. An oracle rung is a control of a particular kind: it replaces one stage with a privileged, impossible-in-practice version and asks what the rest of the pipeline would then do. The privilege is the point. You are deliberately giving the stage information it could never have, precisely so that its own limitations drop out of the measurement and you can see everything else clearly.

And it is cheap in the way that matters, which is that it trains nothing. The expensive thing in research is fitting models, and every rung on the ladder is a substitution followed by a forward pass. The real latents are one encode of data you already have. The identity rung is a lookup. The linear rung is a closed-form least-squares solve, seconds of arithmetic, not a training run. The entire ladder in this project ran on a laptop in minutes and replaced a decision that would otherwise have been made by building a fourth transition over a week and discovering it also scored around $0.65$. The asymmetry is the argument: an afternoon of substitution against a week of training, and the substitution is the one that can tell you the training was pointless before you do it.

The pre-commitment discipline from [Chapter 1](01-what-are-you-measuring.md) applies here too, and it is what keeps the ceiling honest. Decide the decision rule before you read the number. Ours was written down in advance: if the roundtrip ceiling comes back near the current score, the transition is saturated and we stop improving it; if it comes back near the goal, the transition has headroom and is worth another round. Writing that down first is what stops the ceiling from becoming a number you rationalize after the fact. A ceiling read honestly redirects the project. A ceiling read motivatedly just joins the results table as one more plausible number.

## 7. Your bottleneck is in some stage. Go find out which.

Nothing in the last six sections is about biology, or even about generative models. The structure is universal: a pipeline of stages, a metric at the end, and a decision about which stage to invest in. The ceiling is how you make that decision with a measurement instead of a hunch. Here is the same experiment in other fields, so you can build the rung that fits your pipeline.

**In retrieval-augmented question answering**, the stages are a retriever and a generator, and the standard mistake is to tune whichever one you find more interesting. The oracle rung for the retriever is **gold-context**: feed the generator the passage you *know* contains the answer, skipping retrieval, and score the answers. If gold-context accuracy is near your current accuracy, retrieval is not your bottleneck and a better retriever will do nothing; the generator cannot use the right passage even when handed it. If gold-context accuracy is far above, retrieval is exactly your bottleneck. The mirror-image rung, a perfect generator, is harder to build, but gold-context alone routinely overturns a team's assumption about which half to work on.

**In a multi-stage agent**, the stages are plan, tool call, and synthesis. Oracle each in turn by hand-writing the correct intermediate: the correct plan, the correct tool outputs, the correct retrieved state. If the agent succeeds when given a correct plan but fails on its own plans, the planner is the bottleneck. If it fails even with a correct plan and correct tool outputs, the failure is in synthesis and a better planner is wasted effort. A handful of hand-constructed oracle traces decides where the next month goes.

**In reinforcement learning**, the oracle is **privileged information**. Train or evaluate a policy that can see the true environment state, or the true dynamics, that the real agent must infer. If the privileged policy is barely better than yours, your perception stack is not the limit and improving state estimation will not help; the limit is in control. If it is far better, perception is the wall. The gap between a privileged expert and your agent is the single most informative number in an embodied project, and it is measured, not guessed.

**In time-series forecasting**, oracle the exogenous inputs. If your forecaster consumes covariates such as weather or promotions, feed it the *true future* values of those covariates, which you have in the historical data even though you will not have them at deployment. If the score with true future covariates is close to your current score, better covariate forecasting will not help you. If it is far above, the covariate uncertainty is your dominant error and that is where to invest.

**In a classification or perception pipeline**, oracle each preprocessing stage. Feed downstream the ground-truth segmentation, or the ground-truth detection boxes, or the perfectly cleaned input, and see how far the final metric jumps. A face recognizer given oracle-cropped faces, a scene parser given oracle depth, a document classifier given oracle OCR: each isolates whether the front of the pipeline is worth improving or whether the back of it cannot use even perfect input. And for the whole task, the **Bayes error**, the irreducible error that even a perfect model would suffer, is the ultimate ceiling; a human-performance estimate is its usual practical stand-in, and a model already near it has no headroom left to chase.

In every one of these, the recipe is identical. Find a stage. Ask what it is supposed to produce. Obtain the true version from held-out data. Inject it. Read how far the metric moves. The stages where the metric jumps are your bottlenecks. The stages where it does not are done, however unsatisfying it is to stop improving them.

So the practice, stated as a practice:

**Before you improve a stage, oracle it.** Substitute the ground truth for that stage's output and measure the ceiling. It trains nothing and it takes an afternoon.

**Build the ladder, not the single rung.** One oracle gives you a ceiling. A sequence of them, each substituting one more stage, gives you the loss budget and names the bottleneck instead of merely bounding it.

**Gate the ladder with an identity rung.** Feed the metric the truth and confirm it scores perfectly, before you trust any other rung.

**Commit to the decision rule first.** Write down what "saturated" and "has headroom" mean as numbers before you look, so the ceiling redirects the work instead of decorating it.

**And never confuse the ceiling with a result.** It is the best you *could* do with a stage solved for free. It is a compass, not a destination.

---

*Previous: [Failure modes](06-failure-modes.md). Up: [Running an experiment you can trust](index.md). The project this is drawn from: [perturbation response](../../examples/perturbation_response/docs/index.md), whose [results ledger](../../examples/perturbation_response/docs/conditional-flow-jepa/results-ledger.md) records the ceiling as the diagnostic that saturated the transition and relocated the problem to the decoder.*
