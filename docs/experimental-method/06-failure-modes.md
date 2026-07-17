# Chapter 6 — Failure modes

*A catalogue of the ways an experiment lies to you. Every one of these is real and every one of them happened on this project, most of them more than once. They are not presented as confessions. They are presented as specimens, because each has a recognisable shape, a general form that has nothing to do with biology, and a countermeasure that costs less than the mistake did.*

The reason a catalogue is worth having is that these failures do not announce themselves. None of them crashed anything. In every case the pipeline ran to completion, wrote its reports, and produced numbers that were plausible, stable across runs, and wrong. A broken idea is loud. A broken measurement is quiet, and it is wearing a suit by the time you meet it.

Each section below names the failure, gives the concrete instance, states the general lesson in terms that transfer to any method-development project, and gives the countermeasure that now prevents it.

## 1. A better training loss with a worse model

This is the project's signature failure. It happened **three separate times**, in three different subsystems, built by three different lines of reasoning, and it fooled us at least twice. If you take one thing from this chapter, take this one.

The first instance was **optimal-transport coupling**. Our generative model is trained by flow matching, which learns a velocity field carrying one distribution to another, and the training signal depends on how you pair up source and target samples inside a minibatch. Under independent coupling the pairing is random, so the regression target is noisy: a randomly chosen control cell paired with a randomly chosen perturbed cell can point in a direction that helps nobody. Minibatch optimal transport reorders each batch so that each source is matched to its nearest target, which straightens the paths and lowers the variance of the regression target. This is not a hack. It is a well-motivated, theoretically clean improvement, and it is standard practice.

It worked exactly as advertised. The flow-matching training loss fell from about $1.96$ to about $1.48$, a large and unambiguous improvement. And the effect-size score, the thing we actually care about, got **significantly worse**: $-0.021$, with a simultaneous 95% interval of $[-0.039, -0.002]$ that excludes zero. Not "no better". Worse, by the strict standard of [Chapter 5](05-from-difference-to-verdict.md).

The second instance was a **dispersion anchor** in the count decoder, a term added to make the decoder's predicted variance match the observed variance more closely, and built specifically to improve calibration. It improved the negative-binomial likelihood, with the negative log-likelihood falling from $2879$ to $2852$. It **failed entirely to move the calibration metric it was designed to target.** The likelihood, which is the objective, got better. The quantity the objective was a proxy for did not move.

The third instance was the **stochastic action operator**. Our operator arm learns a transformation carrying a control cell to its perturbed counterpart, trained by matching the transformed cloud against the real one under an energy distance. The stochastic variant was built to widen the predicted distribution, which the variance decomposition had identified as the one thing calibration actually needs. Its training energy distance fell from $0.447$ to $0.428$, so it matched the target cloud *better in latent space*. And the latent distribution's share of the predicted spread, $\sigma^2_{\text{bio}}$, **fell** from $0.174$ to $0.081$, and coverage dropped. It got measurably better at its objective while **reducing the very quantity it was built to increase.**

Three times. Three different objectives, three different subsystems, one pattern.

**The general lesson.** Your training objective is a **proxy**. It was chosen because it is differentiable, because it is stable, and because it correlates with the thing you want, and every one of those reasons is a reason it is not the thing you want. Flow-matching loss is a proxy for generative fidelity. Negative log-likelihood is a proxy for calibration. Energy distance in latent space is a proxy for a downstream effect measured after a decoder. Perplexity is a proxy for whether the model is any good. Reward on the training distribution is a proxy for competence. **Optimizing a proxy harder is not the goal, and past some point it is actively how you get further from the goal.**

The trap is that the proxy improving *feels* like progress in a way that is very hard to argue with. The number went down. The plot is monotone. Something is clearly working. And what is clearly working is the optimizer, which is doing exactly what you told it to, on an objective that is not the one you meant.

The sharpest version of the lesson is this: **a loss that improves while the target metric does not is not noise. It is information, and it is telling you that the gap between your proxy and your goal is real and load-bearing.** That divergence is one of the most informative signals an experiment can hand you, and the instinct to shrug it off ("the loss is better, the metric is noisy, let it go") is the instinct that has to be trained out. In our case, all three divergences turned out to be pointing at something true. Optimal-transport coupling straightens paths in a latent space whose geometry the downstream decoder does not care about. The dispersion anchor was pushing a decoder that is under-dispersed in the direction of being narrower still. The stochastic operator's residual displacement is a constant shift, so it adds zero variance and lets the operator relax back toward doing nothing, which is cheap under the objective and useless under the metric. Each divergence was a diagnosis waiting to be read.

**The countermeasure.** Report the **target metric alongside the loss**, on every arm, in the same table, always. Never let a run be summarized by its loss curve alone. And when the two diverge, treat it as a **finding to investigate**, not a discrepancy to explain away. Write it down. Our results ledger now carries a standing note that says a better training objective is not a better model and that this has happened three times, because the third time we recognised it in about an hour instead of a week, and that speedup came entirely from having written down the first two.

## 2. The two-lever arm

The stochastic operator arm from the previous section has a second problem, and it is the one that makes its result unusable rather than merely disappointing.

That arm changed **two things at once**. It made the operator's coefficients stochastic, drawing them from a learned distribution rather than emitting them deterministically, and it added a per-condition **residual displacement**, a learned constant offset applied to every cell of a given condition. Both were motivated. Both were reasonable. They were expected to be complementary, and the whole point of the arm was to widen the predicted latent cloud, which both changes plausibly serve.

The arm regressed. $\sigma^2_{\text{bio}}$, the latent's contribution to predicted variance, fell from $0.174$ to $0.081$, and coverage fell by $0.037$ with an interval excluding zero. That much is clear.

**And the experiment cannot say why.** We have a strong hypothesis, and it is written up as a hypothesis: the residual displacement is a constant shift, so it contributes exactly zero variance, and it can absorb the mean effect of the perturbation cheaply, which frees the operator itself to relax back toward the identity and stop doing the work that generates spread. That story is coherent, it is mechanistically specific, and it fits every number we have. It is also **unfalsifiable by this experiment**, because the experiment moved both levers together and there is no way, even in principle, to attribute a single outcome to one of two simultaneous changes. Had the stochastic coefficients helped and the residual hurt more, we would see exactly what we see. Had both hurt, we would see exactly what we see. The data cannot distinguish these, and no amount of cleverness applied after the fact will make it.

**The general lesson.** **One lever per arm. Always.** Not "usually", not "unless the changes are obviously complementary", and especially not "unless you are confident". Confidence that two changes are complementary is precisely the state of mind in which people bundle them, and it is worth noticing that if you were genuinely certain how the two interact, you would not need to run the experiment.

The rule has an economic edge that makes it easier to follow. A bundled arm that wins tells you almost nothing, because you do not know which half won, and you will have to run the separated arms anyway to find out. A bundled arm that loses tells you *less* than nothing, because it may well contain a change that works, now hidden under one that does not, and you may discard both. Bundling does not save you a run. It costs you the run you already did.

**The countermeasure.** If you genuinely must bundle, because the two changes are architecturally inseparable or because you are exploring rather than testing, then be clear with yourself about what you have produced: **you have run a demo, not an experiment.** Label it as such in the ledger, do not use it as evidence for or against either component, and do not let it into a results table where it will be read as an ablation. A demo is a legitimate object. It is just not a measurement.

## 3. The silently clobbering fetch

This one is not about statistics or modeling at all. It is about a shell command, and it cost more than either of the failures above.

The setup is ordinary and probably describes your project too. Experiments run on a remote GPU, results land on a remote volume, and you pull them back to your laptop for analysis with a directory sync. Ours was a blanket `rsync` of the remote output tree onto the local one.

The remote volume held a **stale copy of the feature-selection artifact**, the file that determines which genes the metric is computed over, which [Chapter 2](02-the-scoring-seam.md) calls the **support**. So the reports generated on the remote machine were computed on the wrong support: correct code, correct models, correct data, wrong scoring seam, and therefore wrong numbers that looked entirely normal.

Then the blanket sync ran, and it did what a blanket sync does. It **silently overwrote seven locally correct result files with the stale remote ones.** No error, no warning, no prompt. The local results had been right. After the fetch, they were wrong, and nothing anywhere in the system recorded that a substitution had taken place.

The way it was caught is the most instructive part of the whole episode, and the least reassuring. It was caught because **the same analysis command, run twice, returned a different number for the same arm.** That is all. Somebody re-ran something they had already run, noticed the number had moved, and pulled the thread. Had nobody happened to re-run it, the stale numbers would have gone into the ledger, and they would have been plausible, stable, and wrong, which is the exact failure this series exists to prevent.

**The general lesson.** There are two, and both are worth having.

The first: **a number that changes between two runs of the same command is a five-alarm fire.** Not a curiosity, not a caching quirk, not something to note and move past. Determinism is the floor. If an analysis is not reproducible against a fixed set of inputs, then you do not have a measurement, you have a sample from an unknown process, and every conclusion you have drawn from it is provisional. Drop everything and find out why.

The second: **any process that can overwrite results must be explicit about direction and scope.** A blanket directory sync is a loaded weapon pointed at your results. It has no idea which side is authoritative, it will happily propagate staleness in whichever direction you last typed, and it fails silently by design, because overwriting files without comment is precisely its job. This generalises well past `rsync`: it covers a shared checkpoint bucket, a cache directory that is keyed on too little, an artifact store that resolves "latest" differently on two machines, and a notebook that writes back into the directory it read from.

**The countermeasure**, in three parts, all of which we now do.

**Version or hash the artifacts a result depends on, and record them inside the result.** A report should not merely contain numbers. It should contain the identity of the support it was scored on, the checkpoint it came from, and the data split it used, ideally as content hashes. Then a stale artifact is not an invisible substitution, it is a mismatched hash that the next stage can refuse to accept.

**Never blanket-sync a results directory.** Make the fetch **selective**: name the run, name the files, and pull those. A fetch that can only add and can never silently replace is a fetch that cannot cost you a week.

**Make direction explicit.** Know which machine is authoritative for which artifact, write it down, and never run a command whose effect depends on remembering it correctly at the moment you type it.

## 4. The meaningless silence

Two variants of the same failure, from the same week, pointing in opposite directions.

**A silence that looked like failure.** A training job was running on a remote pod, streaming its log back over an ssh session. The stream went empty and the local command exited non-zero. Every visible signal said the job had died. The job was **completely fine**: it ran to completion on the remote machine, wrote its checkpoints, and finished successfully. What had actually died was the ssh session, because a laptop went to sleep. The local exit code was reporting the health of the *stream*, and it was reporting it accurately. It had nothing whatsoever to say about the health of the *job*, and we read it as though it did.

**A message that looked like success.** A provisioning wrapper, used to bring up the GPU pod, printed `pod left running` at the end of its output. It printed this **even when provisioning had failed**, because the message was emitted from a cleanup path that ran unconditionally rather than from a success path that ran only on success. So a failure produced a reassuring message, and the reassuring message was believed, and time was spent looking for a problem in the wrong place.

Put those two together and the shape of the failure is clear. In the first case, absence of a signal was read as evidence of failure. In the second, presence of a signal was read as evidence of success. Both readings were wrong, and both were wrong for the same underlying reason: **the thing being observed was not the thing being asked about.** A log stream is not a job. A wrapper's summary line is not a status. Each is a convenient shadow of the truth, and shadows go missing for reasons that have nothing to do with the object casting them.

**The general lesson.** **Absence of output is not evidence of failure, and a success message you did not verify is not evidence of success.** Every layer of tooling between you and the work introduces a channel that can fail independently of the work, and every such channel will eventually fail in the direction that misleads you, because you only notice the channel when it disagrees with reality and by then you have already believed it.

This generalises directly to anything you run remotely or asynchronously, which is nearly everything: a CI job whose logs got truncated, a training run whose metrics stopped reporting to the dashboard, a worker whose heartbeat stopped, a batch inference call whose progress bar froze. In each case there is a **convenience layer** telling you a story, and an **authority** that knows the truth.

**The countermeasure.** **Check the authoritative source.** For us that is the job queue on the remote machine, which knows whether the job reached `SUCCEEDED`, and which is unaffected by whether a laptop stayed awake. Ask it. Never ask the wrapper, never ask the log stream, and never ask the summary line, because those are all convenience, and the moment a convenience layer disagrees with the authority is the exact moment you needed the authority. This is now written into the operations guide as an explicit instruction: a stream drop is not a failure, confirm with the queue.

## 5. Pseudoreplication

Brief, because [Chapter 1](01-what-are-you-measuring.md) is entirely about the underlying idea, but it belongs in the catalogue because the temptation was real and it is by far the most dangerous item on this list.

Our dataset contains **thousands of cells**. Our test set contains **twenty perturbations**. Every metric in this project could technically be computed per cell, and the resulting sample would have been four orders of magnitude larger. Every confidence interval would have been narrow. Every contrast would have been significant, by a mile, in whatever direction it happened to point.

**And every one of those intervals would have been fiction.** The cells within a perturbation are not independent observations of the thing we are comparing. They are repeated measurements of *the same* thing, which is the model's response to that one perturbation, and they share everything: the same condition embedding, the same operator, the same decoder, the same biology, the same batch. Treating them as independent draws inflates $n$ by a factor of a thousand or more, and shrinks every interval by roughly the square root of that factor, which is how you manufacture a $p$-value of $10^{-9}$ for an effect that does not exist.

**The general lesson.** **Your $n$ is the number of independent units of the thing you are comparing, and it is almost always much smaller than the number of rows in your dataframe.** In an LLM eval it is the number of tasks, not the number of generations you sampled per task. In RL it is the number of environment seeds, not the number of episodes and certainly not the number of steps. In forecasting it is the number of series, not the number of timestamps. The seductive thing about pseudoreplication is that it is the only failure on this list that makes your results *better*, which is exactly why nobody catches it in their own work.

**The countermeasure.** Decide the unit of analysis **before** you compute anything, write it at the top of the analysis script, and make the script physically incapable of computing a statistic at any other level. Ours does: it loads per-perturbation reports, and there is no code path by which a cell-level number could reach the bootstrap.

## The unifying countermeasure

Read the five failures together and they have one shape.

In every case, a stage of the pipeline received input it should have rejected, and processed it happily. The flow trainer received a coupling that was optimizing away from the goal, and trained. The operator arm received two levers, and ran. The report generator received a stale feature-selection artifact, and scored against the wrong support without comment. The fetch received a directory full of correct results, and overwrote them. In not one of these cases did a stage look at what it had been handed, ask whether it made sense, and refuse.

So here is the habit that all of this reduces to, and it is the same one the series index opens with:

**Every stage should assert something about its own output before the next stage consumes it.**

The reason is not that assertions catch bugs, although they do, and the reason is not tidiness. The reason is about **where** you find out. Every bug in this chapter, if not caught at its own stage, is caught at **the metric**, and the metric is the worst possible place in the entire system to discover anything. A metric does not have opinions. It does not know what it is scoring. **It will hand you a confident, plausible, stable number no matter how badly you have fed it**, and it will do so with exactly the same air of authority whether the pipeline above it is correct or in ruins. A bug that reaches the metric does not surface as an error. It surfaces as a *result*, and results get believed.

An assertion moves the discovery **upstream**, to the stage that actually has the context to know what is wrong. The feature-selection stage knows what a valid gene selection looks like. The metric does not, and never will.

The concrete gate, from this project. The feature-selection stage picks the genes the metric will be scored on, and it now **refuses to write its cache** if the selected features are not actually detected in the underlying data. It does not warn. It does not fall back to a default. It does not proceed with a reduced set and a note in the log that nobody reads. It **raises**, and the pipeline stops, and the stale or empty selection never reaches the scorer, because the scorer would have accepted it without complaint and returned a number.

That is the whole discipline, and it fits in a sentence. **Make each stage the last place its own bugs can hide**, because the alternative is a results table, and by then the number is wearing a suit.

---

*Previous: [Chapter 5, From a difference to a verdict](05-from-difference-to-verdict.md). Up: [Running an experiment you can trust](index.md). Next: [The ceiling](07-the-ceiling.md).*

*The project these failures are drawn from is written up in full at [conditional-flow + JEPA](../../examples/perturbation_response/docs/conditional-flow-jepa/index.md), and its running scoreboard, including the standing negative result, is the [results ledger](../../examples/perturbation_response/docs/conditional-flow-jepa/results-ledger.md).*
