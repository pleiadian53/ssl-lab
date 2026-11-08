# Round 6 — the bar itself: does the baseline's encoder need the condition?

*Part of [the results ledger](index.md), which holds the standing verdict, the scoreboard and the settings every round shares. Previous: [round 5](round-05-koopman-probe.md) · Next: [round 7](round-07-additive-reference.md).*

**What changed.** Nothing in the method. This round tested the baseline. Every earlier round measured the method against an NB-VAE whose encoder is blind to the perturbation, $q(z \mid x)$, where $x$ is a cell's expression profile, $z$ its latent and $c$ the perturbation's embedding: the perturbation reaches that model only through its decoder. That design follows CPA's basal-state reading, without CPA's adversarial classifier. The textbook conditional VAE also conditions the encoder, $q(z \mid x, c)$. If that form is stronger, the bar every round was measured against was understated, and the program's next step, co-adaptation Phase 1, would be judged against the wrong number. [Chapter 3a](../conditional-flow-jepa/03a-the-models-in-the-head-to-head.md) sets out the design difference.

**The pre-committed endpoint.** Both forms were trained in this round, at seeds 0, 1 and 2, with everything else identical, so the contrast is within the round. The primary endpoint is $\Delta r$ on the twenty held-out combinations, contrast conditioned minus blind, by joint paired bootstrap with a simultaneous interval. Calibration metrics are secondary and reported as intervals only. The decision rule, fixed before any numbers: if the conditioned encoder scores significantly higher, it becomes the bar.

**Verdict: the conditioned encoder wins, and the bar rises to $0.858$.**

| encoder | seed 0 | seed 1 | seed 2 | mean $\Delta r$ |
|---|---|---|---|---|
| blind to the condition, $q(z \mid x)$ | 0.768 | 0.777 | 0.780 | 0.775 |
| reads the condition, $q(z \mid x, c)$ | 0.862 | 0.852 | 0.859 | **0.858** |

| contrast | $\Delta$ | simultaneous 95% interval | verdict |
|---|---|---|---|
| **conditioned − blind** | $\mathbf{+0.083}$ | $[+0.053, +0.113]$ | **significant** |

Every seed of the conditioned encoder beats every seed of the blind one. The secondary metrics move the same way: spread correlation rises from $0.519$ to $0.674$, the 1-Wasserstein distance falls from $0.964$ to $0.865$, and the energy distance from $3.99$ to $3.21$. Coverage barely moves, from $0.324$ to $0.340$, so the readout remains as under-dispersed as before.

**Two checks.** The blind arm reproduces round 1: its mean is $0.775$ against round 1's $0.766$, and seed 0 matches exactly at $0.768$. And no test information can reach the model through the new input. Generation never runs the encoder: it draws $z$ from the prior and feeds it to the decoder with $c$. The held-out combinations therefore never touch the encoder, and the flag changes only what training teaches the decoder.

**What it means.** The method's deficit was larger than the ledger reported. Against the stronger bar, the transport flow's $0.648$ sits about $0.21$ below, not $0.12$, read across rounds through the bar. Co-adaptation Phase 1 was then to be judged against $0.858$; since round 7 it is also compared against the additive reference, as the ledger's [standing verdict](index.md#the-standing-verdict) and [rules](index.md#two-rules-of-this-ledger) describe. The conditioned VAE also lands almost exactly on the linear-readout diagnostic's $0.852$. That is a coincidence across different representations, and the diagnostic is handed real latents, so it should not be read as anything more.

**Why, as a hypothesis.** A blind encoder sees expression that already contains the perturbation's effect, so part of the effect can leak into $z$. The decoder learns to use it, and at generation that part is replaced by a draw from the prior, so the predicted effect is weakened. Telling the encoder the condition lets it account for the effect and keep $z$ for cell-to-cell variation. This round measured the outcome, not the mechanism. A direct test would measure how much perturbation information $z$ carries in each form, for example by probing $z$ for the perturbation's identity.

**Limits.** The interval reflects the twenty test perturbations, not training-seed variance; the seeds were averaged first. The seed ranges, $0.768$ to $0.780$ against $0.852$ to $0.862$, are far apart, so seed noise is not a plausible explanation. The twenty perturbations share genes, which if anything makes the interval too narrow.

**Script and report.** [`run_vae_encoder_ablation_pod.sh`](../../run_vae_encoder_ablation_pod.sh) trains both arms with `08_train_cvae_baseline.py`, the conditioned one with `--encoder-cond`. The arms were graded locally with `09_eval_cvae_baseline.py` and `10_eval_calibration.py`, and compared with `12_compare_arms.py`, which writes `output/vae_encoder_ablation_compare.json`.

---

*Up: [the results ledger](index.md). Previous: [round 5](round-05-koopman-probe.md) · Next: [round 7](round-07-additive-reference.md).*
