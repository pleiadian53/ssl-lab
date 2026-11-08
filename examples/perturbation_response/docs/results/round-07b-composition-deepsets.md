# Round 7b — the composition ablation: can a non-additive gene-set embedding learn interactions?

*Part of [the results ledger](index.md), which holds the standing verdict, the scoreboard and the settings every round shares. Previous: [round 7](round-07-additive-reference.md) · Next: [round 8](round-08-size-aware-endpoint.md).*

**What changed.** Only how a two-gene perturbation is embedded. Every learned model in this project conditions on a gene-set embedding that composes a combination from its genes. In the form used so far, the *additive* composition, the embedding of a perturbation $p$ is the sum of the learned vectors of the genes it targets,

$$
z_p = \sum_{g \in p} e_g , \qquad \text{so} \qquad z_{A+B} = e_A + e_B ,
$$

where $e_g \in \mathbb{R}^{64}$ is gene $g$'s learned vector. This is what lets a model embed a combination it has never seen, from parts it has, but it also means a pair's embedding is fixed by its singles. Whatever the model predicts for a pair, it cannot be told that this pair behaves differently from the sum of its parts. The *DeepSets* composition removes that constraint:

$$
z_p = \phi\Big(\sum_{g \in p} \psi(e_g)\Big),
$$

where $\psi$ and $\phi$ are two small neural networks, each a two-layer MLP of width 128 with a SiLU activation, applied to every gene's vector and to the pooled sum respectively. The sum keeps the embedding independent of gene order and still builds an unseen combination from trained parts, but $\phi$ is nonlinear, so $z_{A+B}$ no longer has to equal $z_A + z_B$. The round asks whether that freedom lets a model predict how a held-out pair departs from additivity.

**Why it was worth testing, and why the odds were modest.** Round 7 showed that correlation cannot see interactions, so this round could not use $\Delta r$ as its endpoint. And the training data offer little to learn interaction patterns from: the combo split's training side has 91 two-gene combinations over 67 genes, a median of two combinations per gene, and 27 genes that appear in only one. A held-out pair's interaction can be predicted only if interactions follow patterns in the gene vectors that transfer from pair to pair.

## Experimental settings

| arm | model | composition | experiments, seeds 0, 1, 2 |
|---|---|---|---|
| `flow_add` | transport flow | additive | `norman_sweep_control_s{0,1,2}` (existing) |
| `flow_ds` | transport flow | DeepSets | `norman_sweep_control_ds_s{0,1,2}` (new) |
| `vae_add` | conditioned NB-VAE | additive | `norman_vae_cond_s{0,1,2}` (existing, round 6) |
| `vae_ds` | conditioned NB-VAE | DeepSets | `norman_vae_cond_ds_s{0,1,2}` (new) |

- **The change, and nothing else.** Each DeepSets run is its additive twin with `--compose deepsets` added. The flows: `04_train_cond_flow.py --split combo --cond-type geneset --flow-base control --coupling independent`, the transport flow of round 1, on the frozen Stage A encoder (`norman_stage_a`, md5 `38f51067`). The VAEs: `08_train_cvae_baseline.py --split combo --encoder-cond`, the conditioned NB-VAE of round 6. Every run used 60 epochs. The six new DeepSets runs, three per family, were trained on one A40 GPU with [`run_compose_ablation_pod.sh`](../../run_compose_ablation_pod.sh).
- **Reused additive twins.** The additive arms were not retrained. No training code had changed since they ran, and before the round two of them were re-graded from their checkpoints into a fresh output folder: both reproduced their stored $\Delta r$ exactly, $0.6729$ and $0.8619$.
- **Decoder.** Every flow is read out through the same count decoder (md5 `52a44337`), byte-identical across the flow runs.
- **Predictions.** Each run's mean profiles were generated exactly as the standing evaluation generates them: 200 cells per pair, generation seed 0, and for flows 100 Euler steps with guidance 1. The scoring script stops unless its $\Delta r$ reproduces each run's stored report to $10^{-9}$; all twelve did. Each model's own predictions for the single genes, used by one secondary, come from a separate generator seeded 1.
- **Benchmark.** The twenty held-out combinations, each pair's top 20 genes ranked by Wilcoxon $\lvert z \rvert$, expression in log1p-CP10K, the change measured against the mean of all control cells, and each single's measured effect from its training cells. These are the settings every round shares, listed on the [ledger's front page](index.md#what-every-round-shares).

## The pre-committed endpoint

The protocol was fixed in the scoring script's docstring before any DeepSets run was scored. For each pair $p$, write $\Delta_p$ for the measured change, $\hat{\Delta}_p$ for a model's predicted change, $\Delta_A$ and $\Delta_B$ for the measured singles, and

$$
I_p = \Delta_p - \Delta_A - \Delta_B
$$

for the measured interaction, the part of the pair's change that additivity misses. The primary endpoint is the log error ratio

$$
\log r_p = \log \lVert \hat{\Delta}_p - \Delta_p \rVert_2 - \log \lVert I_p \rVert_2 ,
$$

on the pair's top 20 genes. The denominator is exactly the additive reference's own error, so $r_p$ is the model's error as a multiple of additivity's: $\log r_p = 0$ is the additive reference, and negative values beat it. Measured against the singles, a model's error in predicting the interaction equals its error in predicting the pair, which is why this ratio is the size-aware test of interaction prediction.

Aggregation is per pair and seed, then the mean over seeds per pair, then the mean over pairs. Two contrasts form one family, DeepSets minus additive within the flow and within the VAE, by joint paired bootstrap over pairs, 10,000 resamples, seed 0, with max-$t$ simultaneous 95% intervals. The rule: DeepSets improves on its additive predecessor if the interval lies wholly below 0, is worse if wholly above, and otherwise the result is inconclusive. If neither family improves, the additive composition stays the default. Whether any arm beats the additive reference itself is a separate question, reported descriptively and answered later by [round 8](round-08-size-aware-endpoint.md).

## Verdict: DeepSets does not help, and it hurts the VAE

| contrast | mean $\log r$ difference | simultaneous 95% interval | error ratio | pairs where DeepSets is lower | verdict |
|---|---|---|---|---|---|
| flow: DeepSets − additive | $+0.022$ | $[-0.002, +0.045]$ | $1.022$ $[0.998, 1.046]$ | $6/20$ | inconclusive |
| **VAE: DeepSets − additive** | $\mathbf{+0.126}$ | $[+0.031, +0.220]$ | $1.134$ $[1.032, 1.246]$ | $5/20$ | **DeepSets worse** |

The max-$t$ critical value is $2.735$, and all twenty pairs were evaluable. In the flow, DeepSets changes the error by about $2\%$, an interval that just touches zero. In the conditioned VAE, DeepSets raises the error by about $13\%$, significantly. By the rule, the additive composition stays the default in both families.

**Against the additive reference, descriptive only.** Every arm's error is several times additivity's, on every pair: geometric-mean ratios of $4.31$ for the additive flow, $4.41$ for the DeepSets flow, $3.63$ for the additive VAE and $4.12$ for the DeepSets VAE, with no arm below additivity on any pair. For scale, comparing two random halves of each pair's measured cells gives a ratio of about $0.23$, the error a perfect predictor would still show from sampling noise alone.

**Secondary results, intervals only, no verdicts.**

- *Size-aware error* $e_p = \lVert \hat{\Delta}_p - \Delta_p \rVert / \lVert \Delta_p \rVert$, round 8's endpoint: flow $1.261 \to 1.283$, VAE $1.053 \to 1.211$, both rises with intervals above zero.
- *Interaction recovery*, the correlation between predicted and measured departure from additivity against the measured singles: flow $0.222 \to 0.212$, VAE $0.319 \to 0.274$, both intervals including zero.
- *The model's own composition*, the same correlation computed against each model's own predicted singles: the VAE's rises from $-0.113$ to $-0.006$, an interval of $[+0.040, +0.179]$ for the change, and the flow's barely moves, $-0.039$ to $-0.034$. DeepSets stops the VAE from composing its singles in the wrong direction, but it does not make it compose them in the right one.
- *$\Delta$-correlation*: flow $0.648 \to 0.628$, VAE $0.858 \to 0.752$, both falls with intervals below zero.
- *The ten most strongly interacting pairs*: the same pattern, flow $+0.009$ $[-0.006, +0.025]$ and VAE $+0.062$ $[+0.001, +0.129]$ in mean $\log r$.

**What it means.** Giving the embedding the freedom to depart from additivity did not let either model predict held-out interactions, and in the stronger model it cost accuracy on every endpoint. A likely reason, not tested here, is the data: with a median of two training combinations per gene, there is little evidence from which a nonlinear composition could learn interaction patterns that transfer to unseen pairs, and the extra flexibility is spent fitting the training pairs instead. The constraint the additive sum imposes is not what keeps the learned models above additivity's error.

**What it does not mean.** It does not show that interactions are unlearnable, only that this composition, on 91 training combinations, does not learn them. It tests one non-additive form; others, for example ones that use prior knowledge about genes, are untested. It is scoped to the twenty held-out pairs and their top 20 genes.

**Script and report.** [`run_compose_ablation_pod.sh`](../../run_compose_ablation_pod.sh) trains the DeepSets arms; [`19_score_interaction.py`](../../19_score_interaction.py) scores every run; [`20_compare_composition.py`](../../20_compare_composition.py) holds the frozen protocol in its docstring and writes `output/composition_ablation_compare.json`. The composition lives in `GeneSetEmbedding` in `ssllab.generative.condition`, the metrics in `ssllab.eval.interaction` and the bootstrap in `ssllab.eval.bootstrap`.

---

*Up: [the results ledger](index.md). Previous: [round 7](round-07-additive-reference.md) · Next: [round 8](round-08-size-aware-endpoint.md).*
