# Round 12 — co-adaptation Phase 1: does training the encoder and the decoder together raise the stack's readout ceiling?

*Part of [the results ledger](index.md), which holds the standing verdict, the scoreboard and the settings every round shares. Previous: [round 11](round-11-library-size.md) · Companion: [the re-baseline after round 11](rebaseline-after-round-11.md).*

**What changed.** Until now the JEPA encoder was frozen after pretraining, and the count decoder learned to read whatever representation it was handed. This round lets the two shape each other. Starting from the pretrained encoder, a fresh count decoder and the encoder are trained together by reconstruction, with the negative-binomial likelihood and the library size corrected in [round 11](round-11-library-size.md). A second arm does the same while maintaining the JEPA predictive objective during reconstruction fine-tuning. The flow is not co-trained; it is retrained afterwards on each moved encoder's latents, with the round-1 recipe.

**Why this question.** After round 11, the stack's readout ceiling, real held-out cells read through the frozen encoder and the corrected decoder, scores $0.865$ in $\Delta$-correlation. The conditioned cVAE scores $0.939$ by generating. The transition has about $0.03$ of headroom under that ceiling, so even a perfect flow could not reach the cVAE. About $0.135$ is lost before the transition: by what the frozen encoder keeps of each cell, as read by either the decoder or a linear map. Joint training aims at that loss directly.

## Experimental settings

| arm | what it is | runs, seeds 0, 1, 2 |
|---|---|---|
| `dec` | the encoder frozen, decoder only (within-round control); round 11's runs, trained by the same trainer and recipe | `norman_r11_dec_hvg_s{s}` |
| `joint` | every encoder parameter trains with the decoder; the JEPA predictor is unused | `norman_r12_joint_s{s}` |
| `joint_anchor` | joint, while maintaining the JEPA predictive objective with a fixed weight | `norman_r12_joint_anchor_s{s}` |

- **Trainer.** [`co-adaptation/02_coadapt_decoder.py`](../../co-adaptation/02_coadapt_decoder.py), the module `ssllab.coadapt`. The encoder starts from `norman_stage_a` (md5 `38f51067`), with a fresh count decoder of the default architecture. Training runs 30 epochs at batch 256 on the combo split's 96,322 training cells, with AdamW. The decoder's learning rate is $10^{-3}$; the encoder's, and the predictor's in the anchored arm, is $10^{-4}$. Gradients are clipped at 5.0, and the library size counts the modeled genes.
- **The anchor.** The anchored arm adds the original JEPA objective, smooth-L1 prediction plus $0.04$ VICReg, with weight $w$. The weight is fixed by rule before any update. It is the mean NB loss of the reference decoder `norman_r11_dec_hvg_s0` divided by the mean JEPA loss of the loaded model, both on the first 20 training batches, in evaluation mode and without gradients. At that weight the two terms start equal. The rule gave $w$ of about $28{,}800$, $21{,}500$ and $20{,}500$ for the three seeds, from initial JEPA losses near $0.10$; the seeds differ because each draws its own batches and masks. The EMA teacher is rebuilt from the loaded encoder and follows the student, so the anchor keeps an objective satisfied and does not penalize distance from the original encoder. Stage A's teacher was never saved, so this is a new fine-tuning start, not a resumed run.
- **Readouts.** [`14_ceiling_analysis.py`](../../14_ceiling_analysis.py) with the identity, roundtrip and linear arms on each arm's own encoder and decoder (`norman_ceiling_r12_{arm}_s{s}`).
- **Full pipeline.** The round-1 transport flow recipe (`04_train_cond_flow.py --split combo --cond-type geneset --flow-base control --coupling independent --epochs 60`) is retrained on each moved encoder's latents (`norman_r12_flow_{joint,joint_anchor}_s{s}`). The `dec` arm reuses `norman_sweep_control_s{s}`, trained on its unchanged encoder. Every pipeline reads out through its own arm's decoder. Predictions as every round: 200 cells, generation seed 0, 100 Euler steps, guidance 1.
- **Population sampling.** At the new defaults: sampled cells normalized like the data, at the median real total over the modeled genes, 4,291.
- **Compute.** One A40 GPU. The six co-adaptation trainings took 23 minutes per joint run and 51 minutes per anchored run, three at a time. The whole job took about two hours: an on-pod smoke run, this round, [the re-baseline](rebaseline-after-round-11.md) and both comparison scripts. Nothing was scored on a laptop.
- **Gates, all passed before any arm was read.**
  - Each `dec` ceiling reproduces round 11's roundtrip for the same seed to within $10^{-4}$.
  - Every identity arm scores 1.000.
  - Every pipeline's $\Delta r$, recomputed from its saved profiles, matches its stored report per pair to within $10^{-6}$.
  - Every joint-arm report records its mode and the corrected library size, and so does its decoder checkpoint.

## The pre-committed endpoint

The protocol is the docstring of [`26_score_round12.py`](../../26_score_round12.py), committed before any round-12 run. The primary endpoint is the readout ceiling's $\Delta r$, the roundtrip of real held-out cells, under the measured control on the 20 held-out pairs. One max-$t$ family holds two contrasts, `joint − dec` and `joint_anchor − dec`, with seeds averaged per pair and a joint paired bootstrap over pairs (10,000 resamples, seed 0).

A contrast passes if its estimate is at least $+0.02$ and its simultaneous interval lies wholly above zero. That establishes a positive gain under the rule. A gain above $0.02$ needs the lower bound above $0.02$. A contrast is worse if its interval lies wholly below zero, and otherwise it does not pass.

Reconstruction discourages complete information loss, but rank, reconstruction quality and task-relevant information were monitored, and are reported below as secondaries.

The pod job ran a smoke pass on limited data before the full round. Its numbers were not read. Its first launch stopped because an input, the additive reference's reports, had not been uploaded to the pod's volume. The byte-identical reports were uploaded and the job was resubmitted, with no change to the protocol.

## Verdict: joint training raises the readout ceiling; the anchored arm does not

| readout ceiling, $\Delta r$ | estimate | contrast with `dec` | simultaneous 95% interval | verdict |
|---|---|---|---|---|
| `dec` | $0.865$ | | | |
| `joint` | $0.936$ | $\mathbf{+0.071}$ | $[+0.010, +0.133]$ | **passes** |
| `joint_anchor` | $0.869$ | $+0.004$ | $[-0.007, +0.015]$ | does not pass |

The max-$t$ critical value is $3.352$. The joint arm passes: it establishes a positive gain in the readout ceiling. Its lower bound, $+0.010$, is below $0.02$, so the round does not establish a gain larger than $0.02$, even though the estimate is $0.071$. The anchored arm's estimate is small and its interval includes zero.

## Secondary results, intervals only, no verdicts

- **The representation itself changed, not only the decoder.** The linear readout of each arm's encoder, a ridge map fit on training cells, rises with the joint encoder in step with the roundtrip.

  | readout of each arm's encoder | $\Delta r$ | $e_p$ | $\Delta r$, own control | $e_p$, own control |
  |---|---|---|---|---|
  | roundtrip, `dec` | 0.865 | 0.491 | 0.885 | 0.531 |
  | roundtrip, `joint` | 0.936 | 0.340 | 0.945 | 0.320 |
  | roundtrip, `joint_anchor` | 0.869 | 0.477 | 0.890 | 0.510 |
  | linear, `dec` | 0.852 | 0.561 | 0.831 | 0.600 |
  | linear, `joint` | 0.916 | 0.383 | 0.913 | 0.390 |
  | linear, `joint_anchor` | 0.858 | 0.540 | 0.838 | 0.572 |

  Against `dec`, the joint roundtrip's $e_p$ changes by $-0.151$ $[-0.189, -0.119]$, and the joint linear readout rises by $0.064$ $[+0.035, +0.101]$ in $\Delta r$. The same contrasts under the own-control convention point the same way.

- **The full pipeline carries the gain.** A transport flow retrained on the joint encoder's latents, read through the joint decoder, scores $0.900$ against $0.837$ for the `dec` pipeline. That is $+0.063$ $[+0.033, +0.104]$ in $\Delta r$ and $-0.119$ $[-0.148, -0.089]$ in $e_p$, which moves from $0.591$ to $0.472$. The anchored pipeline moves by $+0.004$ $[-0.004, +0.013]$.

  | full pipeline against the references | $\Delta r$ | $e_p$ |
  |---|---|---|
  | joint pipeline − conditioned cVAE ($0.939$, $e_p$ $0.455$) | $-0.039$ $[-0.075, -0.009]$ | $+0.017$ $[-0.017, +0.050]$ |
  | joint pipeline − additive reference ($0.961$, $e_p$ $0.289$) | $-0.061$ $[-0.100, -0.030]$ | $+0.183$ $[+0.122, +0.249]$ |
  | `dec` pipeline − conditioned cVAE | $-0.102$ $[-0.177, -0.045]$ | $+0.137$ $[+0.103, +0.169]$ |

  The joint pipeline remains below the cVAE in $\Delta r$. On the size-aware error it is level with the cVAE within the interval. Both remain below additivity.

- **Transition headroom stays small.** Readout ceiling minus full pipeline is $0.028$ for `dec`, $0.037$ $[0.023, 0.052]$ for `joint` and $0.029$ for `joint_anchor`, all with intervals above zero. The flow tracks a raised ceiling about as closely as it tracked the old one.

- **Population quality and calibration of the full pipelines.** Definitions as in the protocol:
  - coverage is the fraction of true (cell, gene) entries inside each gene's central 80% predicted interval, pooled over the top 20 genes, with the nominal 0.80 as the target;
  - spread is the correlation of per-gene standard deviations across the 20 genes;
  - $W_1$ is per gene, averaged;
  - energy distance is joint over the 20 genes.

  | pipeline | coverage (gap to 0.80) | spread | $W_1$ | energy |
  |---|---|---|---|---|
  | `dec` | $0.771$ ($-0.029$) | $0.660$ | $0.412$ | $0.919$ |
  | `joint` | $0.797$ ($-0.003$) | $0.701$ | $0.315$ | $0.564$ |
  | `joint_anchor` | $0.775$ ($-0.025$) | $0.651$ | $0.398$ | $0.886$ |

  Against `dec`, the joint pipeline's $W_1$ changes by $-0.097$ $[-0.138, -0.059]$ and its energy distance by $-0.355$ $[-0.537, -0.200]$. Spread changes by $+0.041$ $[+0.003, +0.084]$ and coverage by $+0.027$ $[+0.012, +0.042]$, to within $0.003$ of nominal. The conditioned cVAE, at the same defaults, has spread $0.819$, $W_1$ $0.262$ and energy $0.355$ ([the re-baseline](rebaseline-after-round-11.md)). The joint pipeline approaches it but does not reach it.

- **How far the representation moved.** Linear CKA between each encoder's cell vectors and the original encoder's, on 2,000 training cells:

  | encoder | CKA to the original | effective rank | feature spread | perturbation probe, train / test accuracy |
  |---|---|---|---|---|
  | `dec`, the original | $1.00$ | $213$ | $0.76$ | $0.79$ / $0.05$ |
  | `joint` | $0.49$ to $0.51$ | $228$ to $229$ | $0.25$ to $0.26$ | $0.77$ / $0.18$ |
  | `joint_anchor` | $0.90$ to $0.91$ | $216$ | $0.68$ | $0.81$ / $0.06$ |

  The joint encoder moved a long way: half of its linear structure is no longer shared with the pretrained encoder. Its cell vectors spread over slightly more directions, at about a third of the scale, which the decoder absorbs. Its perturbation probe, a logistic regression from the cell vector to the perturbation label, generalizes better: $0.18$ against $0.05$ on cells held out from the probe's fit, at about the same training accuracy, where chance is about $0.004$. Those cells are a random per-cell hold-out, not the held-out pairs, so the encoder may have trained on them; the probe measures how separable the labels are, not generalization to unseen data. No sign of collapse appears on any of these measures.

- **Training diagnostics.** The joint arm's final NB loss is about $2{,}803$ per cell, against about $2{,}872$ for the anchored arm and $2{,}883$ for the reference decoder on the frozen encoder at the start. In the anchored arm, the anchor's gradient on the encoder was about $13$ to $26$ times the NB term's at the last epoch ($720$ to $1{,}530$ against $51$ to $59$). The anchor, at the weight its rule set, dominated the encoder's updates.

## What it means

Letting the encoder and the decoder shape each other raised how much of each held-out cell the stack keeps, and the gain survives into the full pipeline once the flow is retrained on the new latents. The stack's ceiling moved from below the conditioned cVAE ($0.865$ against $0.939$) to level with it ($0.936$). The full pipeline rose from $0.837$ to $0.900$, and its size-aware error is now level with the cVAE's within the interval. The gain came with a large move away from the pretrained representation, and the change shows up in the linear readout as well as the decoder, so the representation itself became more informative about held-out cells.

The anchored arm stayed close to where it started. At the weight the rule fixed, the JEPA objective's gradient was more than ten times the reconstruction's, and the encoder moved only slightly. The arm therefore does not test whether maintaining the JEPA objective and gaining from reconstruction are compatible; it shows that at this weight the anchor holds the encoder in place.

## What it does not mean

- It does not show that JEPA pretraining helped. The joint encoder started from the pretrained one, but no arm trained the same encoder by reconstruction from a random start, so the round cannot separate the pretraining's contribution from the fine-tuning's.
- It does not show a gain above $0.02$ in the readout ceiling. The rule establishes a positive gain; the lower bound is $0.010$.
- The readout ceiling is a diagnostic handed real held-out cells, not a model. The full pipeline is the model, and its gain is a secondary result.
- No learned model passes additivity. The joint pipeline sits $0.061$ below it in $\Delta r$, and its $e_p$ is $0.183$ higher.
- The anchored arm's result applies to the weight its rule produced, not to the anchor in general.

## Limits

- The `dec` arm is round 11's runs, trained by the same trainer and recipe and re-read here.
- Each pipeline's flow was trained once per seed. The intervals describe the twenty pairs, with seeds averaged per pair first.
- Pairs share genes, so the twenty are not independent.
- The encoder's learning rate, $10^{-4}$, and the 30 epochs were set before the round and were not tuned. A different schedule could move the joint encoder less or more.

## Script and report

- [`co-adaptation/run_coadapt_pod.sh`](../../co-adaptation/run_coadapt_pod.sh) trains the joint arms, reads the ceilings and probes, and retrains and grades the flows. [`run_rebaseline_round12_pod.sh`](../../run_rebaseline_round12_pod.sh) runs it on the pod with the re-baseline, after an on-pod smoke run.
- [`26_score_round12.py`](../../26_score_round12.py) holds the frozen protocol and writes `output/round12_compare.json`.

---

*Up: [the results ledger](index.md). Previous: [round 11](round-11-library-size.md) · Companion: [the re-baseline after round 11](rebaseline-after-round-11.md).*
