# Round 13 — does the JEPA pretraining contribute to round 12's gain? Joint training from a random start

*Part of [the results ledger](index.md), which holds the standing verdict, the scoreboard and the settings every round shares. Previous: [round 12](round-12-coadaptation.md).*

**What changed.** Round 12 trained the pretrained JEPA encoder and a fresh count decoder together by reconstruction, and the stack's readout ceiling rose from $0.865$ to $0.936$. The encoder moved a long way from its pretrained form on the way, keeping about half of its linear structure, so that round could not say whether the pretraining had contributed or whether joint reconstruction training alone would have got there. This round trains the same architecture, with the same trainer and recipe, from freshly initialized weights.

**Why this question.** The method's premise is that a self-supervised representation is a better starting point than none. Once the encoder is trained by reconstruction anyway, that premise reduces to a narrower claim: starting from the JEPA weights reaches a better encoder than starting from the initialization the JEPA pretraining itself began from. Round 12 left exactly that untested.

## Experimental settings

| arm | what it is | runs, seeds 0, 1, 2 |
|---|---|---|
| `joint` | the encoder starts from the pretrained `norman_stage_a` (md5 `38f51067`), encoder learning rate $10^{-4}$; round 12's runs, re-read and not retrained | `norman_r12_joint_s{s}` |
| `random_lr4` | the encoder starts from fresh weights, encoder learning rate $10^{-4}$: round 12's recipe exactly | `norman_r13_random_lr4_s{s}` |
| `random_lr3` | the same fresh start, encoder learning rate $10^{-3}$, the decoder's | `norman_r13_random_lr3_s{s}` |
| `dec`, context | the encoder frozen at its pretrained weights, decoder only; round 11's runs as re-read in round 12 | `norman_r11_dec_hvg_s{s}` |

- **Trainer.** [`co-adaptation/02_coadapt_decoder.py`](../../co-adaptation/02_coadapt_decoder.py) `--mode joint`, with `--init random` for the two new arms. Every arm trains a fresh count decoder of the default architecture together with every encoder parameter. Training runs 30 epochs at batch 256 on the combo split's 96,322 training cells, with AdamW, decoder learning rate $10^{-3}$, gradients clipped at 5.0 and the library size counted over the modeled genes. The JEPA predictor is unused.
- **The random start.** The fresh encoder has the pretrained checkpoint's architecture and the JEPA constructor's own initialization, the one Stage A pretraining started from. Only the checkpoint's configuration is read, never its weights. The fresh start draws from the random-number generator exactly as loading the checkpoint does, so at the same seed the decoder's initialization and the order of the batches match the `joint` arm's. In `random_lr4`, the encoder's starting weights are therefore the only difference from `joint`. A unit test checks the draw, and that the fresh weights equal the constructor's and differ from the checkpoint's.
- **Why two random arms.** The encoder's learning rate of $10^{-4}$ was chosen for fine-tuning a pretrained encoder and could under-train a random one. The second arm gives the random start the decoder's rate. The pretrained start has to beat both to establish a contribution.
- **Readouts.** [`14_ceiling_analysis.py`](../../14_ceiling_analysis.py) with the identity, roundtrip and linear arms on each arm's own encoder and decoder (`norman_ceiling_r13_{arm}_s{s}`), and the perturbation probe [`02_probe_cell_encoder.py`](../../02_probe_cell_encoder.py) on each encoder.
- **Full pipeline.** The round-1 transport flow recipe (`04_train_cond_flow.py --split combo --cond-type geneset --flow-base control --coupling independent --epochs 60`) is trained on each random encoder's latents (`norman_r13_flow_{arm}_s{s}`) and read out through its arm's decoder. Predictions as every round: 200 cells, generation seed 0, 100 Euler steps, guidance 1. Populations at the defaults: sampled cells normalized like the data, at the median real total over the modeled genes, 4,291.
- **Compute.** One A40 GPU. Each random-start training took about 30 minutes with up to four running at once, and 16 minutes for the last two. Each flow took about 6 minutes. The whole job, an on-pod smoke run included, took about an hour and a half. Nothing was scored on a laptop.
- **Gates, all passed before any arm was read.**
  - Every readout ceiling's roundtrip $\Delta r$, recomputed from its saved profiles, matches its stored report to within $10^{-6}$, and every `joint` encoder's md5 matches the one round 12 recorded.
  - Every identity arm scores 1.000.
  - Every pipeline's $\Delta r$, recomputed from its saved profiles, matches its stored report per pair to within $10^{-6}$.
  - Every random-arm report records mode joint, a random start, the corrected library size and its arm's encoder learning rate, with every other training setting equal to the `joint` arm's. The `joint` reports record a pretrained start from md5 `38f51067`, and every decoder checkpoint records the corrected library size.

## The pre-committed endpoint

The protocol is the docstring of [`27_score_round13.py`](../../27_score_round13.py), committed before any round-13 run. The primary endpoint is the readout ceiling's $\Delta r$, the roundtrip of real held-out cells, under the measured control on the 20 held-out pairs. One max-$t$ family holds two contrasts, `joint − random_lr4` and `joint − random_lr3`, with seeds averaged per pair and a joint paired bootstrap over pairs (10,000 resamples, seed 0).

Each contrast passes if its estimate is at least $+0.02$ and its simultaneous interval lies wholly above zero. It is worse if the interval lies wholly below zero, and otherwise it does not pass. The round's verdict follows from the pair of contrasts:

- **Pretraining's contribution is established** only if both contrasts pass, because the pretrained start must beat every from-scratch configuration tried.
- **A from-scratch arm reads held-out cells better** if either contrast is worse.
- **Otherwise, no contribution of pretraining is established.** That is not evidence that it contributes nothing: the intervals say how large a contribution the data can rule out.

Reconstruction discourages complete information loss, but rank, reconstruction quality and task-relevant information were monitored, and are reported below as secondaries.

The pod job ran a smoke pass on limited data before the full round. Its numbers were not read.

## Verdict: no contribution of pretraining is established

| readout ceiling, $\Delta r$ | estimate | `joint` minus this arm | simultaneous 95% interval | verdict |
|---|---|---|---|---|
| `joint`, pretrained start | $0.936$ | | | |
| `random_lr4` | $0.948$ | $-0.012$ | $[-0.057, +0.034]$ | does not pass |
| `random_lr3` | $0.961$ | $-0.025$ | $[-0.080, +0.030]$ | does not pass |
| `dec`, context | $0.865$ | | | |

The max-$t$ critical value is $5.546$. Neither contrast passes, and neither is worse under the rule, so no contribution of pretraining is established. The simultaneous intervals rule out a contribution larger than about $+0.03$ to the readout ceiling, at this recipe.

Both estimates point the other way. Each from-scratch arm scores above the pretrained start, and so does each of its seeds: the three `joint` seeds score $0.936$ to $0.937$, the `random_lr4` seeds $0.947$ to $0.950$ and the `random_lr3` seeds $0.959$ to $0.964$.

**Why the simultaneous intervals are wide.** The critical value is much larger than round 12's ($3.352$) because one pair carries most of the difference. On OSR2+UBASH3B, the pair the frozen encoder reads worst ($0.385$), the pretrained start reaches $0.728$ and the random starts $0.889$ and $0.932$, differences of $-0.161$ and $-0.204$. On the other nineteen pairs the differences are small: their means are $-0.004$ and $-0.016$. The max-$t$ procedure studentizes each bootstrap resample by its own standard error. Resamples that leave out that pair have a very small standard error, so their studentized statistic is large, and the critical value grows. That is the procedure fixed in advance, and the verdict stands on it.

Descriptively, outside the rule, `random_lr3` scores above the pretrained start on 19 of the 20 pairs and `random_lr4` on 14, with median differences of $-0.013$ and $-0.004$. The plain percentile intervals, which carry no verdict, are $[-0.030, -0.000]$ and $[-0.047, -0.011]$. These are observations about this round's data, not results.

## Secondary results, intervals only, no verdicts

- **Every readout of each arm's encoder.**

  | readout | $\Delta r$ | $e_p$ | $\Delta r$, own control | $e_p$, own control |
  |---|---|---|---|---|
  | roundtrip, `joint` | 0.936 | 0.340 | 0.945 | 0.320 |
  | roundtrip, `random_lr4` | 0.948 | 0.325 | 0.962 | 0.270 |
  | roundtrip, `random_lr3` | 0.961 | 0.296 | 0.970 | 0.238 |
  | roundtrip, `dec` | 0.865 | 0.491 | 0.885 | 0.531 |
  | linear, `joint` | 0.916 | 0.383 | 0.913 | 0.390 |
  | linear, `random_lr4` | 0.946 | 0.318 | 0.943 | 0.325 |
  | linear, `random_lr3` | 0.951 | 0.305 | 0.949 | 0.308 |
  | linear, `dec` | 0.852 | 0.561 | 0.831 | 0.600 |

  The `joint` minus random contrasts on these readouts, measured control:

  | readout | `joint − random_lr4`, $\Delta r$ | $e_p$ | `joint − random_lr3`, $\Delta r$ | $e_p$ |
  |---|---|---|---|---|
  | roundtrip | $-0.012$ $[-0.030, -0.000]$ | $+0.015$ $[-0.002, +0.037]$ | $-0.025$ $[-0.047, -0.011]$ | $+0.044$ $[+0.022, +0.073]$ |
  | linear | $-0.030$ $[-0.056, -0.013]$ | $+0.065$ $[+0.042, +0.093]$ | $-0.034$ $[-0.061, -0.017]$ | $+0.078$ $[+0.055, +0.110]$ |
  | full pipeline | $-0.009$ $[-0.029, +0.005]$ | $+0.002$ $[-0.019, +0.026]$ | $-0.022$ $[-0.050, -0.000]$ | $+0.011$ $[-0.017, +0.041]$ |

  Under the own-control convention the same contrasts point the same way: roundtrip $\Delta r$ $-0.017$ $[-0.034, -0.006]$ and $-0.025$ $[-0.045, -0.012]$, pipeline $-0.023$ $[-0.050, -0.007]$ and $-0.027$ $[-0.060, -0.004]$.

  The linear readout separates the arms more than the roundtrip does. A ridge map from the cell vector reads $0.03$ more out of each random encoder than out of the pretrained start, with a size-aware error about $0.07$ lower.

- **Joint training from scratch against the frozen pretrained encoder.** Each random arm against `dec`, measured control:

  | contrast | readout ceiling, $\Delta r$ | $e_p$ | full pipeline, $\Delta r$ | $e_p$ |
  |---|---|---|---|---|
  | `random_lr4 − dec` | $+0.083$ $[+0.045, +0.137]$ | $-0.166$ $[-0.212, -0.127]$ | $+0.072$ $[+0.033, +0.130]$ | $-0.121$ $[-0.162, -0.082]$ |
  | `random_lr3 − dec` | $+0.096$ $[+0.053, +0.156]$ | $-0.195$ $[-0.250, -0.148]$ | $+0.085$ $[+0.037, +0.152]$ | $-0.130$ $[-0.174, -0.084]$ |

  For comparison, round 12's `joint − dec` was $+0.071$ on the readout ceiling and $+0.063$ on the full pipeline.

- **The full pipeline against the references.** Each pipeline is a transport flow trained on its arm's latents, read through its arm's decoder.

  | pipeline | $\Delta r$ | $e_p$ | minus the conditioned cVAE ($0.939$, $e_p$ $0.455$), $\Delta r$ | $e_p$ | minus additivity ($0.961$, $e_p$ $0.289$), $\Delta r$ | $e_p$ |
  |---|---|---|---|---|---|---|
  | `joint` | 0.900 | 0.472 | $-0.039$ $[-0.075, -0.009]$ | $+0.017$ $[-0.017, +0.050]$ | $-0.061$ $[-0.100, -0.030]$ | $+0.183$ $[+0.122, +0.249]$ |
  | `random_lr4` | 0.909 | 0.470 | $-0.030$ $[-0.057, -0.006]$ | $+0.015$ $[-0.020, +0.049]$ | $-0.051$ $[-0.082, -0.025]$ | $+0.181$ $[+0.114, +0.253]$ |
  | `random_lr3` | 0.921 | 0.461 | $-0.017$ $[-0.030, -0.005]$ | $+0.006$ $[-0.029, +0.043]$ | $-0.039$ $[-0.059, -0.021]$ | $+0.172$ $[+0.109, +0.243]$ |

  Every pipeline remains below the conditioned cVAE in $\Delta r$ and level with it in $e_p$ within the interval. Every pipeline remains below additivity on both.

- **Transition headroom.** Readout ceiling minus full pipeline is $0.037$ $[0.023, 0.052]$ for `joint`, $0.039$ $[0.022, 0.059]$ for `random_lr4` and $0.040$ $[0.024, 0.063]$ for `random_lr3`. The flow trails each arm's ceiling by about the same amount.

- **Population quality and calibration of the full pipelines.** Definitions as in the protocol:
  - coverage is the fraction of true (cell, gene) entries inside each gene's central 80% predicted interval, pooled over the top 20 genes, with the nominal 0.80 as the target;
  - spread is the correlation of per-gene standard deviations across the 20 genes;
  - $W_1$ is per gene, averaged, so it compares marginals only;
  - energy distance is joint over the 20 genes, Euclidean, as a V-statistic.

  | pipeline | coverage (gap to 0.80) | spread | $W_1$ | energy |
  |---|---|---|---|---|
  | `joint` | $0.797$ ($-0.003$) | $0.701$ | $0.315$ | $0.564$ |
  | `random_lr4` | $0.808$ ($+0.008$) | $0.726$ | $0.296$ | $0.462$ |
  | `random_lr3` | $0.811$ ($+0.011$) | $0.738$ | $0.304$ | $0.478$ |

  Against `random_lr4`, the `joint` pipeline's energy distance is higher by $0.102$ $[0.025, 0.212]$, its $W_1$ by $0.019$ $[0.003, 0.038]$, and its spread correlation lower by $0.025$ $[0.011, 0.041]$. Against `random_lr3` the energy distance is higher by $0.086$ $[0.013, 0.169]$, spread lower by $0.037$ $[0.016, 0.061]$, and $W_1$ higher by $0.012$ $[-0.004, 0.028]$. All three pipelines cover within about $0.01$ of nominal. The conditioned cVAE, at the same defaults, has spread $0.819$, $W_1$ $0.262$ and energy $0.355$ ([the re-baseline](rebaseline-after-round-11.md)), still ahead of every pipeline.

- **The representations.** Seed means. Linear CKA is computed on 2,000 training cells, drawn with seed 0.

  | encoder | CKA to the pretrained encoder | CKA to the same seed's `joint` | effective rank | feature spread | perturbation probe, train / test accuracy |
  |---|---|---|---|---|---|
  | pretrained, frozen | $1.00$ | | $213$ | $0.76$ | $0.79$ / $0.05$ |
  | `joint` | $0.50$ | $1.00$ | $229$ | $0.26$ | $0.77$ / $0.18$ |
  | `random_lr4` | $0.23$ | $0.60$ | $238$ | $0.23$ | $0.81$ / $0.23$ |
  | `random_lr3` | $0.23$ | $0.61$ | $236$ | $0.18$ | $0.77$ / $0.28$ |

  The random-start encoders share about $0.6$ of their linear structure with the pretrained start's trained encoder and less with the pretrained encoder itself. They spread over slightly more directions than the `joint` encoder, at a similar or smaller scale. The perturbation probe, a logistic regression from the cell vector to one of 237 labels, scores higher on its held-out cells for both random encoders. The cautions given in [round 12](round-12-coadaptation.md) apply in full. Always answering "control" would score about $0.10$, so the pretrained encoder's $0.05$ is below that level. The probe overfits heavily, with training accuracy near $0.8$. Its fixed penalty interacts with the features' scale, which is smallest for `random_lr3`, the encoder with the highest test accuracy. And the test cells are a random per-cell hold-out that every encoder here trained on. The probe ranking is therefore not evidence on its own about how much perturbation information each representation carries. No sign of collapse appears on any of these measures.

- **Training diagnostics.** The final NB loss per cell is about $2{,}804$ for `joint`, $2{,}782$ for `random_lr4` and $2{,}765$ for `random_lr3`. Every arm was still improving when it stopped: the mean loss over the last five epochs is $16.5$, $17.2$ and $18.1$ below the five epochs before them. The norm of the NB term's gradient on the encoder, on the first batch of the last epoch and before clipping, averaged $91$, $221$ and $88$.

## What it means

At this recipe, the JEPA pretraining is not established to help the encoder that joint reconstruction training produces. The two random starts reach readout ceilings of $0.948$ and $0.961$, above the pretrained start's $0.936$ at every seed, and the round's rule rules out a pretrained advantage larger than about $0.03$. Training from scratch with round 12's recipe beats the frozen pretrained encoder by about as much as round 12's fine-tuning did ($+0.083$ and $+0.096$ against $+0.071$). So round 12's gain is accounted for by joint reconstruction training, and the pretrained weights are not shown to add to it.

The direction of the estimates is consistent across every readout of the representation: the roundtrip, the linear readout, the full pipeline, and the spread, $W_1$ and energy distance of its populations all favor a random start, the linear readout most clearly. The perturbation probe points the same way but is confounded, as described above. Coverage does not separate the arms: all three sit within about $0.01$ of nominal. Those are secondaries, chosen in advance but without a verdict, and the primary does not establish a difference in either direction. If the pattern is real, a candidate account is that the pretrained start keeps structure the reconstruction objective does not need. The `joint` encoder still shares half of its linear structure with the pretrained one after 30 epochs. That account is a hypothesis, not something this round tested.

The best full pipeline built so far, the transport flow on the `random_lr3` encoder, scores $0.921$ with a size-aware error of $0.461$. It is $0.017$ below the conditioned cVAE in $\Delta r$ and level with it in $e_p$, and both remain below additivity.

## What it does not mean

- It does not show that pretraining hurts. Neither contrast meets the rule's criterion for a from-scratch arm being better, and one pair carries most of the difference.
- It does not show that self-supervised pretraining has no value. Stage A pretrains on the training side of the same split, the same cells the co-adaptation then trains on, so the pretraining had no unlabeled data the reconstruction lacked. The premise it was meant to test, that a representation learned on abundant unlabeled cells pays off when labeled cells are scarce, needs a data-efficiency experiment, and this round does not test it.
- It applies to this JEPA, this architecture and this recipe: 30 epochs, two encoder learning rates, one decoder. A longer schedule, a different pretraining or a frozen-then-unfrozen schedule could change it.
- The readout ceiling is a diagnostic handed real held-out cells, not a model. The full pipeline is the model, and its numbers are secondaries.
- No learned model passes additivity.

## Limits

- The `joint` arm is round 12's runs, re-read here. Its configuration, recorded in its reports, matches the random arms' in every setting except the start and, for `random_lr3`, the encoder's learning rate.
- Every arm was still improving at 30 epochs, so the comparison is between arms at a fixed budget, not at convergence.
- Each pipeline's flow was trained once per seed. The intervals describe the twenty pairs, with seeds averaged per pair first. Pairs share genes, so the twenty are not independent.
- The verdict is driven by the studentized max-$t$ interval, which one dominant pair makes wide. The procedure was fixed before the round and is reported as it ran.

## Script and report

- [`co-adaptation/run_round13_pod.sh`](../../co-adaptation/run_round13_pod.sh) checks every input on the pod, runs a smoke pass at tiny settings, then trains the random arms, reads their ceilings and probes, trains and grades their flows, and runs the comparison.
- [`27_score_round13.py`](../../27_score_round13.py) holds the frozen protocol and writes `output/round13_compare.json`.

---

*Up: [the results ledger](index.md). Previous: [round 12](round-12-coadaptation.md).*
