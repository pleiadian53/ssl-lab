# 5. Vary: adding an arm without breaking comparability

*Not a script but a pattern, implemented across the shell wrappers and a handful of reuse flags. A round of this project is one pass of this workflow: change exactly one thing, reuse everything else, grade with the identical harness, and record the result where the previous rounds can be read against it.*

---

## The pattern

```
                      ┌── reuse ────────────────────────┐
   cache  ──→  encoder.pt  ──→  [ THE LEVER ]  ──→  count_decoder.pt  ──→  same harness
   (00)         (01, frozen)      retrain this        (03, reused)          (06 / 10)
```

Retrain **only** the lever. Everything upstream and downstream is reused, byte for byte, from artifacts that already exist. Three things follow.

**The comparison isolates the lever.** If the encoder and decoder are literally the same files, a difference between two arms is a difference in the thing that changed.

**A round is cheap.** Stage B on this data is minutes on an A40, not hours, because Stages A and C are not retrained.

**Arms stay comparable across rounds.** Round 4's operator can be read against Round 1's flow because both sit on the same frozen encoder and are graded through the same decoder and metric.

## How the reuse is expressed in the code

Two flags carry it, and they are the whole mechanism:

```bash
# train only the lever
python examples/perturbation_response/16_train_operator_algebra.py \
    --experiment norman_operator_algebra_s0 \
    --encoder output/norman_stage_a/checkpoints/encoder.pt \
    --split combo --epochs 60 --seed 0

# grade it: pick the Stage B, and REUSE the canonical decoder
python examples/perturbation_response/06_eval_effect_size.py \
    --experiment norman_operator_algebra_s0 --stage-b operator_algebra --split combo \
    --decoder output/norman_flow_control/checkpoints/count_decoder.pt
```

`--stage-b {flow,operator,operator_algebra}` selects which generative middle to grade. `--decoder <path>` points at an existing readout instead of the experiment's own. A Stage-B lever does not change the readout, so reusing it is not a shortcut, it is the thing that makes the comparison mean something.

Adding a new Stage B is correspondingly small: a module, a loader in [`perturb.py`](../../../../src/ssllab/generative/perturb.py) that returns the same bundle shape, and a choice in the flag. Because the loader returns the same shape, the sampling path and the entire evaluation stack are untouched.

## One lever per arm

The rule with the most scar tissue behind it.

An arm that changes two things cannot explain its own result. Round 3's stochastic-operator arm bundled a stochastic coefficient **and** a residual displacement, and it regressed. The likely cause was the residual, since a per-condition constant adds zero variance and can absorb the mean effect cheaply. But "likely" is where the analysis had to stop, permanently, because no measurement in that arm can separate the two changes. The only fix is to re-run them apart, and that is a second round spent recovering information the first round threw away for free.

The related trap is subtler: **a lever aimed the wrong way is not a null, it is a wasted round.** Round 2 built a decoder lever to *narrow* a distribution believed to be over-dispersed. The distribution was in fact under-dispersed on every model. The result carries no information about the design it was meant to test, because the design was pointed the opposite direction from the problem. That is not a statistics failure, it is a diagnosis failure, and it is what the [Diagnose](03-diagnose.md) workflow is for.

## Orchestration: the shell wrappers

Each `run_*.sh` is one round's arms, and they share a shape worth copying:

```bash
ARMS="${ARMS:-operator operator_stoch}"   # env-overridable, so a subset can be re-run
SEEDS="${SEEDS:-0 1 2}"
# preflight: fail in seconds, not after twenty minutes of training
python -c "import torch; assert torch.cuda.is_available()" || fail=1
[ -d "$DATA/norman2019" ] || { echo "FATAL: cache missing"; echo "  fix: python examples/ops/ops_stage_data.py data/norman2019"; fail=1; }
[ "$fail" -eq 1 ] && { echo "preflight FAILED; nothing was trained."; exit 1; }
```

**The preflight is the important part.** It checks every input the round needs *before* training anything, and when one is missing it prints the specific fix rather than a stack trace. A round that dies twenty minutes in because a checkpoint was never staged costs a GPU-hour and a context switch; the same failure in three seconds costs neither.

The wrappers **train only**. Evaluation happens locally after fetch, because a pod volume can hold a stale gene list and grading there once produced numbers that silently disagreed with the corrected local ones.

## The pod loop

Fixed order, each step failing fast: **provision → stage → launch → fetch → grade locally**.

```bash
export PATH="<env>/bin:$PATH"                      # sky must be on PATH or the driver no-ops
python examples/ops/ops_run_pipeline.py --execute --gpu a40 -- true          # 1. provision, kept alive
python examples/ops/ops_stage_data.py data/norman2019 output/norman_stage_a/checkpoints   # 2. stage once
python examples/ops/ops_run_pipeline.py --execute --gpu a40 -- \
    bash examples/perturbation_response/run_operator_algebra_pod.sh          # 3. launch
rsync -Pavz -u <cluster>:/runpod-volume/ssl-lab/output/<exp>/ output/<exp>/  # 4. fetch (-u is load-bearing)
sky down <cluster> -y                                                        # 5. release
```

Four things that have each cost real time:

- **`rsync -u` is not a nicety.** It skips any file newer on the receiver, so a fetch can never move a result *backwards*. Without it, a pod with stale inputs once overwrote seven corrected reports.
- **A silent local log means nothing.** Laptop standby drops the stream while the job keeps running. Confirm with `sky queue <cluster>` before concluding failure.
- **GPU capacity is intermittent.** A failed provision is a clean no-op that bills nothing; retry rather than replanning.
- **Tear down when done.** The volume persists, so the cache and checkpoints survive the pod.

## Closing a round

A round ends in the [ledger](../conditional-flow-jepa/results-ledger.md), and the ledger's own rules are the checklist: state what changed and which chapter specified it, pre-commit the endpoint and the decision rule, report the round's arms together while carrying the baseline forward as the fixed bar, say the seed count, and report failures **with their diagnosis**. A failure with a named cause is more useful than an omission, and it is often more useful than a success.

---

*Previous: [Decide](04-decide.md). Up: [the workflow map](index.md). Where rounds are recorded: [the results ledger](../conditional-flow-jepa/results-ledger.md).*
