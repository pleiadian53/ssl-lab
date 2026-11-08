# Round 2 — the decoder levers (superseded, being re-run)

*Part of [the results ledger](index.md), which holds the standing verdict, the scoreboard and the settings every round shares. Previous: [round 1](round-01-head-to-head.md) · Next: [round 3](round-03-action-operator.md).*

**What was tested.** The two levers of [Chapter 8](../conditional-flow-jepa/08-modeling-the-readout-count-decoder.md), as opt-in flags on `CountDecoder`: **B1**, an identity-anchored mean head targeting effect size, and **B2**, a state-aware dispersion with a moment-matching anchor targeting calibration. Four arms, single seed, retraining only Stage C.

**Status: the numbers from this round are not carried forward.** The round was designed against a reading of the calibration axis that the current gene selection does not support. B2 was built to *narrow* a decoder believed to be over-dispersed, and the decoder is in fact **under-dispersed** on every model measured (coverage $0.33$ to $0.38$, far below nominal). A lever pushing in that direction is aimed the wrong way, so its result carries no information about the design it was meant to test. B1, which targets effect size and is unaffected by the dispersion question, is the part worth re-measuring.

**Re-running:** the four arms on the current gene selection, seeded, so that B1's contribution can be read against the same simultaneous-interval standard as Round 1. The arms' checkpoints persist on the pod volume.

**Carry forward regardless:** B2's failure mode is instructive independent of its sign. Its NB likelihood *improved* while its target metric did not, which is the same trap optimal-transport coupling sprang in Round 1. **A better training loss is not a better model**, and that has now happened twice.

---

*Up: [the results ledger](index.md). Previous: [round 1](round-01-head-to-head.md) · Next: [round 3](round-03-action-operator.md).*
