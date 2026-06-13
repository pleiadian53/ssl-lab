# Time-Series JEPA

*JEPA, put on a clock: learning the structure of time series by predicting the future in latent space.*

Most of machine learning's progress on images came from a simple idea: hide part of the data and learn to predict it. **JEPA** — the Joint-Embedding Predictive Architecture — does this in a particularly clean way, predicting the *meaning* (the embedding) of a hidden region rather than its raw pixels. **Time-Series JEPA** is that same idea pointed at *time*: hide the future, and learn to predict it from the past — again in latent space.

This series is a self-contained tutorial on Time-Series JEPA as a method in its own right. It is the gentlest entry point in this corpus: if you know roughly what self-supervised learning is, you have enough to start. You do **not** need any of the action-operator or "world model" material to follow it — that comes later, and Time-Series JEPA is what it will be built on.

---

## What you will get

A working mental model of how to learn from unlabeled time series, told as one continuous story:

1. **From I-JEPA to Time-Series JEPA** — how "predict a masked patch of an image" becomes "predict the next stretch of a time series," and why doing it in latent space (not raw signal) is the right call.
2. **Multimodal temporal channels** — the real-world complication: several signals at once (a wearable, a phone), sampled at different rates, often missing. Handling this *is* the research problem, and it is where Time-Series JEPA earns its keep.
3. **The limits of pure Time-Series JEPA** — one honest blind spot that pure prediction cannot fix, which is exactly the door into the rest of the corpus.

Throughout, we follow **one running example** — a person's day-to-day behavioral rhythm — so each idea lands on something concrete you can picture, not just notation.

---

## Before you start: what to read, and what you can skip

You need only a passing familiarity with **self-supervised learning** and the original **I-JEPA** (image JEPA). We recap the essentials in Part 1 in fresh words, so a quick memory is enough; for a fuller treatment see the JEPA encoder chapter in the [Generative JEPA](../generative_jepa/01-the-jepa-encoder.md) series.

You can **skip, for now**, the [Action Operators](../action_operator/00-from-actions-to-operators.md) foundation and the [Operator World Models](../operator_world_models/index.md) series. Time-Series JEPA stands on its own. Those become relevant only once we hit the blind spot in Part 3 — at which point this series hands you forward to them.

---

## Reading order

| Part | Topic | What you get |
|---|---|---|
| **[1 — From I-JEPA to Time-Series JEPA](01-from-ijepa-to-tjepa.md)** | the temporal reinterpretation; the four pieces; the forward pass | how masked-patch prediction becomes next-step prediction, with a worked example |
| **2 — Multimodal temporal channels** *(coming next)* | async, irregular, missing streams; per-modality encoders; masking curricula | how Time-Series JEPA handles many real signals at once |
| **3 — The limits of pure Time-Series JEPA** *(coming next)* | the "what acted?" blind spot; the contaminated surprise signal | why pure prediction is not enough — and where to go next |

New to a symbol? The [notation reference](notation.md) defines every one.

---

## Where this leads

Time-Series JEPA gives you a model that predicts *what comes next*. By Part 3 you will see that it cannot, on its own, say *what acted* to bring that next state about — and for problems like deciding which intervention helps a patient, that gap is everything. Closing it is the job of the [Action Operators](../action_operator/00-from-actions-to-operators.md) foundation and the [Operator World Models](../operator_world_models/index.md) series, for which this one is the natural runway.

## Related work, and what is new here

Applying JEPA to time series is an idea several groups have arrived at independently — which is a good sign it is a natural step, not a niche trick. For the published lines, see **S-JEPA** (*Signal-JEPA*, an early JEPA applied to EEG signals, [arXiv:2403.11772](https://arxiv.org/abs/2403.11772)), **TS-JEPA** / *Joint Embeddings Go Temporal* ([arXiv:2509.25449](https://arxiv.org/abs/2509.25449)), and a **Time-Series JEPA** applied to predictive remote control ([arXiv:2406.04853](https://arxiv.org/abs/2406.04853)). For the multimodal setting of Part 2, see *Giving Sensors a Voice: Multimodal JEPA for Semantic Time-Series Embeddings* ([OpenReview](https://openreview.net/forum?id=RtHXBIJfYG)), which introduces **CHARM**, a channel-aware multimodal model.

A naming note: the acronym *T-JEPA* in the literature refers to unrelated work on **tabular** ([arXiv:2410.05016](https://arxiv.org/abs/2410.05016)) and **trajectory** ([arXiv:2406.12913](https://arxiv.org/abs/2406.12913)) data — which is why this series uses the descriptive name *Time-Series JEPA* throughout.

This series is an accessible, example-driven exposition of that shared foundation. Its distinctive contribution is not the temporal predictor itself, but what we build *on* it: conditioning the predictor with an **action operator** — the synthesis developed in the [Action Operators](../action_operator/00-from-actions-to-operators.md) foundation and the [Operator World Models](../operator_world_models/index.md) series, which goes beyond the prediction-only models above.

---

*Start with [Part 1 — From I-JEPA to Time-Series JEPA](01-from-ijepa-to-tjepa.md).*
