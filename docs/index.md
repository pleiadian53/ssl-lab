# ssl-lab documentation

A research lab for **self-supervised learning (SSL)** — tracking the SSL frontier and going deep on the methods most worth mastering. The current focus is **JEPA** (joint-embedding predictive architectures) and two ways to extend it.

This site is built from the `docs/` folder of [pleiadian53/ssl-lab](https://github.com/pleiadian53/ssl-lab) and rendered with full LaTeX math support.

## Research directions

**1. Generative JEPA — make the representation *sampleable*.** Add a prior over the latent and a decoder back to data space, turning a representation learner into a generative model. Built as a walking skeleton on MNIST (modality-agnostic core). → **[Full tutorial: Generative JEPA](generative_jepa/index.md)**.

```mermaid
flowchart LR
    NOISE(["noise ε"]) --> PRIOR["flow-matching<br/>prior p(z)"]
    PRIOR -- "sample z ~ p(z)" --> DEC["decoder<br/>z → x"]
    DEC --> GEN(["generated sample"])
    ENC["JEPA encoder<br/>(frozen)"] -. "defines the latent z<br/>the prior is fit to" .-> PRIOR

    classDef accent fill:#eef2ff,stroke:#6366f1,color:#1e1b4b;
    classDef io fill:#f8fafc,stroke:#94a3b8,color:#0f172a;
    class ENC,PRIOR,DEC accent;
    class NOISE,GEN io;
```

**2. Action operators on JEPA — make prediction *active*.** Promote JEPA's fixed "predict region $p$" mask to a **learned operator the model chooses** — *sensing* (where to look) and *perturbing* (what an edit means) — so the system can form and test hypotheses rather than only in-fill what's masked. This builds on the action-operator formalism from the sibling project [GRL](https://github.com/pleiadian53/GRL).

## Read next

- **[Generative JEPA](generative_jepa/index.md)** — a four-part tutorial on extending a JEPA encoder into a sampleable generative model: the encoder, the flow-matching prior, the decoder, and sampling + evaluation.
- [JEPA as an Action-Operator World Model](action_operator/01-jepa-action-operators.md) — the synthesis: how JEPA's predictor is a special case of an action-operator world-model.
- [Notation reference](action_operator/notation.md) — every symbol used in the write-up.

## About

ssl-lab is a spin-off of [genai-lab](https://github.com/pleiadian53/genai-lab) (generative AI for computational biology). Methods are kept deliberately modality-agnostic so they can mature here and feed back into genomic generative models. See the repository [README](https://github.com/pleiadian53/ssl-lab#readme) for setup and the code layout.
