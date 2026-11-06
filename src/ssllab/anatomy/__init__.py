"""Anatomy: walk through any component of the perturbation model and see its data flow.

Name a component (``jepa-encoder``, ``student-branch``, ``time-embedding``,
``condition-embedding``, ``conditional-flow``, ``count-decoder``, ...) and get the
operations it performs in order, the shape of the tensor after each one, and the
parameters of the layer that ran. Options select a variant (how the condition is built,
where the flow starts, which decoder levers are on), so alternatives can be compared side
by side.

Truthful by construction. Each walkthrough builds the real modules from ``ssllab`` and
replays their forward pass one operation at a time with the modules' own layers. It then
runs the real forward on the same inputs and reports the largest difference between the
two; the tests require it to be zero for every component and variant, so a walkthrough
cannot drift from the code it describes.

    >>> from ssllab.anatomy import walk, render
    >>> print(render(walk("velocity-field")))
    >>> print(render(walk("condition", cond_type="table"), fmt="markdown"))
    >>> print(render(walk("pipeline", experiment="norman_flow_control")))   # a trained run

By default the modules are freshly initialized. With ``experiment=...`` they are loaded from
``<output_root>/<experiment>/checkpoints/``, together with the run's variant and sizes and,
where the checkpoint has them, its real control latents (see :mod:`ssllab.anatomy.source`).

The command-line entry point is ``examples/perturbation_response/00c_show_architecture.py``.
"""

from __future__ import annotations

from typing import Any

import torch

from ssllab.anatomy import components  # noqa: F401  (registers the walkthroughs)
from ssllab.anatomy.geometry import Geometry
from ssllab.anatomy.options import Options
from ssllab.anatomy.registry import ComponentSpec, all_components, get
from ssllab.anatomy.render import FORMATS, render, render_catalog
from ssllab.anatomy.source import FreshModels, TrainedModels, make_source, runs_with
from ssllab.anatomy.trace import Walkthrough
from ssllab.anatomy.vocab import Vocab, load_vocab, toy_vocab

__all__ = ["walk", "render", "render_catalog", "all_components", "get", "Geometry", "Options",
           "Walkthrough", "Vocab", "load_vocab", "toy_vocab", "ComponentSpec", "FORMATS",
           "FreshModels", "TrainedModels", "make_source", "runs_with"]


def walk(
    name: str,
    geometry: Geometry | None = None,
    vocab: Vocab | None = None,
    seed: int = 0,
    experiment: str | None = None,
    output_root: str = "output",
    **options: Any,
) -> Walkthrough:
    """Run the walkthrough of component ``name`` (or an alias) with the given options.

    ``geometry`` defaults to the production sizes, ``vocab`` to the Norman cache under
    ``data/`` (the toy vocabulary if absent). ``experiment`` loads that run's checkpoints:
    its variant and sizes then come from the checkpoints, and an option or size passed
    explicitly that contradicts them raises ``ValueError``. Options a component does not use
    are ignored. Runs on CPU without gradients; fresh weights are initialized from ``seed``.
    """
    spec = get(name)
    src = make_source(experiment, output_root, seed)
    voc = vocab or load_vocab()
    opts, geo = src.reconcile(spec.needs, options, geometry or Geometry())
    torch.manual_seed(seed)
    with torch.no_grad():
        w = spec.fn(geo, voc, opts, src)
    w.weights = src.describe(spec.needs) if spec.needs else "none (this component has no weights)"
    w.notes = list(dict.fromkeys(src.notes))
    return w
