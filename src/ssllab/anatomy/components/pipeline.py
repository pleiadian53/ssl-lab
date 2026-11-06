"""Walkthrough of the whole generation path, one row per stage: perturbation name →
condition → flow → decoder → predicted mean profile. Checked against
``ssllab.generative.perturb.predicted_expression``, the function the effect-size
evaluation (``06``) calls.
"""

from __future__ import annotations

import torch

from ssllab.anatomy.components._common import rng_checkpoint
from ssllab.anatomy.components.flow import FLOW_VARIANTS, inputs_note, replay_flow
from ssllab.anatomy.geometry import Geometry
from ssllab.anatomy.options import Options
from ssllab.anatomy.registry import component
from ssllab.anatomy.trace import Trace, Walkthrough
from ssllab.anatomy.vocab import Vocab
from ssllab.generative.perturb import predicted_expression


@component("pipeline", group="end to end", aliases=("generate", "end-to-end"),
           summary="perturbation name → condition → flow → decoder → predicted mean profile",
           options=("pert", "flow_base", "cond_type", "compose", "guidance", "zinb", "anchored_mean",
                    "state_dispersion"),
           variants=FLOW_VARIANTS + ({"zinb": True},), needs=("cond_flow", "count_decoder"))
def pipeline(geo: Geometry, vocab: Vocab, opts: Options, src) -> Walkthrough:
    name = vocab.resolve(opts.pert)
    bundle = src.flow_bundle(geo, vocab, opts)
    flow, cond, dec = bundle["flow"], bundle["cond"], src.decoder(geo, opts)
    pid0, n = vocab.pert_id(name), geo.n
    tr = Trace()
    with rng_checkpoint() as rng:
        z = replay_flow(tr, bundle, vocab, opts, name, pid0, n, geo.steps, torch.Generator().manual_seed(0))
        tr.section("decode")
        rho = tr.record("rho", dec(z, library_size=torch.ones(n))["rho"],
                        "count decoder: gene shares per cell (see count-decoder)", module=dec, label="decoder")
        expr = tr.record("expr", torch.log1p(1e4 * rho), "log1p(10⁴·rho)", label="log1p")
        pred = tr.record("pred_mean", expr.mean(0), "average over the cells: the predicted profile", label="mean",
                         note="scored by 06: Δ = pred_mean − control_mean on the top-20 genes, Pearson r with the real Δ")
        rng.rewind()
        ref = predicted_expression(bundle, dec, pid0, n, guidance=opts.guidance, steps=geo.steps,
                                   generator=torch.Generator().manual_seed(0))
    shown = {k: getattr(opts, k) for k in ("flow_base", "cond_type", "compose", "guidance", "zinb",
                                           "anchored_mean", "state_dispersion")}
    return Walkthrough(
        component="pipeline", title=f"generating the predicted response to {name}", options=shown | {"pert": name},
        geometry=geo.as_dict(), steps=tr.steps,
        modules={"condition module": cond, "velocity field": flow, "decoder": dec},
        output=pred, reference=ref, checked="the predicted mean profile",
        inputs=f"{n} generated cells under {name} (cells per perturbation, --n); " + inputs_note(bundle),
    )
