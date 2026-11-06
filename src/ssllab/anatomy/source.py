"""Where a walkthrough's modules come from: freshly initialized, or a trained run's checkpoints.

``FreshModels`` (the default) builds every module from the geometry with new random
weights and gives the flow random stand-in control latents. Shapes and operations are
exactly those of the trained models; values are not.

``TrainedModels`` loads a run from ``<output_root>/<experiment>/checkpoints/`` with the same
loaders the evaluation scripts use, so the walkthrough shows that run's actual sizes,
variant and weights. Where the checkpoint carries real inputs (the flow checkpoint stores
the standardized latents of the training control cells and the standardization
statistics), the walkthroughs use them, so the flow, velocity, condition, decoder and
pipeline views compute the trained model's real outputs. The JEPA views still use random
stand-in tokens: real cells would mean reading the expression matrix (gigabytes) for a
handful of rows.

A run's variant and sizes are facts about the run, so they come from its checkpoints. An
option or size passed explicitly that contradicts a checkpoint is an error rather than
being silently replaced.

One gap in the checkpoints: ``encoder.pt`` stores the student encoder and the predictor but
not the EMA teacher (``EMA`` holds the teacher outside the module tree, so
``state_dict`` does not see it). A loaded JEPA's teacher is therefore its random
initialization. The trained student stands in for it here, and the header says so.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from ssllab.anatomy.components.cvae import build_cvae
from ssllab.anatomy.components.decoder import build_decoder
from ssllab.anatomy.components.flow import build_condition, build_flow
from ssllab.anatomy.components.jepa import build as build_jepa_model
from ssllab.anatomy.geometry import Geometry
from ssllab.anatomy.options import Options
from ssllab.anatomy.vocab import Vocab
from ssllab.data.perturbseq import token_dim_for
from ssllab.generative.count_decoder import CountDecoder
from ssllab.generative.cvae import ConditionalNBVAE
from ssllab.generative.perturb import load_cond_flow, load_count_decoder
from ssllab.checkpoint import load_jepa
from ssllab.jepa.model import JEPA

CHECKPOINTS = {
    "encoder": "encoder.pt",
    "cond_flow": "cond_flow.pt",
    "count_decoder": "count_decoder.pt",
    "cvae": "cvae_baseline.pt",
}


class FreshModels:
    """Newly initialized modules built from the geometry (the default)."""

    trained = False

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.notes: list[str] = []

    def describe(self, needs: tuple[str, ...]) -> str:
        return f"freshly initialized (seed {self.seed}); shapes are the real ones, values are not a trained model's"

    def reconcile(self, needs: tuple[str, ...], explicit: dict[str, Any], geo: Geometry) -> tuple[Options, Geometry]:
        return Options(**explicit), geo

    def jepa(self, geo: Geometry, opts: Options, teacher: bool = False) -> JEPA:
        return build_jepa_model(geo, opts)

    def flow_bundle(self, geo: Geometry, vocab: Vocab, opts: Options) -> dict[str, Any]:
        """The same keys ``load_cond_flow`` returns, with random stand-ins for the control pool and statistics."""
        return {"flow": build_flow(geo), "cond": build_condition(geo, vocab, opts), "flow_base": opts.flow_base,
                "dim": geo.embed_dim, "ctrl_pool": torch.randn(geo.n_ctrl, geo.embed_dim),
                "mean": torch.randn(geo.embed_dim), "std": torch.rand(geo.embed_dim) + 0.5, "real_inputs": False}

    def decoder(self, geo: Geometry, opts: Options) -> CountDecoder:
        return build_decoder(geo, opts)

    def decoder_input(self, geo: Geometry, n: int) -> tuple[torch.Tensor, str]:
        return torch.randn(n, geo.embed_dim), f"{n} random stand-in latents"

    def cvae(self, geo: Geometry, vocab: Vocab, opts: Options) -> ConditionalNBVAE:
        return build_cvae(geo, vocab, opts)


def runs_with(output_root: str | Path, filename: str) -> list[str]:
    """Experiments under ``output_root`` whose checkpoints include ``filename``."""
    root = Path(output_root)
    return sorted(p.parent.parent.name for p in root.glob(f"*/checkpoints/{filename}"))


class TrainedModels:
    """Modules and settings loaded from ``<output_root>/<experiment>/checkpoints/``."""

    trained = True

    def __init__(self, experiment: str, output_root: str | Path = "output") -> None:
        self.experiment = experiment
        self.output_root = Path(output_root)
        self.dir = self.output_root / experiment / "checkpoints"
        if not self.dir.is_dir():
            runs = sorted({r for f in CHECKPOINTS.values() for r in runs_with(self.output_root, f)})
            raise KeyError(f"no checkpoints for {experiment!r} under {self.output_root}/; "
                           f"runs with checkpoints this tool reads: {', '.join(runs) or 'none'}")
        self.notes: list[str] = []
        self._raw: dict[str, dict] = {}

    # -- files -------------------------------------------------------------
    def path(self, key: str) -> Path:
        p = self.dir / CHECKPOINTS[key]
        if not p.exists():
            others = runs_with(self.output_root, CHECKPOINTS[key])
            raise KeyError(f"{self.experiment} has no {CHECKPOINTS[key]}, which this component needs; "
                           f"runs that have it: {', '.join(others) or 'none'}")
        return p

    def raw(self, key: str) -> dict:
        if key not in self._raw:
            self._raw[key] = torch.load(self.path(key), map_location="cpu", weights_only=False)
        return self._raw[key]

    def has(self, key: str) -> bool:
        return (self.dir / CHECKPOINTS[key]).exists()

    def describe(self, needs: tuple[str, ...]) -> str:
        files = ", ".join(CHECKPOINTS[k] for k in needs) or "none needed"
        return f"trained, from {self.dir}/ ({files})"

    # -- settings ------------------------------------------------------------
    def _fixed(self, needs: tuple[str, ...]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Options and geometry fields each needed checkpoint determines; disagreements raise."""
        opts: dict[str, Any] = {}
        geo: dict[str, Any] = {}

        def put(store: dict, key: str, value: Any, source: str) -> None:
            if key in store and store[key][0] != value:
                raise ValueError(f"{self.experiment}: {store[key][1]} says {key}={store[key][0]} but {source} "
                                 f"says {key}={value}")
            store[key] = (value, source)

        if "encoder" in needs:
            cfg = self.raw("encoder")["cfg"]
            if cfg.get("mask_kind", "random") != "random":
                raise NotImplementedError("walkthroughs replay random masks only; this encoder used span masks")
            src = CHECKPOINTS["encoder"]
            put(opts, "reg_coef", cfg["reg_coef"], src)
            for key, ck_key in (("embed_dim", "embed_dim"), ("enc_depth", "enc_depth"), ("pred_depth", "pred_depth"),
                                ("n_heads", "n_heads"), ("n_target", "n_target"), ("n_tokens", "n_tokens")):
                put(geo, key, cfg[ck_key], src)
            geo["_token_dim"] = (cfg["token_dim"], src)
        if "cond_flow" in needs:
            ck = self.raw("cond_flow")
            src = CHECKPOINTS["cond_flow"]
            put(opts, "flow_base", ck.get("flow_base", "gaussian"), src)       # the loader's defaults for old files
            put(opts, "cond_type", ck.get("cond_type", "table"), src)
            put(opts, "compose", ck.get("compose", "additive"), src)
            for key, ck_key in (("embed_dim", "dim"), ("cond_dim", "cond_dim"), ("pert_dim", "pert_dim"),
                                ("flow_hidden", "hidden"), ("flow_layers", "n_layers")):
                put(geo, key, ck[ck_key], src)
            put(geo, "gene_dim", ck.get("gene_dim", 64), src)
            put(geo, "n_ctrl", len(ck["ctrl_pool"]), src)
        if "count_decoder" in needs:
            ck = self.raw("count_decoder")
            src = CHECKPOINTS["count_decoder"]
            for key in ("zinb", "anchored_mean", "state_dispersion"):
                put(opts, key, ck.get(key, False), src)
            put(geo, "embed_dim", ck["latent_dim"], src)
            put(geo, "n_hvg", ck["n_genes"], src)
            hidden = tuple(v.shape[0] for k, v in ck["state"].items() if k.startswith("trunk.") and v.dim() == 2)
            put(geo, "dec_hidden", hidden, src)
        if "cvae" in needs:
            ck = self.raw("cvae")
            src = CHECKPOINTS["cvae"]
            put(opts, "encoder_cond", ck.get("encoder_cond", False), src)
            put(opts, "compose", ck["compose"], src)
            for key, ck_key in (("n_hvg", "n_genes"), ("vae_latent", "latent_dim"), ("cond_dim", "cond_dim"),
                                ("gene_dim", "gene_dim"), ("vae_hidden", "hidden")):
                put(geo, key, ck[ck_key], src)
        return opts, geo

    def reconcile(self, needs: tuple[str, ...], explicit: dict[str, Any], geo: Geometry) -> tuple[Options, Geometry]:
        """Options and geometry for this run; an explicit value contradicting a checkpoint raises."""
        fixed_opts, fixed_geo = self._fixed(needs)
        for key, (value, source) in fixed_opts.items():
            if key in explicit and explicit[key] != value:
                flag = "--" + key.replace("_", "-")
                raise ValueError(f"{self.experiment} was trained with {key}={value} ({source}); "
                                 f"drop {flag} or pass the same value")
        defaults = Geometry()
        token_dim = fixed_geo.pop("_token_dim", None)
        updates = {}
        for key, (value, source) in fixed_geo.items():
            current = getattr(geo, key)
            if current != getattr(defaults, key) and current != value:
                raise ValueError(f"{self.experiment} has {key}={value} ({source}); drop --set {key}=… "
                                 f"or pass the same value")
            updates[key] = value
        out_geo = replace(geo, **updates)
        if token_dim is not None and token_dim_for(out_geo.n_hvg, out_geo.n_tokens) != token_dim[0]:
            out_geo = replace(out_geo, n_hvg=token_dim[0] * out_geo.n_tokens)
        opts = Options(**{**explicit, **{k: v for k, (v, _) in fixed_opts.items()}})
        return opts, out_geo

    # -- modules -----------------------------------------------------------
    def jepa(self, geo: Geometry, opts: Options, teacher: bool = False) -> JEPA:
        """The trained JEPA; with ``teacher=True`` the student's weights also fill the (unsaved) teacher."""
        jepa = load_jepa(self.path("encoder"))
        if teacher:
            jepa.ema.teacher.load_state_dict(jepa.encoder.state_dict())
            self.notes.append("encoder.pt does not store the EMA teacher; the trained student stands in for it")
        return jepa

    def _check_vocab(self, ck: dict, vocab: Vocab, what: str) -> None:
        pert_gene = ck.get("pert_gene")
        if pert_gene is not None and not (pert_gene.shape == vocab.pert_gene.shape
                                          and torch.equal(pert_gene.float(), vocab.pert_gene.float())):
            raise ValueError(f"{what} was trained on a different perturbation vocabulary than {vocab.source} "
                             f"({tuple(pert_gene.shape)} vs {tuple(vocab.pert_gene.shape)}); point "
                             "--data-dir/--artifact at the cache it was trained on")
        n_perts = ck.get("n_perts")
        if n_perts is not None and n_perts != vocab.n_perts:
            raise ValueError(f"{what} has {n_perts} perturbations, {vocab.source} has {vocab.n_perts}")

    def flow_bundle(self, geo: Geometry, vocab: Vocab, opts: Options) -> dict[str, Any]:
        self._check_vocab(self.raw("cond_flow"), vocab, CHECKPOINTS["cond_flow"])
        bundle = load_cond_flow(self.path("cond_flow"))
        bundle["real_inputs"] = True
        return bundle

    def decoder(self, geo: Geometry, opts: Options) -> CountDecoder:
        return load_count_decoder(self.path("count_decoder"))

    def decoder_input(self, geo: Geometry, n: int) -> tuple[torch.Tensor, str]:
        """Real control latents in encoder units when the run has a flow checkpoint, else stand-ins."""
        if not self.has("cond_flow"):
            return torch.randn(n, geo.embed_dim), f"{n} random stand-in latents (the run has no cond_flow.pt)"
        ck = self.raw("cond_flow")
        pool = ck["ctrl_pool"]
        idx = torch.randint(len(pool), (n,), generator=torch.Generator().manual_seed(0))
        return pool[idx] * ck["std"] + ck["mean"], f"{n} real control-cell latents (from cond_flow.pt)"

    def cvae(self, geo: Geometry, vocab: Vocab, opts: Options) -> ConditionalNBVAE:
        ck = self.raw("cvae")
        self._check_vocab(ck, vocab, CHECKPOINTS["cvae"])
        return ConditionalNBVAE.from_checkpoint(ck)


def make_source(experiment: str | None, output_root: str | Path = "output", seed: int = 0):
    return TrainedModels(experiment, output_root) if experiment else FreshModels(seed)


__all__ = ["FreshModels", "TrainedModels", "make_source", "runs_with", "CHECKPOINTS"]
