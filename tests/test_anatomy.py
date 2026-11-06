"""Tests for ``ssllab.anatomy``: every walkthrough must be a faithful account of the code.

The central test replays every registered component under every declared variant and
requires the replay's output to equal the real module's forward on the same inputs. If a
model's ``forward`` changes and its walkthrough is not updated, this fails.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from ssllab.anatomy import FORMATS, Geometry, all_components, get, render, render_catalog, toy_vocab, walk

CASES = [(spec.name, variant) for spec in all_components() for variant in spec.variants]


@pytest.mark.parametrize("name,variant", CASES, ids=[f"{n}-{v}" for n, v in CASES])
def test_replay_matches_real_forward(name, variant):
    w = walk(name, geometry=Geometry.small(), vocab=toy_vocab(), **variant)
    assert w.output is not None and w.reference is not None
    assert w.output.shape == w.reference.shape
    assert torch.allclose(w.output, w.reference, atol=1e-6), f"{name} {variant}: max |diff| {w.max_abs_diff}"
    assert any(s.shape is not None for s in w.steps)
    for fmt in FORMATS:
        assert render(w, fmt, stats=True)


def test_every_component_listed_and_aliases_resolve():
    catalog = render_catalog()
    for spec in all_components():
        assert spec.name in catalog and spec.summary in catalog
        for alias in spec.aliases:
            assert get(alias) is spec
    assert get("Jepa_Encoder").name == "jepa-encoder"
    with pytest.raises(KeyError, match="did you mean"):
        get("jepa-encodr")


def test_production_shapes_and_parameter_counts():
    """Spot-check the production geometry against the sizes reported in the docs."""
    vel = walk("velocity-field", vocab=toy_vocab())
    assert vel.param_summary == {"velocity field": 970_880}
    assert vel.steps[-1].shape == (200, 256)

    dec = walk("count-decoder", vocab=toy_vocab())
    assert dec.param_summary == {"decoder": 2_966_288}
    assert next(s for s in dec.steps if s.name == "rho").shape == (200, 5000)

    pred = walk("predictor", geometry=Geometry().with_overrides({"batch": 4}), vocab=toy_vocab())
    assert pred.param_summary == {"predictor": 1_593_088}
    assert pred.steps[-1].shape == (4, 12, 256)

    enc = walk("jepa-encoder", geometry=Geometry().with_overrides({"batch": 4}), vocab=toy_vocab())
    assert enc.param_summary == {"encoder": 4_777_728}
    assert enc.steps[-1].shape == (4, 256)


def test_condition_names_the_target_genes():
    w = walk("condition-embedding", geometry=Geometry.small(), vocab=toy_vocab(), pert="b+a")
    assert w.options["pert"] == "A+B"                           # case and gene order ignored
    multi_hot = next(s for s in w.steps if s.name == "multi-hot")
    assert "1 at A, B" in multi_hot.op
    with pytest.raises(KeyError):
        walk("condition-embedding", geometry=Geometry.small(), vocab=toy_vocab(), pert="Z+Q")


def test_cli_catalog_runs(capsys, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "examples/perturbation_response/00c_show_architecture.py"
    spec = importlib.util.spec_from_file_location("show_architecture", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr("sys.argv", ["00c_show_architecture.py"])
    mod.main()
    out = capsys.readouterr().out
    assert "velocity-field" in out and "aliases:" in out


# --------------------------------------------------------------------------- #
# Trained runs (--from-experiment).
# --------------------------------------------------------------------------- #
from ssllab.anatomy import Options  # noqa: E402
from ssllab.anatomy.components.cvae import build_cvae  # noqa: E402
from ssllab.anatomy.components.decoder import build_decoder  # noqa: E402
from ssllab.anatomy.components.flow import build_condition, build_flow  # noqa: E402
from ssllab.anatomy.components.jepa import build as build_jepa_model  # noqa: E402
from ssllab.checkpoint import save_jepa  # noqa: E402

# The loaders rebuild the flow's time MLP and the decoder trunk at their default widths
# (neither is stored in the checkpoint), so a synthetic run must use those widths.
RUN_GEO = Geometry.small().with_overrides({"time_dim": 64, "dec_hidden": (512, 512)})
RUN_OPTS = Options(flow_base="gaussian", cond_type="geneset", compose="deepsets", anchored_mean=True,
                   encoder_cond=True, reg_coef=0.1)


def _write_run(root: Path, name: str = "toy_run") -> Path:
    """Save freshly built small modules in the formats 01, 03, 04 and 08 write."""
    geo, vocab, opts = RUN_GEO, toy_vocab(), RUN_OPTS
    ck = root / name / "checkpoints"
    ck.mkdir(parents=True)
    torch.manual_seed(123)
    save_jepa(build_jepa_model(geo, opts), ck / "encoder.pt")
    flow, cond = build_flow(geo), build_condition(geo, vocab, opts)
    torch.save({"flow": flow.state_dict(), "cond": cond.state_dict(), "dim": geo.embed_dim,
                "cond_dim": geo.cond_dim, "pert_dim": geo.pert_dim, "hidden": geo.flow_hidden,
                "n_layers": geo.flow_layers, "n_perts": vocab.n_perts, "cond_type": opts.cond_type,
                "compose": opts.compose, "gene_dim": geo.gene_dim, "flow_base": opts.flow_base,
                "pert_gene": vocab.pert_gene, "mean": torch.randn(geo.embed_dim),
                "std": torch.rand(geo.embed_dim) + 0.5, "ctrl_pool": torch.randn(geo.n_ctrl, geo.embed_dim)},
               ck / "cond_flow.pt")
    dec = build_decoder(geo, opts)
    torch.save({"state": dec.state_dict(), "latent_dim": geo.embed_dim, "n_genes": geo.n_hvg, "zinb": opts.zinb,
                "anchored_mean": opts.anchored_mean, "state_dispersion": opts.state_dispersion}, ck / "count_decoder.pt")
    vae = build_cvae(geo, vocab, opts)
    torch.save({"state": vae.state_dict(), "n_genes": geo.n_hvg, "latent_dim": geo.vae_latent,
                "cond_dim": geo.cond_dim, "gene_dim": geo.gene_dim, "hidden": geo.vae_hidden,
                "compose": opts.compose, "pert_gene": vocab.pert_gene, "encoder_cond": opts.encoder_cond},
               ck / "cvae_baseline.pt")
    return root


@pytest.mark.parametrize("name", [s.name for s in all_components()])
def test_trained_run_replays_exactly_with_its_own_settings(tmp_path, name):
    root = _write_run(tmp_path)
    for extra in ({}, {"mode": "train"}) if name == "cvae" else ({},):
        w = walk(name, geometry=RUN_GEO, vocab=toy_vocab(), experiment="toy_run", output_root=str(root), **extra)
        assert torch.allclose(w.output, w.reference, atol=1e-6), f"{name}: max |diff| {w.max_abs_diff}"
        if get(name).needs:
            assert w.weights.startswith("trained")
        for key in ("flow_base", "compose", "anchored_mean", "encoder_cond", "reg_coef"):
            if key in w.options:                                # the run's settings, not the defaults
                assert w.options[key] == getattr(RUN_OPTS, key), (name, key)


def test_trained_weights_are_the_saved_ones(tmp_path):
    root = _write_run(tmp_path)
    w = walk("velocity-field", geometry=RUN_GEO, vocab=toy_vocab(), experiment="toy_run", output_root=str(root))
    saved = torch.load(root / "toy_run/checkpoints/cond_flow.pt", weights_only=False)["flow"]
    loaded = w.modules["velocity field"].state_dict()
    assert all(torch.equal(saved[k], loaded[k]) for k in saved)
    assert w.steps[[s.name for s in w.steps].index("c")].shape == (RUN_GEO.n, RUN_GEO.cond_dim)


def test_trained_run_rejects_contradicting_flags_and_names_missing_files(tmp_path):
    root = _write_run(tmp_path)
    kw = dict(geometry=RUN_GEO, vocab=toy_vocab(), output_root=str(root))
    with pytest.raises(ValueError, match="trained with flow_base=gaussian"):
        walk("flow", experiment="toy_run", flow_base="control", **kw)
    walk("flow", experiment="toy_run", flow_base="gaussian", **kw)          # agreeing flag is fine
    with pytest.raises(ValueError, match="cond_dim"):
        walk("flow", experiment="toy_run", **{**kw, "geometry": RUN_GEO.with_overrides({"cond_dim": 99})})
    (root / "toy_run/checkpoints/cvae_baseline.pt").unlink()
    with pytest.raises(KeyError, match="no cvae_baseline.pt"):
        walk("cvae", experiment="toy_run", **kw)
    with pytest.raises(KeyError, match="runs with checkpoints this tool reads: toy_run"):
        walk("flow", experiment="nope", **kw)


def test_teacher_caveat_only_where_the_teacher_is_used(tmp_path):
    root = _write_run(tmp_path)
    kw = dict(geometry=RUN_GEO, vocab=toy_vocab(), experiment="toy_run", output_root=str(root))
    assert walk("teacher-branch", **kw).notes and walk("jepa-step", **kw).notes
    assert not walk("jepa-encoder", **kw).notes and not walk("predictor", **kw).notes


@pytest.mark.skipif(not Path("output/norman_flow_control/checkpoints/cond_flow.pt").exists(),
                    reason="needs the trained transport-flow run")
def test_real_transport_flow_run():
    w = walk("pipeline", geometry=Geometry().with_overrides({"n": 8, "steps": 5}), experiment="norman_flow_control")
    assert w.options["flow_base"] == "control" and w.max_abs_diff == 0.0
    assert w.steps[-1].shape == (5000,)
