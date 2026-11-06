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
