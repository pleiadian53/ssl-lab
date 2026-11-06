"""The record of one walkthrough: an ordered list of steps, each a tensor and how it was made.

A walkthrough replays a component's forward pass one operation at a time, using the
component's own submodules, and calls :meth:`Trace.record` after each operation. The
recorder keeps the tensor's shape, a plain-language description of the operation, the
parameter count of the submodule that ran (0 for inline arithmetic such as FiLM or a
softmax), and optionally the tensor's mean and standard deviation. It returns the tensor
unchanged, so recording never alters the computation it describes.

The finished :class:`Walkthrough` also carries the replay's final output and the output of
the component's *real* forward on the same inputs. Their maximum absolute difference is the
evidence that the steps shown are the steps the code actually takes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


def count_params(module: nn.Module | None) -> int:
    """All parameters of ``module`` (trainable or frozen); 0 for ``None``."""
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters())


@dataclass
class Step:
    """One row of a walkthrough.

    ``name`` is the tensor's name (blank for an intermediate inside a small MLP),
    ``shape`` its shape (``None`` for a note row that produces no tensor), ``op`` the
    operation that produced it, ``label`` a short form of ``op`` for the one-line chain
    (``None``: use ``op``; ``""``: an input, drawn without an incoming arrow),
    ``params`` the parameters of the submodule that ran, ``section`` the group it belongs
    to (e.g. "time branch"), and ``stats`` its (mean, std) when it is a float tensor.
    """

    name: str
    shape: tuple[int, ...] | None
    op: str
    label: str | None = None
    params: int = 0
    note: str = ""
    section: str = ""
    stats: tuple[float, float] | None = None


class Trace:
    """Collects :class:`Step` rows in order; ``section`` starts a new group of rows."""

    def __init__(self) -> None:
        self.steps: list[Step] = []
        self._section = ""

    def section(self, title: str) -> None:
        self._section = title

    def record(
        self,
        name: str,
        x: Any,
        op: str,
        *,
        module: nn.Module | None = None,
        params: int | None = None,
        label: str | None = None,
        note: str = "",
    ) -> Any:
        """Record ``x`` (a tensor) and return it unchanged.

        ``params`` overrides the count taken from ``module``, for a bare ``nn.Parameter``
        such as a positional table.
        """
        shape = tuple(x.shape) if isinstance(x, torch.Tensor) else None
        stats = None
        if isinstance(x, torch.Tensor) and x.is_floating_point() and x.numel() > 1:
            xd = x.detach()
            stats = (float(xd.mean()), float(xd.std()))
        self.steps.append(Step(
            name=name, shape=shape, op=op, label=label,
            params=count_params(module) if params is None else params,
            note=note, section=self._section, stats=stats,
        ))
        return x

    def note(self, text: str) -> None:
        """A row with no tensor: a repetition, a pointer to another component, a caveat."""
        self.steps.append(Step(name="", shape=None, op=text, section=self._section))


@dataclass
class Walkthrough:
    """A finished walkthrough of one component under one set of options.

    ``modules`` maps a display name to each top-level module involved, for the parameter
    summary; ``frozen`` names the ones that receive no gradient (the EMA teacher).
    ``output`` is the replay's result and ``reference`` the real forward's result on the
    same inputs; ``checked`` says which output the two represent.
    """

    component: str
    title: str
    options: dict[str, Any]
    geometry: dict[str, Any]
    steps: list[Step]
    modules: dict[str, nn.Module] = field(default_factory=dict)
    frozen: tuple[str, ...] = ()
    output: torch.Tensor | None = None
    reference: torch.Tensor | None = None
    checked: str = ""
    inputs: str = ""

    @property
    def max_abs_diff(self) -> float | None:
        if self.output is None or self.reference is None:
            return None
        return float((self.output.detach().float() - self.reference.detach().float()).abs().max())

    @property
    def param_summary(self) -> dict[str, int]:
        return {name: count_params(m) for name, m in self.modules.items()}
