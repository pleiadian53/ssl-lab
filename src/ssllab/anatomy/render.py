"""Turn a :class:`~ssllab.anatomy.trace.Walkthrough` into text.

Four formats:

    table      an aligned terminal table, one row per step, grouped by section (default)
    chain      one line per section: tensor (shape) -op-> tensor (shape) ...
    markdown   a pipe table, for pasting into notes and docs
    json       everything, machine-readable

:func:`render_catalog` prints the list of components, grouped by family, with their
aliases, descriptions and options, so the names do not have to be remembered.
"""

from __future__ import annotations

import json

from ssllab.anatomy.registry import ComponentSpec, all_components
from ssllab.anatomy.trace import Walkthrough

FORMATS = ("table", "chain", "markdown", "json")
GROUP_ORDER = ("data", "jepa", "flow", "decoder", "baseline", "end to end")


def _shape(shape: tuple[int, ...] | None) -> str:
    if shape is None:
        return ""
    if len(shape) == 0:
        return "scalar"
    return "(" + ", ".join(f"{d}" for d in shape) + ("," if len(shape) == 1 else "") + ")"


def _params(n: int) -> str:
    return f"{n:,}" if n else ""


def _header(w: Walkthrough) -> list[str]:
    lines = [f"{w.component}: {w.title}"]
    if w.options:
        lines.append("variant: " + ", ".join(f"{k}={v}" for k, v in w.options.items()))
    if w.weights:
        lines.append(f"weights: {w.weights}")
    lines += [f"note: {n}" for n in w.notes]
    if w.inputs:
        lines.append(f"inputs: {w.inputs}")
    if w.modules:
        parts = []
        for name, n in w.param_summary.items():
            frozen = " (frozen)" if name in w.frozen else ""
            parts.append(f"{name} {n:,}{frozen}")
        lines.append("parameters: " + " · ".join(parts))
    diff = w.max_abs_diff
    if diff is not None:
        verdict = "matches" if diff < 1e-5 else "DOES NOT MATCH"
        lines.append(f"check: replay {verdict} the real forward on {w.checked} (max |diff| = {diff:.1e})")
    return lines


def render_table(w: Walkthrough, stats: bool = False) -> str:
    rows = []
    i = 0
    for s in w.steps:
        stat = f"{s.stats[0]:+.3f} ± {s.stats[1]:.3f}" if (stats and s.stats) else ""
        i += s.shape is not None                                # note rows are not numbered
        rows.append((str(i) if s.shape is not None else "", s.name, _shape(s.shape), s.op,
                     _params(s.params), stat, s.note, s.section))
    head = ("#", "tensor", "shape", "operation", "params", "mean ± std")
    show_stats = stats and any(r[5] for r in rows)
    ncol = 6 if show_stats else 5
    data_rows = [r for r in rows if r[2]]                       # note rows span the table instead
    widths = [max([len(head[c])] + [len(r[c]) for r in data_rows]) for c in range(ncol)]

    def line(cells: tuple[str, ...]) -> str:
        out = []
        for c in range(ncol):
            text = cells[c]
            align = str.rjust if c in (0, 4) else str.ljust
            out.append(align(text, widths[c]))
        return "  ".join(out).rstrip()

    out = _header(w) + ["", line(head), "  ".join("-" * wd for wd in widths)]
    section = None
    for r in rows:
        if r[7] != section:
            section = r[7]
            if section:
                out.append(f"[{section}]")
        if not r[2]:                                            # a note row
            out.append(" " * (widths[0] + 2) + "· " + r[3])
            continue
        out.append(line(r[:6]))
        if r[6]:
            out.append(" " * (widths[0] + widths[1] + widths[2] + 6) + "↳ " + r[6])
    return "\n".join(out)


def render_chain(w: Walkthrough) -> str:
    """One line per section; each arrow is labelled with the step's short operation.

    Inputs (steps whose label is ``""``) appear as bare nodes; every other step is drawn
    with an incoming arrow, including the first step of a section, whose input is the
    previous section's output.
    """
    out = _header(w) + [""]
    sections: dict[str, list] = {}
    for s in w.steps:
        if s.shape is not None:
            sections.setdefault(s.section, []).append(s)
    for section, steps in sections.items():
        parts = []
        for s in steps:
            node = f"{s.name or '·'} {_shape(s.shape)}"
            label = s.op if s.label is None else s.label
            parts.append(node if label == "" else f"─{label}─▶ {node}")
        prefix = f"[{section}] " if section else ""
        out.append(prefix + " ".join(parts))
    return "\n".join(out)


def render_markdown(w: Walkthrough) -> str:
    out = [f"**{w.component}**: {w.title}", ""]
    out += [f"- {line}" for line in _header(w)[1:]]
    out += ["", "| # | tensor | shape | operation | params |", "|---|---|---|---|---|"]
    section = None
    i = 0
    for s in w.steps:
        if s.section != section:
            section = s.section
            if section:
                out.append(f"| | **{section}** | | | |")
        if s.shape is None:
            out.append(f"| | | | *{s.op}* | |")
            continue
        i += 1
        op = s.op + (f" ({s.note})" if s.note else "")
        name = f"`{s.name}`" if s.name else ""
        out.append(f"| {i} | {name} | {_shape(s.shape)} | {op} | {_params(s.params)} |")
    return "\n".join(out)


def render_json(w: Walkthrough) -> str:
    return json.dumps({
        "component": w.component, "title": w.title, "options": w.options, "geometry": w.geometry,
        "inputs": w.inputs, "weights": w.weights, "notes": w.notes, "parameters": w.param_summary, "frozen": list(w.frozen),
        "check": {"on": w.checked, "max_abs_diff": w.max_abs_diff},
        "steps": [{"name": s.name, "shape": list(s.shape) if s.shape is not None else None,
                   "op": s.op, "params": s.params, "note": s.note, "section": s.section,
                   "mean_std": list(s.stats) if s.stats else None} for s in w.steps],
    }, indent=2, default=str)


def render(w: Walkthrough, fmt: str = "table", stats: bool = False) -> str:
    if fmt == "table":
        return render_table(w, stats=stats)
    if fmt == "chain":
        return render_chain(w)
    if fmt == "markdown":
        return render_markdown(w)
    if fmt == "json":
        return render_json(w)
    raise ValueError(f"format must be one of {FORMATS}, got {fmt!r}")


def _option_flags(spec: ComponentSpec) -> str:
    return " ".join("--" + o.replace("_", "-") for o in spec.options)


def render_catalog(prog: str = "") -> str:
    """Every component, grouped by family, with aliases, a description and its options."""
    specs = all_components()
    groups = {g: [s for s in specs if s.group == g] for g in GROUP_ORDER}
    groups.update({s.group: [t for t in specs if t.group == s.group] for s in specs if s.group not in groups})
    width = max(len(s.name) for s in specs)
    out = ["Components (pass the name or any alias):"]
    for group, members in groups.items():
        if not members:
            continue
        out.append(f"\n  {group}")
        for s in members:
            out.append(f"    {s.name.ljust(width)}  {s.summary}")
            extra = []
            if s.aliases:
                extra.append("aliases: " + ", ".join(s.aliases))
            if s.options:
                extra.append("options: " + _option_flags(s))
            if extra:
                out.append(f"    {''.ljust(width)}  " + "  ·  ".join(extra))
    if prog:
        out += ["", "Examples:",
                f"  {prog} velocity-field",
                f"  {prog} condition --cond-type table --flow-base gaussian",
                f"  {prog} decoder --zinb --format markdown",
                f"  {prog} encoder --expand --stats",
                f"  {prog} pipeline --from-experiment norman_flow_control   # a trained run"]
    return "\n".join(out)
