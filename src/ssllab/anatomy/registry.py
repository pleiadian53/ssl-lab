"""Component registry: a name (and aliases) for each walkthrough, its options and variants.

A walkthrough is a function ``fn(geo, vocab, opts) -> Walkthrough`` registered with
:func:`component`. ``options`` lists the :class:`~ssllab.anatomy.options.Options` fields it
reads (shown in its header), and ``variants`` lists option settings that exercise every
structural branch it has, which the tests replay one by one. Adding a variant of a
component (a new time embedding, a new condition encoder) means registering what the new
code does, once that code exists in the model.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ssllab.anatomy.trace import Walkthrough


@dataclass
class ComponentSpec:
    name: str
    fn: Callable[..., Walkthrough]
    summary: str
    group: str
    aliases: tuple[str, ...] = ()
    options: tuple[str, ...] = ()
    variants: tuple[dict[str, Any], ...] = field(default_factory=lambda: ({},))


_REGISTRY: dict[str, ComponentSpec] = {}
_ALIASES: dict[str, str] = {}


def component(
    name: str,
    *,
    summary: str,
    group: str,
    aliases: tuple[str, ...] = (),
    options: tuple[str, ...] = (),
    variants: tuple[dict[str, Any], ...] = ({},),
) -> Callable[[Callable[..., Walkthrough]], Callable[..., Walkthrough]]:
    def register(fn: Callable[..., Walkthrough]) -> Callable[..., Walkthrough]:
        if name in _REGISTRY:
            raise ValueError(f"component {name!r} registered twice")
        _REGISTRY[name] = ComponentSpec(name, fn, summary, group, aliases, options, variants)
        for alias in aliases:
            _ALIASES[alias] = name
        return fn

    return register


def get(name: str) -> ComponentSpec:
    """Look up a component by name or alias; unknown names get close-match suggestions."""
    key = name.strip().lower().replace("_", "-")
    key = _ALIASES.get(key, key)
    if key in _REGISTRY:
        return _REGISTRY[key]
    close = difflib.get_close_matches(key, list(_REGISTRY) + list(_ALIASES), n=4)
    hint = f"; did you mean {', '.join(close)}?" if close else ""
    raise KeyError(f"unknown component {name!r}{hint} (list them with --list)")


def all_components() -> list[ComponentSpec]:
    return list(_REGISTRY.values())
