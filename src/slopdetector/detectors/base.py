from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


@dataclass
class Hit:
    pattern_id: str
    label: str
    span: tuple[int, int] | None = None
    detail: str = ""
    weight: float = 1.0


class Detector(Protocol):
    """A detector scores one text unit against one pattern node's `data`."""

    def match(self, text: str, node: dict) -> list[Hit]: ...


_REGISTRY: dict[str, Detector] = {}


def register(name: str):
    def _wrap(cls):
        _REGISTRY[name] = cls()
        return cls

    return _wrap


def should_skip(text: str, data: dict) -> bool:
    """Standard opt-out hook for any detector: node.data.skip_if is a regex
    (optionally node.data.skip_if_flags, e.g. ["MULTILINE"]) — if it matches
    the unit's text at all, the whole node is skipped for this unit. Used to
    exempt structural text (footnote definitions, bibliography entries) from
    patterns meant for prose, without hard-coding document structure into
    the pipeline itself.
    """
    pattern = data.get("skip_if")
    if not pattern:
        return False
    flags = 0
    for f in data.get("skip_if_flags", []):
        flags |= getattr(re, f)
    return bool(re.search(pattern, text, flags))


def get_detector(name: str) -> Detector:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown detector '{name}'. Registered: {sorted(_REGISTRY)}. "
            "Add a new detector by creating detectors/<name>.py with @register(\"<name>\")."
        )
    return _REGISTRY[name]
