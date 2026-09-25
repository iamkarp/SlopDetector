from __future__ import annotations

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


def get_detector(name: str) -> Detector:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown detector '{name}'. Registered: {sorted(_REGISTRY)}. "
            "Add a new detector by creating detectors/<name>.py with @register(\"<name>\")."
        )
    return _REGISTRY[name]
