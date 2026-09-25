"""Auto-discovers pattern nodes from graph/nodes/*.yaml. Adding a new pattern
category is: drop a new YAML file here, no code changes. Each file holds a
top-level `patterns:` list; each entry is one node.

Node schema:
  id: str (unique)
  category: str
  detector: str (registered detector name)
  label: str
  data: dict (detector-specific: pattern/words/bands/...)
  weight: float (default 1.0)
  applicable_granularity: [paragraph, sentence, line]  (default: both)
  genre_modifiers: {genre: weight_multiplier}  (optional)
  source: str (provenance, e.g. "book-forge/90-anti-ai-patterns.md")
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import yaml

_REQUIRED = ("id", "category", "detector", "data")


def _validate(node: dict, path: Path) -> None:
    missing = [k for k in _REQUIRED if k not in node]
    if missing:
        raise ValueError(f"Node in {path} missing required fields: {missing} -> {node}")


def _load_dir(dir_path: Path) -> list[dict]:
    nodes: list[dict] = []
    if not dir_path.is_dir():
        return nodes
    for yml in sorted(dir_path.glob("*.yaml")):
        doc = yaml.safe_load(yml.read_text()) or {}
        for node in doc.get("patterns", []):
            _validate(node, yml)
            node.setdefault("weight", 1.0)
            node.setdefault("applicable_granularity", ["paragraph", "sentence", "line"])
            node.setdefault("genre_modifiers", {})
            node.setdefault("source", yml.stem)
            node.setdefault("label", node["id"])
            nodes.append(node)
    return nodes


def load_nodes(extra_dirs: list[str] | None = None) -> list[dict]:
    built_in = resources.files("slopdetector.graph") / "nodes"
    nodes = _load_dir(Path(str(built_in)))
    seen_ids = {n["id"] for n in nodes}
    for extra in extra_dirs or []:
        for node in _load_dir(Path(extra)):
            if node["id"] in seen_ids:
                raise ValueError(f"Duplicate pattern id '{node['id']}' from {extra}")
            seen_ids.add(node["id"])
            nodes.append(node)
    return nodes


def load_edges() -> list[dict]:
    edges_path = Path(str(resources.files("slopdetector.graph") / "edges.yaml"))
    if not edges_path.is_file():
        return []
    doc = yaml.safe_load(edges_path.read_text()) or {}
    return doc.get("edges", [])
