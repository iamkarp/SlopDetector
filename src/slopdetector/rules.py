"""Stage A: run every applicable pattern node against one text unit."""

from __future__ import annotations

from .config import Config
from .detectors import Hit, get_detector
from .graph import load_nodes


def _resolve_weight(node: dict, hit: Hit, config: Config) -> float:
    w = hit.weight
    modifiers = node.get("genre_modifiers", {})
    if config.genre and config.genre in modifiers:
        w *= modifiers[config.genre]
    elif "_default" in modifiers:
        # No --genre given (or genre isn't one this node has an opinion on):
        # fall back to the node's own default multiplier instead of
        # silently assuming full weight. Fiction-only patterns (e.g. the
        # phrase fingerprints) set _default: 0.0 so they don't fire on
        # nonfiction just because nobody passed --genre.
        w *= modifiers["_default"]
    return w


# graph/edges.yaml records em_dash_hard_zero "supersedes" em_dash_cluster
# under profile=surface-gate. Enacted here: profile em_dash_max_per_window
# picks exactly one of the two nodes, so switching --profile actually
# changes scoring instead of being a config field nobody reads.
_EM_DASH_ZERO_TOLERANCE_NODE = "em_dash_hard_zero"
_EM_DASH_CLUSTER_TOLERANCE_NODE = "em_dash_cluster"


def _profile_skips_node(node_id: str, config: Config) -> bool:
    max_per_window = config.profile_settings().get("em_dash_max_per_window", 1)
    if node_id == _EM_DASH_ZERO_TOLERANCE_NODE:
        return max_per_window >= 1  # tolerant profile: skip the zero-tolerance rule
    if node_id == _EM_DASH_CLUSTER_TOLERANCE_NODE:
        return max_per_window == 0  # zero-tolerance profile: hard-zero rule already covers it
    return False


def score_unit(text: str, granularity: str, config: Config, nodes: list[dict] | None = None) -> tuple[float, list[Hit]]:
    """Returns (rule_score, hits). rule_score is an unbounded positive sum of
    weighted hits — not a probability. The pipeline squashes it into [0,1]
    only as context for the JEV prompt / as a fallback when use_llm=False.
    """
    nodes = nodes if nodes is not None else load_nodes(config.extra_node_dirs)
    hits: list[Hit] = []
    total = 0.0
    for node in nodes:
        if granularity not in node.get("applicable_granularity", ["paragraph", "sentence", "line"]):
            continue
        if _profile_skips_node(node["id"], config):
            continue
        if node["detector"] == "contraction_ratio":
            node = {**node, "_formality": config.formality}
        detector = get_detector(node["detector"])
        for hit in detector.match(text, node):
            weighted = _resolve_weight(node, hit, config)
            hits.append(hit)
            total += weighted
    return total, hits


def rule_score_to_pseudo_probability(rule_score: float) -> float:
    """Cheap monotonic squash (1 - e^-x) so a JEV-free run still returns a
    0-1 number. Calibrated loosely against slopmonster's 0-5 scale: a
    rule_score of ~3 (roughly one flag per category) lands around 0.55,
    the default drill-down threshold.
    """
    import math

    return 1 - math.exp(-rule_score / 3.5)
