"""Runtime configuration: thresholds, genre bands, model name, profiles.

Nothing here holds a secret. The OpenRouter key is read once, at call time,
by llm/openrouter_client.py, never stored on a Config object.
"""

from __future__ import annotations

from dataclasses import dataclass, field

GENRE_BANDS = {
    "literary-fiction": (3, 6),
    "memoir": (4, 7),
    "commercial-fiction": (8, 11),
    "prescriptive-nf": (12, 15),
}

# Strictness profiles resolve conflicts between source skills' tolerances.
# "surface-gate" matches book-forge's 5/5-required marketing-copy behavior.
# "prose-advisory" matches book-forge's own default for manuscript prose.
# "marketing" matches slopmonster's tolerance-based (not zero-tolerance) defaults.
PROFILES = {
    "prose-advisory": {"em_dash_max_per_window": 1, "gates": False},
    "surface-gate": {"em_dash_max_per_window": 0, "gates": True},
    "marketing": {"em_dash_max_per_window": 1, "gates": False},
}

DEFAULT_MODEL = "typesafe/jev-1.13"
DEFAULT_THRESHOLD = 0.55
DEFAULT_GRANULARITY = "paragraph"


@dataclass
class Config:
    model: str = DEFAULT_MODEL
    threshold: float = DEFAULT_THRESHOLD
    granularity: str = DEFAULT_GRANULARITY
    genre: str | None = None
    profile: str = "prose-advisory"
    formality: str = "neutral"  # "formal" | "neutral" | "casual" — gates contraction weight
    concurrency: int = 6  # concurrent JEV *requests* in flight, each request may hold batch_size units
    batch_size: int = 10  # units judged per JEV decisions call; 1 = one call per unit (old behavior)
    include_reason: bool = True  # companion "why" category per unit; ~2x tokens, --no-reason to skip
    use_llm: bool = True
    env_file: str | None = None
    cache_dir: str | None = ".slopdetector-cache"
    extra_node_dirs: list[str] = field(default_factory=list)

    def profile_settings(self) -> dict:
        return PROFILES.get(self.profile, PROFILES["prose-advisory"])
