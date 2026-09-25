"""Importing this package registers every built-in detector.

To add a new detector kind: create detectors/<name>.py, decorate the class
with @register("<name>"), import it below. A pattern YAML node then sets
`detector: <name>` to use it — no pipeline code changes needed.
"""

from . import regex_detector  # noqa: F401
from . import wordlist_detector  # noqa: F401
from . import density_detector  # noqa: F401
from . import contraction_ratio  # noqa: F401
from . import anaphora_detector  # noqa: F401

from .base import Detector, Hit, get_detector, register  # noqa: F401
