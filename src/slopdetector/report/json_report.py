from __future__ import annotations

import json


def render(result: dict) -> str:
    return json.dumps(result, indent=2, ensure_ascii=False)
