from __future__ import annotations

import json
from typing import Any


def serialize_conversation(messages: list[dict[str, Any]]) -> str:
    """Serialize without losing roles, language, or earlier-turn facts."""
    return json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
