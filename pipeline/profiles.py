from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PatientProfile:
    id: str
    label: str
    perspective: str
    modified_query: str
    preserved_facts: tuple[str, ...]
    focus_areas: tuple[str, ...]


def profile_schema(count: int) -> dict[str, Any]:
    string_array = {"type": "array", "items": {"type": "string"}}
    profile = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string"},
            "label": {"type": "string"},
            "perspective": {"type": "string"},
            "modified_query": {"type": "string"},
            "preserved_facts": string_array,
            "focus_areas": string_array,
        },
        "required": ["id", "label", "perspective", "modified_query", "preserved_facts", "focus_areas"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {"profiles": {"type": "array", "minItems": count, "maxItems": count, "items": profile}},
        "required": ["profiles"],
    }


def parse_profiles(value: dict[str, Any], expected_count: int) -> list[PatientProfile]:
    raw = value.get("profiles")
    if not isinstance(raw, list) or len(raw) != expected_count:
        raise ValueError(f"expected exactly {expected_count} profiles")
    profiles: list[PatientProfile] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("profile must be an object")
        fields = ("id", "label", "perspective", "modified_query")
        if any(not isinstance(item.get(field), str) or not item[field].strip() for field in fields):
            raise ValueError("profile text fields must be non-empty strings")
        facts = item.get("preserved_facts")
        focus = item.get("focus_areas")
        if not isinstance(facts, list) or not all(isinstance(x, str) for x in facts):
            raise ValueError("preserved_facts must be a string array")
        if not isinstance(focus, list) or not focus or not all(isinstance(x, str) for x in focus):
            raise ValueError("focus_areas must be a non-empty string array")
        profiles.append(PatientProfile(
            id=item["id"].strip(), label=item["label"].strip(),
            perspective=item["perspective"].strip(), modified_query=item["modified_query"].strip(),
            preserved_facts=tuple(facts), focus_areas=tuple(focus),
        ))
    if len({p.id for p in profiles}) != len(profiles):
        raise ValueError("profile IDs must be unique")
    if len({tuple(x.lower() for x in p.focus_areas) for p in profiles}) != len(profiles):
        raise ValueError("profiles must have distinguishable focus areas")
    return profiles
