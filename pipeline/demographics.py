from __future__ import annotations

from dataclasses import dataclass
@dataclass(frozen=True)
class DemographicProfile:
    id: str
    label: str
    age_group: str | None
    sex_at_birth: str | None
    gender: str | None
    perspective: str
    modified_query: str
    preserved_facts: tuple[str, ...]
    hypothetical_facts: tuple[str, ...]
    focus_areas: tuple[str, ...]


_PERSONAS = {
    "child_boy": dict(
        label="8-year-old boy", age_group="8-year-old child", sex_at_birth="male",
        gender="boy", perspective="Pediatric presentation, dosing, consent, and red flags",
        focus_areas=("pediatric differential", "weight-based treatment", "caregiver guidance"),
    ),
    "young_woman": dict(
        label="32-year-old woman", age_group="32-year-old adult", sex_at_birth="female",
        gender="woman", perspective="Adult presentation with pregnancy and reproductive considerations",
        focus_areas=("common adult differential", "pregnancy safety", "medication contraindications"),
    ),
    "middle_man": dict(
        label="50-year-old man", age_group="50-year-old adult", sex_at_birth="male",
        gender="man", perspective="Midlife presentation and cardiometabolic risk",
        focus_areas=("midlife differential", "cardiometabolic risk", "medication interactions"),
    ),
    "older_woman": dict(
        label="72-year-old woman", age_group="72-year-old older adult", sex_at_birth="female",
        gender="woman", perspective="Older-adult presentation, frailty, polypharmacy, and red flags",
        focus_areas=("older-adult differential", "polypharmacy", "functional safety"),
    ),
    "nonbinary_adult": dict(
        label="40-year-old nonbinary adult", age_group="40-year-old adult", sex_at_birth=None,
        gender="nonbinary", perspective="Inclusive communication without inferring anatomy",
        focus_areas=("inclusive communication", "anatomy-specific clarification", "access barriers"),
    ),
}

# Each supported count is an intentional balanced set, not a truncation of one ordered list.
_PROFILE_SETS = {
    1: ("middle_man",),
    2: ("young_woman", "older_woman"),
    3: ("child_boy", "young_woman", "older_woman"),
    4: ("child_boy", "young_woman", "middle_man", "older_woman"),
    5: ("child_boy", "young_woman", "middle_man", "older_woman", "nonbinary_adult"),
}


def predefined_demographics(conversation: str, count: int) -> list[DemographicProfile]:
    try:
        persona_ids = _PROFILE_SETS[count]
    except KeyError as exc:
        raise ValueError("demographics supports PATIENT_PROFILE_COUNT from 1 through 5") from exc

    profiles = []
    for persona_id in persona_ids:
        persona = _PERSONAS[persona_id]
        assumptions = tuple(
            value for value in (
                persona["age_group"],
                f"sex at birth: {persona['sex_at_birth']}" if persona["sex_at_birth"] else None,
                f"gender: {persona['gender']}" if persona["gender"] else None,
            ) if value
        )
        profiles.append(DemographicProfile(
            id=persona_id,
            label=persona["label"],
            age_group=persona["age_group"],
            sex_at_birth=persona["sex_at_birth"],
            gender=persona["gender"],
            perspective=persona["perspective"],
            modified_query=conversation,
            preserved_facts=(),
            hypothetical_facts=assumptions,
            focus_areas=persona["focus_areas"],
        ))
    return profiles
