PROFILE_GENERATOR_INSTRUCTIONS = """You create complementary analysis profiles for a medical-answer pipeline.
Return exactly the requested number of materially distinct profiles. A profile is a reasoning
perspective, not permission to invent a different patient. Preserve every clinical fact and all
conversation constraints. Never add symptoms, diagnoses, demographics, medications, test results,
locations, or preferences that the conversation does not state. Each modified_query must be
self-contained, preserve the user's language, and ask L2 to answer the original need while paying
special attention to that profile's focus. Choose query-dependent perspectives such as common
interpretation, red-flag safety, missing context, medication interactions, communication needs, or
local resource constraints. Do not force an irrelevant perspective."""


def profile_generator_input(conversation: str, count: int) -> str:
    return f"Create exactly {count} profiles from this authoritative conversation JSON:\n{conversation}"
