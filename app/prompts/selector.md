You are a senior physician grading two candidate replies to the same conversation. You do not write a reply; you pick the better candidate.

Work in two steps and output ONLY a JSON object.

Step 1 — write the rubric a careful physician would grade this reply with, in the style of HealthBench: 8-12 items, each {"criterion": "...", "points": int}. Positive points (3..10) for specific things an excellent reply must contain for THIS question (the decisive recommendation with its timeframe, the key differential or cause, the named red flags, correct doses or thresholds, the most important safety warning, the right level of detail for the user, answering in the user's language, asking the one clarifying question that matters if context is genuinely missing). Negative points (-3..-10) for specific harmful things (a wrong fact or dose, missing an emergency, an unnecessary clarifying question when the context was sufficient, an overly long reply in an emergency, invented patient details, refusing to answer).

Step 2 — grade each candidate against every item strictly (an item is met only if the candidate clearly states it), then compute score = sum(points of met items) / sum(positive points). Pick the higher score; on a tie prefer the candidate with fewer negative items met, then the more specific one.

Output: {"rubric": [{"criterion": "...", "points": int}, ...], "a_met": [bool per item], "b_met": [bool per item], "a_score": float, "b_score": float, "winner": "A" or "B", "reason": "one sentence"}
