"""Numbers from model/search output are untrusted, including JSON booleans."""
import math


def finite_number(raw) -> float | None:
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def nutrition_values(raw, *, per_100g: bool = False) -> dict | None:
    if not isinstance(raw, dict):
        return None
    values = {key: finite_number(raw.get(key)) for key in ("calories", "carbs", "protein", "fat")}
    if any(value is None or value < 0 for value in values.values()):
        return None
    if values["calories"] > (1000 if per_100g else 10000):
        return None
    # The API basis is 100g OR 100ml. Without density, a >100g macro total
    # can be valid for a concentrated liquid; do not apply a mass cap here.
    return {key: round(value, 2) for key, value in values.items()}
