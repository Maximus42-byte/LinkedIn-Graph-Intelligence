"""Keep JSON-facing floating point results stable across BLAS implementations."""

import math


def stable_numbers(value):
    """Round only output scalars; preserve integer counts and missing values."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 9) + 0.0
    if isinstance(value, dict):
        return {key: stable_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [stable_numbers(item) for item in value]
    return value
