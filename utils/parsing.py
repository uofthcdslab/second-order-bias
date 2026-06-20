import ast
import pandas as pd
from utils.scoring import is_unknown_value

def normalize_attr_value(v):
    """
    Case-insensitive normalization for attribution values.
    """
    return str(v).strip().lower()

def parse_response_dict(x):
    """
    Parse response into a cleaned dictionary.

    Returns None for:
    - Unknown
    - refusal
    - null
    - invalid/non-dict responses
    - dicts where all values are Unknown/null
    - dicts containing list/tuple/set values

    If any value is list-like, the entire response is ignored.
    """
    if x is None:
        return None

    if isinstance(x, str):
        if x.strip().lower() in {"unknown", "refusal"}:
            return None

        try:
            x = ast.literal_eval(x)
        except Exception:
            return None

    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    if not isinstance(x, dict):
        return None

    # Ignore entire response if any value is list-like
    for v in x.values():
        if isinstance(v, (list, tuple, set)):
            return None

    cleaned = {
        k: v
        for k, v in x.items()
        if not is_unknown_value(v)
    }

    if len(cleaned) == 0:
        return None

    return cleaned

def clean_response_dict(x):
    # Handle actual null response cells
    if pd.isna(x):
        return "Unknown"

    # If response is stored as a string, convert it to a dict
    if isinstance(x, str):
        try:
            x = ast.literal_eval(x)
        except Exception:
            return x

    # Only process dictionaries
    if not isinstance(x, dict):
        return x

    cleaned = {
        k: v
        for k, v in x.items()
        if not (
            pd.isna(v)
            or str(v).strip().lower() == "unknown"
        )
    }

    # If every value was Unknown/null
    if len(cleaned) == 0:
        return "Unknown"

    return cleaned
