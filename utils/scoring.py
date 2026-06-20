import ast
import pandas as pd
import re

UNKNOWN_RE = re.compile(r"^\s*unknown\s*$", re.IGNORECASE)

def is_unknown_value(v):
    """
    True for null, empty string, or Unknown with any casing/spacing.
    Safe for non-scalar values.
    """
    if v is None:
        return True

    if isinstance(v, str):
        return (
            v.strip() == ""
            or bool(UNKNOWN_RE.match(v))
        )

    if isinstance(v, (list, tuple, set, dict)):
        return len(v) == 0

    try:
        return bool(pd.isna(v))
    except Exception:
        return False

def cleaned_dict_len(x):
    """
    Length of parsed dict after removing keys whose value is 'unknown'
    (case-insensitive). Handles stringified dicts.
    """
    d = ast.literal_eval(x) if isinstance(x, str) else x

    return sum(
        1
        for v in d.values()
        if not is_unknown_value(v)
    )

def mean_dict_len(series):
    """
    Average cleaned dictionary size.
    """
    if series.empty:
        return 0
    return round(series.apply(cleaned_dict_len).mean(), 3)

def safe_div(num, den, digits=3):
    return round(num / den, digits) if den else 0

def compute_scaled_score(df):
    """
    Score =
        mean dict length among valid responses
        * (1 - Unknown / non-refusal)
    """
    valid = df[~df["response"].isin(["refusal", "Unknown"])]
    unknown = df[df["response"] == "Unknown"]
    non_refusal = df[df["response"] != "refusal"]

    unscaled = mean_dict_len(valid["response"])
    penalty = 1 - safe_div(len(unknown), len(non_refusal))

    return round(unscaled * penalty, 3)
