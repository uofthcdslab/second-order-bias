import pandas as pd
import numpy as np


def stratified_sample(df: pd.DataFrame, col: str, n: int, random_state: int = 42) -> pd.DataFrame:
    """
    Proportional stratified sampling from `df` based on `col`.
    """
    proportions = df[col].value_counts(normalize=True)

    # Allocate samples per category
    n_per_cat = (proportions * n).round().astype(int)

    # Fix rounding so total == n
    diff = n - n_per_cat.sum()
    if diff != 0:
        adjust_cats = proportions.sort_values(ascending=False).index[:abs(diff)]
        n_per_cat[adjust_cats] += np.sign(diff)

    return (
        df
        .groupby(col, group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), n_per_cat[x.name]), random_state=random_state))
    )


# ── IS Hate ────────────────────────────────────────────────────────────────────
# 0. only test split
# 1. only HS
# 2. 186 implicit HS (all kept)
# 3. total 500 max cap
# 4. remaining explicit HS filled via stratified sampling on target proportions

data = pd.read_parquet("datasets/downloaded/ishate_test.parquet.gzip")

hate = data[data['hateful_layer'] == 'HS']

implicit_hate = hate[hate['implicit_layer'] == 'Implicit HS']
explicit_hate = hate[hate['implicit_layer'] == 'Explicit HS']

n_explicit = 500 - len(implicit_hate)
sampled_explicit = stratified_sample(explicit_hate, col="target", n=n_explicit)

ishate_sampled = pd.concat([implicit_hate, sampled_explicit], axis=0, ignore_index=True)

ishate_sampled.to_csv('datasets/sampled/ishate_sampled_500.csv', index=False)
print(f"IS Hate saved: {ishate_sampled.shape}")


# ── Toxigen ────────────────────────────────────────────────────────────────────
# 0. only test split
# 1. filtered data with toxicity_human > 2.5 - for only hateful texts
# 2. no sampling since total filtered values < 500

toxigen = pd.read_csv("datasets/downloaded/toxigen_annotated_test.csv")

hate_toxigen = toxigen[toxigen['toxicity_human'] > 2.5]
hate_toxigen.to_csv('datasets/sampled/toxigen_sampled_500.csv', index=False)
print(f"Toxigen saved: {hate_toxigen.shape}")


# ── Hate Check ─────────────────────────────────────────────────────────────────
# 0. only test split
# 1. only hateful texts
# 2. target and directed column distributions were almost uniform
# 3. so just did a random sample of 500 samples

hatecheck = pd.read_csv('datasets/downloaded/hatecheck_test.csv')

hatecheck_hate = hatecheck[hatecheck['label_gold'] == 'hateful']
sampled_hatecheck = hatecheck_hate.sample(n=500, random_state=42)

sampled_hatecheck.to_csv('datasets/sampled/hatecheck_sampled_500.csv', index=False)
print(f"Hate Check saved: {sampled_hatecheck.shape}")


# ── DynaB ──────────────────────────────────────────────────────────────────────
# 0. only test split
# 1. only hateful texts
# 2. filtered where target is absent
# 3. stratified sampling based on hate type column to get 500 samples

dynab = pd.read_csv('datasets/downloaded/Dynamically Generated Hate Dataset v0.2.3.csv')

dynab      = dynab[dynab['split'] == 'test']
dynab_hate = dynab[dynab['label'] == 'hate']
dynab_hate = dynab_hate[~dynab_hate['target'].isin(['notargetrecorded', 'notgiven'])]

sampled_dynab = stratified_sample(dynab_hate, col="type", n=500)

sampled_dynab.to_csv('datasets/sampled/dynab_sampled_500.csv', index=False)
print(f"DynaB saved: {sampled_dynab.shape}")


# ── LingHate ───────────────────────────────────────────────────────────────────

ling = pd.read_csv('datasets/downloaded/linguistic_informed_data.csv', sep='\t')
ling.to_csv('datasets/downloaded/transformed_linguistic_informed_data.csv', index=False)

sampled_ling = stratified_sample(ling, col="TARGET", n=500)

sampled_ling.to_csv('datasets/sampled/linghate_sampled_500.csv', index=False)
print(f"LingHate saved: {sampled_ling.shape}")
