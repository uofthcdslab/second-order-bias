import json
from pathlib import Path

import pandas as pd

# ── Load Datasets ───────────────────────────────────────────────────────────────

dynab     = pd.read_csv('datasets/sampled/dynab_sampled_500.csv')
hatecheck = pd.read_csv('datasets/sampled/hatecheck_sampled_500.csv')
ishate    = pd.read_csv('datasets/sampled/ishate_sampled_500.csv')
linghate  = pd.read_csv('datasets/sampled/linghate_sampled_500.csv')
toxigen   = pd.read_csv('datasets/sampled/toxigen_sampled_500.csv')


# ── Normalizing ─────────────────────────────────────────────────────────────────

# Column name mapping across datasets
COLUMN_MAP = {
    "dynab":     {"text": "text",                 "demographic": "target",        "type": "type"},
    "hatecheck": {"text": "test_case",            "demographic": "target_ident",  "type": "label_gold"},
    "ishate":    {"text": "text",                 "demographic": "target",        "type": "implicit_layer"},
    "linghate":  {"text": "#SENTENCE_FROM_TWEET", "demographic": "TARGET",        "type": "LABEL"},
    "toxigen":   {"text": "text",                 "demographic": "target_group",  "type": "framing"},
}

datasets = {
    "dynab":     dynab,
    "hatecheck": hatecheck,
    "ishate":    ishate,
    "linghate":  linghate,
    "toxigen":   toxigen,
}

# Normalization lookup: raw demographic value -> normalized category - manually done
target_categories = list(set(list(dynab['target'].unique()) + list(hatecheck['target_ident'].unique()) + list(ishate['target'].unique()) + list(linghate['TARGET'].unique()) + list(toxigen['target_group'].unique())))
# saved as demographic_normalizing.csv

target_normalized = pd.read_csv('demographic_normalizing.csv')
norm_cols = [c for c in target_normalized.columns if c not in ("original", "original2")]
norm_map = (
    target_normalized
    .groupby("original")
    .apply(lambda g: g[norm_cols].to_dict(orient="records"))
    .to_dict()
)


# ── Build Master DataFrame ──────────────────────────────────────────────────────

chunks = []

for dataset_name, col_map in COLUMN_MAP.items():
    df = datasets[dataset_name]

    chunk = pd.DataFrame({
        "source": dataset_name,
        **{
            output_name: df[source_col] if source_col in df.columns else None
            for output_name, source_col in col_map.items()
        }
    })

    # Map -> list of dicts, explode to multiple rows, unpack dict into columns
    chunk["_norm"] = chunk["demographic"].map(norm_map)
    chunk = chunk.explode("_norm", ignore_index=True)

    # Unpack the dict — only adds columns that exist in normalization_df
    norm_unpacked = chunk.pop("_norm").apply(lambda x: x if isinstance(x, dict) else {})
    chunk = pd.concat([chunk, pd.DataFrame(norm_unpacked.tolist(), index=chunk.index)], axis=1)

    chunks.append(chunk)

master_df = pd.concat(chunks, ignore_index=True)


# ── Sanity Checks ───────────────────────────────────────────────────────────────

# Infer which normalization columns were unpacked
norm_output_cols = [c for c in master_df.columns if c not in
                    ["source", "text", "demographic", "type"] + list(COLUMN_MAP.keys())]

total_raw_rows = sum(len(d) for d in datasets.values())
print(f"Raw rows across all datasets : {total_raw_rows}")
print(f"Rows after explode           : {len(master_df)}")
print(f"\nNormalization columns added  : {norm_output_cols}")
for col in norm_output_cols:
    print(f"  Nulls in '{col}': {master_df[col].isna().sum()}")
print(f"\nColumns in master_df         : {list(master_df.columns)}")
print(f"\nSource breakdown:\n{master_df['source'].value_counts()}")
print(f"\nSample:\n{master_df.head()}")

master_df.to_csv("target_maped_before_author_annot.csv", index=False)
print(f"\nMaster DataFrame saved - target_maped_before_author_annot.csv  {master_df.shape}")


# ── Manual annotation  ──────────────────────────────────────────────────
# religion
#   is hate: religious group --> muslim, jew
#   is hate: some of the immigrant targets also include muslims, mexicans
# race
#   dynab and ishate
#   most of them are non white etc.
#   some are kept as race - no clear target
#   some are mixed race - no specific target race
# nation
#   ishate - changed nation to mexican, asian, etc. where applicable

# the processed file is saved as target_mapped.csv

master_df_annotated = pd.read_csv('target_mapped.csv')
master_df_annotated.drop_duplicates(subset=['source', 'text'], inplace=True)

# Per-source sequential index
master_df_annotated['tix'] = master_df_annotated.groupby('source').cumcount() + 1
print(master_df_annotated.groupby('source')['tix'].agg(['min', 'max', 'count']))

# Select only relevant columns
master_df_annotated = master_df_annotated[[
    'tix', 'source', 'text', 'demographic', 'type',
    'normalized', 'normalized_secondary', 'normalized_tertiary'
]]


# ── Demographic Category Mapping ────────────────────────────────────────────────
# | Demographic Group  | Primary Category       | Notes                                      |
# |--------------------|------------------------|--------------------------------------------|
# | economic status    | Socio-economic status  |                                            |
# | working class      | Socio-economic status  |                                            |
# | professionals      | Socio-economic status  |                                            |
# | immigrant          | Socio-economic status  | Also overlaps with Nationality             |
# | refugee            | Socio-economic status  | Also overlaps with Nationality             |
# | lgbtq              | Sexual orientation     | Also overlaps with Gender identity         |
# | muslim             | Religion               |                                            |
# | jew                | Religion               |                                            |
# | nazi               | Religion / Political   | Ideological; may need its own category     |
# | black              | Race                   |                                            |
# | white              | Race                   |                                            |
# | non white          | Race                   |                                            |
# | mixed race         | Race                   |                                            |
# | race               | Race                   |                                            |
# | arab               | Race                   | Also commonly treated as Ethnicity         |
# | ethnic             | Ethnicity              |                                            |
# | hispanic           | Ethnicity              | Also overlaps with Nationality             |
# | mexican            | Ethnicity              | Also overlaps with Nationality             |
# | polish             | Ethnicity              | Also overlaps with Nationality             |
# | indigenous         | Ethnicity              |                                            |
# | asian              | Ethnicity              |                                            |
# | chinese            | Ethnicity              | Also overlaps with Nationality             |
# | non specified      | Physical appearance    | Meta-category; may need its own bucket     |
# | nation             | Nationality            |                                            |
# | women              | Gender identity        |                                            |
# | disabled           | Disability status      |                                            |
# | old                | Age                    |                                            |
# | political          | —                      | Does not fit cleanly into any category     |

demographic_category_map = {
    "economic status": "Socio Economic Status",
    "working class":   "Socio Economic Status",
    "professionals":   "Socio Economic Status",
    "immigrant":       "Socio Economic Status",
    "refugee":         "Socio Economic Status",
    "lgbtq":           "Sexual Orientation",
    "muslim":          "Religion",
    "jew":             "Religion",
    "nazi":            "Ethnicity",
    "black":           "Race",
    "white":           "Race",
    "non white":       "Race",
    "mixed race":      "Race",
    "race":            "Race",
    "arab":            "Race",
    "ethnic":          "Ethnicity",
    "hispanic":        "Ethnicity",
    "mexican":         "Nationality",
    "polish":          "Nationality",
    "indigenous":      "Ethnicity",
    "asian":           "Ethnicity",
    "chinese":         "Nationality",
    "non specified":   "Physical Appearance",
    "nation":          "Nationality",
    "women":           "Gender Identity",
    "disabled":        "Disability Status",
    "old":             "Age",
    "political":       "Nationality",
}

master_df_annotated['normalized_dem_category']           = master_df_annotated['normalized'].str.lower().str.strip().map(demographic_category_map)
master_df_annotated['normalized_dem_secondary_category'] = master_df_annotated['normalized_secondary'].str.lower().str.strip().map(demographic_category_map, na_action='ignore')
master_df_annotated['normalized_dem_tertiary_category']  = master_df_annotated['normalized_tertiary'].str.lower().str.strip().map(demographic_category_map, na_action='ignore')

master_df_annotated.to_csv("target_mapped.csv", index=False)
print(f"Annotated DataFrame saved  -  target_mapped.csv  {master_df_annotated.shape}")
