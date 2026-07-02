import os
import pandas as pd
from collections import Counter, defaultdict
from config.settings import settings
from utils.scoring import safe_div, compute_scaled_score
from utils.parsing import normalize_attr_value, parse_response_dict, is_unknown_value
from utils.latex import make_side_by_side_heatmap_table, export_topn_heatmap_latex

# Ensure output directory exists
os.makedirs(settings.paths["figures_dir"], exist_ok=True)
os.makedirs(os.path.join(settings.paths["figures_dir"], "appendix"), exist_ok=True)

# Load data
parsed_results = pd.read_csv(settings.paths["sob_results_csv"])
target_vals = parsed_results['target_dem_val'].value_counts().index.tolist()[:10]

TARGET_DEM_VALS = [
    "muslim", "lgbtq", "women", "jew", "immigrant", 
    "black", "disabled", "asian", "mexican", "arab"
]

def generate_model_scores(df, prompt_name):
    filtered_results = df.loc[
        (df["prompt_name"] == prompt_name) & df["response"].notna()
    ]
    
    rows = []
    for model_name, model_df in filtered_results.groupby("model_name"):
        row = {"model_name": model_name}
        refusals = (model_df["response"] == "refusal").sum()
        unknowns = (model_df["response"] == "Unknown").sum()

        row["% Refusals"] = safe_div(refusals, len(model_df))
        row["% Attribution"] = round(1 - safe_div(unknowns, len(model_df) - refusals), 3)
        row["Overall"] = compute_scaled_score(model_df)

        for target in target_vals:
            target_df = model_df[model_df["target_dem_val"] == target]
            row[target] = compute_scaled_score(target_df)

        rows.append(row)
        
    return pd.DataFrame(rows).set_index("model_name")

accep_df = generate_model_scores(parsed_results, "acceptable_ours")
non_accep_df = generate_model_scores(parsed_results, "non_acceptable_ours")

score_cols = ["Overall"] + [c for c in TARGET_DEM_VALS if c in target_vals]

model_order = [
    "gpt51_instruct", "gpt51_think", "sonnet46_instruct", "sonnet46_think",
    "llama70b_instruct", "qwen35b_instruct", "qwen35b_think", "olmo32b_instruct",
    "olmo32b_think", "gemma27b_instruct", "phi4_14b_think", "llama8b_instruct",
]

model_labels = [
    "gpt5.1-I", "gpt5.1-T", "sonnet4.6-I", "sonnet4.6-T", "llama70b",
    "qwen35b-I", "qwen35b-T", "olmo32b-I", "olmo32b-T", "gemma27b",
    "phi4-14b-T", "llama8b",
]

# Generate Overall side-by-side table
make_side_by_side_heatmap_table(
    table1=accep_df,
    table2=non_accep_df,
    model_order=model_order,
    model_labels=model_labels,
    score_cols=score_cols,
    table1_name="Acceptability",
    table2_name="Non-acceptability",
    output_path=os.path.join(settings.paths["figures_dir"], "overall_results.tex"),
    rotate_angle=60,
)

def make_rowwise_topn_attr_and_pct_tables(
    df, prompt_name, model_name=None, target_vals=TARGET_DEM_VALS,
    group_col="target_dem_val", response_col="response", top_n=5
):
    sub = df.loc[
        (df["prompt_name"] == prompt_name)
        & (df[group_col].isin(target_vals))
        & df[response_col].notna()
    ].copy()
    
    if model_name:
        sub = sub[sub["model_name"] == model_name]

    sub["_parsed_response"] = sub[response_col].apply(parse_response_dict)
    valid = sub[sub["_parsed_response"].notna()].copy()

    display_rows, pct_rows = [], []

    for target in target_vals:
        target_df = valid[valid[group_col] == target]
        denom = len(target_df)

        display_row, pct_row = {group_col: target}, {group_col: target}

        if denom == 0:
            for i in range(1, top_n + 1):
                display_row[f"Top-{i}"] = "--"
                pct_row[f"Top-{i}"] = 0.0
            display_rows.append(display_row)
            pct_rows.append(pct_row)
            continue

        counter = Counter()
        display_forms = defaultdict(Counter)

        for d in target_df["_parsed_response"]:
            normalized_values_in_response = set()
            for v in d.values():
                if is_unknown_value(v): continue
                raw_value = str(v).strip()
                norm_value = normalize_attr_value(raw_value)
                normalized_values_in_response.add(norm_value)
                display_forms[norm_value][raw_value] += 1
            counter.update(normalized_values_in_response)

        top_items = counter.most_common(top_n)

        for i in range(1, top_n + 1):
            col = f"Top-{i}"
            if i <= len(top_items):
                norm_value, count = top_items[i - 1]
                display_value = display_forms[norm_value].most_common(1)[0][0]
                pct = 100 * count / denom
                display_row[col] = display_value
                pct_row[col] = pct
            else:
                display_row[col] = "--"
                pct_row[col] = 0.0

        display_rows.append(display_row)
        pct_rows.append(pct_row)

    display_df = pd.DataFrame(display_rows).set_index(group_col)
    pct_df = pd.DataFrame(pct_rows).set_index(group_col)

    display_df.columns = [str(i) for i in range(1, top_n + 1)]
    pct_df.columns = [str(i) for i in range(1, top_n + 1)]
    display_df.index = [str(x) for x in display_df.index]
    pct_df.index = display_df.index

    return display_df, pct_df

# Generate Appendix per-model and overall plots
runs = [(m, f"{m}_") for m in parsed_results["model_name"].unique()]
runs.append((None, "overall_"))

for model_name, prefix in runs:
    acc_display, acc_pct = make_rowwise_topn_attr_and_pct_tables(
        parsed_results, "acceptable_ours", model_name=model_name
    )
    nonacc_display, nonacc_pct = make_rowwise_topn_attr_and_pct_tables(
        parsed_results, "non_acceptable_ours", model_name=model_name
    )

    export_topn_heatmap_latex(
        acc_display, acc_pct,
        os.path.join(settings.paths["figures_dir"], "appendix", f"{prefix}acceptability_top5_heatmap.tex"),
        hide_index=False
    )
    export_topn_heatmap_latex(
        nonacc_display, nonacc_pct,
        os.path.join(settings.paths["figures_dir"], "appendix", f"{prefix}non_acceptability_top5_heatmap.tex"),
        hide_index=True
    )
