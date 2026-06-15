import pandas as pd
import numpy as np

def latex_escape(s):
    """
    Escape LaTeX-sensitive characters for labels.
    """
    s = str(s)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s

def rotated_label(label, angle=60):
    """
    Rotated LaTeX column label.
    """
    return rf"\rotatebox{{{angle}}}{{\scriptsize {latex_escape(label)}}}"

def remove_numeric_header_row(latex, n_cols, has_index=True):
    """
    Remove the visible column header row like:
        & 1 & 2 & ... & 5 \\
    or
        {} & 1 & 2 & ... & 5 \\
    For hide_index=True tables, also handles:
        1 & 2 & ... & 5 \\
    """
    expected_labels = [str(i) for i in range(1, n_cols + 1)]

    def is_numeric_header_line(line):
        stripped = line.strip()
        stripped = stripped.replace(r"\\", "").strip()
        parts = [p.strip() for p in stripped.split("&")]

        if has_index:
            if len(parts) != n_cols + 1:
                return False
            first_cell_ok = parts[0] in {"", "{}"}
            numeric_cells_ok = parts[1:] == expected_labels
            return first_cell_ok and numeric_cells_ok
        else:
            if len(parts) != n_cols:
                return False
            return parts == expected_labels

    lines = latex.splitlines()
    lines = [line for line in lines if not is_numeric_header_line(line)]
    return "\n".join(lines)

def make_side_by_side_heatmap_table(
    table1,
    table2,
    model_order,
    model_labels,
    score_cols,
    attr_col="% Attribution",
    refusal_col="% Refusals",
    table1_name="Acceptability",
    table2_name="Non-acceptability",
    score_cmap="YlOrRd",
    attr_cmap="Blues",
    output_path="side_by_side_heatmap.tex",
    rotate_angle=60,
):
    if len(model_order) != len(model_labels):
        raise ValueError("model_order and model_labels must have the same length.")

    table1 = table1.copy().drop(columns=[refusal_col], errors="ignore")
    table2 = table2.copy().drop(columns=[refusal_col], errors="ignore")

    t1 = table1.reindex(model_order)
    t2 = table2.reindex(model_order)

    attr_label = rotated_label("Attr.", rotate_angle)
    rotated_score_cols = {col: rotated_label(col, rotate_angle) for col in score_cols}

    combined = pd.DataFrame(index=model_labels)
    combined.index.name = "Model"

    combined[(table1_name, attr_label)] = t1[attr_col].to_numpy()
    for col in score_cols:
        combined[(table1_name, rotated_score_cols[col])] = t1[col].to_numpy()

    for col in score_cols:
        combined[(table2_name, rotated_score_cols[col])] = t2[col].to_numpy()
    combined[(table2_name, attr_label)] = t2[attr_col].to_numpy()

    combined.columns = pd.MultiIndex.from_tuples(combined.columns)
    combined = combined.round(2)

    attr_cols = [(table1_name, attr_label), (table2_name, attr_label)]
    score_heatmap_cols = (
        [(table1_name, rotated_score_cols[col]) for col in score_cols]
        + [(table2_name, rotated_score_cols[col]) for col in score_cols]
    )

    styler = combined.style.format("{:.2f}", na_rep="--")
    styler = styler.background_gradient(cmap=attr_cmap, subset=attr_cols, vmin=0, vmax=1)

    score_values = combined.loc[:, score_heatmap_cols].to_numpy(dtype=float)
    if np.all(np.isnan(score_values)):
        score_max = 1
    else:
        score_max = np.nanmax(score_values)

    if score_max == 0 or np.isnan(score_max):
        score_max = 1

    styler = styler.background_gradient(cmap=score_cmap, subset=score_heatmap_cols, vmin=0, vmax=score_max)

    n_left = 1 + len(score_cols)
    n_right = len(score_cols) + 1
    column_format = "l" + "r" * n_left + r"@{\hspace{5pt}}!{\vrule width 0.8pt}@{\hspace{5pt}}" + "r" * n_right

    latex = styler.to_latex(convert_css=True, hrules=True, column_format=column_format, multicol_align="c")

    with open(output_path, "w") as f:
        f.write(latex)

    return combined, latex

def export_topn_heatmap_latex(
    display_df,
    pct_df,
    output_path,
    cmap="YlOrRd",
    vmin=0,
    vmax=100,
    remove_column_header=True,
    hide_index=False,
):
    styled = display_df.style.format(escape="latex").background_gradient(
        cmap=cmap, gmap=pct_df, vmin=vmin, vmax=vmax, axis=None
    )

    if hide_index:
        if hasattr(styled, "hide"):
            styled = styled.hide(axis="index")
        else:
            styled = styled.hide_index()

    column_format = "c" * display_df.shape[1] if hide_index else "l" + "c" * display_df.shape[1]

    latex = styled.to_latex(convert_css=True, hrules=True, column_format=column_format)

    if remove_column_header:
        latex = remove_numeric_header_row(latex, n_cols=display_df.shape[1], has_index=not hide_index)

    with open(output_path, "w") as f:
        f.write(latex)

    return latex
