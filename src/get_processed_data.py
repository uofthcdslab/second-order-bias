"""
get_processed_data.py — One-shot pipeline that builds the final
`data/input_biased_text.csv` from the raw downloaded datasets.

This is the canonical entry point for reproducing the processed
inputs that feed into `run_sob.py`. It combines the logic of the two
underlying scripts:

    src/input_sampling.py     (5 dataset → data/sampled/*_sampled_500.csv)
    src.input_normalizing.py  (data/sampled/* + demographic_normalizing.csv
                               → data/input_biased_text.csv)

The intermediate file `data/input_biased_text_before_author_annot.csv`
is also written, so a human reviewer can compare it against the
manually-annotated final.

The second half of `input_normalizing.py` (author-annotation pass)
is **not** replicated here — it is a manual step that requires a
human to apply the religion / race / nation annotations described
in the code comments of `input_normalizing.py`. To run it manually,
open `input_normalizing.py` and uncomment the Section B block, or
see the README for details.

Usage:
    python -m src.get_processed_data
"""

from __future__ import annotations

from config.settings import PROJECT_ROOT

# We import the two scripts as modules so the logic stays single-sourced
# and we don't duplicate the per-dataset code.
from src import input_sampling        # noqa: F401  (run-on-import side effects)
from src import input_normalizing     # noqa: F401  (run-on-import side effects)


def main() -> None:
    print("=" * 60)
    print("get_processed_data — building data/input_biased_text.csv")
    print("=" * 60)
    print(f"  Sampled datasets    : {PROJECT_ROOT / 'data/sampled'}")
    print(f"  Intermediate output : {PROJECT_ROOT / 'data/input_biased_text_before_author_annot.csv'}")
    print(f"  Final output (manual annotation required) :")
    print(f"      {PROJECT_ROOT / 'data/input_biased_text.csv'}")
    print()
    print("Sampling and normalizing completed.")
    print()
    print("Next step:")
    print("  • For automated re-run from scratch:     python main.py --all")
    print("  • To skip sampling/normalizing:          python main.py --from-processed")
    print("  • To start from a pre-built file:        python main.py --from-processed")


if __name__ == "__main__":
    main()
