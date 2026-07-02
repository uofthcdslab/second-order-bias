import argparse
import os
import subprocess
import sys
from pathlib import Path

# Ensure the parent process can print Unicode glyphs (▶, [OK], etc.) on
# Windows consoles that default to cp1252.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from config.settings import settings

# Define the sequence of pipeline steps
PIPELINE_STEPS = {
    1: ("Sampling inputs", "src.input_sampling"),
    2: ("Normalizing and mapping demographics", "src.input_normalizing"),
    3: ("Running Second-Order Bias (SOB)", "src.run_sob"),
    4: ("Parsing SOB results", "src.parse_sob_results"),
    5: ("Addressing parsing errors", "src.address_parsing"),
    6: ("Generating plots and LaTeX tables", "src.do_plots"),
}

# Steps that run when the user starts from a pre-built `input_biased_text.csv`.
# Sampling + normalizing are skipped.
FROM_PROCESSED_STEPS = {
    3: ("Running Second-Order Bias (SOB)", "src.run_sob"),
    4: ("Parsing SOB results", "src.parse_sob_results"),
    5: ("Addressing parsing errors", "src.address_parsing"),
    6: ("Generating plots and LaTeX tables", "src.do_plots"),
}


def run_step(step_num, step_table):
    desc, module = step_table[step_num]
    print(f"\n{'='*60}")
    print(f"▶ Step {step_num}: {desc}")
    print(f"▶ Running: python -m {module}")
    print(f"{'='*60}\n")

    try:
        # Force UTF-8 in the child process so the ▶ / OK / [X] glyphs
        # don't blow up on Windows consoles that default to cp1252.
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        result = subprocess.run(
            [sys.executable, "-m", module], check=True, env=env
        )
    except subprocess.CalledProcessError as e:
        print(f"\n[X] Pipeline failed at Step {step_num} ({module}).")
        print(f"Exit code: {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n\n[!] Pipeline interrupted by user.")
        sys.exit(1)


def _check_processed_csv() -> Path:
    """Verify `data/input_biased_text.csv` exists before starting the
    'from processed' pipeline."""
    csv_path = Path(settings.paths["input_biased_text_csv"])
    if not csv_path.exists():
        print(f"[X] Processed data not found at: {csv_path}")
        print()
        print("You have two options:")
        print("  1. Run the full pipeline (sampling + normalizing) first:")
        print("       python main.py --all")
        print("  2. Or run sampling + normalizing in one shot:")
        print("       python -m src.get_processed_data")
        print()
        print("If you want to understand how input_biased_text.csv was")
        print("produced, see src/get_processed_data.py and the comments in")
        print("src/input_normalizing.py (Section A is replicable; Section B")
        print("requires manual author annotation).")
        sys.exit(1)
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Second-Order Bias Pipeline")
    parser.add_argument(
        "--step",
        type=int,
        choices=list(PIPELINE_STEPS.keys()),
        help="Run a specific step (1-6).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run the entire pipeline from start to finish.",
    )
    parser.add_argument(
        "--from-processed",
        action="store_true",
        help=(
            "Skip sampling and normalizing. Use this when "
            "`data/input_biased_text.csv` already exists (e.g. the file "
            "shipped in the repo or one you've previously generated via "
            "`python -m src.get_processed_data`). Steps 3→6 will run."
        ),
    )

    args = parser.parse_args()

    # Default: print help
    if not args.step and not args.all and not args.from_processed:
        parser.print_help()
        print("\nAvailable Steps:")
        for k, v in PIPELINE_STEPS.items():
            print(f"  {k}: {v[0]} ({v[1]})")
        print("\nQuick-starts:")
        print("  python main.py --all             # full pipeline from raw datasets")
        print("  python main.py --from-processed  # skip sampling+normalizing")
        print("  python main.py --step 3          # run only model inference")
        sys.exit(0)

    # Branch: --from-processed
    if args.from_processed:
        csv_path = _check_processed_csv()
        print(f"[OK] Using pre-built processed data: {csv_path}")
        print("   (Skipping sampling and normalizing.)")
        print("   (For reproducibility, see src/get_processed_data.py.)")
        for step_num in sorted(FROM_PROCESSED_STEPS.keys()):
            run_step(step_num, FROM_PROCESSED_STEPS)
        print("\n[OK] Pipeline completed (from processed data)!")
        return

    # --step
    if args.step:
        run_step(args.step, PIPELINE_STEPS)
        return

    # --all
    if args.all:
        print("Starting full pipeline execution...\n")
        for step_num in sorted(PIPELINE_STEPS.keys()):
            run_step(step_num, PIPELINE_STEPS)
        print("\n[OK] Full pipeline completed successfully!")


if __name__ == "__main__":
    main()
