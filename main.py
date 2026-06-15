import argparse
import subprocess
import sys

# Define the sequence of pipeline steps
PIPELINE_STEPS = {
    1: ("Sampling inputs", "src.input_sampling"),
    2: ("Normalizing and mapping demographics", "src.input_normalizing"),
    3: ("Running Second-Order Bias (SOB) models", "src.run_sob"),
    4: ("Parsing SOB results", "src.parse_sob_results"),
    5: ("Addressing parsing errors", "src.address_parsing"),
    6: ("Generating plots and LaTeX tables", "src.do_plots"),
}

def run_step(step_num):
    desc, module = PIPELINE_STEPS[step_num]
    print(f"\n{'='*60}")
    print(f"▶ Step {step_num}: {desc}")
    print(f"▶ Running: python -m {module}")
    print(f"{'='*60}\n")
    
    try:
        # Run the module using subprocess to ensure a clean execution environment
        result = subprocess.run([sys.executable, "-m", module], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Pipeline failed at Step {step_num} ({module}).")
        print(f"Exit code: {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline interrupted by user.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Second-Order Bias Pipeline Orchestrator")
    parser.add_argument(
        "--step", 
        type=int, 
        choices=list(PIPELINE_STEPS.keys()), 
        help="Run a specific step (1-6)."
    )
    parser.add_argument(
        "--all", 
        action="store_true", 
        help="Run the entire pipeline from start to finish."
    )

    args = parser.parse_args()

    # If no arguments provided, print help
    if not args.step and not args.all:
        parser.print_help()
        print("\nAvailable Steps:")
        for k, v in PIPELINE_STEPS.items():
            print(f"  {k}: {v[0]} ({v[1]})")
        print("\nExample usage:")
        print("  python main.py --all       # Run the full pipeline")
        print("  python main.py --step 3    # Run only the model inference step")
        sys.exit(0)

    if args.step:
        run_step(args.step)
    elif args.all:
        print("Starting full pipeline execution...\n")
        for step_num in sorted(PIPELINE_STEPS.keys()):
            run_step(step_num)
        print("\n✅ Full pipeline completed successfully!")

if __name__ == "__main__":
    main()
