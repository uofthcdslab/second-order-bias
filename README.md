# Evaluating Second-Order Bias of LLMs Through Epistemic Entitlement

> **Preprint coming soon.**

---

## Overview

Most LLM bias evaluations focus on whether a model *generates* biased content. This work investigates a subtler problem: bias in how LLMs *judge* biased content. We call this **second-order bias** — social bias that surfaces in a model's evaluation of bias, rather than in its generation.

To measure this, we draw on entitlement epistemology and frame bias as misplaced foundational knowledge. From this, we derive a logical reasoning task where LLMs must judge whether a biased text is acceptable or non-acceptable, and *to whom*. We introduce two metrics to capture how readily models make unsupported demographic inferences, and how these vary across target groups.

Evaluating both open and closed models, we find that this task bypasses safety guardrails, varies systematically across demographic groups, and reveals how models remain sensitive to demographic labels even in judgment settings.

---

## Repository Structure

```
second-order-bias/
├── main.py                  # Pipeline orchestrator CLI
├── config/                  # Configuration files
│   ├── config.yaml          # Pipeline tunables (models, prompts, paths)
│   └── settings.py          # Python config loader
├── src/                     # Pipeline source code
│   ├── input_sampling.py
│   ├── input_normalizing.py
│   ├── run_sob.py
│   ├── parse_sob_results.py
│   ├── address_parsing.py
│   └── do_plots.py
├── utils/                   # Shared helpers
│   ├── api_client.py
│   ├── latex.py
│   ├── parsing.py
│   └── scoring.py
└── data/                    # Datasets and outputs
    ├── downloaded/          # Raw dataset downloads
    ├── sampled/             # Sub-sampled datasets
    ├── demographic_normalizing.csv
    ├── input_biased_text.csv
    └── sob_results.csv
```

---

## Replication

You can easily run the pipeline using the `main.py` orchestrator script. This script automatically runs the pipeline stages in the correct sequence.

To see all available options:
```bash
python main.py --help
```

To run a specific step (e.g., just the model inference):
```bash
python main.py --step 3
```

To run the entire pipeline from start to finish:
```bash
python main.py --all
```

The pipeline executes these 6 sequential steps under the hood:

1. **`src.input_sampling`**: Samples ~500 hateful texts from source datasets.
2. **`src.input_normalizing`**: Normalizes demographic labels into standardized categories.
3. **`src.run_sob`**: Runs the inference across models via OpenRouter.
4. **`src.parse_sob_results`**: Parses raw LLM JSON outputs into structured dictionaries using a parser model.
5. **`src.address_parsing`**: Cleans errors and missing values.
6. **`src.do_plots`**: Generates LaTeX heatmap figures from the parsed data.

### Alternative: Manual Execution

If you prefer not to use `main.py`, you can manually run each step in order using python's module execution:

```bash
python -m src.input_sampling
python -m src.input_normalizing
python -m src.run_sob
python -m src.parse_sob_results
python -m src.address_parsing
python -m src.do_plots
```

### Requirements

- `OPENROUTER_API_KEY` must be set in your `.env` file (see `.env.example`).
- Install dependencies via `pip install -r requirements.txt`.

### Base Models Evaluated

| # | Base model | Instruct variant | Think variant |
|---|------------|------------------|---------------|
| 1 | OLMo 3.1 32B | `olmo_instruct` | `olmo_think` (removed from OpenRouter March 2026) |
| 2 | GPT-5.1 | `gpt51_instruct` | `gpt51_think` |
| 3 | Qwen 3.5 35B-A3B | `qwen_instruct` | `qwen_think` |
| 4 | Claude Sonnet 4.6 | `sonnet46_instruct` | `sonnet46_think` |

### Manual Interventions

There are two steps in the pipeline that rely on manual human intervention in the original study. These are guarded in the code to skip gracefully if the manual artifacts are missing:

1. **Author Annotation**: `input_normalizing.py` requires an author-annotated `input_biased_text.csv` file for its final step.
2. **Manual Parsing Fixes**: `address_parsing.py` applies manual corrections stored in `data/parsing_formatting_errors_new_models_fixed.json`.

---

## Authors

**Ramaravind Kommiya Mothilal¹, Terry Jingchen Zhang¹²³, Raiyan Ahmed¹, Zhijing Jin¹²³⁴, Shion Guha¹, Syed Ishtiaque Ahmed¹**

¹ University of Toronto &nbsp; ² Vector Institute &nbsp; ³ EuroSafeAI &nbsp; ⁴ Max Planck Institute for Intelligent Systems, Tübingen, Germany

Correspondence: [ram.mothilal@mail.utoronto.ca](mailto:ram.mothilal@mail.utoronto.ca)
