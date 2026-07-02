# Evaluating Second-Order Bias of LLMs Through Epistemic Entitlement

**Preprint:** https://arxiv.org/pdf/2606.17506

## Overview

Most LLM bias evaluations focus on whether a model *generates* biased content. This work investigates a subtler problem: bias in how LLMs *judge* biased content. We call this **second-order bias** — social bias that surfaces in a model's evaluation of bias, rather than in its generation.

To measure this, we draw on entitlement epistemology and frame bias as misplaced foundational knowledge. From this, we derive a logical reasoning task where LLMs must judge whether a biased text is acceptable or non-acceptable, and *to whom*. We introduce two metrics to capture how readily models make unsupported demographic inferences, and how these vary across target groups.

Evaluating both open and closed models, we find that this task bypasses safety guardrails, varies systematically across demographic groups, and reveals how models remain sensitive to demographic labels even in judgment settings.

---

## Repository Structure

```
second-order-bias/
├── main.py                 
├── config/                  # Configuration files
│   ├── config.yaml          # Pipeline tunables (models, prompts, paths)
│   └── settings.py          # Python config loader
├── src/                     # Pipeline source code
│   ├── input_sampling.py
│   ├── input_normalizing.py
│   ├── get_processed_data.py   # one-shot: steps 1+2 (sampling + normalizing)
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

`main.py` runs the pipeline. To see all available options:

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

To skip the sampling + normalizing steps (steps 1+2) and start from an existing `data/input_biased_text.csv`:
```bash
python main.py --from-processed
```

If you want to understand how `input_biased_text.csv` was created (and run the sampling + normalizing in one shot), use:
```bash
python -m src.get_processed_data
```
That script's source is the single point of reference for steps 1+2.

The pipeline executes these 6 sequential steps under the hood:

1. **`src.input_sampling`**: Samples ~500 hateful texts from source datasets.
2. **`src.input_normalizing`**: Normalizes and maps demographic labels across datasets.
3. **`src.run_sob`**: Runs the second-order bias task across models via OpenRouter.
4. **`src.parse_sob_results`**: Parses raw LLM JSON outputs into structured dictionaries using a parser model.
5. **`src.address_parsing`**: Cleans errors and missing values.
6. **`src.do_plots`**: Generates LaTeX heatmap figures from the parsed data for the paper.

**The SOB results will be saved to `data/sob_results.csv`**

## Requirements

- `OPENROUTER_API_KEY` must be set in your `.env` file (see `.env.example`).
- Install dependencies via `pip install -r requirements.txt`.

---

## Citing

**BibTeX:**

```bibtex
@article{kommiya2026sob,
  title={Evaluating Second-Order Bias of LLMs Through Epistemic Entitlement},
  author={Kommiya Mothilal, Ramaravind and Zhang, Terry Jingchen and Ahmed, Raiyan and Jin, Zhijing and Guha, Shion and Ahmed, Syed Ishtiaque},
  journal={arXiv e-prints},
  eprint={2606.17506},
  year={2025}
}
```

**Text:**
Mothilal, R. K., Zhang, T. J., Ahmed, R., Jin, Z., Guha, S., & Ahmed, S. I. (2026). Evaluating Second-Order Bias of LLMs Through Epistemic Entitlement. arXiv preprint:2606.17506.
