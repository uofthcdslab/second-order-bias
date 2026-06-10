# Evaluating Second-Order Bias of LLMs Through Epistemic Entitlement

> 🚧 This repository is under active development. Code is being cleaned and reorganized.
> **Preprint coming soon.**

---

## Overview

Most LLM bias evaluations focus on whether a model *generates* biased content. This work investigates a subtler problem: bias in how LLMs *judge* biased content. We call this **second-order bias** — social bias that surfaces in a model's evaluation of bias, rather than in its generation.

To measure this, we draw on entitlement epistemology and frame bias as misplaced foundational knowledge. From this, we derive a logical reasoning task where LLMs must judge whether a biased text is acceptable or non-acceptable, and *to whom*. We introduce two metrics to capture how readily models make unsupported demographic inferences, and how these vary across target groups.

Evaluating both open and closed models, we find that this task bypasses safety guardrails, varies systematically across demographic groups, and reveals how models remain sensitive to demographic labels even in judgment settings.

---

## Repository Structure

| File | Description |
|------|-------------|
| `input_sampling.py` | Samples hate speech inputs from source datasets |
| `input_normalizing.py` | Normalizes and maps demographic labels across datasets |
| `run_sob.py` | Runs the second-order bias task across models |
| `parse_sob_results.py` | Parses and structures raw model outputs |
| `address_parsing.py` | Final parsing corrections and formatting |
| `sob_results.csv` | Full results file used in the analysis |
| `target_mapped.csv` | Normalized input dataset with demographic mappings |
| `demographic_normalizing.csv` | Demographic label normalization reference |

The final model responses used in the analysis in the paper are at [`sob_results.csv`](./sob_results.csv).

---

## Authors

**Ramaravind Kommiya Mothilal¹, Terry Jingchen Zhang¹²³, Raiyan Ahmed¹, Zhijing Jin¹²³⁴, Shion Guha¹, Syed Ishtiaque Ahmed¹**

¹ University of Toronto &nbsp; ² Vector Institute &nbsp; ³ EuroSafeAI &nbsp; ⁴ Max Planck Institute for Intelligent Systems, Tübingen, Germany

Correspondence: [ram.mothilal@mail.utoronto.ca](mailto:ram.mothilal@mail.utoronto.ca)
