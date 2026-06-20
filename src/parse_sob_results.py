import asyncio
import json
import os
import time
from datetime import datetime

import pandas as pd
from config.settings import settings
from utils.api_client import get_client
from openai import RateLimitError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from tqdm.asyncio import tqdm

client = get_client()

CONCURRENCY = settings.parse_sob["concurrency"]

# Use the same parser prompt for all
parser_prompts = {
    name: settings.prompts["parser"]
    for name in settings.parse_sob["prompt_types"]
}

# Filter the registered model labels to the subset that was actually run.
# (Honors `run_sob.selected_models`; falls back to all parse_sob models if
# the file list ends up empty for one reason or another.)
_all_parse_models = settings.parse_sob["models"]
_selected = settings.run_sob.get("selected_models", []) or []
models_openrouter = (
    [m for m in _all_parse_models if m in _selected]
    if _selected else _all_parse_models
)
prompt_types      = settings.parse_sob["prompt_types"]

results_path = settings.paths["results_dir"]
if os.path.isdir(results_path):
    json_files = [
        pos_json
        for pos_json in os.listdir(results_path)
        if pos_json.endswith(".json") and not pos_json.startswith("stats")
    ]
else:
    print(f"[warn] {results_path} not found — parse_sob_results.py needs raw model JSONs to run.")
    json_files = []


# API call — retries only on 429 (rate limit), not other errors
@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(5),
)
async def _call_api(system_prompt: str, text: str):
    kwargs = dict(
        model=settings.parse_sob["parser_model"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": text},
        ],
    )

    return await client.chat.completions.create(**kwargs)


# hardcoded "2457 rows" message text is replaced with n_rows
async def fetch_inference(
    semaphore: asyncio.Semaphore,
    system_prompt: str,
    row: dict
) -> dict:
    async with semaphore:
        t0 = time.monotonic()
        try:
            response   = await _call_api(system_prompt, row['response'])
            message    = response.choices[0].message
            latency_ms = round((time.monotonic() - t0) * 1000)

            return {
                "id":         row["id"],
                "data_name":  row["data_name"],
                "model":      row["model_name"],
                "prompt":     row["prompt_name"],
                "response":   row["response"],
                "parsed":   message.content,
                "latency_ms": latency_ms,
                "error":      None,
            }
        except Exception as e:
            return {
                "id":         row["id"],
                "data_name":  row["data_name"],
                "model":      row["model_name"],
                "prompt":     row["prompt_name"],
                "response":   row["response"],
                "parsed":     None,
                "latency_ms": round((time.monotonic() - t0) * 1000),
                "error":      str(e),
            }


async def main():
    semaphore = asyncio.Semaphore(CONCURRENCY)

    tasks = []
    rows_per_prompt: int | None = None

    def _filename_matches(fname: str, model_label: str, prompt_name: str) -> bool:
        """Match `{model}__{prompt}_{timestamp}.json` exactly.

        Plain substring matching is unsafe because e.g. `"acceptable_ours"`
        is a substring of `"non_acceptable_ours"`.
        """
        base = fname[:-5] if fname.endswith(".json") else fname
        parts = base.split("__", 1)
        if len(parts) != 2 or parts[0] != model_label:
            return False
        return parts[1].startswith(prompt_name + "_")

    for model_name in models_openrouter:
        for prompt_name in prompt_types:
            matching = sorted(
                [f for f in json_files if _filename_matches(f, model_name, prompt_name)],
                reverse=True,
            )
            if not matching:
                raise FileNotFoundError(
                    f"No JSON file in {results_path} matches model={model_name!r} prompt={prompt_name!r}. "
                    f"Available: {json_files}"
                )
            model_file = matching[0]
            print(f"  using {model_file} for {model_name} / {prompt_name}")

            with open(os.path.join(results_path, model_file) , 'r', encoding='utf-8') as file:
                data = json.load(file)

            if rows_per_prompt is None:
                rows_per_prompt = len(data)

            for one_instance in data:
                row = {'id': one_instance['id'], 'data_name': one_instance['data_name'], 'model_name': model_name, 'prompt_name': prompt_name, 'response': one_instance['response']}
                tasks.append(fetch_inference(semaphore, parser_prompts[prompt_name], row))

    total = len(tasks)
    rows_per_prompt = rows_per_prompt or 0
    print(f"Queue built: {len(models_openrouter)} models x {len(prompt_types)} prompts x {rows_per_prompt} rows = {total:,} requests\n")

    os.makedirs(settings.paths["parse_results_dir"], exist_ok=True)
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = time.monotonic()

    # buckets: flushed + cleared as soon as each (model) is complete
    buckets = {m: [] for m in models_openrouter}

    total_latency = 0
    total_ok      = 0
    total_fail    = 0
    per_model_lat = {m: [] for m in models_openrouter}
    per_model_err = {m: 0  for m in models_openrouter}

    futures = [asyncio.create_task(t) for t in tasks]
    expected_per_model = total // len(models_openrouter)

    with tqdm(total=total, desc="Inferencing") as pbar:
        for future in asyncio.as_completed(futures):
            result = await future

            # route into bucket
            key = result["model"]
            buckets[key].append(result)

            # flush + free memory as soon as bucket is complete
            if len(buckets[key]) == expected_per_model:
                model_label = key
                fname = f"{settings.paths['parse_results_dir']}/{model_label}_{timestamp}.json"
                with open(fname, "w") as f:
                    json.dump(buckets[key], f, indent=2)
                pbar.write(f"  Saved: {fname}")
                buckets[key].clear()  

            # accumulate
            if result["error"] is None:
                total_ok      += 1
                total_latency += result["latency_ms"]
                per_model_lat[result["model"]].append(result["latency_ms"])
            else:
                total_fail += 1
                per_model_err[result["model"]] += 1

            pbar.update(1)

    elapsed = time.monotonic() - start_time

    # compute stats
    stats = {
        "total_requests": total,
        "error_rate_pct": round(total_fail / total * 100, 2) if total else 0.0,
        "wall_time_min":  round(elapsed / 60, 2),
        "avg_latency_ms": round(total_latency / total_ok, 1) if total_ok else None,
        "per_model": {
            label: {
                "errors":         per_model_err[label],
                "avg_latency_ms": round(sum(lats) / len(lats), 1) if lats else None,
            }
            for label, lats in per_model_lat.items()
        },
    }

    stats_file = f"{settings.paths['parse_results_dir']}/stats_{timestamp}.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    # print summary
    print(f"\n{'='*50}")
    print(f"  INFERENCE RUN COMPLETE")
    print(f"{'='*50}")
    print(f"  Total requests : {stats['total_requests']:,}")
    print(f"  Error rate     : {stats['error_rate_pct']}%")
    print(f"  Wall time      : {stats['wall_time_min']} min")
    print(f"  Avg latency    : {stats['avg_latency_ms']} ms")
    print(f"\n  Per-model avg latency:")
    for label, m in stats["per_model"].items():
        print(f"    {label:20s}  {m['avg_latency_ms']} ms  ({m['errors']} errors)")
    print(f"{'='*50}")
    print(f"  Stats -> {stats_file}")

if __name__ == "__main__":
    asyncio.run(main())