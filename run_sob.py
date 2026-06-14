import asyncio
import json
import os
import time
from datetime import datetime

import pandas as pd
from settings import settings
from openai import AsyncOpenAI, RateLimitError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from tqdm.asyncio import tqdm

client = AsyncOpenAI(
    base_url=settings.base_url,
    api_key=settings.api_key,
)

CONCURRENCY  = settings.run_sob["concurrency"]
TEXT_COLUMN  = settings.run_sob["text_column"]
# How many rows to sample for a smoke test. None = full run.
SAMPLE_N     = settings.run_sob["sample_n"]
RANDOM_STATE = settings.run_sob["random_state"]

PROMPTS = {
    name: text
    for name, text in settings.prompts.items()
    if name in settings.run_sob["prompt_names"]
}
MODELS  = settings.models

# API call — retries only on 429 (rate limit), not other errors
@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(5),
)
async def _call_api(model: dict, system_prompt: str, text: str):
    kwargs = dict(
        model=model["id"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": text},
        ],
    )
    
    if "extra_body" in model and model["extra_body"]:
        kwargs["extra_body"] = model["extra_body"]


    return await client.chat.completions.create(**kwargs)


async def fetch_inference(
    semaphore: asyncio.Semaphore,
    model: dict,
    prompt_name: str,
    system_prompt: str,
    row: dict
) -> dict:
    async with semaphore:
        t0 = time.monotonic()
        try:
            response   = await _call_api(model, system_prompt, row[TEXT_COLUMN])
            message    = response.choices[0].message
            latency_ms = round((time.monotonic() - t0) * 1000)

            # extract reasoning text if present
            reasoning = None
            if hasattr(message, "model_extra") and message.model_extra:
                reasoning = message.model_extra.get("reasoning")

            return {
                "id":         row["tix"],
                "text":       row[TEXT_COLUMN],
                "data_name":  row["source"],
                "model":      model["label"],
                "prompt":     prompt_name,
                "response":   message.content,
                "reasoning":  reasoning,
                "latency_ms": latency_ms,
                "error":      None,
            }
        except Exception as e:
            return {
                "id":         row["tix"],
                "text":       row[TEXT_COLUMN],
                "data_name":  row["source"],
                "model":      model["label"],
                "prompt":     prompt_name,
                "response":   None,
                "reasoning":  None,
                "latency_ms": round((time.monotonic() - t0) * 1000),
                "error":      str(e),
            }


async def main():
    df = pd.read_csv(settings.paths["target_mapped_csv"])
    if SAMPLE_N is not None:
        df = df.sample(n=SAMPLE_N, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"SMOKE TEST: sampled {SAMPLE_N} rows")
    rows = df.to_dict(orient="records")

    semaphore = asyncio.Semaphore(CONCURRENCY)
    n_rows    = len(rows)

    tasks = [
        fetch_inference(semaphore, model, prompt_name, sys_prompt, row)
        for model                   in MODELS
        for prompt_name, sys_prompt in PROMPTS.items()
        for row                     in rows
    ]

    total = len(tasks)
    print(f"Queue built: {len(MODELS)} models x {len(PROMPTS)} prompts x {n_rows} rows = {total:,} requests\n")

    os.makedirs(settings.paths["results_dir"], exist_ok=True)
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = time.monotonic()

    # buckets: flushed + cleared as soon as each (model, prompt) is complete
    buckets = {(m["label"], p): [] for m in MODELS for p in PROMPTS}

    total_latency = 0
    total_ok      = 0
    total_fail    = 0
    per_model_lat = {m["label"]: [] for m in MODELS}
    per_model_err = {m["label"]: 0  for m in MODELS}

    futures = [asyncio.create_task(t) for t in tasks]

    with tqdm(total=total, desc="Inferencing") as pbar:
        for future in asyncio.as_completed(futures):
            result = await future

            # route into bucket
            key = (result["model"], result["prompt"])
            buckets[key].append(result)

            # flush + free memory as soon as bucket is complete
            if len(buckets[key]) == n_rows:
                model_label, prompt_name = key
                fname = f"{settings.paths['results_dir']}/{model_label}__{prompt_name}_{timestamp}.json"
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

    stats_file = f"{settings.paths['results_dir']}/stats_{timestamp}.json"
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