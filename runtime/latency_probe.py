"""
Quick latency probe for LiteLLM model routes.

Usage examples:
  python runtime/latency_probe.py
  python runtime/latency_probe.py --tiers 1 2 3 --runs 5 --warmup 1
  python runtime/latency_probe.py --models tier1 tier2 tier3
  python runtime/latency_probe.py --prompt "Classify: remind me at 7pm"

What it measures per model/alias:
  - success/failure counts
  - min / avg / p50 / p95 / max latency (seconds)

This script calls LiteLLM /chat/completions directly, so it isolates model/API
latency from the rest of the Kairos pipeline.
"""

import argparse
import asyncio
import math
import os
import statistics
import time
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv

load_dotenv()


DEFAULT_PROMPT = "Return ONLY this JSON: {\"ok\":true}"


@dataclass
class RunResult:
    model: str
    ok: bool
    latency_s: float
    error: str = ""


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    idx = (len(values) - 1) * pct
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (idx - lo)


async def one_call(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    timeout_s: float,
    prompt: str,
    max_tokens: int,
) -> RunResult:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": 0,
    }

    t0 = time.perf_counter()
    try:
        response = await client.post(url, json=payload, timeout=timeout_s)
        response.raise_for_status()
        _ = response.json()
        return RunResult(model=model, ok=True, latency_s=time.perf_counter() - t0)
    except Exception as e:
        return RunResult(model=model, ok=False, latency_s=time.perf_counter() - t0, error=str(e))


async def probe_model(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    runs: int,
    warmup: int,
    timeout_s: float,
    prompt: str,
    max_tokens: int,
) -> list[RunResult]:
    results: list[RunResult] = []

    for _ in range(max(0, warmup)):
        await one_call(client, base_url, model, timeout_s, prompt, max_tokens)

    for _ in range(runs):
        results.append(
            await one_call(client, base_url, model, timeout_s, prompt, max_tokens)
        )

    return results


def summarize(model: str, results: list[RunResult]) -> str:
    oks = [r.latency_s for r in results if r.ok]
    fails = [r for r in results if not r.ok]

    lines = [f"\n=== {model} ==="]
    lines.append(f"runs={len(results)} success={len(oks)} fail={len(fails)}")

    if oks:
        s = sorted(oks)
        lines.append(
            "latency_s "
            f"min={min(s):.3f} avg={statistics.mean(s):.3f} "
            f"p50={percentile(s, 0.50):.3f} p95={percentile(s, 0.95):.3f} max={max(s):.3f}"
        )

    if fails:
        lines.append("errors:")
        for i, fail in enumerate(fails[:3], start=1):
            lines.append(f"  {i}. {fail.error[:240]}")

    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Probe per-model latency via LiteLLM")
    parser.add_argument("--base-url", default=os.getenv("LITELLM_BASE_URL", "http://localhost:4000"))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=float(os.getenv("LLM_COMPLETE_TIMEOUT", "60")))
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--tiers", nargs="*", type=int, default=[1, 2, 3])
    parser.add_argument("--models", nargs="*", help="Model aliases accepted by LiteLLM, e.g. tier1 tier2 tier3")
    args = parser.parse_args()

    models = args.models if args.models else [f"tier{t}" for t in args.tiers]

    print(f"LiteLLM base URL: {args.base_url}")
    print(f"Models: {', '.join(models)}")
    print(f"Runs/model={args.runs}, warmup={args.warmup}, timeout={args.timeout}s")

    async with httpx.AsyncClient() as client:
        for model in models:
            results = await probe_model(
                client=client,
                base_url=args.base_url,
                model=model,
                runs=args.runs,
                warmup=args.warmup,
                timeout_s=args.timeout,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
            )
            print(summarize(model, results))


if __name__ == "__main__":
    asyncio.run(main())
