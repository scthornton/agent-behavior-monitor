"""Measure the local /score round trip, including all four detectors."""

import argparse
import json
import random
import time
from pathlib import Path

import httpx
import numpy as np


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--out", type=Path, default=Path("run"))
    args.add_argument("--port", type=int, default=8765)
    opts = args.parse_args()
    source = opts.out / "sessions.jsonl"
    if not source.exists():
        raise SystemExit(
            f"{source} not found. The reference folder ships models only. "
            f"Run `python -m monitor generate --out {opts.out}` first."
        )
    with source.open() as f:
        cases = [s for line in f if (s := json.loads(line))["split"] == "test"]
    decisions = [(s, i) for s in cases for i in range(len(s["labels"]))]
    samples = random.Random(1741).sample(decisions, 260)
    timings = []
    with httpx.Client(
        base_url=f"http://127.0.0.1:{opts.port}", trust_env=False
    ) as client:
        for s, i in samples:
            body = {
                "history": s["events"][1 : 2 * i : 2],
                "proposed": s["events"][2 * i],
            }
            start = time.perf_counter_ns()
            resp = client.post("/score", json=body)
            resp.raise_for_status()
            result = resp.json()
            timings.append((time.perf_counter_ns() - start) / 1e6)
            assert result["score"] is not None and len(result["scores"]) == 4
    p50, p95 = np.percentile(timings[20:], [50, 95])
    report = {
        "scope": "localhost HTTP, four detectors, full history transport, sequential",
        "seed": 1741,
        "warmup": 20,
        "samples": 240,
        "p50_ms": float(p50),
        "p95_ms": float(p95),
    }
    (opts.out / "http-latency.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
