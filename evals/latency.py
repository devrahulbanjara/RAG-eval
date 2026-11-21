"""Measure the latency of the retriever, the generator and the whole pipeline.

Runs every golden query through the real pipeline and reports min, p50, p90,
p99, max and mean latency per stage, then appends the summary to
``evals/results/latency.jsonl`` so runs can be compared over time.

Usage:
    uv run python -m evals.latency               # every golden query once
    uv run python -m evals.latency --runs 3      # every golden query 3 times
    uv run python -m evals.latency --trace       # keep LangSmith tracing on
"""

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np

STAGES = ["retriever", "generator", "end_to_end"]
PERCENTILES = [50, 90, 99]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="How many times to run every golden query (default: 1).",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Keep LangSmith tracing on. Off by default so it does not add to the timings.",
    )
    return parser.parse_args()


def summarize(latencies: list[float]) -> dict:
    values = np.array(latencies)
    summary = {"min": values.min(), "max": values.max(), "mean": values.mean()}
    for p in PERCENTILES:
        summary[f"p{p}"] = np.percentile(values, p)
    return {key: round(float(value), 3) for key, value in summary.items()}


def print_table(summaries: dict, samples: int) -> None:
    columns = ["min", *(f"p{p}" for p in PERCENTILES), "max", "mean"]
    print(f"\nLatency in seconds over {samples} queries\n")
    print(f"{'stage':<12}" + "".join(f"{c:>9}" for c in columns))
    print("-" * (12 + 9 * len(columns)))
    for stage in STAGES:
        row = summaries[stage]
        print(f"{stage:<12}" + "".join(f"{row[c]:>9.3f}" for c in columns))


def main() -> None:
    args = parse_args()
    if not args.trace:
        # Must be set before anything imports langsmith; load_dotenv does not override it.
        os.environ["LANGSMITH_TRACING"] = "false"

    from evals.common import load_goldens
    from evals.run_log import NEPAL_TIME, RUN_LOG_DIR, current_commit
    from src.generate_response import format_as_context, retrieve
    from src.llm import LLM_MODEL, generate_response_from_context

    queries = [golden["query"] for golden in load_goldens("retriever_goldens")]

    # Load the embedding model and open the Pinecone and OpenAI connections
    # once up front, so the first query is not charged for the cold start.
    print("Warming up...")
    generate_response_from_context(queries[0], format_as_context(retrieve(queries[0])))

    latencies = {stage: [] for stage in STAGES}
    total = len(queries) * args.runs
    for run in range(args.runs):
        for position, query in enumerate(queries, start=1):
            start = time.perf_counter()
            context = format_as_context(retrieve(query))
            retrieved = time.perf_counter()
            generate_response_from_context(query, context)
            generated = time.perf_counter()

            latencies["retriever"].append(retrieved - start)
            latencies["generator"].append(generated - retrieved)
            latencies["end_to_end"].append(generated - start)

            done = run * len(queries) + position
            print(
                f"[{done}/{total}] retriever {retrieved - start:.3f}s  "
                f"generator {generated - retrieved:.3f}s  {query[:60]}"
            )

    summaries = {stage: summarize(latencies[stage]) for stage in STAGES}
    print_table(summaries, total)

    record = {
        "timestamp": datetime.now(NEPAL_TIME).isoformat(timespec="seconds"),
        "commit": current_commit(),
        "llm_model": LLM_MODEL,
        "langsmith_tracing": args.trace,
        "samples": total,
        **summaries,
    }
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUN_LOG_DIR / "latency.jsonl"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"\nLogged to {log_path}")


if __name__ == "__main__":
    main()
