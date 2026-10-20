import json
from functools import cache
from pathlib import Path

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, ErrorConfig
from deepeval.models import GeminiModel

from evals.run_log import RUN_LOG_DIR, log_run
from src.config import setting

JUDGE_MODEL = "gemini-3.1-flash-lite"
THRESHOLD = 0.7

# Gemini free tier allows 15 requests/min. Each eval runs one test case at a time
# and sleeps between them for its metrics' requests per case, keeping us at 12 requests/min.
REQUESTS_PER_MINUTE = 12

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDENS_DIR = PROJECT_ROOT / "goldens"


@cache
def judge_model() -> GeminiModel:
    return GeminiModel(
        model=JUDGE_MODEL,
        api_key=setting.GEMINI_API_KEY,
        temperature=0,
        cost_per_input_token=0.00000125,
        cost_per_output_token=0.00000500,
    )


def load_goldens(name: str) -> list[dict]:
    with (GOLDENS_DIR / f"{name}.json").open(encoding="utf-8") as f:
        return json.load(f)


def run_eval(
    test_cases, metrics, requests_per_test_case: int, run_log_dir: Path = RUN_LOG_DIR
) -> list[dict]:
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(
            max_concurrent=1,
            throttle_value=60 * requests_per_test_case // REQUESTS_PER_MINUTE,
        ),
        cache_config=CacheConfig(write_cache=False),
        # A single hung or rate-limited Gemini call would otherwise crash the whole run and lose every score so far.
        error_config=ErrorConfig(ignore_errors=True),
    )

    runs = log_run(result, run_log_dir)
    for run in runs:
        if not run["scored"]:
            print(
                f"{run['metric']}: run not logged, judge unavailable for all {run['total']} queries "
                f"(likely a free-tier rate or quota limit): {run['errored_queries'][0]['error']}"
            )
            continue
        print(
            f"{run['metric']}: {run['passed']}/{run['scored']} passed, "
            f"{run['errored']} errored -> {run_log_dir}"
        )
        if run["errored"]:
            print(
                f"  WARNING: {run['errored']} of {run['total']} queries were not scored, so this run is partial."
            )
    return runs
