import json
from functools import cache
from pathlib import Path

from deepeval import evaluate
from deepeval.evaluate.configs import CacheConfig, ErrorConfig
from deepeval.models import OpenAIModel

from evals.run_log import RUN_LOG_DIR, log_run
from src.config import setting

JUDGE_MODEL = "gpt-4o-mini"
THRESHOLD = 0.7

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDENS_DIR = PROJECT_ROOT / "goldens"


@cache
def judge_model() -> OpenAIModel:
    return OpenAIModel(
        model=JUDGE_MODEL,
        api_key=setting.OPENAI_API_KEY,
        temperature=0,
        cost_per_input_token=0.00000015,
        cost_per_output_token=0.00000060,
    )


def load_goldens(name: str) -> list[dict]:
    with (GOLDENS_DIR / f"{name}.json").open(encoding="utf-8") as f:
        return json.load(f)


def run_eval(test_cases, metrics, run_log_dir: Path = RUN_LOG_DIR) -> list[dict]:
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        cache_config=CacheConfig(write_cache=False),
        # A single hung or rate-limited judge call would otherwise crash the whole run and lose every score so far.
        error_config=ErrorConfig(ignore_errors=True),
    )

    runs = log_run(result, run_log_dir)
    for run in runs:
        if not run["scored"]:
            print(
                f"{run['metric']}: run not logged, judge unavailable for all {run['total']} queries "
                f"(likely a rate or quota limit): {run['errored_queries'][0]['error']}"
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
