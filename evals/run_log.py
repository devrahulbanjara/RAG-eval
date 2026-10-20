import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

NEPAL_TIME = timezone(timedelta(hours=5, minutes=45), "NPT")

RUN_LOG_DIR = Path(__file__).resolve().parent / "results"
PIPELINE_RUN_LOG_DIR = RUN_LOG_DIR / "pipeline"

MAX_ERROR_LENGTH = 200


def current_commit() -> str | None:
    """Return the short commit hash of HEAD.

    Returns:
        The short hash, suffixed with ``-dirty`` if the working tree has
        uncommitted changes, or ``None`` if git is unavailable.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except OSError, subprocess.CalledProcessError:
        return None
    return f"{commit}-dirty" if dirty else commit


def log_run(result, run_log_dir: Path = RUN_LOG_DIR) -> list[dict]:
    """Summarize an evaluation result and append it to the run history.

    Writes one JSON line per metric to ``<run_log_dir>/<metric_name>.jsonl``
    (e.g. Contextual Precision -> ``contextual_precision.jsonl``).

    Each test case is classified per metric as exactly one of:

    - **passed**: scored at or above the threshold.
    - **failed**: scored below the threshold.
    - **errored**: the judge call raised an error and produced no score.

    Errored cases are excluded from ``pass_rate`` and ``average_score``.
    Metrics with no scored cases are returned but not written.

    Args:
        result: The ``EvaluationResult`` returned by ``deepeval.evaluate``.
        run_log_dir: Directory containing the per-metric JSONL files.

    Returns:
        One summary dict per metric, with the fields written to the log.
    """
    collected = {}
    for test_result in result.test_results:
        for metric_data in test_result.metrics_data:
            summary = collected.setdefault(
                metric_data.name,
                {
                    "judge_model": metric_data.evaluation_model,
                    "threshold": metric_data.threshold,
                    "scores": [],
                    "passed": 0,
                    "failed_queries": [],
                    "errored_queries": [],
                },
            )
            if metric_data.error:
                summary["errored_queries"].append(
                    {
                        "query": test_result.input,
                        "error": metric_data.error[:MAX_ERROR_LENGTH],
                    }
                )
                continue
            summary["scores"].append(metric_data.score)
            if metric_data.success:
                summary["passed"] += 1
            else:
                summary["failed_queries"].append(test_result.input)

    run_log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(NEPAL_TIME).isoformat(timespec="seconds")
    commit = current_commit()
    total = len(result.test_results)
    runs = []

    for name, summary in collected.items():
        scores = [score for score in summary["scores"] if score is not None]
        passed = summary["passed"]
        scored = passed + len(summary["failed_queries"])
        run = {
            "timestamp": timestamp,
            "commit": commit,
            "metric": name,
            "judge_model": summary["judge_model"],
            "threshold": summary["threshold"],
            "total": total,
            "scored": scored,
            "errored": len(summary["errored_queries"]),
            "passed": passed,
            "pass_rate": round(passed / scored, 3) if scored else None,
            "average_score": round(sum(scores) / len(scores), 3) if scores else None,
            "failed_queries": summary["failed_queries"],
            "errored_queries": summary["errored_queries"],
        }
        if scored:
            run_log = run_log_dir / f"{name.lower().replace(' ', '_')}.jsonl"
            with run_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(run) + "\n")
        runs.append(run)

    return runs
