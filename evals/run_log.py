import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

NEPAL_TIME = timezone(timedelta(hours=5, minutes=45), "NPT")

RUN_LOG_DIR = Path("evals/results")


def log_run(result, run_log_dir: Path = RUN_LOG_DIR) -> list[dict]:
    """Append one line per metric per run so scores can be compared over time.

    Each metric gets its own file, so an eval that measures several metrics
    writes a line to each: Contextual Precision -> contextual_precision.jsonl.
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
                    "failed_queries": [],
                },
            )
            summary["scores"].append(metric_data.score)
            if not metric_data.success:
                summary["failed_queries"].append(test_result.input)

    run_log_dir.mkdir(parents=True, exist_ok=True)
    total = len(result.test_results)
    runs = []

    for name, summary in collected.items():
        scores = [score for score in summary["scores"] if score is not None]
        passed = total - len(summary["failed_queries"])
        run = {
            "timestamp": datetime.now(NEPAL_TIME).isoformat(timespec="seconds"),
            "metric": name,
            "judge_model": summary["judge_model"],
            "threshold": summary["threshold"],
            "total": total,
            "passed": passed,
            "pass_rate": round(passed / total, 3),
            "average_score": round(sum(scores) / len(scores), 3) if scores else None,
            "failed_queries": summary["failed_queries"],
        }
        run_log = run_log_dir / f"{name.lower().replace(' ', '_')}.jsonl"
        with run_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(run) + "\n")
        runs.append(run)

    return runs
