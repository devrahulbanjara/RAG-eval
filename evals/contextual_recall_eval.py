import json

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, CacheConfig
from deepeval.metrics import ContextualRecallMetric
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase

from evals.run_log import RUN_LOG_DIR, log_run
from src.config import setting
from src.generate_response import retrieve_context

JUDGE_MODEL = "gemini-3.1-flash-lite"
THRESHOLD = 0.7

REQUESTS_PER_MINUTE = 12
REQUESTS_PER_TEST_CASE = 4
SECONDS_BETWEEN_TEST_CASES = 60 * REQUESTS_PER_TEST_CASE // REQUESTS_PER_MINUTE

model = GeminiModel(
    model=JUDGE_MODEL,
    api_key=setting.GEMINI_API_KEY,
    temperature=0,
    cost_per_input_token=0.00000125,
    cost_per_output_token=0.00000500,
)

metric = ContextualRecallMetric(threshold=THRESHOLD, model=model, include_reason=True)

with open("goldens/retriever_goldens.json", "r") as f:
    contextual_recall_goldens = json.load(f)

test_cases = []

for golden in contextual_recall_goldens:
    query = golden["query"]
    retrieval_context = retrieve_context(query)
    expected_output = golden["ideal_answer"]
    test_case = LLMTestCase(
        input=query,
        retrieval_context=retrieval_context,
        actual_output="(generator not evaluated in this run)",
        expected_output=expected_output,
    )
    test_cases.append(test_case)

result = evaluate(
    test_cases=test_cases,
    metrics=[metric],
    async_config=AsyncConfig(
        max_concurrent=1, throttle_value=SECONDS_BETWEEN_TEST_CASES
    ),
    cache_config=CacheConfig(write_cache=False),
)

for run in log_run(result):
    print(f"{run['metric']}: {run['passed']}/{run['total']} passed -> {RUN_LOG_DIR}")
