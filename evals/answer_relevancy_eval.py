import json

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, CacheConfig
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase

from evals.run_log import RUN_LOG_DIR, log_run
from src.config import setting
from src.llm import generate_response_from_context

JUDGE_MODEL = "gemini-3.1-flash-lite"
THRESHOLD = 0.7

# Gemini free tier allows 15 requests/min, and answer relevancy spends 3 requests per test case (statements, verdicts, reason).
# Running one case at a time and sleeping between them keeps us at 12 requests/min.
REQUESTS_PER_MINUTE = 12
REQUESTS_PER_TEST_CASE = 3
SECONDS_BETWEEN_TEST_CASES = 60 * REQUESTS_PER_TEST_CASE // REQUESTS_PER_MINUTE

model = GeminiModel(
    model=JUDGE_MODEL,
    api_key=setting.GEMINI_API_KEY,
    temperature=0,
    cost_per_input_token=0.00000125,
    cost_per_output_token=0.00000500,
)

metric = AnswerRelevancyMetric(threshold=THRESHOLD, model=model, include_reason=True)

# The generator gets the golden context rather than the retriever's, so a low score is the generator's fault alone.
with open("goldens/faithfulness_goldens.json", "r") as f:
    answer_relevancy_goldens = json.load(f)

test_cases = []

for golden in answer_relevancy_goldens:
    query = golden["query"]
    test_case = LLMTestCase(
        input=query,
        actual_output=generate_response_from_context(query, golden["ideal_context"]),
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
