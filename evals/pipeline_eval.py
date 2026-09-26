import json

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, ErrorConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase

from evals.run_log import RUN_LOG_DIR, log_run
from src.config import setting
from src.generate_response import TOP_K, retrieve_context
from src.llm import generate_response_from_context

JUDGE_MODEL = "gemini-3.1-flash-lite"
THRESHOLD = 0.7
PIPELINE_RUN_LOG_DIR = RUN_LOG_DIR / "pipeline"

# Gemini free tier allows 15 requests/min, and deepeval runs all three metrics of a test case at once:
# contextual relevancy spends one request per retrieved chunk plus a reason, faithfulness 4, answer relevancy 3.
# Running one case at a time and sleeping between them keeps us at 12 requests/min.
REQUESTS_PER_MINUTE = 12
REQUESTS_PER_TEST_CASE = (TOP_K + 1) + 4 + 3
SECONDS_BETWEEN_TEST_CASES = 60 * REQUESTS_PER_TEST_CASE // REQUESTS_PER_MINUTE

model = GeminiModel(
    model=JUDGE_MODEL,
    api_key=setting.GEMINI_API_KEY,
    temperature=0,
    cost_per_input_token=0.00000125,
    cost_per_output_token=0.00000500,
)

metrics = [
    ContextualRelevancyMetric(threshold=THRESHOLD, model=model, include_reason=True),
    FaithfulnessMetric(threshold=THRESHOLD, model=model, include_reason=True),
    AnswerRelevancyMetric(threshold=THRESHOLD, model=model, include_reason=True),
]

with open("goldens/retriever_goldens.json", "r") as f:
    pipeline_goldens = json.load(f)

test_cases = []

# The generator gets what the real retriever pulled, not the golden context.
for golden in pipeline_goldens:
    query = golden["query"]
    retrieval_context = retrieve_context(query)
    test_case = LLMTestCase(
        input=query,
        actual_output=generate_response_from_context(query, retrieval_context),
        retrieval_context=retrieval_context,
    )
    test_cases.append(test_case)

result = evaluate(
    test_cases=test_cases,
    metrics=metrics,
    async_config=AsyncConfig(
        max_concurrent=1, throttle_value=SECONDS_BETWEEN_TEST_CASES
    ),
    cache_config=CacheConfig(write_cache=False),
    # A single hung Gemini call would otherwise crash the whole run and lose every score so far.
    error_config=ErrorConfig(ignore_errors=True),
)

for run in log_run(result, PIPELINE_RUN_LOG_DIR):
    print(
        f"{run['metric']}: {run['passed']}/{run['total']} passed -> {PIPELINE_RUN_LOG_DIR}"
    )
