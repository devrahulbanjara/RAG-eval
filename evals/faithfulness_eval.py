import json

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, CacheConfig
from deepeval.metrics import FaithfulnessMetric
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase

from src.config import setting
from src.llm import generate_response_from_context

model = GeminiModel(
    model="gemini-3.1-flash-lite",
    api_key=setting.GEMINI_API_KEY,
    temperature=0,
    cost_per_input_token=0.00000125,
    cost_per_output_token=0.00000500,
)

metric = FaithfulnessMetric(threshold=0.7, model=model, include_reason=True)

with open("goldens/faithfulness_goldens.json", "r") as f:
    faithfulness_goldens = json.load(f)

test_cases = []

for golden in faithfulness_goldens:
    query = golden["query"]
    retrieval_context = golden["ideal_context"]
    test_case = LLMTestCase(
        input=query,
        actual_output=generate_response_from_context(query, retrieval_context),
        retrieval_context=retrieval_context,
    )
    test_cases.append(test_case)

# Gemini free tier allows 15 requests/min, and faithfulness spends 4 requests per
# test case (truths, claims, verdicts, reason). Running one case at a time and
# sleeping between them keeps us at 12 requests/min.
REQUESTS_PER_MINUTE = 12
REQUESTS_PER_TEST_CASE = 4
SECONDS_BETWEEN_TEST_CASES = 60 * REQUESTS_PER_TEST_CASE // REQUESTS_PER_MINUTE

evaluate(
    test_cases=test_cases,
    metrics=[metric],
    async_config=AsyncConfig(
        max_concurrent=1, throttle_value=SECONDS_BETWEEN_TEST_CASES
    ),
    cache_config=CacheConfig(write_cache=False),
)
