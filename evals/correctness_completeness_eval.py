import json

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, ErrorConfig
from deepeval.metrics import GEval
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase, SingleTurnParams

from evals.run_log import RUN_LOG_DIR, log_run
from src.config import setting
from src.generate_response import retrieve_context
from src.llm import generate_response_from_context

JUDGE_MODEL = "gemini-3.1-flash-lite"
THRESHOLD = 0.7
PIPELINE_RUN_LOG_DIR = RUN_LOG_DIR / "pipeline"

REQUESTS_PER_MINUTE = 12
REQUESTS_PER_TEST_CASE = 2
SECONDS_BETWEEN_TEST_CASES = 60 * REQUESTS_PER_TEST_CASE // REQUESTS_PER_MINUTE

model = GeminiModel(
    model=JUDGE_MODEL,
    api_key=setting.GEMINI_API_KEY,
    temperature=0,
    cost_per_input_token=0.00000125,
    cost_per_output_token=0.00000500,
)

evaluation_params = [
    SingleTurnParams.INPUT,
    SingleTurnParams.ACTUAL_OUTPUT,
    SingleTurnParams.EXPECTED_OUTPUT,
]

correctness = GEval(
    name="Correctness",
    evaluation_steps=[
        "Compare the facts, numbers and conditions stated in the actual output against the expected output.",
        "Heavily penalize any statement that contradicts the expected output.",
        "Judge truth only. Do not penalize the actual output for leaving out points from the expected output.",
        "Do not penalize extra information unless it contradicts the expected output.",
    ],
    evaluation_params=evaluation_params,
    model=model,
    threshold=THRESHOLD,
    _include_g_eval_suffix=False,
)

completeness = GEval(
    name="Completeness",
    evaluation_steps=[
        "List the key facts, numbers, conditions and exceptions in the expected output that answer the input.",
        "Check which of them the actual output covers, allowing different wording.",
        "Penalize each key point the actual output omits or only covers vaguely.",
        "Judge coverage only. Do not penalize a covered point for being stated incorrectly, or extra information.",
    ],
    evaluation_params=evaluation_params,
    model=model,
    threshold=THRESHOLD,
    _include_g_eval_suffix=False,
)

with open("goldens/retriever_goldens.json", "r") as f:
    goldens = json.load(f)

test_cases = []

for golden in goldens:
    query = golden["query"]
    test_case = LLMTestCase(
        input=query,
        actual_output=generate_response_from_context(query, retrieve_context(query)),
        expected_output=golden["ideal_answer"],
    )
    test_cases.append(test_case)

result = evaluate(
    test_cases=test_cases,
    metrics=[correctness, completeness],
    async_config=AsyncConfig(
        max_concurrent=1, throttle_value=SECONDS_BETWEEN_TEST_CASES
    ),
    cache_config=CacheConfig(write_cache=False),
    error_config=ErrorConfig(ignore_errors=True),
)

for run in log_run(result, PIPELINE_RUN_LOG_DIR):
    print(
        f"{run['metric']}: {run['passed']}/{run['total']} passed -> {PIPELINE_RUN_LOG_DIR}"
    )
