from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from evals.common import THRESHOLD, judge_model, load_goldens, run_eval
from evals.run_log import PIPELINE_RUN_LOG_DIR
from src.generate_response import retrieve_context
from src.llm import generate_response_from_context

# Each GEval metric is given fixed evaluation steps, so it spends 1 request per test case.
REQUESTS_PER_TEST_CASE = 2

EVALUATION_PARAMS = [
    SingleTurnParams.INPUT,
    SingleTurnParams.ACTUAL_OUTPUT,
    SingleTurnParams.EXPECTED_OUTPUT,
]


def main() -> None:
    model = judge_model()
    correctness = GEval(
        name="Correctness",
        evaluation_steps=[
            "Compare the facts, numbers and conditions stated in the actual output against the expected output.",
            "Heavily penalize any statement that contradicts the expected output.",
            "Judge truth only. Do not penalize the actual output for leaving out points from the expected output.",
            "Do not penalize extra information unless it contradicts the expected output.",
        ],
        evaluation_params=EVALUATION_PARAMS,
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
        evaluation_params=EVALUATION_PARAMS,
        model=model,
        threshold=THRESHOLD,
        _include_g_eval_suffix=False,
    )

    test_cases = []
    for golden in load_goldens("retriever_goldens"):
        query = golden["query"]
        test_cases.append(
            LLMTestCase(
                input=query,
                actual_output=generate_response_from_context(
                    query, retrieve_context(query)
                ),
                expected_output=golden["ideal_answer"],
            )
        )

    run_eval(
        test_cases,
        [correctness, completeness],
        REQUESTS_PER_TEST_CASE,
        PIPELINE_RUN_LOG_DIR,
    )


if __name__ == "__main__":
    main()
