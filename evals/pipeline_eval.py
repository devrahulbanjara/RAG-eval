from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from evals.common import THRESHOLD, judge_model, load_goldens, run_eval
from evals.run_log import PIPELINE_RUN_LOG_DIR
from src.generate_response import TOP_K, retrieve_context
from src.llm import generate_response_from_context

# deepeval runs all three metrics of a test case at once: contextual relevancy spends
# one request per retrieved chunk plus a reason, faithfulness 4, answer relevancy 3.
REQUESTS_PER_TEST_CASE = (TOP_K + 1) + 4 + 3


def main() -> None:
    model = judge_model()
    metrics = [
        ContextualRelevancyMetric(
            threshold=THRESHOLD, model=model, include_reason=True
        ),
        FaithfulnessMetric(threshold=THRESHOLD, model=model, include_reason=True),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=model, include_reason=True),
    ]

    # The generator gets what the real retriever pulled, not the golden context.
    test_cases = []
    for golden in load_goldens("retriever_goldens"):
        query = golden["query"]
        retrieval_context = retrieve_context(query)
        test_cases.append(
            LLMTestCase(
                input=query,
                actual_output=generate_response_from_context(query, retrieval_context),
                retrieval_context=retrieval_context,
            )
        )

    run_eval(test_cases, metrics, REQUESTS_PER_TEST_CASE, PIPELINE_RUN_LOG_DIR)


if __name__ == "__main__":
    main()
