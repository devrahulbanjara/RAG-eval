from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from evals.common import THRESHOLD, judge_model, load_goldens, run_eval
from src.llm import generate_response_from_context


def main() -> None:
    metric = FaithfulnessMetric(
        threshold=THRESHOLD, model=judge_model(), include_reason=True
    )

    test_cases = []
    for golden in load_goldens("faithfulness_goldens"):
        query = golden["query"]
        retrieval_context = golden["ideal_context"]
        test_cases.append(
            LLMTestCase(
                input=query,
                actual_output=generate_response_from_context(query, retrieval_context),
                retrieval_context=retrieval_context,
            )
        )

    run_eval(test_cases, [metric])


if __name__ == "__main__":
    main()
