from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase

from evals.common import THRESHOLD, judge_model, load_goldens, run_eval
from src.generate_response import retrieve_context


def main() -> None:
    metric = ContextualPrecisionMetric(
        threshold=THRESHOLD, model=judge_model(), include_reason=True
    )

    test_cases = []
    for golden in load_goldens("retriever_goldens"):
        query = golden["query"]
        test_cases.append(
            LLMTestCase(
                input=query,
                retrieval_context=retrieve_context(query),
                actual_output="(generator not evaluated in this run)",
                expected_output=golden["ideal_answer"],
            )
        )

    run_eval(test_cases, [metric])


if __name__ == "__main__":
    main()
