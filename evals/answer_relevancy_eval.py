from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from evals.common import THRESHOLD, judge_model, load_goldens, run_eval
from src.llm import generate_response_from_context


def main() -> None:
    metric = AnswerRelevancyMetric(
        threshold=THRESHOLD, model=judge_model(), include_reason=True
    )

    # The generator gets the golden context rather than the retriever's, so a low score is the generator's fault alone.
    test_cases = []
    for golden in load_goldens("faithfulness_goldens"):
        query = golden["query"]
        test_cases.append(
            LLMTestCase(
                input=query,
                actual_output=generate_response_from_context(
                    query, golden["ideal_context"]
                ),
            )
        )

    run_eval(test_cases, [metric])


if __name__ == "__main__":
    main()
