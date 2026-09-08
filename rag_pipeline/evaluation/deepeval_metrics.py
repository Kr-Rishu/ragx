from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics.g_eval import Rubric

from ..config import JUDGE_MODEL_NAME

def _cost_of(metric):
    return getattr(metric, 'evaluation_cost', None) or 0.0

def score_correctness(question, expected_answer, generated_answer, threshold=0.5):
    metric = GEval(
        name='Correctness',
        criteria="""
        Judge whether the actual output correctly answers the question
        according to the expected output.

        Extra information MUST NOT reduce the score. A longer answer,
        additional details, repetition, alternative valid answers, or
        different wording are acceptable.

        Reduce the score only if an important expected point is missing.
        """,
        evaluation_steps=[
            "Identify the essential answer and facts in the expected output.",
            "Check whether those essential facts are correctly present in the actual output.",
            "Reduce the score only for missing information. Ignore all extra information.",
            "A factually accurate answer must score at least 9 even if it is shorter or covers fewer points than the expected output.",
            "Additional information must NEVER lower the score."
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="Addresses essentially all key points in the expected output."),
            Rubric(score_range=(5, 8), expected_outcome="Covers the main key points but missing minor details."),
            Rubric(score_range=(0, 4), expected_outcome="Misses several key points; only partially covers the expected output."),
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=threshold,
        model=JUDGE_MODEL_NAME
    )
    test_case = LLMTestCase(input=question, actual_output=generated_answer, expected_output=expected_answer)
    try:
        metric.measure(test_case)
        return metric.score, metric.reason, _cost_of(metric)
    except Exception as e:
        return None, f"correctness scoring failed: {e}", 0.0

def score_faithfulness(question, generated_answer, retrieval_context, threshold = 0.5):
    metric = FaithfulnessMetric(threshold=threshold, model=JUDGE_MODEL_NAME, include_reason=True)
    test_case = LLMTestCase(input=question, actual_output=generated_answer, retrieval_context=retrieval_context or ['no context retrieved'])
    try:
        metric.measure(test_case)
        return metric.score, metric.reason, _cost_of(metric)
    except Exception as e:
        return None, f'faithfulness scoring failed: {e}', 0.0

def score_answer_relevancy(question, generated_answer, threshold = 0.5):
    metric = AnswerRelevancyMetric(threshold=threshold, model=JUDGE_MODEL_NAME, include_reason=True)
    test_case = LLMTestCase(input=question, actual_output=generated_answer)
    try:
        metric.measure(test_case)
        return metric.score, metric.reason, _cost_of(metric)
    except Exception as e:
        return None, f'answer relevancy scoring failed: {e}', 0.0
