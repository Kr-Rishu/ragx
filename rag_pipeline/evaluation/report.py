from typing import Optional
import numpy as np
from ..models import EvalRowResult, EvalReport
from .operational import usd_to_inr
import pandas as pd
from io import BytesIO

COLD_START_EXCLUDE_ROWS = 2

def _percentiles(values):
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.array(values)
    return float(np.percentile(arr, 50)), float(np.percentile(arr, 95)), float(np.percentile(arr, 99))

def build_passed_qa_excel(rows):
    passed_rows = [r for r in rows if r.passed]

    data = []

    for r in passed_rows:
        data.append({
            'Row ID': r.row_id,
            'Question': r.question,
            'Expected Answer': r.expected_answer,
            'Generated Answer': r.generated_answer,
            'Correctness Score': r.correctness_score,
            'Faithfulness Score': r.faithfulness_score,
            'Answer Relevancy Score': r.answer_relevancy_score,
            'Passed': r.passed
        })

    df = pd.DataFrame(data)

    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(
            writer,
            index=False,
            sheet_name="Passed Q&A"
        )

    output.seek(0)

    return output.getvalue()

def build_report(mode, rows):
    n = len(rows)
    failed = [r for r in rows if not r.passed]

    correctness_scores = [r.correctness_score for r in rows if r.correctness_score is not None]
    faith_scores = [r.faithfulness_score for r in rows if r.faithfulness_score is not None]
    relevancy_scores = [r.answer_relevancy_score for r in rows if r.answer_relevancy_score is not None]

    warm_rows = rows[COLD_START_EXCLUDE_ROWS:] if n > COLD_START_EXCLUDE_ROWS else rows
    retrieval_latencies = [r.retrieval_latency_sec for r in warm_rows]
    generation_latencies = [r.generation_latency_sec for r in warm_rows]
    total_latencies = [r.retrieval_latency_sec + r.generation_latency_sec for r in warm_rows]
    n_warm = len(warm_rows) or 1

    r_p50, r_p95, r_p99 = _percentiles(retrieval_latencies)
    g_p50, g_p95, g_p99 = _percentiles(generation_latencies)
    t_p50, t_p95, t_p99 = _percentiles(total_latencies)

    passed_qa_excel = build_passed_qa_excel(rows)

    total_generation_cost = sum(r.generation_cost_usd for r in rows)
    total_eval_cost = sum(r.eval_cost_usd for r in rows)
    total_cost = total_generation_cost + total_eval_cost

    return EvalReport(
        mode=mode,
        n_evaluated=n,
        n_failed=len(failed),
        avg_correctness=(sum(correctness_scores) / len(correctness_scores)) if correctness_scores else None,
        avg_faithfulness=(sum(faith_scores) / len(faith_scores)) if faith_scores else None,
        avg_answer_relevancy=(sum(relevancy_scores) / len(relevancy_scores)) if relevancy_scores else None,
        avg_retrieval_latency_sec=(sum(retrieval_latencies) / n_warm),
        avg_generation_latency_sec=(sum(generation_latencies) / n_warm),
        retrieval_latency_p50=r_p50, retrieval_latency_p95=r_p95, retrieval_latency_p99=r_p99,
        generation_latency_p50=g_p50, generation_latency_p95=g_p95, generation_latency_p99=g_p99,
        total_latency_p50=t_p50, total_latency_p95=t_p95, total_latency_p99=t_p99,
        total_generation_cost_usd=round(total_generation_cost, 6),
        total_eval_cost_usd=round(total_eval_cost, 6),
        total_cost_usd=round(total_cost, 6),
        total_cost_inr=usd_to_inr(total_cost),
        failed_rows=failed,
        all_rows=rows,
        passed_qa_excel=passed_qa_excel
    )

def _failure_reason_label(score, reason):
    if score is None:
        return f'SCORING ERROR (no score returned): {reason}'
    return f'score {score:.3f} below threshold - {reason}'

def render_markdown(report):
    lines = [
        f'# Evaluation Report ({report.mode})',
        '',
        f'- Questions evaluated: **{report.n_evaluated}**',
        f'- Failed threshold: **{report.n_failed}**' + (f' ({report.n_failed / report.n_evaluated * 100:.1f}%)' if report.n_evaluated else ''),
    ]
    if report.avg_correctness is not None:
        lines.append(f'- Average correctness: **{report.avg_correctness:.3f}**')
    if report.avg_faithfulness is not None:
        lines.append(f'- Average faithfulness: **{report.avg_faithfulness:.3f}**')
    if report.avg_answer_relevancy is not None:
        lines.append(f'- Average answer relevancy: **{report.avg_answer_relevancy:.3f}**')

    lines += [
        '',
        f'## Latency (seconds, first {COLD_START_EXCLUDE_ROWS} rows excluded as cold-start)',
        f'- Retrieval — avg: {report.avg_retrieval_latency_sec:.3f} | p50: {report.retrieval_latency_p50:.3f} | p95: {report.retrieval_latency_p95:.3f} | p99: {report.retrieval_latency_p99:.3f}',
        f'- Generation — avg: {report.avg_generation_latency_sec:.3f} | p50: {report.generation_latency_p50:.3f} | p95: {report.generation_latency_p95:.3f} | p99: {report.generation_latency_p99:.3f}',
        f'- Total — p50: {report.total_latency_p50:.3f} | p95: {report.total_latency_p95:.3f} | p99: {report.total_latency_p99:.3f}',
        '',
        '## Cost breakdown',
        f'- Generation (answer LLM): ${report.total_generation_cost_usd:.4f} (RS {usd_to_inr(report.total_generation_cost_usd):.2f})',
        f'- Evaluation (judge LLM): ${report.total_eval_cost_usd:.4f} (RS {usd_to_inr(report.total_eval_cost_usd):.2f})',
        f'- **Total: ${report.total_cost_usd:.4f} (RS {report.total_cost_inr:.2f})**',
        '',
        '*1 USD = 94.49 INR*',
        '',
        '## Failed / flagged rows',
        '',
    ]
    if not report.failed_rows:
        lines.append('None - every row met the configured threshold(s)')
    else:
        for r in report.failed_rows:
            lines.append(f'### Row {r.row_id}: {r.question}')
            lines.append(f'- Generated answer: {r.generated_answer}')
            if r.expected_answer:
                lines.append(f'- Expected answer: {r.expected_answer}')
            if r.correctness_score is not None or r.correctness_reason:
                lines.append(f'- Correctness: {_failure_reason_label(r.correctness_score, r.correctness_reason)}')
            if r.faithfulness_score is not None or r.faithfulness_reason:
                lines.append(f'- Faithfulness: {_failure_reason_label(r.faithfulness_score, r.faithfulness_reason)}')
            if r.answer_relevancy_score is not None or r.answer_relevancy_reason:
                lines.append(f'- Answer relevancy: {_failure_reason_label(r.answer_relevancy_score, r.answer_relevancy_reason)}')
            lines.append('')
    return '\n'.join(lines)
