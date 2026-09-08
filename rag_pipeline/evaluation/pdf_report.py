from io import BytesIO

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

from .report import COLD_START_EXCLUDE_ROWS
from .operational import usd_to_inr

def render_pdf(report):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    styles = getSampleStyleSheet()
    small = ParagraphStyle('small', parent=styles['Normal'], fontSize=9, leading=12)
    story = []

    story.append(Paragraph('RAG Pipeline - Evaluation Report', styles['Title']))
    story.append(Paragraph(f'Mode: {report.mode}', styles['Normal']))
    story.append(Spacer(1, 16))

    # summary table
    summary_rows = [['Metric', 'Value']]
    summary_rows.append(['Questions evaluated', str(report.n_evaluated)])
    summary_rows.append(['Failed threshold', f'{report.n_failed} ({(report.n_failed / report.n_evaluated * 100) if report.n_evaluated else 0:.1f}%)'])
    if report.avg_correctness is not None:
        summary_rows.append(['Average correctness', f'{report.avg_correctness:.3f}'])
    if report.avg_faithfulness is not None:
        summary_rows.append(['Average faithfulness', f'{report.avg_faithfulness:.3f}'])
    if report.avg_answer_relevancy is not None:
        summary_rows.append(['Average answer relevancy', f'{report.avg_answer_relevancy:.3f}'])

    summary_table = Table(summary_rows, colWidths=[2.6 * inch, 3.4 * inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f2937')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d1d5db')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fafb')]),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 20))

    # latency table
    story.append(Paragraph(f'Latency (seconds, first {COLD_START_EXCLUDE_ROWS} rows excluded as cold-start)', styles['Heading2']))
    latency_rows = [
        ['Stage', 'Avg', 'P50', 'P95', 'P99'],
        ['Retrieval', f'{report.avg_retrieval_latency_sec:.3f}', f'{report.retrieval_latency_p50:.3f}',
         f'{report.retrieval_latency_p95:.3f}', f'{report.retrieval_latency_p99:.3f}"'],
        ['Generation', f'{report.avg_generation_latency_sec:.3f}', f'{report.generation_latency_p50:.3f}',
         f'{report.generation_latency_p95:.3f}', f'{report.generation_latency_p99:.3f}'],
        ['Total', '-', f'{report.total_latency_p50:.3f}', f'{report.total_latency_p95:.3f}', f'{report.total_latency_p99:.3f}'],
    ]
    latency_table = Table(latency_rows, colWidths=[1.3 * inch] + [1.175 * inch] * 4)
    latency_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f2937')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d1d5db')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fafb')]),
        ('FONTSIZE', (0, 0), (-1, -1), 9)
    ]))
    story.append(latency_table)
    story.append(Spacer(1, 20))

    # cost table 
    story.append(Paragraph('Cost breakdown', styles['Heading2']))
    cost_rows = [
        ['Category"', 'USD', 'INR (approx.)'],
        ['Generation (answer LLM)', f'${report.total_generation_cost_usd:.4f}', f'INR {usd_to_inr(report.total_generation_cost_usd):.2f}'],
        ['Evaluation (judge LLM)', f'${report.total_eval_cost_usd:.4f}', f'INR {usd_to_inr(report.total_eval_cost_usd):.2f}'],
        ['Total', f'${report.total_cost_usd:.4f}', f'INR {report.total_cost_inr:.2f}']
    ]
    cost_table = Table(cost_rows, colWidths=[2.6 * inch, 1.7 * inch, 1.7 * inch])
    cost_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), "Helvetica-Bold"),
        ('FONTNAME', (0, -1), (-1, -1), "Helvetica-Bold"),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, colors.HexColor("#f9fafb")]),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
    ]))
    story.append(cost_table)
    story.append(Paragraph('<i>1 USD = 94.49 INR</i>', small))
    story.append(PageBreak())

    # failed rows
    story.append(Paragraph('Failed / flagged rows', styles['Heading2']))
    if not report.failed_rows:
        story.append(Paragraph('None, every row met the configured threshold(s)', styles['Normal']))
    else:
        for r in report.failed_rows:
            story.append(Paragraph(f'Row {r.row_id}: {r.question}', styles['Heading3']))
            story.append(Paragraph(f'<b>Generated answer:</b> {r.generated_answer}', small))
            if r.expected_answer:
                story.append(Paragraph(f'<b>Expected answer:</b> {r.expected_answer}', small))
            if r.correctness_score is not None or r.correctness_reason:
                story.append(Paragraph(f'<b>Correctness:</b> score={r.correctness_score} - {r.correctness_reason}', small))
            if r.faithfulness_score is not None or r.faithfulness_reason:
                story.append(Paragraph(f'<b>Faithfulness:</b> score={r.faithfulness_score} - {r.faithfulness_reason}', small))
            if r.answer_relevancy_score is not None or r.answer_relevancy_reason:
                story.append(Paragraph(f'<b>Answer relevancy:</b> score={r.answer_relevancy_score} - {r.answer_relevancy_reason}', small))
            story.append(Spacer(1, 10))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()
