import tempfile
from pathlib import Path
import pandas as pd
import streamlit as st
from rag_pipeline.pipeline import RAGPipeline
from rag_pipeline.models import EvalConfig
from rag_pipeline.evaluation import render_markdown, render_pdf
from rag_pipeline.reports_store import SharedReportStore
import io
import zipfile

from rag_pipeline import DocSegmenter

st.set_page_config(page_title='RAG Assistant', layout='wide')

SHARED_COLLECTION_NAME = 'shared_knowledge_base'
SHARED_COLLECTION_DIR = './rag_collections'

st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; max-width: 1100px; }
    h1 { font-weight: 700; letter-spacing: -0.5px; }
    div[data-testid="stMetric"] {
        background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 10px;
        padding: 12px 16px;
    }
    div[data-testid="stExpander"] { border: 1px solid #e5e7eb; border-radius: 10px; }
    .stButton > button { border-radius: 8px; font-weight: 600; padding: 0.5rem 1.2rem; }
    .stButton > button[kind="primary"] { background-color: #111827; }
    div[data-testid="stChatInput"] {
        position: fixed;
        bottom: 0;
        left: 15rem;
        right: 0;
        background: white;
        padding: 1rem 2rem;
        z-index: 999;
        border-top: 1px solid #e5e7eb;
    }
    .main .block-container { padding-bottom: 6rem; }
</style> 
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align: center; margin-bottom: 0;'>RAG Assistant</h1>
<p style='text-align: center; font-size:20px; color: #555; margin-top: 0;'>Ask questions. Get grounded answers.</p>
""", unsafe_allow_html=True)
st.markdown("---")

@st.cache_resource
def get_pipeline():
    segmenter = DocSegmenter()
    return RAGPipeline(
        segmenter=segmenter,
        collection_name=SHARED_COLLECTION_NAME,
        storage_backend="turbovec_sqlite",
        storage_kwargs={"collection_dir": SHARED_COLLECTION_DIR}
    )

@st.cache_resource
def get_report_store():
    return SharedReportStore(db_path=f'{SHARED_COLLECTION_DIR}/shared_reports.sqlite')

pipeline = get_pipeline()
report_store = get_report_store()

# session init per-browser-session chat history only
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.subheader('Shared knowledge base')
    st.caption('Documents added here are visible to everyone using this app')

    uploaded_files = st.file_uploader('Add PDFs', type=['pdf'], accept_multiple_files=True)
    asset_names = []
    if uploaded_files:
        st.caption('Mark table-heavy datasheets as asset-category:')
        for f in uploaded_files:
            if st.checkbox(f'{f.name}', key=f'asset_{f.name}'):
                asset_names.append(f.name)

        if st.button('Add to shared knowledge base', type='primary', use_container_width=True):
            with tempfile.TemporaryDirectory() as tmp_dir:
                progress = st.progress(0.0, text='Indexing...')
                for i, f in enumerate(uploaded_files):
                    tmp_path = Path(tmp_dir) / f.name
                    tmp_path.write_bytes(f.getvalue())
                    result = pipeline.index_document(str(tmp_path), f.name, is_asset_category=(f.name in asset_names))
                    st.caption(f"'{f.name}': {result.new_chunks_indexed} new chunks "
                               f"({result.skipped_chunks} unchanged) · ${result.ingestion_total_cost_usd:.5f}")
                    progress.progress((i + 1) / len(uploaded_files), text=f"Indexed {f.name}")
                pipeline.finalize_index()
            progress.empty()
            st.success("Added to the shared knowledge base.")

    st.markdown('---')
    if st.button('Clear my chat', use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

tab_chat, tab_eval, tab_reports = st.tabs(['Chat', 'Evaluate', 'Past reports'])

with tab_chat:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    if prompt := st.chat_input('Ask a question about the shared documents...'):
        st.session_state.chat_history.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.markdown(prompt)

        with st.chat_message('assistant'):
            with st.spinner('Thinking...'):
                try:
                    result = pipeline.query(prompt)
                    answer_text = result.answer
                    if not result.has_sufficient_context:
                        answer_text += '\n\n*(The knowledge base may not have enough information for this question.)*'
                except Exception:
                    answer_text = 'Something went wrong answering that - please try again'
            st.markdown(answer_text)

        st.session_state.chat_history.append({'role': 'assistant', 'content': answer_text})

# EVALUATE
with tab_eval:
    st.subheader('Run an evaluation')

    eval_mode_label = st.radio('Mode', ['Question + Expected Answer pairs', 'Questions only'], horizontal=True)
    mode = 'qa_pairs' if eval_mode_label == 'Question + Expected Answer pairs' else 'questions_only'

    input_method = st.radio('How do you want to provide questions?', ['Upload a file', 'Type directly'], horizontal=True)

    df = None
    if input_method == 'Upload a file':
        eval_file = st.file_uploader('Questions file (.xlsx/.csv)', type=['xlsx', 'csv'], key='eval_file')
        if eval_file is not None:
            df = pd.read_csv(eval_file) if eval_file.name.endswith('.csv') else pd.read_excel(eval_file)
    else:
        if mode == 'qa_pairs':
            c1, c2 = st.columns(2)
            with c1:
                questions_text = st.text_area('Questions (one per line)', height=200, key='typed_q')
            with c2:
                answers_text = st.text_area('Expected answers (one per line, SAME order)', height=200, key='typed_a')
            if questions_text.strip() and answers_text.strip():
                qs = [q.strip() for q in questions_text.strip().split("\n") if q.strip()]
                ans = [a.strip() for a in answers_text.strip().split("\n") if a.strip()]
                if len(qs) != len(ans):
                    st.error(f'{len(qs)} questions but {len(ans)} answers - counts must match, one pair per line')
                else:
                    df = pd.DataFrame({'Question': qs, 'Expected Answer': ans})
        else:
            questions_text = st.text_area('Questions (one per line)', height=200, key='typed_q_only')
            if questions_text.strip():
                qs = [q.strip() for q in questions_text.strip().split("\n") if q.strip()]
                df = pd.DataFrame({'Question': qs})

    n_questions = st.number_input('How many to evaluate? (0 = all)', min_value=0, value=0, step=1)

    if df is not None and st.button("Run evaluation", type="primary"):
        cfg = EvalConfig(mode=mode, n_questions=(n_questions or None))
        with st.spinner(f"Evaluating {n_questions or len(df)} questions..."):
            report = pipeline.evaluate(df, cfg)
            pdf_bytes = render_pdf(report)
            markdown = render_markdown(report)
            report_id = report_store.save_report(report, pdf_bytes, markdown)

        st.success(f'Evaluation complete - saved to the shared report archive (ID: {report_id[:8]})')

        c1, c2, c3 = st.columns(3)
        c1.metric("Evaluated", report.n_evaluated)
        c2.metric("Below threshold", report.n_failed)
        c3.metric("Total cost", f"${report.total_cost_usd:.4f} · RS {report.total_cost_inr:.2f}")

        score_cols = st.columns(3)
        if report.avg_correctness is not None:
            score_cols[0].metric("Avg correctness", f"{report.avg_correctness:.3f}")
        if report.avg_faithfulness is not None:
            score_cols[1].metric("Avg faithfulness", f"{report.avg_faithfulness:.3f}")
        if report.avg_answer_relevancy is not None:
            score_cols[2].metric("Avg answer relevancy", f"{report.avg_answer_relevancy:.3f}")

        # st.download_button(
        #     "Download this report as PDF", data=pdf_bytes,
        #     file_name=f"eval_report_{report_id[:8]}.pdf", mime="application/pdf", type="primary",
        # )

        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr(f"eval_report_{report_id[:8]}.pdf", pdf_bytes)

            if report.passed_qa_excel:
                zip_file.writestr('passed_questions_answers.xlsx', report.passed_qa_excel)

        zip_buffer.seek(0)

        st.download_button(
            label='Download Complete Evaluation Report',
            data=zip_buffer.getvalue(),
            file_name=f"evaluation_report_{report_id[:8]}.zip",
            mime='application/zip',
            type='primary'
        )

        with st.expander("Full markdown report"):
            st.markdown(markdown)

# PAST REPORTS (shared archive)
with tab_reports:
    st.subheader("Past evaluation reports")
    st.caption("Every evaluation anyone runs on this app is saved here automatically.")

    reports = report_store.list_reports()
    if not reports:
        st.info("No evaluations have been run yet.")
    else:
        for r in reports:
            title = f"{r['created_at'][:19].replace('T', ' ')} · {r['mode']} · {r['n_evaluated']} questions · {r['n_failed']} failed"
            with st.expander(title):
                cols = st.columns(4)
                if r["avg_correctness"] is not None:
                    cols[0].metric("Correctness", f"{r['avg_correctness']:.3f}")
                if r["avg_faithfulness"] is not None:
                    cols[1].metric("Faithfulness", f"{r['avg_faithfulness']:.3f}")
                if r["avg_answer_relevancy"] is not None:
                    cols[2].metric("Answer relevancy", f"{r['avg_answer_relevancy']:.3f}")
                cols[3].metric("Total cost", f"${r['total_cost_usd']:.4f}")

                pdf_bytes = report_store.get_pdf(r["report_id"])
                if pdf_bytes:
                    st.download_button(
                        "Download PDF", data=pdf_bytes,
                        file_name=f"eval_report_{r['report_id'][:8]}.pdf",
                        mime="application/pdf", key=f"dl_{r['report_id']}",
                    )
