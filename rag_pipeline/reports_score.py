import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .models import EvalReport

class SharedReportStore:
    def __init__(self, db_path = './rag_collections/shared_reports.sqlite'):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                report_id TEXT PRIMARY KEY,
                created_at TEXT,
                mode TEXT,
                n_evaluated INTEGER,
                n_failed INTEGER,
                avg_correctness REAL,
                avg_faithfulness REAL,
                avg_answer_relevancy REAL,
                total_cost_usd REAL,
                total_cost_inr REAL,
                markdown TEXT,
                pdf_blob BLOB
            )
        """)
        self.conn.commit()

    def save_report(self, report, pdf_bytes, markdown):
        report_id = str(uuid.uuid4())
        with self._lock:
            self.conn.execute(
                "INSERT INTO reports (report_id, created_at, mode, n_evaluated, n_failed, "
                "avg_correctness, avg_faithfulness, avg_answer_relevancy, total_cost_usd, "
                "total_cost_inr, markdown, pdf_blob) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (report_id, datetime.now(timezone.utc).isoformat(), report.mode,
                 report.n_evaluated, report.n_failed, report.avg_correctness,
                 report.avg_faithfulness, report.avg_answer_relevancy,
                 report.total_cost_usd, report.total_cost_inr, markdown, pdf_bytes),
            )
            self.conn.commit()
        return report_id

    def list_reports(self, limit = 50):
        with self._lock:
            rows = self.conn.execute(
                "SELECT report_id, created_at, mode, n_evaluated, n_failed, avg_correctness, "
                "avg_faithfulness, avg_answer_relevancy, total_cost_usd, total_cost_inr "
                "FROM reports ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        columns = ["report_id", "created_at", "mode", "n_evaluated", "n_failed",
                   "avg_correctness", "avg_faithfulness", "avg_answer_relevancy",
                   "total_cost_usd", "total_cost_inr"]
        return [dict(zip(columns, row)) for row in rows]

    def get_pdf(self, report_id):
        with self._lock:
            row = self.conn.execute("SELECT pdf_blob FROM reports WHERE report_id = ?", (report_id,)).fetchone()
        return row[0] if row else None

    def get_markdown(self, report_id):
        with self._lock:
            row = self.conn.execute("SELECT markdown FROM reports WHERE report_id = ?", (report_id,)).fetchone()
        return row[0] if row else None

    def close(self):
        self.conn.close()
