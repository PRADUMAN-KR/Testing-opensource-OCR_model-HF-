import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


TaskStatus = Literal["processing", "completed", "failed"]
VALID_TASK_STATUSES = {"processing", "completed", "failed"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class OCRTaskStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._lock = threading.RLock()

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ocr_tasks (
                    task_id TEXT PRIMARY KEY,
                    filename TEXT,
                    status TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                UPDATE ocr_tasks
                SET status = 'failed',
                    error = 'Server restarted before task completed.',
                    updated_at = ?
                WHERE status = 'processing'
                """,
                (utc_now_iso(),),
            )
            conn.commit()

    def health(self) -> dict[str, Any]:
        with self._lock, self._connect() as conn:
            conn.execute("SELECT 1").fetchone()
            rows = conn.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM ocr_tasks
                GROUP BY status
                """
            ).fetchall()

        counts = {status: 0 for status in VALID_TASK_STATUSES}
        for row in rows:
            counts[row["status"]] = row["count"]

        return {
            "ok": True,
            "path": str(self.db_path),
            "task_counts": counts,
        }

    def create_task(self, task_id: str, filename: str | None) -> dict[str, Any]:
        now = utc_now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ocr_tasks (
                    task_id, filename, status, result_json, error, created_at, updated_at
                )
                VALUES (?, ?, 'processing', NULL, NULL, ?, ?)
                """,
                (task_id, filename, now, now),
            )
            conn.commit()
        return self.get_task(task_id) or {
            "task_id": task_id,
            "filename": filename,
            "status": "processing",
            "result": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
        }

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT task_id, filename, status, result_json, error, created_at, updated_at
                FROM ocr_tasks
                WHERE task_id = ?
                """,
                (task_id,),
            ).fetchone()

        if row is None:
            return None

        result = json.loads(row["result_json"]) if row["result_json"] else None
        return {
            "task_id": row["task_id"],
            "filename": row["filename"],
            "status": row["status"],
            "result": result,
            "error": row["error"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def update_task(
        self,
        task_id: str,
        *,
        status: TaskStatus,
        result: Any | None = None,
        error: str | None = None,
    ) -> None:
        if status not in VALID_TASK_STATUSES:
            raise ValueError(f"Invalid OCR task status '{status}'. Expected one of: {sorted(VALID_TASK_STATUSES)}")

        result_json = json.dumps(result, ensure_ascii=False, default=str) if result is not None else None
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE ocr_tasks
                SET status = ?,
                    result_json = ?,
                    error = ?,
                    updated_at = ?
                WHERE task_id = ?
                """,
                (status, result_json, error, utc_now_iso(), task_id),
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
