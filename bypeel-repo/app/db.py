"""
Local persistence layer — SQLite (Python's built-in `sqlite3`, zero extra
dependency, zero network, single file on disk). Every time a user processes
a dataset through the Python ML backend, the full result bundle is saved as
a "case" so it survives server restarts and can be reloaded instantly
without re-running the pipeline.

Database file: data/bypeel.db (created automatically on first run).
"""
import sqlite3
import json
import os
import uuid
import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "bypeel.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    source_files  TEXT,
    total_tx      INTEGER,
    total_alerts  INTEGER,
    flagged_tx    INTEGER,
    bundle_json   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dossiers (
    id            TEXT PRIMARY KEY,
    case_id       TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    investigator  TEXT,
    notes         TEXT,
    report_hash   TEXT,
    dossier_json  TEXT NOT NULL,
    FOREIGN KEY(case_id) REFERENCES cases(id)
);
"""


def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def save_case(name: str, source_files: list, bundle: dict) -> str:
    case_id = str(uuid.uuid4())
    conn = get_conn()
    conn.execute(
        "INSERT INTO cases (id, name, created_at, source_files, total_tx, total_alerts, flagged_tx, bundle_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (case_id, name, datetime.datetime.utcnow().isoformat() + "Z", json.dumps(source_files),
         bundle["overview"]["kpis"]["total_tx"], len(bundle["alerts"]),
         bundle["overview"]["kpis"]["flagged_tx"], json.dumps(bundle)),
    )
    conn.commit()
    conn.close()
    return case_id


def list_cases() -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, name, created_at, source_files, total_tx, total_alerts, flagged_tx FROM cases ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_case(case_id: str) -> dict | None:
    conn = get_conn()
    row = conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d["bundle"] = json.loads(d.pop("bundle_json"))
    d["source_files"] = json.loads(d["source_files"]) if d["source_files"] else []
    return d


def delete_case(case_id: str) -> bool:
    conn = get_conn()
    cur = conn.execute("DELETE FROM cases WHERE id = ?", (case_id,))
    conn.execute("DELETE FROM dossiers WHERE case_id = ?", (case_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


def save_dossier(case_id: str, investigator: str, notes: str, report_hash: str, dossier_payload: dict) -> str:
    dossier_id = str(uuid.uuid4())
    conn = get_conn()
    conn.execute(
        "INSERT INTO dossiers (id, case_id, created_at, investigator, notes, report_hash, dossier_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (dossier_id, case_id, datetime.datetime.utcnow().isoformat() + "Z", investigator, notes,
         report_hash, json.dumps(dossier_payload)),
    )
    conn.commit()
    conn.close()
    return dossier_id


def list_dossiers(case_id: str = None) -> list:
    conn = get_conn()
    if case_id:
        rows = conn.execute("SELECT id, case_id, created_at, investigator, report_hash FROM dossiers WHERE case_id = ? ORDER BY created_at DESC", (case_id,)).fetchall()
    else:
        rows = conn.execute("SELECT id, case_id, created_at, investigator, report_hash FROM dossiers ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]
