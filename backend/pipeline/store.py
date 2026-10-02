"""M13 storage. Postgres (Neon) when DATABASE_URL is set, otherwise local SQLite.

Heatmap PNGs live in the database, not on disk, so a deploy with an ephemeral filesystem keeps them.
Both backends expose the same methods: get_analysis, save_analysis.
"""
from __future__ import annotations

import base64
import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

from pipeline.config import data_dir

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    scores TEXT, metrics TEXT, heatmap BLOB, result TEXT
);
CREATE TABLE IF NOT EXISTS comparisons (
    id TEXT PRIMARY KEY, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    analysis_ids TEXT, winner_id TEXT
);
CREATE TABLE IF NOT EXISTS gaze_sessions (
    id TEXT PRIMARY KEY, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    thumbnail_id TEXT, method TEXT, points TEXT, cc REAL, nss REAL
);
CREATE TABLE IF NOT EXISTS pilot_thumbnails (
    id TEXT PRIMARY KEY, file_path TEXT, channel TEXT, niche TEXT
);
"""

PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY, created_at TIMESTAMPTZ DEFAULT now(),
    scores JSONB, metrics JSONB, heatmap BYTEA, result JSONB
);
CREATE TABLE IF NOT EXISTS comparisons (
    id TEXT PRIMARY KEY, created_at TIMESTAMPTZ DEFAULT now(),
    analysis_ids JSONB, winner_id TEXT
);
CREATE TABLE IF NOT EXISTS gaze_sessions (
    id TEXT PRIMARY KEY, created_at TIMESTAMPTZ DEFAULT now(),
    thumbnail_id TEXT, method TEXT, points JSONB, cc DOUBLE PRECISION, nss DOUBLE PRECISION
);
CREATE TABLE IF NOT EXISTS pilot_thumbnails (
    id TEXT PRIMARY KEY, file_path TEXT, channel TEXT, niche TEXT
);
"""


def _with_heatmap(result: dict, heatmap: bytes | None) -> dict:
    if heatmap:
        result["heatmap_png"] = base64.b64encode(bytes(heatmap)).decode()
    return result


def _split_heatmap(result: dict) -> tuple[dict, bytes | None]:
    stored = dict(result)
    heatmap = stored.pop("heatmap_png", None)
    return stored, (base64.b64decode(heatmap) if heatmap else None)


class SqliteStore:
    def __init__(self, root: Path | None = None):
        root = Path(root) if root else data_dir()
        root.mkdir(parents=True, exist_ok=True)
        self.db_path = root / "app.db"
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SQLITE_SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def get_analysis(self, analysis_id: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT result, heatmap FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
        return None if row is None else _with_heatmap(json.loads(row["result"]), row["heatmap"])

    def save_analysis(self, result: dict) -> None:
        stored, heatmap = _split_heatmap(result)
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO analyses (id, scores, metrics, heatmap, result) VALUES (?,?,?,?,?)",
                (
                    result["id"],
                    json.dumps(result.get("scores")),
                    json.dumps(result.get("metrics")),
                    heatmap,
                    json.dumps(stored),
                ),
            )


    def save_gaze_session(self, session_id: str, thumbnail_id: str, method: str, payload: dict, cc: float, nss: float) -> None:
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO gaze_sessions (id, thumbnail_id, method, points, cc, nss) VALUES (?,?,?,?,?,?)",
                      (session_id, thumbnail_id, method, json.dumps(payload), cc, nss))

    def list_gaze_sessions(self, thumbnail_id: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT id, method, points, cc, nss FROM gaze_sessions WHERE thumbnail_id = ? ORDER BY created_at, rowid",
                             (thumbnail_id,)).fetchall()
        return [{"id": r["id"], "method": r["method"], "payload": json.loads(r["points"]), "cc": r["cc"], "nss": r["nss"]} for r in rows]

    def save_comparison(self, comparison_id: str, analysis_ids: list[str], winner_id: str | None) -> None:
        with self._lock, self._conn() as c:
            c.execute("INSERT OR REPLACE INTO comparisons (id, analysis_ids, winner_id) VALUES (?,?,?)",
                      (comparison_id, json.dumps(analysis_ids), winner_id))


class PgStore:
    def __init__(self, url: str):
        from psycopg_pool import ConnectionPool

        # check= revalidates connections, since Neon drops idle ones when the compute scales to zero
        self.pool = ConnectionPool(
            url,
            min_size=1,
            max_size=5,
            kwargs={"autocommit": True},
            check=ConnectionPool.check_connection,
            open=True,
        )
        with self.pool.connection() as c:
            c.execute(PG_SCHEMA)

    def get_analysis(self, analysis_id: str) -> dict | None:
        with self.pool.connection() as c:
            row = c.execute("SELECT result, heatmap FROM analyses WHERE id = %s", (analysis_id,)).fetchone()
        return None if row is None else _with_heatmap(row[0], row[1])

    def save_analysis(self, result: dict) -> None:
        from psycopg.types.json import Jsonb

        stored, heatmap = _split_heatmap(result)
        with self.pool.connection() as c:
            c.execute(
                """INSERT INTO analyses (id, scores, metrics, heatmap, result) VALUES (%s,%s,%s,%s,%s)
                   ON CONFLICT (id) DO UPDATE SET scores = EXCLUDED.scores, metrics = EXCLUDED.metrics,
                   heatmap = EXCLUDED.heatmap, result = EXCLUDED.result""",
                (result["id"], Jsonb(result.get("scores")), Jsonb(result.get("metrics")), heatmap, Jsonb(stored)),
            )

    def save_gaze_session(self, session_id: str, thumbnail_id: str, method: str, payload: dict, cc: float, nss: float) -> None:
        from psycopg.types.json import Jsonb

        with self.pool.connection() as c:
            c.execute("INSERT INTO gaze_sessions (id, thumbnail_id, method, points, cc, nss) VALUES (%s,%s,%s,%s,%s,%s)",
                      (session_id, thumbnail_id, method, Jsonb(payload), cc, nss))

    def list_gaze_sessions(self, thumbnail_id: str) -> list[dict]:
        with self.pool.connection() as c:
            rows = c.execute("SELECT id, method, points, cc, nss FROM gaze_sessions WHERE thumbnail_id = %s ORDER BY created_at", (thumbnail_id,)).fetchall()
        return [{"id": r[0], "method": r[1], "payload": r[2], "cc": r[3], "nss": r[4]} for r in rows]

    def save_comparison(self, comparison_id: str, analysis_ids: list[str], winner_id: str | None) -> None:
        from psycopg.types.json import Jsonb

        with self.pool.connection() as c:
            c.execute(
                """INSERT INTO comparisons (id, analysis_ids, winner_id) VALUES (%s,%s,%s)
                   ON CONFLICT (id) DO UPDATE SET analysis_ids = EXCLUDED.analysis_ids, winner_id = EXCLUDED.winner_id""",
                (comparison_id, Jsonb(analysis_ids), winner_id))

    def close(self) -> None:
        self.pool.close()


Store = SqliteStore | PgStore


def make_store() -> Store:
    url = os.environ.get("DATABASE_URL", "").strip()
    return PgStore(url) if url else SqliteStore()
