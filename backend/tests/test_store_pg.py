"""Runs only when TEST_DATABASE_URL points at a throwaway Postgres/Neon branch."""
import base64
import os
import uuid

import pytest

from pipeline.store import PgStore, SqliteStore

PG_URL = os.environ.get("TEST_DATABASE_URL")


def sample(heatmap=b"\x89PNGfake"):
    return {
        "id": uuid.uuid4().hex * 2,
        "heatmap_png": base64.b64encode(heatmap).decode(),
        "scores": {"overall": 70},
        "metrics": {"face": {"count": 1}},
        "unavailable": [],
    }


def roundtrip(store):
    r = sample()
    store.save_analysis(r)
    got = store.get_analysis(r["id"])
    assert got == r
    store.save_analysis({**r, "scores": {"overall": 71}})  # upsert
    assert store.get_analysis(r["id"])["scores"]["overall"] == 71
    assert store.get_analysis("missing") is None


def test_sqlite_roundtrip(tmp_path):
    roundtrip(SqliteStore(tmp_path))


@pytest.mark.skipif(not PG_URL, reason="TEST_DATABASE_URL not set")
def test_postgres_roundtrip():
    store = PgStore(PG_URL)
    try:
        roundtrip(store)
    finally:
        store.close()
