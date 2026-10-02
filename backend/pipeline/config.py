from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env") 
CONFIG_DIR = BACKEND_DIR / "config"
MODELS_DIR = BACKEND_DIR / "models"


def data_dir() -> Path:
    d = Path(os.environ.get("THUMB_DATA_DIR", BACKEND_DIR.parent / "data"))
    d.mkdir(parents=True, exist_ok=True)
    return d


@lru_cache(maxsize=None)
def load_yaml(name: str) -> dict[str, Any]:
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def analyzers_config() -> dict[str, Any]:
    return load_yaml("analyzers.yaml")
