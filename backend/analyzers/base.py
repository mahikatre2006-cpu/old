"""Common analyzer interface. Every module implements this so it can be swapped via config."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from pipeline.context import AnalyzerResult, Context


class Analyzer(ABC):
    name: str

    def load(self) -> None:
        """Load models once at startup. Default: nothing to load."""

    @abstractmethod
    def run(self, img: np.ndarray, ctx: Context) -> AnalyzerResult: ...
