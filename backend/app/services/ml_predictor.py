from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from textblob import TextBlob

logger = logging.getLogger(__name__)

URGENT_KEYWORDS: tuple[str, ...] = (
    "refund",
    "broken",
    "cancel",
    "down",
    "help",
    "urgent",
    "asap",
    "fix",
    "issue",
    "problem",
    "stuck",
    "error",
    "not working",
    "wrong",
    "charged",
    "money",
    "scam",
    "fraud",
    "disappointed",
    "angry",
    "furious",
    "terrible",
    "worst",
    "useless",
    "hopeless",
)


class MLPredictor:
    """Load trained classifier and run priority inference."""

    @staticmethod
    @lru_cache(maxsize=1)
    def get_instance() -> "MLPredictor":
        return MLPredictor()

    def __init__(self) -> None:
        project_root = Path(__file__).resolve().parents[3]
        self.model_path = project_root / "models" / "priority_classifier.pkl"
        self.features_path = project_root / "models" / "feature_columns.json"

        if not self.model_path.exists():
            raise FileNotFoundError(f"ML model not found: {self.model_path}")
        if not self.features_path.exists():
            raise FileNotFoundError(f"Feature columns file not found: {self.features_path}")

        self.model: Any = joblib.load(self.model_path)
        self.feature_columns: list[str] = json.loads(self.features_path.read_text(encoding="utf-8"))
        if not self.feature_columns:
            raise ValueError("feature_columns.json is empty.")

        logger.info(
            "Initialized MLPredictor model='%s' features=%d",
            self.model_path.name,
            len(self.feature_columns),
        )

    @staticmethod
    def _sentiment_score(text: str) -> float:
        return float(TextBlob(text).sentiment.polarity)

    def _extract_features(self, text: str) -> pd.DataFrame:
        cleaned = text.strip()
        words = cleaned.split()
        word_count = len(words)
        char_count = len(cleaned)
        exclamation_count = cleaned.count("!")
        question_count = cleaned.count("?")

        letters_only = re.sub(r"[^A-Za-z]", "", cleaned)
        letter_count = len(letters_only)
        uppercase_count = sum(1 for char in letters_only if char.isupper())
        caps_ratio = (uppercase_count / letter_count) if letter_count > 0 else 0.0

        lowered = cleaned.lower()
        keyword_count = 0
        for kw in URGENT_KEYWORDS:
            keyword_count += len(re.findall(rf"\b{re.escape(kw)}\b", lowered))

        raw_features: dict[str, float | int] = {
            "word_count": word_count,
            "char_count": char_count,
            "exclamation_count": exclamation_count,
            "question_count": question_count,
            "caps_ratio": caps_ratio,
            "keyword_count": keyword_count,
            "has_exclamation": int(exclamation_count > 0),
            "has_question": int(question_count > 0),
            "has_urgent_keyword": int(keyword_count > 0),
            "avg_word_length": char_count / max(1, word_count),
            "sentiment_score": self._sentiment_score(cleaned),
        }

        # Keep exact training column order and default missing values to 0.
        ordered = {column: raw_features.get(column, 0) for column in self.feature_columns}
        return pd.DataFrame([ordered], columns=self.feature_columns)

    def predict_priority(self, text: str) -> tuple[str, float]:
        features = self._extract_features(text)
        pred = int(self.model.predict(features)[0])

        confidence = None
        if hasattr(self.model, "predict_proba"):
            confidence = float(self.model.predict_proba(features)[0][1])
        else:
            confidence = float(pred)

        priority = "urgent" if pred == 1 else "normal"
        return priority, confidence

    async def predict_priority_async(self, text: str) -> tuple[str, float]:
        return await asyncio.to_thread(self.predict_priority, text)

    async def predict_with_metrics(self, text: str) -> tuple[str, float, float, float]:
        start = time.perf_counter()
        priority, confidence = await self.predict_priority_async(text)
        latency_ms = (time.perf_counter() - start) * 1000.0
        cost_dollars = 0.0
        return priority, confidence, latency_ms, cost_dollars

