"""
RecommendationRecorder — persists analysis recommendations to SQLite for backtesting.
"""
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS recommendations (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol            TEXT    NOT NULL,
    recommendation    TEXT    NOT NULL,
    confidence        REAL,
    sentiment_score   REAL,
    risk_score        REAL,
    composite_score   REAL,
    timestamp         TEXT    NOT NULL,
    evaluated_at      TEXT,
    actual_return     REAL,
    actual_return_1d  REAL,
    actual_return_7d  REAL,
    actual_return_30d REAL,
    outcome           TEXT,
    created_at        TEXT    NOT NULL
)
"""


class RecommendationRecorder:
    """
    Persists stock recommendations to a local SQLite database so they can be
    evaluated against subsequent price movements for backtesting.

    Usage:
        rec = RecommendationRecorder()
        row_id = rec.save("AAPL", "BUY", confidence=0.8, scores={...})
        pending = rec.get_pending(days_elapsed=7)
        rec.mark_evaluated(pending[0]["id"], actual_return=0.05, outcome="correct")
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Args:
            db_path: Path to the SQLite database file.  Falls back to the
                     ``BACKTESTING_DB_PATH`` env var, then ``data/backtesting.db``.
        """
        self.db_path = db_path or os.getenv("BACKTESTING_DB_PATH", "data/backtesting.db")
        dir_part = os.path.dirname(self.db_path)
        if dir_part:
            os.makedirs(dir_part, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        symbol: str,
        recommendation: str,
        confidence: float,
        scores: Dict[str, Any],
        timestamp: Optional[datetime] = None,
    ) -> int:
        """
        Persist a recommendation for later evaluation.

        Args:
            symbol: Ticker symbol (e.g. ``"AAPL"``).
            recommendation: One of STRONG BUY / BUY / HOLD / SELL / AVOID.
            confidence: Confidence score in [0, 1].
            scores: Dict with ``sentiment_score``, ``risk_score``, and
                    ``composite_score`` keys (extra keys are ignored).
            timestamp: When the recommendation was made; defaults to now.

        Returns:
            Inserted row ID.
        """
        ts = (timestamp or datetime.now()).isoformat()
        now = datetime.now().isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO recommendations
                    (symbol, recommendation, confidence,
                     sentiment_score, risk_score, composite_score,
                     timestamp, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol.upper(),
                    recommendation,
                    confidence,
                    scores.get("sentiment_score"),
                    scores.get("risk_score"),
                    scores.get("composite_score"),
                    ts,
                    now,
                ),
            )
            row_id = cursor.lastrowid
        logger.debug(f"Backtest: saved #{row_id} — {symbol} {recommendation}")
        return row_id

    def get_pending(self, days_elapsed: int = 7) -> List[Dict[str, Any]]:
        """
        Return unevaluated recommendations old enough to assess.

        Args:
            days_elapsed: Minimum age in days before a record is considered
                          ready for evaluation (default 7, matching the 7d
                          price-return window).

        Returns:
            List of row dicts ordered by ``timestamp`` ascending.
        """
        cutoff = (datetime.now() - timedelta(days=days_elapsed)).isoformat()
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM recommendations
                WHERE evaluated_at IS NULL
                  AND timestamp <= ?
                ORDER BY timestamp ASC
                """,
                (cutoff,),
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_evaluated(
        self,
        id: int,
        actual_return: float,
        outcome: str,
        actual_return_1d: Optional[float] = None,
        actual_return_7d: Optional[float] = None,
        actual_return_30d: Optional[float] = None,
    ) -> None:
        """
        Record the evaluation result for a pending recommendation.

        Args:
            id: Primary key of the recommendation row.
            actual_return: Primary return used for accuracy scoring (7d window).
            outcome: ``"correct"``, ``"incorrect"``, or ``"neutral"``.
            actual_return_1d: 1-day price return (optional).
            actual_return_7d: 7-day price return (optional; mirrors
                              ``actual_return`` when not supplied separately).
            actual_return_30d: 30-day price return (optional).
        """
        now = datetime.now().isoformat()
        r7d = actual_return_7d if actual_return_7d is not None else actual_return
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE recommendations
                SET evaluated_at      = ?,
                    actual_return     = ?,
                    actual_return_1d  = ?,
                    actual_return_7d  = ?,
                    actual_return_30d = ?,
                    outcome           = ?
                WHERE id = ?
                """,
                (now, actual_return, actual_return_1d, r7d, actual_return_30d, outcome, id),
            )
        logger.debug(f"Backtest: marked #{id} as {outcome} (return={actual_return:.2%})")

    def get_evaluated(self) -> List[Dict[str, Any]]:
        """Return all evaluated recommendations."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM recommendations WHERE evaluated_at IS NOT NULL"
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE)
