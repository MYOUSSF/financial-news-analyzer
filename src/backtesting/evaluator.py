"""
BacktestEvaluator — fetches actual price returns and scores recommendation accuracy.
"""
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger

from src.backtesting.recorder import RecommendationRecorder


_BULLISH = {"STRONG BUY", "BUY"}
_BEARISH = {"SELL", "AVOID"}


def _classify_outcome(recommendation: str, return_7d: Optional[float]) -> str:
    """Map a recommendation and its realized 7-day return to an outcome label."""
    if return_7d is None:
        return "unknown"
    rec = recommendation.upper()
    if rec in _BULLISH:
        return "correct" if return_7d > 0 else "incorrect"
    if rec in _BEARISH:
        return "correct" if return_7d < 0 else "incorrect"
    return "neutral"  # HOLD


class BacktestEvaluator:
    """
    Evaluates pending recommendations by comparing them against subsequent
    price movements fetched via yfinance.

    Usage:
        ev = BacktestEvaluator()
        ev.run_all_pending()
        metrics = ev.compute_metrics()
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Args:
            db_path: Passed through to RecommendationRecorder.  Uses
                     ``BACKTESTING_DB_PATH`` env var or ``data/backtesting.db``
                     when not supplied.
        """
        self.recorder = RecommendationRecorder(db_path=db_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_recommendation(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch 1d, 7d, and 30d price returns for a single recommendation and
        mark it evaluated in the database.

        Args:
            record: Row dict from ``RecommendationRecorder.get_pending()``.

        Returns:
            Dict with ``id``, ``actual_return_1d``, ``actual_return_7d``,
            ``actual_return_30d``, and ``outcome``.  A window's value is
            ``None`` when the price data is not yet available.
        """
        symbol = record["symbol"]
        rec_date = datetime.fromisoformat(record["timestamp"]).date()

        hist = self._fetch_history(symbol, rec_date)
        if hist is None or hist.empty:
            logger.warning(f"No price history for {symbol} from {rec_date}")
            return {"id": record["id"], "error": "no price data"}

        base_price = self._price_on_or_after(hist, rec_date)
        if base_price is None:
            logger.warning(f"No base price for {symbol} on {rec_date}")
            return {"id": record["id"], "error": "no base price"}

        def _return_at(days: int) -> Optional[float]:
            target = rec_date + timedelta(days=days)
            price = self._price_on_or_after(hist, target)
            if price is None:
                return None
            return (price - base_price) / base_price

        r1d = _return_at(1)
        r7d = _return_at(7)
        r30d = _return_at(30)
        outcome = _classify_outcome(record["recommendation"], r7d)

        self.recorder.mark_evaluated(
            id=record["id"],
            actual_return=r7d if r7d is not None else 0.0,
            outcome=outcome,
            actual_return_1d=r1d,
            actual_return_7d=r7d,
            actual_return_30d=r30d,
        )

        if r7d is not None:
            logger.info(
                f"Backtest: #{record['id']} {symbol} {record['recommendation']}"
                f" → 7d={r7d:.2%} ({outcome})"
            )
        else:
            logger.info(f"Backtest: #{record['id']} {symbol} — 7d return unavailable")

        return {
            "id": record["id"],
            "actual_return_1d": r1d,
            "actual_return_7d": r7d,
            "actual_return_30d": r30d,
            "outcome": outcome,
        }

    def run_all_pending(self, days_elapsed: int = 7) -> List[Dict[str, Any]]:
        """
        Evaluate every pending recommendation that is old enough.

        Args:
            days_elapsed: Minimum age in days for a record to be evaluated.

        Returns:
            List of evaluation result dicts, one per processed record.
        """
        pending = self.recorder.get_pending(days_elapsed=days_elapsed)
        logger.info(f"Backtest: {len(pending)} pending recommendation(s) to evaluate")
        results = []
        for record in pending:
            try:
                results.append(self.evaluate_recommendation(record))
            except Exception as exc:
                logger.error(f"Failed to evaluate record #{record['id']}: {exc}")
                results.append({"id": record["id"], "error": str(exc)})
        return results

    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute accuracy statistics over all evaluated recommendations.

        Returns:
            Dict with:

            - ``overall_accuracy``: float in [0, 1], or ``None`` if no
              scoreable records (HOLD-only or empty).
            - ``accuracy_by_recommendation``: ``{rec_type: float | None}``.
            - ``avg_return_by_recommendation``: ``{rec_type: float | None}``.
            - ``total_evaluated``: total count of evaluated records.
        """
        rows = self.recorder.get_evaluated()
        if not rows:
            return {
                "overall_accuracy": None,
                "accuracy_by_recommendation": {},
                "avg_return_by_recommendation": {},
                "total_evaluated": 0,
            }

        # Overall accuracy — HOLD (neutral/unknown) excluded from denominator
        scoreable = [r for r in rows if r["outcome"] in ("correct", "incorrect")]
        n_correct = sum(1 for r in scoreable if r["outcome"] == "correct")
        overall_accuracy = n_correct / len(scoreable) if scoreable else None

        # Per-recommendation breakdown
        rec_types = sorted({r["recommendation"] for r in rows})
        acc_by_rec: Dict[str, Any] = {}
        avg_ret_by_rec: Dict[str, Any] = {}
        for rec in rec_types:
            group = [r for r in rows if r["recommendation"] == rec]
            group_scoreable = [r for r in group if r["outcome"] in ("correct", "incorrect")]
            n_group_correct = sum(1 for r in group_scoreable if r["outcome"] == "correct")
            acc_by_rec[rec] = (
                n_group_correct / len(group_scoreable) if group_scoreable else None
            )
            returns = [
                r["actual_return_7d"]
                for r in group
                if r["actual_return_7d"] is not None
            ]
            avg_ret_by_rec[rec] = sum(returns) / len(returns) if returns else None

        return {
            "overall_accuracy": overall_accuracy,
            "accuracy_by_recommendation": acc_by_rec,
            "avg_return_by_recommendation": avg_ret_by_rec,
            "total_evaluated": len(rows),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_history(self, symbol: str, from_date) -> Optional[Any]:
        """Download ~35 days of daily price history starting at from_date."""
        try:
            import yfinance as yf
            end_date = from_date + timedelta(days=35)
            ticker = yf.Ticker(symbol)
            return ticker.history(
                start=from_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
            )
        except Exception as exc:
            logger.error(f"yfinance fetch failed for {symbol}: {exc}")
            return None

    @staticmethod
    def _price_on_or_after(hist, target_date) -> Optional[float]:
        """
        Return the closing price on the first available trading day on or
        after ``target_date``.  Handles both tz-aware and tz-naive indices.
        """
        from datetime import date as date_type
        target = (
            target_date if isinstance(target_date, date_type) else target_date.date()
        )
        for i, ts in enumerate(hist.index):
            row_date = ts.date() if hasattr(ts, "date") else ts.to_pydatetime().date()
            if row_date >= target:
                return float(hist["Close"].iloc[i])
        return None
