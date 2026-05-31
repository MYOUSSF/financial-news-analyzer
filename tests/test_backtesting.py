"""
Unit tests for the backtesting module.

Run with:
    pytest tests/test_backtesting.py -v
"""
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.backtesting.recorder import RecommendationRecorder
from src.backtesting.evaluator import BacktestEvaluator, _classify_outcome
from src.backtesting.report import generate_backtest_report


# ===========================================================================
# Shared helpers
# ===========================================================================

def _make_price_history(base_price: float, daily_returns: list, start_date) -> pd.DataFrame:
    """
    Build a tz-naive daily Close-price DataFrame for mocking yfinance.

    Args:
        base_price: Opening price (day 0).
        daily_returns: Fractional return per day, e.g. [0.01]*34.
        start_date: First date in the index (datetime or date).
    """
    prices = [base_price]
    for r in daily_returns:
        prices.append(prices[-1] * (1 + r))
    dates = pd.date_range(start=start_date, periods=len(prices), freq="D")
    return pd.DataFrame({"Close": prices}, index=dates)


def _seed_record(
    recorder: RecommendationRecorder,
    symbol: str = "AAPL",
    recommendation: str = "BUY",
    timestamp: datetime = None,
) -> int:
    """Insert one record; timestamp defaults to 10 days ago."""
    ts = timestamp or datetime.now() - timedelta(days=10)
    return recorder.save(
        symbol=symbol,
        recommendation=recommendation,
        confidence=0.8,
        scores={"sentiment_score": 0.7, "risk_score": 0.3, "composite_score": 0.7},
        timestamp=ts,
    )


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def recorder(tmp_path):
    """RecommendationRecorder backed by a fresh temporary SQLite DB."""
    return RecommendationRecorder(db_path=str(tmp_path / "bt.db"))


@pytest.fixture
def evaluator(tmp_path):
    """BacktestEvaluator backed by a fresh temporary SQLite DB."""
    return BacktestEvaluator(db_path=str(tmp_path / "bt.db"))


# ===========================================================================
# RecommendationRecorder — schema & persistence
# ===========================================================================

class TestRecommendationRecorder:

    def test_save_returns_integer_id(self, recorder):
        row_id = _seed_record(recorder)
        assert isinstance(row_id, int) and row_id >= 1

    def test_save_stores_all_score_fields(self, recorder):
        ts = datetime(2024, 1, 15, 12, 0, 0)
        recorder.save(
            "TSLA", "SELL", 0.6,
            scores={"sentiment_score": 0.3, "risk_score": 0.8, "composite_score": 0.4},
            timestamp=ts,
        )
        rows = recorder.get_pending(days_elapsed=1)
        assert len(rows) == 1
        r = rows[0]
        assert r["symbol"] == "TSLA"
        assert r["recommendation"] == "SELL"
        assert r["confidence"] == pytest.approx(0.6)
        assert r["sentiment_score"] == pytest.approx(0.3)
        assert r["risk_score"] == pytest.approx(0.8)
        assert r["composite_score"] == pytest.approx(0.4)

    def test_save_upcases_symbol(self, recorder):
        recorder.save("aapl", "BUY", 0.8, {}, timestamp=datetime.now() - timedelta(days=5))
        rows = recorder.get_pending(days_elapsed=1)
        assert rows[0]["symbol"] == "AAPL"

    def test_save_accepts_missing_scores(self, recorder):
        """Partial or empty scores dict should not raise."""
        recorder.save("AAPL", "HOLD", 0.5, {}, timestamp=datetime.now() - timedelta(days=5))
        rows = recorder.get_pending(days_elapsed=1)
        assert rows[0]["sentiment_score"] is None

    # ── get_pending ──────────────────────────────────────────────────

    def test_get_pending_returns_old_records(self, recorder):
        _seed_record(recorder, timestamp=datetime.now() - timedelta(days=10))
        assert len(recorder.get_pending(days_elapsed=7)) == 1

    def test_get_pending_excludes_too_recent_records(self, recorder):
        _seed_record(recorder, timestamp=datetime.now() - timedelta(days=3))
        assert len(recorder.get_pending(days_elapsed=7)) == 0

    def test_get_pending_excludes_already_evaluated(self, recorder):
        row_id = _seed_record(recorder)
        recorder.mark_evaluated(row_id, actual_return=0.05, outcome="correct")
        assert len(recorder.get_pending(days_elapsed=7)) == 0

    def test_get_pending_mixes_old_and_recent(self, recorder):
        old_id = _seed_record(recorder, timestamp=datetime.now() - timedelta(days=10))
        _seed_record(recorder, timestamp=datetime.now() - timedelta(days=3))
        pending = recorder.get_pending(days_elapsed=7)
        assert len(pending) == 1
        assert pending[0]["id"] == old_id

    def test_get_pending_ordered_by_timestamp(self, recorder):
        t1 = datetime.now() - timedelta(days=20)
        t2 = datetime.now() - timedelta(days=15)
        id1 = _seed_record(recorder, timestamp=t1)
        id2 = _seed_record(recorder, timestamp=t2)
        ids = [r["id"] for r in recorder.get_pending(days_elapsed=7)]
        assert ids == [id1, id2]

    # ── mark_evaluated ───────────────────────────────────────────────

    def test_mark_evaluated_sets_outcome_and_return(self, recorder):
        row_id = _seed_record(recorder)
        recorder.mark_evaluated(row_id, actual_return=0.05, outcome="correct")
        evaluated = recorder.get_evaluated()
        assert len(evaluated) == 1
        r = evaluated[0]
        assert r["outcome"] == "correct"
        assert r["actual_return"] == pytest.approx(0.05)
        assert r["evaluated_at"] is not None

    def test_mark_evaluated_stores_all_windows(self, recorder):
        row_id = _seed_record(recorder)
        recorder.mark_evaluated(
            row_id, actual_return=0.05, outcome="correct",
            actual_return_1d=0.01, actual_return_7d=0.05, actual_return_30d=0.12,
        )
        r = recorder.get_evaluated()[0]
        assert r["actual_return_1d"] == pytest.approx(0.01)
        assert r["actual_return_7d"] == pytest.approx(0.05)
        assert r["actual_return_30d"] == pytest.approx(0.12)

    def test_mark_evaluated_defaults_7d_to_actual_return(self, recorder):
        """When actual_return_7d is omitted, it mirrors actual_return."""
        row_id = _seed_record(recorder)
        recorder.mark_evaluated(row_id, actual_return=0.08, outcome="correct")
        r = recorder.get_evaluated()[0]
        assert r["actual_return_7d"] == pytest.approx(0.08)


# ===========================================================================
# _classify_outcome
# ===========================================================================

class TestClassifyOutcome:

    @pytest.mark.parametrize("rec", ["BUY", "STRONG BUY"])
    def test_bullish_positive_return_correct(self, rec):
        assert _classify_outcome(rec, 0.05) == "correct"

    @pytest.mark.parametrize("rec", ["BUY", "STRONG BUY"])
    def test_bullish_negative_return_incorrect(self, rec):
        assert _classify_outcome(rec, -0.05) == "incorrect"

    @pytest.mark.parametrize("rec", ["SELL", "AVOID"])
    def test_bearish_negative_return_correct(self, rec):
        assert _classify_outcome(rec, -0.05) == "correct"

    @pytest.mark.parametrize("rec", ["SELL", "AVOID"])
    def test_bearish_positive_return_incorrect(self, rec):
        assert _classify_outcome(rec, 0.05) == "incorrect"

    def test_hold_is_always_neutral(self):
        assert _classify_outcome("HOLD", 0.10) == "neutral"
        assert _classify_outcome("HOLD", -0.10) == "neutral"
        assert _classify_outcome("HOLD", 0.0) == "neutral"

    def test_none_return_is_unknown(self):
        assert _classify_outcome("BUY", None) == "unknown"

    def test_case_insensitive(self):
        assert _classify_outcome("buy", 0.05) == "correct"
        assert _classify_outcome("sell", -0.05) == "correct"


# ===========================================================================
# BacktestEvaluator — price-fetch mocking
# ===========================================================================

# Fixed recommendation date well in the past so get_pending always returns it.
_REC_DATE = datetime(2024, 1, 8)


def _ticker_mock(hist: pd.DataFrame) -> MagicMock:
    """Return a mock yfinance.Ticker whose history() returns hist."""
    mock = MagicMock()
    mock.history.return_value = hist
    return mock


class TestBacktestEvaluator:

    def _save_old(self, evaluator: BacktestEvaluator, recommendation: str = "BUY") -> dict:
        """Save a record dated _REC_DATE (old enough for get_pending) and return it."""
        evaluator.recorder.save(
            "AAPL", recommendation, 0.8, {}, timestamp=_REC_DATE,
        )
        return evaluator.recorder.get_pending(days_elapsed=7)[0]

    # ── evaluate_recommendation ──────────────────────────────────────

    def test_buy_correct_when_price_rises(self, evaluator):
        record = self._save_old(evaluator, "BUY")
        hist = _make_price_history(100.0, [0.01] * 34, _REC_DATE)
        with patch("yfinance.Ticker", return_value=_ticker_mock(hist)):
            result = evaluator.evaluate_recommendation(record)
        assert result["outcome"] == "correct"
        assert result["actual_return_7d"] > 0

    def test_buy_incorrect_when_price_falls(self, evaluator):
        record = self._save_old(evaluator, "BUY")
        hist = _make_price_history(100.0, [-0.01] * 34, _REC_DATE)
        with patch("yfinance.Ticker", return_value=_ticker_mock(hist)):
            result = evaluator.evaluate_recommendation(record)
        assert result["outcome"] == "incorrect"
        assert result["actual_return_7d"] < 0

    def test_sell_correct_when_price_falls(self, evaluator):
        record = self._save_old(evaluator, "SELL")
        hist = _make_price_history(100.0, [-0.01] * 34, _REC_DATE)
        with patch("yfinance.Ticker", return_value=_ticker_mock(hist)):
            result = evaluator.evaluate_recommendation(record)
        assert result["outcome"] == "correct"

    def test_hold_outcome_is_neutral(self, evaluator):
        record = self._save_old(evaluator, "HOLD")
        hist = _make_price_history(100.0, [0.01] * 34, _REC_DATE)
        with patch("yfinance.Ticker", return_value=_ticker_mock(hist)):
            result = evaluator.evaluate_recommendation(record)
        assert result["outcome"] == "neutral"

    def test_all_three_windows_computed(self, evaluator):
        record = self._save_old(evaluator)
        hist = _make_price_history(100.0, [0.01] * 34, _REC_DATE)
        with patch("yfinance.Ticker", return_value=_ticker_mock(hist)):
            result = evaluator.evaluate_recommendation(record)
        assert result["actual_return_1d"] is not None
        assert result["actual_return_7d"] is not None
        assert result["actual_return_30d"] is not None

    def test_evaluation_marks_record_in_db(self, evaluator):
        """After evaluation, record no longer appears in get_pending."""
        record = self._save_old(evaluator)
        hist = _make_price_history(100.0, [0.01] * 34, _REC_DATE)
        with patch("yfinance.Ticker", return_value=_ticker_mock(hist)):
            evaluator.evaluate_recommendation(record)
        assert len(evaluator.recorder.get_pending(days_elapsed=7)) == 0
        assert len(evaluator.recorder.get_evaluated()) == 1

    def test_empty_price_history_returns_error(self, evaluator):
        record = self._save_old(evaluator)
        empty_hist = pd.DataFrame(columns=["Close"])
        with patch("yfinance.Ticker", return_value=_ticker_mock(empty_hist)):
            result = evaluator.evaluate_recommendation(record)
        assert "error" in result

    def test_7d_return_approximate_value(self, evaluator):
        """7-day return should reflect ~7 days of 1% daily compounding."""
        record = self._save_old(evaluator)
        hist = _make_price_history(100.0, [0.01] * 34, _REC_DATE)
        with patch("yfinance.Ticker", return_value=_ticker_mock(hist)):
            result = evaluator.evaluate_recommendation(record)
        # 1.01^7 - 1 ≈ 7.2%
        assert result["actual_return_7d"] == pytest.approx(1.01 ** 7 - 1, abs=1e-6)

    # ── compute_metrics ──────────────────────────────────────────────

    def test_compute_metrics_empty(self, evaluator):
        metrics = evaluator.compute_metrics()
        assert metrics["total_evaluated"] == 0
        assert metrics["overall_accuracy"] is None
        assert metrics["accuracy_by_recommendation"] == {}
        assert metrics["avg_return_by_recommendation"] == {}

    def test_compute_metrics_two_correct_one_incorrect(self, evaluator):
        """2 correct + 1 incorrect → overall accuracy = 2/3."""
        rec = evaluator.recorder
        for outcome, ret in [("correct", 0.05), ("correct", 0.03), ("incorrect", -0.02)]:
            rid = rec.save("AAPL", "BUY", 0.8, {}, timestamp=_REC_DATE)
            rec.mark_evaluated(rid, actual_return=ret, outcome=outcome, actual_return_7d=ret)

        metrics = evaluator.compute_metrics()
        assert metrics["total_evaluated"] == 3
        assert metrics["overall_accuracy"] == pytest.approx(2 / 3)
        assert metrics["accuracy_by_recommendation"]["BUY"] == pytest.approx(2 / 3)

    def test_compute_metrics_hold_excluded_from_accuracy(self, evaluator):
        """HOLD (neutral) records do not affect the accuracy denominator."""
        rec = evaluator.recorder
        buy_id = rec.save("AAPL", "BUY", 0.8, {}, timestamp=_REC_DATE)
        hold_id = rec.save("AAPL", "HOLD", 0.5, {}, timestamp=_REC_DATE)
        rec.mark_evaluated(buy_id, actual_return=0.05, outcome="correct", actual_return_7d=0.05)
        rec.mark_evaluated(hold_id, actual_return=0.01, outcome="neutral", actual_return_7d=0.01)

        metrics = evaluator.compute_metrics()
        assert metrics["total_evaluated"] == 2
        assert metrics["overall_accuracy"] == pytest.approx(1.0)

    def test_compute_metrics_avg_return_by_recommendation(self, evaluator):
        rec = evaluator.recorder
        for ret in [0.04, 0.06]:
            rid = rec.save("AAPL", "BUY", 0.8, {}, timestamp=_REC_DATE)
            rec.mark_evaluated(rid, actual_return=ret, outcome="correct", actual_return_7d=ret)

        metrics = evaluator.compute_metrics()
        assert metrics["avg_return_by_recommendation"]["BUY"] == pytest.approx(0.05)

    def test_compute_metrics_multiple_recommendation_types(self, evaluator):
        rec = evaluator.recorder
        buy_id = rec.save("AAPL", "BUY", 0.8, {}, timestamp=_REC_DATE)
        sell_id = rec.save("AAPL", "SELL", 0.7, {}, timestamp=_REC_DATE)
        rec.mark_evaluated(buy_id, actual_return=0.05, outcome="correct", actual_return_7d=0.05)
        rec.mark_evaluated(sell_id, actual_return=-0.03, outcome="correct", actual_return_7d=-0.03)

        metrics = evaluator.compute_metrics()
        assert metrics["accuracy_by_recommendation"]["BUY"] == pytest.approx(1.0)
        assert metrics["accuracy_by_recommendation"]["SELL"] == pytest.approx(1.0)
        assert metrics["overall_accuracy"] == pytest.approx(1.0)


# ===========================================================================
# generate_backtest_report
# ===========================================================================

class TestGenerateBacktestReport:

    def _full_metrics(self) -> dict:
        return {
            "total_evaluated": 10,
            "overall_accuracy": 0.75,
            "accuracy_by_recommendation": {"BUY": 0.80, "SELL": 0.60, "HOLD": None},
            "avg_return_by_recommendation": {"BUY": 0.030, "SELL": -0.020, "HOLD": 0.005},
        }

    def test_report_contains_header(self):
        report = generate_backtest_report(self._full_metrics())
        assert "# Backtesting Report" in report

    def test_report_shows_overall_accuracy(self):
        report = generate_backtest_report(self._full_metrics())
        assert "75.0%" in report

    def test_report_lists_all_recommendation_types(self):
        report = generate_backtest_report(self._full_metrics())
        assert "BUY" in report
        assert "SELL" in report
        assert "HOLD" in report

    def test_report_canonical_row_order(self):
        metrics = {
            "total_evaluated": 3,
            "overall_accuracy": 0.8,
            "accuracy_by_recommendation": {"SELL": 0.7, "BUY": 0.9, "HOLD": None},
            "avg_return_by_recommendation": {"SELL": -0.01, "BUY": 0.03, "HOLD": 0.0},
        }
        report = generate_backtest_report(metrics)
        buy_pos = report.index("BUY")
        hold_pos = report.index("HOLD")
        sell_pos = report.index("SELL")
        # Canonical order: STRONG BUY → BUY → HOLD → SELL → AVOID
        assert buy_pos < hold_pos < sell_pos

    def test_report_empty_metrics_note(self):
        metrics = {
            "total_evaluated": 0,
            "overall_accuracy": None,
            "accuracy_by_recommendation": {},
            "avg_return_by_recommendation": {},
        }
        report = generate_backtest_report(metrics)
        assert "N/A" in report
        assert "No evaluated" in report

    def test_report_returns_string(self):
        report = generate_backtest_report(self._full_metrics())
        assert isinstance(report, str)
        assert report.endswith("\n")


# ===========================================================================
# Session-scoped setup
# ===========================================================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    os.environ.setdefault("TESTING", "true")
    os.environ.setdefault("LOG_LEVEL", "ERROR")
    yield
