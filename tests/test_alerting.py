"""
Unit tests for AlertDispatcher.

Run with:
    pytest tests/test_alerting.py -v
"""
import os
import sys
from datetime import datetime
from unittest.mock import MagicMock, Mock, call, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.alerting import AlertDispatcher


# ===========================================================================
# Fixtures
# ===========================================================================

def _dispatcher(**kwargs) -> AlertDispatcher:
    """Return a dispatcher with explicit config; env vars are isolated."""
    defaults = {
        "webhook_url": "https://hooks.example.com/alerts",
        "smtp_host": "smtp.example.com",
        "smtp_port": 587,
        "smtp_user": "alerts@example.com",
        "smtp_password": "secret",
        "email_to": "ops@example.com",
    }
    defaults.update(kwargs)
    return AlertDispatcher(**defaults)


def _unconfigured() -> AlertDispatcher:
    """Return a dispatcher with no channels configured."""
    return AlertDispatcher(
        webhook_url=None,
        smtp_host=None,
        smtp_port=587,
        smtp_user=None,
        smtp_password=None,
        email_to=None,
    )


def _alert(severity: str = "HIGH", message: str = "Risk detected") -> dict:
    return {
        "severity": severity,
        "type": "OVERALL_RISK",
        "message": message,
        "timestamp": datetime(2024, 1, 15, 12, 0, 0).isoformat(),
    }


def _mock_response(status_code: int = 200) -> Mock:
    resp = Mock()
    resp.status_code = status_code
    resp.raise_for_status = Mock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return resp


# ===========================================================================
# AlertDispatcher — initialisation
# ===========================================================================

class TestAlertDispatcherInit:

    def test_reads_webhook_url_from_env(self, monkeypatch):
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://env.example.com/hook")
        d = AlertDispatcher()
        assert d.webhook_url == "https://env.example.com/hook"

    def test_constructor_overrides_env(self, monkeypatch):
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://env.example.com/hook")
        d = AlertDispatcher(webhook_url="https://explicit.example.com/hook")
        assert d.webhook_url == "https://explicit.example.com/hook"

    def test_smtp_port_defaults_to_587(self):
        d = AlertDispatcher()
        assert d.smtp_port == 587

    def test_smtp_port_read_from_env(self, monkeypatch):
        monkeypatch.setenv("SMTP_PORT", "465")
        d = AlertDispatcher()
        assert d.smtp_port == 465

    def test_all_fields_none_when_not_configured(self, monkeypatch):
        for var in ("ALERT_WEBHOOK_URL", "SMTP_HOST", "SMTP_USER",
                    "SMTP_PASSWORD", "ALERT_EMAIL_TO"):
            monkeypatch.delenv(var, raising=False)
        d = AlertDispatcher()
        assert d.webhook_url is None
        assert d.smtp_host is None
        assert d.email_to is None


# ===========================================================================
# send_webhook
# ===========================================================================

class TestSendWebhook:

    def test_returns_true_on_200(self):
        d = _dispatcher()
        with patch("requests.post", return_value=_mock_response(200)) as mock_post:
            result = d.send_webhook("https://example.com/hook", {"event": "test"})
        assert result is True
        mock_post.assert_called_once()

    def test_payload_sent_as_json(self):
        d = _dispatcher()
        payload = {"symbol": "AAPL", "severity": "HIGH"}
        with patch("requests.post", return_value=_mock_response(200)) as mock_post:
            d.send_webhook("https://example.com/hook", payload)
        _, kwargs = mock_post.call_args
        assert kwargs["json"] == payload

    def test_content_type_header_set(self):
        d = _dispatcher()
        with patch("requests.post", return_value=_mock_response(200)) as mock_post:
            d.send_webhook("https://example.com/hook", {})
        _, kwargs = mock_post.call_args
        assert kwargs["headers"]["Content-Type"] == "application/json"

    def test_returns_true_on_first_attempt_success(self):
        d = _dispatcher()
        with patch("requests.post", return_value=_mock_response(200)), \
             patch("time.sleep") as mock_sleep:
            result = d.send_webhook("https://example.com/hook", {})
        assert result is True
        mock_sleep.assert_not_called()

    def test_retries_on_transient_failure(self):
        """Fails on attempt 1, succeeds on attempt 2."""
        d = _dispatcher()
        responses = [Exception("timeout"), _mock_response(200)]

        with patch("requests.post", side_effect=responses) as mock_post, \
             patch("time.sleep"):
            result = d.send_webhook("https://example.com/hook", {})

        assert result is True
        assert mock_post.call_count == 2

    def test_retries_three_times_then_returns_false(self):
        """Fails on all three attempts."""
        d = _dispatcher()
        with patch("requests.post", side_effect=Exception("connection refused")) as mock_post, \
             patch("time.sleep"):
            result = d.send_webhook("https://example.com/hook", {})

        assert result is False
        assert mock_post.call_count == 3

    def test_exponential_backoff_delays(self):
        """Sleep intervals follow 2^0, 2^1 (1 s, 2 s) between attempts."""
        d = _dispatcher()
        with patch("requests.post", side_effect=Exception("err")), \
             patch("time.sleep") as mock_sleep:
            d.send_webhook("https://example.com/hook", {})

        sleep_calls = [c.args[0] for c in mock_sleep.call_args_list]
        assert sleep_calls == [1, 2]

    def test_returns_false_on_http_error_after_retries(self):
        """HTTP 500 raises on raise_for_status(); should exhaust retries."""
        d = _dispatcher()
        with patch("requests.post", return_value=_mock_response(500)), \
             patch("time.sleep"):
            result = d.send_webhook("https://example.com/hook", {})
        assert result is False

    def test_second_attempt_succeeds_no_further_retries(self):
        """When attempt 2 succeeds, attempt 3 must not be made."""
        d = _dispatcher()
        responses = [Exception("err"), _mock_response(200)]
        with patch("requests.post", side_effect=responses) as mock_post, \
             patch("time.sleep"):
            result = d.send_webhook("https://example.com/hook", {})
        assert result is True
        assert mock_post.call_count == 2


# ===========================================================================
# send_email
# ===========================================================================

class TestSendEmail:

    def _smtp_mock(self):
        """Return a context-manager-compatible SMTP mock."""
        smtp_inst = MagicMock()
        smtp_cls = MagicMock()
        smtp_cls.return_value.__enter__ = Mock(return_value=smtp_inst)
        smtp_cls.return_value.__exit__ = Mock(return_value=False)
        return smtp_cls, smtp_inst

    def test_returns_false_when_smtp_not_configured(self):
        d = _unconfigured()
        result = d.send_email("Test", "Body", "to@example.com")
        assert result is False

    def test_returns_false_when_no_recipient(self):
        d = AlertDispatcher(
            smtp_host="smtp.example.com",
            smtp_user="user@example.com",
            smtp_password="pass",
            email_to=None,
        )
        result = d.send_email("Test", "Body", to_address=None)
        assert result is False

    def test_returns_true_on_successful_delivery(self):
        d = _dispatcher()
        smtp_cls, _ = self._smtp_mock()
        with patch("smtplib.SMTP", smtp_cls):
            result = d.send_email("Alert!", "Risk detected.", "ops@example.com")
        assert result is True

    def test_uses_starttls(self):
        d = _dispatcher()
        smtp_cls, smtp_inst = self._smtp_mock()
        with patch("smtplib.SMTP", smtp_cls):
            d.send_email("Alert!", "Body", "ops@example.com")
        smtp_inst.starttls.assert_called_once()

    def test_logs_in_with_credentials(self):
        d = _dispatcher()
        smtp_cls, smtp_inst = self._smtp_mock()
        with patch("smtplib.SMTP", smtp_cls):
            d.send_email("Alert!", "Body", "ops@example.com")
        smtp_inst.login.assert_called_once_with("alerts@example.com", "secret")

    def test_sends_message(self):
        d = _dispatcher()
        smtp_cls, smtp_inst = self._smtp_mock()
        with patch("smtplib.SMTP", smtp_cls):
            d.send_email("Alert!", "Body", "ops@example.com")
        smtp_inst.send_message.assert_called_once()

    def test_connects_to_configured_host_and_port(self):
        d = _dispatcher(smtp_host="mail.corp.com", smtp_port=465)
        smtp_cls, _ = self._smtp_mock()
        with patch("smtplib.SMTP", smtp_cls):
            d.send_email("Alert!", "Body", "ops@example.com")
        smtp_cls.assert_called_once_with("mail.corp.com", 465)

    def test_returns_false_on_smtp_error(self):
        d = _dispatcher()
        smtp_cls = MagicMock()
        smtp_cls.return_value.__enter__.side_effect = Exception("connection refused")
        smtp_cls.return_value.__exit__ = Mock(return_value=False)
        with patch("smtplib.SMTP", smtp_cls):
            result = d.send_email("Alert!", "Body", "ops@example.com")
        assert result is False

    def test_uses_email_to_when_to_address_omitted(self):
        d = _dispatcher(email_to="default@example.com")
        smtp_cls, smtp_inst = self._smtp_mock()
        with patch("smtplib.SMTP", smtp_cls):
            result = d.send_email("Alert!", "Body")  # no to_address
        assert result is True


# ===========================================================================
# dispatch — routing logic
# ===========================================================================

class TestDispatch:

    def test_critical_triggers_webhook_and_email(self):
        d = _dispatcher()
        with patch.object(d, "send_webhook", return_value=True) as mock_wh, \
             patch.object(d, "send_email", return_value=True) as mock_em:
            d.dispatch([_alert("CRITICAL")], "AAPL")
        mock_wh.assert_called_once()
        mock_em.assert_called_once()

    def test_high_triggers_webhook_only(self):
        d = _dispatcher()
        with patch.object(d, "send_webhook", return_value=True) as mock_wh, \
             patch.object(d, "send_email", return_value=True) as mock_em:
            d.dispatch([_alert("HIGH")], "AAPL")
        mock_wh.assert_called_once()
        mock_em.assert_not_called()

    def test_medium_triggers_neither_channel(self):
        d = _dispatcher()
        with patch.object(d, "send_webhook") as mock_wh, \
             patch.object(d, "send_email") as mock_em:
            d.dispatch([_alert("MEDIUM")], "AAPL")
        mock_wh.assert_not_called()
        mock_em.assert_not_called()

    def test_low_triggers_neither_channel(self):
        d = _dispatcher()
        with patch.object(d, "send_webhook") as mock_wh, \
             patch.object(d, "send_email") as mock_em:
            d.dispatch([_alert("LOW")], "AAPL")
        mock_wh.assert_not_called()
        mock_em.assert_not_called()

    def test_empty_alerts_list_is_noop(self):
        d = _dispatcher()
        with patch.object(d, "send_webhook") as mock_wh, \
             patch.object(d, "send_email") as mock_em:
            d.dispatch([], "AAPL")
        mock_wh.assert_not_called()
        mock_em.assert_not_called()

    def test_multiple_alerts_dispatched_individually(self):
        """Two HIGH alerts → two webhook calls."""
        d = _dispatcher()
        alerts = [_alert("HIGH", "Risk A"), _alert("HIGH", "Risk B")]
        with patch.object(d, "send_webhook", return_value=True) as mock_wh, \
             patch.object(d, "send_email") as mock_em:
            d.dispatch(alerts, "MSFT")
        assert mock_wh.call_count == 2
        mock_em.assert_not_called()

    def test_mixed_severities_routed_correctly(self):
        """CRITICAL + MEDIUM: webhook+email called once, log-only for the second."""
        d = _dispatcher()
        alerts = [_alert("CRITICAL", "Critical risk"), _alert("MEDIUM", "Minor risk")]
        with patch.object(d, "send_webhook", return_value=True) as mock_wh, \
             patch.object(d, "send_email", return_value=True) as mock_em:
            d.dispatch(alerts, "TSLA")
        assert mock_wh.call_count == 1
        assert mock_em.call_count == 1

    def test_webhook_payload_contains_symbol(self):
        d = _dispatcher()
        captured = []
        with patch.object(d, "send_webhook", side_effect=lambda url, payload: captured.append(payload) or True):
            d.dispatch([_alert("HIGH")], "GOOGL")
        assert captured[0]["symbol"] == "GOOGL"

    def test_webhook_payload_contains_severity(self):
        d = _dispatcher()
        captured = []
        with patch.object(d, "send_webhook", side_effect=lambda url, payload: captured.append(payload) or True):
            d.dispatch([_alert("HIGH", "Some risk")], "AAPL")
        assert captured[0]["severity"] == "HIGH"

    def test_no_webhook_call_when_url_not_configured(self):
        """Even for CRITICAL, webhook skipped when no URL is set."""
        d = _unconfigured()
        with patch.object(d, "send_webhook") as mock_wh, \
             patch.object(d, "send_email", return_value=False) as mock_em:
            d.dispatch([_alert("CRITICAL")], "AAPL")
        mock_wh.assert_not_called()
        mock_em.assert_called_once()  # email still attempted

    def test_case_insensitive_severity(self):
        """Severity matching is case-insensitive."""
        d = _dispatcher()
        alert = _alert("high")  # lowercase
        with patch.object(d, "send_webhook", return_value=True) as mock_wh, \
             patch.object(d, "send_email") as mock_em:
            d.dispatch([alert], "AAPL")
        mock_wh.assert_called_once()
        mock_em.assert_not_called()


# ===========================================================================
# analyze_stock integration — enable_alerts wiring
# ===========================================================================

class TestAlertWiringInChain:
    """Verify enable_alerts is threaded through analyze_stock → analyze_stock_async."""

    @pytest.fixture
    def chain(self):
        from src.chains.analysis_chain import FinancialAnalysisChain
        from unittest.mock import AsyncMock

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="ok")

        with patch.object(FinancialAnalysisChain, "_build_tools", return_value=[]), \
             patch.object(FinancialAnalysisChain, "_init_agents"), \
             patch.object(FinancialAnalysisChain, "_init_vector_store", return_value=None):
            c = FinancialAnalysisChain(llm=mock_llm)

        _summary = {
            "symbol": "AAPL", "recommendation": "BUY",
            "executive_summary": "Ok.", "key_positives": [],
            "key_negatives": [], "action_items": [],
            "scores": {"sentiment_score": 0.7, "risk_score": 0.3,
                       "composite_score": 0.7, "confidence": 0.8, "confidence_label": "HIGH"},
            "confidence": 0.8, "confidence_label": "HIGH",
            "analysis_date": "2024-01-01T00:00:00", "period_days": 7,
            "full_report": "", "metadata": {},
        }
        _risk_with_alert = {
            "symbol": "AAPL", "overall_risk_score": 0.8, "risk_level": "HIGH",
            "identified_risks": [],
            "alerts": [_alert("HIGH", "High overall risk")],
            "recommendations": [],
        }
        _risk_no_alert = {
            "symbol": "AAPL", "overall_risk_score": 0.3, "risk_level": "LOW",
            "identified_risks": [], "alerts": [], "recommendations": [],
        }

        c.research_agent = MagicMock()
        c.sentiment_agent = MagicMock()
        c.risk_agent = MagicMock()
        c.summary_agent = MagicMock()
        c.vector_store = None

        c.research_agent.aexecute = AsyncMock(return_value={
            "symbol": "AAPL", "findings": "News.", "sources_used": [], "period_days": 7,
        })
        c.sentiment_agent.aexecute = AsyncMock(return_value={
            "symbol": "AAPL", "overall_sentiment": "POSITIVE", "sentiment_score": 0.7,
            "confidence": 0.8, "text_count": 1, "llm_analysis": {},
        })
        c.risk_agent.aexecute = AsyncMock(return_value=_risk_with_alert)
        c.summary_agent.aexecute = AsyncMock(return_value=_summary)
        c._risk_with_alert = _risk_with_alert
        c._risk_no_alert = _risk_no_alert
        return c

    def test_dispatch_called_when_enable_alerts_true(self, chain):
        with patch("src.utils.alerting.AlertDispatcher.dispatch") as mock_dispatch:
            chain.analyze_stock("AAPL", enable_alerts=True)
        mock_dispatch.assert_called_once()

    def test_dispatch_not_called_when_enable_alerts_false(self, chain):
        with patch("src.utils.alerting.AlertDispatcher.dispatch") as mock_dispatch:
            chain.analyze_stock("AAPL", enable_alerts=False)
        mock_dispatch.assert_not_called()

    def test_dispatch_receives_correct_symbol(self, chain):
        captured = {}
        def _capture(alerts, symbol):
            captured["symbol"] = symbol
        with patch("src.utils.alerting.AlertDispatcher.dispatch", side_effect=_capture):
            chain.analyze_stock("AAPL", enable_alerts=True)
        assert captured["symbol"] == "AAPL"

    def test_dispatch_not_called_when_no_alerts(self, chain):
        from unittest.mock import AsyncMock
        chain.risk_agent.aexecute = AsyncMock(return_value=chain._risk_no_alert)
        with patch("src.utils.alerting.AlertDispatcher.dispatch") as mock_dispatch:
            chain.analyze_stock("AAPL", enable_alerts=True)
        mock_dispatch.assert_not_called()

    def test_sync_wrapper_passes_enable_alerts(self, chain):
        """analyze_stock passes enable_alerts through to analyze_stock_async."""
        from unittest.mock import AsyncMock
        expected = {"symbol": "AAPL", "recommendation": "HOLD", "_elapsed_seconds": 0.5}
        with patch.object(
            chain, "analyze_stock_async", new_callable=AsyncMock, return_value=expected
        ) as mock_async:
            chain.analyze_stock("AAPL", days_back=3, enable_alerts=True)
        _, kwargs = mock_async.call_args
        assert kwargs["enable_alerts"] is True


# ===========================================================================
# Session-scoped setup
# ===========================================================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    os.environ.setdefault("TESTING", "true")
    os.environ.setdefault("LOG_LEVEL", "ERROR")
    # Prevent real env vars from leaking into tests
    for var in ("ALERT_WEBHOOK_URL", "SMTP_HOST", "SMTP_USER",
                "SMTP_PASSWORD", "ALERT_EMAIL_TO"):
        os.environ.pop(var, None)
    yield
