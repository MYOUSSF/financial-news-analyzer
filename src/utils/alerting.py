"""
AlertDispatcher — routes RiskAgent alerts to webhook and/or email channels.

Channel selection by alert severity:
    CRITICAL → webhook + email
    HIGH     → webhook only
    MEDIUM / LOW → log only (no external dispatch)

Usage:
    dispatcher = AlertDispatcher()          # reads config from env vars
    dispatcher.dispatch(risk_result["alerts"], symbol="AAPL")
"""
import os
import smtplib
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

import requests
from loguru import logger


class AlertDispatcher:
    """
    Dispatches risk alerts to external channels (webhook and/or email).

    All configuration is read from environment variables at construction time;
    explicit constructor arguments override env vars when provided.

    Environment variables (see .env.example):
        ALERT_WEBHOOK_URL  — target URL for webhook POST requests
        SMTP_HOST          — SMTP server hostname
        SMTP_PORT          — SMTP server port (default: 587)
        SMTP_USER          — SMTP login username / sender address
        SMTP_PASSWORD      — SMTP login password
        ALERT_EMAIL_TO     — recipient address for alert emails
    """

    _MAX_WEBHOOK_ATTEMPTS = 3

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        email_to: Optional[str] = None,
    ):
        self.webhook_url = webhook_url or os.getenv("ALERT_WEBHOOK_URL")
        self.smtp_host = smtp_host or os.getenv("SMTP_HOST")
        self.smtp_port = int(smtp_port or os.getenv("SMTP_PORT", "587"))
        self.smtp_user = smtp_user or os.getenv("SMTP_USER")
        self.smtp_password = smtp_password or os.getenv("SMTP_PASSWORD")
        self.email_to = email_to or os.getenv("ALERT_EMAIL_TO")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_webhook(self, url: str, payload: Dict[str, Any]) -> bool:
        """
        POST ``payload`` as JSON to ``url``.

        Retries up to ``_MAX_WEBHOOK_ATTEMPTS`` times with exponential
        backoff (1 s, 2 s between attempts).

        Args:
            url: Target webhook URL.
            payload: Data to send as the JSON body.

        Returns:
            ``True`` on success, ``False`` after all attempts are exhausted.
        """
        for attempt in range(1, self._MAX_WEBHOOK_ATTEMPTS + 1):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=10,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                logger.info(
                    f"Webhook delivered to {url} "
                    f"(attempt {attempt}/{self._MAX_WEBHOOK_ATTEMPTS}, "
                    f"status={response.status_code})"
                )
                return True
            except Exception as exc:
                if attempt < self._MAX_WEBHOOK_ATTEMPTS:
                    delay = 2 ** (attempt - 1)  # 1 s, 2 s
                    logger.warning(
                        f"Webhook attempt {attempt}/{self._MAX_WEBHOOK_ATTEMPTS} "
                        f"failed: {exc}. Retrying in {delay}s…"
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Webhook failed after {self._MAX_WEBHOOK_ATTEMPTS} "
                        f"attempts ({url}): {exc}"
                    )
        return False

    def send_email(self, subject: str, body: str, to_address: Optional[str] = None) -> bool:
        """
        Send a plain-text email via STARTTLS SMTP.

        Returns ``False`` immediately (without raising) if SMTP credentials or
        the recipient address are not configured.

        Args:
            subject: Email subject line.
            body: Plain-text message body.
            to_address: Recipient address.  Falls back to ``ALERT_EMAIL_TO``
                        env var when not supplied.

        Returns:
            ``True`` on successful delivery, ``False`` otherwise.
        """
        recipient = to_address or self.email_to
        if not all([self.smtp_host, self.smtp_user, self.smtp_password, recipient]):
            logger.info(
                "Email alert skipped: SMTP not fully configured "
                "(need SMTP_HOST, SMTP_USER, SMTP_PASSWORD, ALERT_EMAIL_TO)"
            )
            return False

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.smtp_user
            msg["To"] = recipient
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.ehlo()
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)

            logger.info(f"Email alert sent to {recipient} — {subject!r}")
            return True
        except Exception as exc:
            logger.error(f"Email delivery failed: {exc}")
            return False

    def dispatch(self, alerts: List[Dict[str, Any]], symbol: str) -> None:
        """
        Examine each alert and route it to the appropriate channel(s).

        Severity routing:
            - ``CRITICAL`` → webhook **and** email
            - ``HIGH``     → webhook only
            - ``MEDIUM`` / ``LOW`` → log only

        Args:
            alerts: List of alert dicts from ``RiskAgent`` (each has at minimum
                    ``severity`` and ``message`` keys).
            symbol: Ticker symbol included in outbound payloads.
        """
        if not alerts:
            return

        for alert in alerts:
            severity = alert.get("severity", "LOW").upper()
            message = alert.get("message", "No message")

            payload: Dict[str, Any] = {
                "symbol": symbol,
                "severity": severity,
                "alert_type": alert.get("type", "RISK_ALERT"),
                "message": message,
                "timestamp": alert.get("timestamp", datetime.now().isoformat()),
            }

            if severity == "CRITICAL":
                logger.warning(f"CRITICAL alert for {symbol}: {message}")
                if self.webhook_url:
                    self.send_webhook(self.webhook_url, payload)
                self.send_email(
                    subject=f"CRITICAL Risk Alert: {symbol}",
                    body=message,
                    to_address=self.email_to,
                )

            elif severity == "HIGH":
                logger.warning(f"HIGH alert for {symbol}: {message}")
                if self.webhook_url:
                    self.send_webhook(self.webhook_url, payload)

            else:
                logger.info(f"Alert [{severity}] for {symbol}: {message}")
