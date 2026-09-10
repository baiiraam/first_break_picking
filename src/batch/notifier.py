# file location: src/batch/notifier.py

"""
Email and Slack notification utilities.
"""

import json
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

import requests
from loguru import logger


def send_email_notification(subject: str, body: str, config: dict):
    """Send email notification."""
    if not config or not config.get("enabled", False):
        return

    try:
        msg = MIMEMultipart()
        msg["From"] = config["sender"]
        msg["To"] = config["recipient"]
        msg["Subject"] = subject

        msg.attach(MIMEText(body, "plain"))

        password = os.environ.get(config.get("password_env_var", "EMAIL_PASSWORD"))
        if not password:
            logger.warning("⚠️ Email password not found in environment")
            return

        server = smtplib.SMTP(config["smtp_server"], config["smtp_port"])
        server.starttls()
        server.login(config["sender"], password)
        server.send_message(msg)
        server.quit()
        logger.info(f"📧 Email sent to {config['recipient']}")
    except (smtplib.SMTPException, ConnectionError, OSError) as e:
        logger.warning(f"⚠️ Failed to send email: {e}")


def send_slack_notification(message: str, config: dict):
    """Send Slack notification."""
    if not config or not config.get("enabled", False):
        return

    webhook_url = os.environ.get(config.get("webhook_url_env_var", "SLACK_WEBHOOK_URL"))
    if not webhook_url:
        logger.warning("⚠️ Slack webhook URL not found in environment")
        return

    try:
        payload = {
            "channel": config.get("channel", "#ml-training"),
            "text": message,
            "username": "Batch Training Bot",
        }
        response = requests.post(webhook_url, json=payload)
        if response.status_code == 200:
            logger.info("📨 Slack notification sent")
        else:
            logger.warning(f"⚠️ Slack notification failed: {response.status_code}")
    except (smtplib.SMTPException, ConnectionError, OSError) as e:
        logger.warning(f"⚠️ Failed to send Slack notification: {e}")


class NotificationDispatcher:
    """
    Handles sending notifications for batch training results.
    """

    def __init__(self, monitoring_config: dict[str, Any] | None = None):
        """
        Initialize notification dispatcher.

        Args:
            monitoring_config: Configuration dictionary with email/slack settings
        """
        self.monitoring = monitoring_config or {}

    def send_failure_notification(
        self,
        summary: Any,
        errors: list[dict[str, Any]],
    ) -> None:
        """Send notification for failed datasets."""
        if not summary.failed_datasets:
            return

        body = self._format_failure_message(summary, errors)

        # ✅ Handle None/empty configs safely
        email_config = self.monitoring.get("email") if self.monitoring else None
        slack_config = self.monitoring.get("slack") if self.monitoring else None

        if email_config and email_config.get("enabled", False):
            send_email_notification(
                f"Batch Training Report - {len(summary.failed_datasets)} failures",
                body,
                email_config,
            )

        if slack_config and slack_config.get("enabled", False):
            send_slack_notification(body, slack_config)

    def _format_failure_message(
        self, summary: Any, errors: list[dict[str, Any]]
    ) -> str:
        """Format failure notification message."""
        return f"""
Batch Training Summary
====================
Time: {summary.timestamp}
Duration: {summary.total_duration_seconds / 60:.1f} minutes
Mode: {summary.mode}

Successful: {len(summary.successful_datasets)}/{summary.total_datasets}
  • {", ".join(summary.successful_datasets) if summary.successful_datasets else "None"}

Failed: {len(summary.failed_datasets)}/{summary.total_datasets}
  • {", ".join(summary.failed_datasets)}

Errors:
{json.dumps(errors, indent=2)}
"""
