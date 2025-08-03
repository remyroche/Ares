import logging

try:
    import aiosmtplib

    AIOSMTP_AVAILABLE = True
except ImportError:
    aiosmtplib = None
    AIOSMTP_AVAILABLE = False
from email.mime.text import MIMEText

from src.config import settings


class AresMailer:
    """
    This class handles sending email alerts for critical system events.
    It now uses aiosmtp for non-blocking email sending.
    """

    def __init__(self, config):
        self.email_config = config.get("email_config", {})
        self.logger = logging.getLogger(self.__class__.__name__)
        self.smtp_server = self.email_config.get("smtp_server")
        self.smtp_port = self.email_config.get("smtp_port")
        self.smtp_user = self.email_config.get("sender_email")
        self.smtp_password = self.email_config.get("app_password")
        self.recipients = [self.email_config.get("recipient_email")]

    async def send_alert(self, subject: str, body: str):
        """Asynchronously sends an email alert."""
        if not all(
            [
                self.smtp_server,
                self.smtp_port,
                self.smtp_user,
                self.smtp_password,
                self.recipients,
            ],
        ):
            self.logger.error("Email configuration is incomplete. Cannot send alert.")
            return

        msg = MIMEText(body)
        msg["Subject"] = f"[ARES ALERT] {subject}"
        msg["From"] = self.smtp_user
        msg["To"] = ", ".join(self.recipients)

        if not AIOSMTP_AVAILABLE:
            self.logger.warning("aiosmtp not available. Email alerts disabled.")
            return

        try:
            self.logger.info(f"Sending alert: {subject}")
            async with aiosmtplib.SMTP(
                hostname=self.smtp_server,
                port=self.smtp_port,
                use_tls=True,
            ) as smtp:
                await smtp.login(self.smtp_user, self.smtp_password)
                await smtp.send_message(msg)
            self.logger.info("Alert sent successfully.")
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}", exc_info=True)


# Helper function for scripts that might not have an AresMailer instance
async def send_email(subject, body):
    """Standalone async function to send an email."""
    mailer = AresMailer(config=settings.dict())
    await mailer.send_alert(subject, body)
