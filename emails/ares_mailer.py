import logging
import smtplib
from email.mime.text import MIMEText

class AresMailer:
    """
    This class handles sending email alerts for critical system events,
    such as large drawdowns, errors, or kill switch activation.
    """
    def __init__(self, config):
        self.config = config.get("emails", {})
        self.logger = logging.getLogger(self.__class__.__name__)
        self.smtp_server = self.config.get("smtp_server")
        self.smtp_port = self.config.get("smtp_port")
        self.smtp_user = self.config.get("smtp_user")
        self.smtp_password = self.config.get("smtp_password")
        self.recipients = self.config.get("recipients", [])

    def send_alert(self, subject: str, body: str):
        if not all([self.smtp_server, self.smtp_port, self.smtp_user, self.smtp_password, self.recipients]):
            self.logger.error("Email configuration is incomplete. Cannot send alert.")
            return

        msg = MIMEText(body)
        msg['Subject'] = f"[ARES ALERT] {subject}"
        msg['From'] = self.smtp_user
        msg['To'] = ', '.join(self.recipients)

        try:
            self.logger.info(f"Sending alert: {subject}")
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as server:
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.smtp_user, self.recipients, msg.as_string())
            self.logger.info("Alert sent successfully.")
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}", exc_info=True)


