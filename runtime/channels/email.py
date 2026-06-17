import asyncio
import logging
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)

class EmailChannel:
    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.username = os.getenv("SMTP_USERNAME")
        self.password = os.getenv("SMTP_PASSWORD")
        self.email_from = os.getenv("EMAIL_FROM")
        self.email_to = os.getenv("EMAIL_TO")

        if not self.username or not self.password:
            raise ValueError("Gmail SMTP credentials missing in .env")

    def _send_sync(self, subject: str, body_html: str):
        """Synchronous runner for secure Gmail SMTP via TLS."""
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.email_from
        msg["To"] = self.email_to

        part = MIMEText(body_html, "html")
        msg.attach(part)

        # Gmail requires explicit connection via Port 587 followed by .starttls()
        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.ehlo()  # Identify yourself to Gmail
            server.starttls()  # Encrypt the connection
            server.ehlo()  # Re-identify over secure link
            server.login(self.username, self.password)
            server.sendmail(self.email_from, self.email_to, msg.as_string())

    async def send_briefing(self, subject: str, content: str):
        """Offloads the network request to an async thread pool."""
        try:
            formatted_content = content.replace("\n", "<br>")
            html_body = f"""
            <html>
                <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #2C3E50; background-color: #F8F9FA; padding: 20px;">
                    <div style="max-width: 600px; margin: 0 auto; background: #FFFFFF; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #E2E8F0;">
                        <h2 style="color: #3182CE; margin-top: 0; font-size: 22px; display: flex; align-items: center;">⏳ KAIROS Daily Briefing</h2>
                        <hr style="border: 0; border-top: 1px solid #E2E8F0; margin: 20px 0;">
                        <div style="font-size: 15px; color: #4A5568;">
                            {formatted_content}
                        </div>
                    </div>
                </body>
            </html>
            """
            await asyncio.to_thread(self._send_sync, subject, html_body)
            logger.info("Gmail briefing sent successfully.")
        except Exception as e:
            logger.error(f"Gmail channel failed to send email: {e}")

    async def run(self):
        """Keeps the engine background task alive."""
        logger.info("Kairos Gmail engine channel loaded and listening.")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            logger.info("Gmail engine stopped.")

    def _verify_sync(self) -> bool:
        """Perform a rapid handshake with Gmail to verify credentials."""
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=5.0) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(self.username, self.password)
                return True
        except Exception as e:
            logger.warning(f"Gmail SMTP verification failed: {e}")
            return False

    async def verify_transport(self) -> bool:
        """Asynchronously checks if the email service is operational."""
        if not self.username or not self.password:
            logger.error("Gmail SMTP configuration missing from environment variables.")
            return False
        
        logger.info("Verifying Gmail SMTP connection status...")
        # Offload blocking socket connection to a worker thread
        is_ok = await asyncio.to_thread(self._verify_sync)
        if is_ok:
            logger.info("Gmail SMTP authentication successful. Email channel ready.")
        else:
            logger.warning("Email channel initialization warning: Check network or App Password.")
        return is_ok