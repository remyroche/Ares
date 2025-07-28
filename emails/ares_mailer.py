import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from config import CONFIG

def send_email(subject, body):
    """Sends an email with the specified subject and body."""
    # Access EMAIL_CONFIG from the main CONFIG dictionary
    email_config = CONFIG['EMAIL_CONFIG']

    if not email_config.get('enabled', False):
        print("Email notifications are disabled in config.py. Skipping.")
        return

    # Basic validation to ensure all required fields are present
    required_keys = ['smtp_server', 'smtp_port', 'sender_email', 'app_password', 'recipient_email']
    if not all(email_config.get(key) for key in required_keys):
        print("Email configuration is incomplete in config.py. Skipping email.")
        return

    print(f"Sending email notification to {email_config['recipient_email']}...")

    try:
        # Set up the message object
        message = MIMEMultipart()
        message['From'] = email_config['sender_email']
        message['To'] = email_config['recipient_email']
        message['Subject'] = subject
        
        # Attach the body as HTML. Using <pre> tags helps preserve the formatting
        # of the console output, making the email much more readable.
        html_body = f"<html><body><pre style='font-family: monospace; font-size: 12px;'>{body}</pre></body></html>"
        message.attach(MIMEText(html_body, 'html'))

        # Connect to the SMTP server and send the email
        with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
            server.starttls()  # Secure the connection
            server.login(email_config['sender_email'], email_config['app_password'])
            text = message.as_string()
            server.sendmail(email_config['sender_email'], email_config['recipient_email'], text)
        
        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email. Error: {e}")

