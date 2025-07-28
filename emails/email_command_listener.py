# emails/email_command_listener.py
import imaplib
import email
import subprocess
import sys
import time
import os
import signal
from src.config import CONFIG
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

ALLOWED_COMMANDS = {
    "STATUS": lambda: get_bot_status(),
    "SHUTDOWN": lambda: shutdown_bot(),
    "CANCEL_ALL": lambda: cancel_all_orders(),
    "PAUSE TRADING",
    "RESUME TRADING",
    "GIT PULL",
    "PROMOTE CHALLENGER",
    "RESTART ARES BOT",
}


def write_flag_file(flag_file_path):
    """Writes a generic flag file."""
    try:
        with open(flag_file_path, 'w') as f:
            f.write("1")
        print(f"Flag file '{flag_file_path}' created.")
    except Exception as e:
        print(f"Error writing flag file: {e}")

def check_emails():
    """Checks the configured email inbox for commands."""
    command_email_config = CONFIG['COMMAND_EMAIL_CONFIG']
    if not command_email_config.get('enabled', False):
        return

    state_manager = StateManager()

    try:
        mail = imaplib.IMAP4_SSL(command_email_config['imap_server'], command_email_config['imap_port'])
        mail.login(command_email_config['email_address'], command_email_config['app_password'])
        mail.select('inbox')
        status, email_ids = mail.search(None, 'UNSEEN', 'FROM', command_email_config['allowed_sender'])
        
        email_id_list = email_ids[0].split()
        if not email_id_list:
            return

        for e_id in email_id_list:
            _, msg_data = mail.fetch(e_id, '(RFC822)')
            msg = email.message_from_bytes(msg_data[0][1])
            subject = msg['Subject'].upper()

    
            if not command in ALLOWED_COMMANDS:
                print(f"SECURITY WARNING: Received unauthorized command '{email_body}'. Ignoring.")
                continue

            print(f"Processing command email with subject: '{subject}'")

            if subject == "RESTART ARES BOT":
                write_flag_file(CONFIG['RESTART_FLAG_FILE'])
            elif subject == "PROMOTE CHALLENGER":
                write_flag_file(CONFIG['PROMOTE_CHALLENGER_FLAG_FILE'])
            elif subject == "GIT PULL":
                subprocess.run("git pull", shell=True)
            elif subject == "PAUSE TRADING":
                state_manager.pause_trading()
            elif subject == "RESUME TRADING":
                state_manager.resume_trading()
            else:
                print(f"Unknown command: '{subject}'")

            mail.store(e_id, '+FLAGS', '\\Seen')

    except Exception as e:
        print(f"An error occurred during email check: {e}")
    finally:
        if 'mail' in locals():
            mail.logout()

def main_listener():
    print("--- Ares Email Command Listener Starting ---")
    polling_interval = CONFIG['COMMAND_EMAIL_CONFIG'].get('polling_interval_seconds', 60)
    while True:
        check_emails()
        time.sleep(polling_interval)

if __name__ == "__main__":
    main_listener()
