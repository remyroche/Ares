# emails/email_command_listener.py
import imaplib
import email
import subprocess
import sys
import time
import os
import signal
from src.config import CONFIG
from src.utils.state_manager import StateManager # Import StateManager for pause/resume functionality
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Define allowed commands as a set of strings for efficient lookup.
# The lambda functions were not actually used in the processing logic,
# so they are removed for cleaner syntax.
ALLOWED_COMMANDS = {
    "STATUS",
    "SHUTDOWN",
    "CANCEL_ALL",
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

    state_manager = StateManager() # Initialize StateManager

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

            # Check if the subject is an allowed command
            if subject not in ALLOWED_COMMANDS: # Corrected variable name to 'subject'
                print(f"SECURITY WARNING: Received unauthorized command '{subject}'. Ignoring.") # Corrected variable name
                mail.store(e_id, '+FLAGS', '\\Seen') # Mark as seen even if unauthorized
                continue

            print(f"Processing command email with subject: '{subject}'")

            # Implement specific command actions
            if subject == "RESTART ARES BOT":
                write_flag_file(CONFIG['RESTART_FLAG_FILE'])
                print("RESTART ARES BOT command processed.")
            elif subject == "PROMOTE CHALLENGER":
                write_flag_file(CONFIG['PROMOTE_CHALLENGER_FLAG_FILE'])
                print("PROMOTE CHALLENGER command processed.")
            elif subject == "GIT PULL":
                print("Executing 'git pull'...")
                result = subprocess.run(["git", "pull"], capture_output=True, text=True)
                print(f"Git Pull Output:\n{result.stdout}\n{result.stderr}")
                if result.returncode != 0:
                    print("Git Pull failed.")
            elif subject == "PAUSE TRADING":
                state_manager.pause_trading()
                print("PAUSE TRADING command processed.")
            elif subject == "RESUME TRADING":
                state_manager.resume_trading()
                print("RESUME TRADING command processed.")
            elif subject == "STATUS":
                # In a real scenario, you'd fetch and send bot status via email
                print("STATUS command received. (Status reporting not yet implemented)")
            elif subject == "SHUTDOWN":
                # In a real scenario, you'd trigger a graceful shutdown of the main bot process
                print("SHUTDOWN command received. (Shutdown trigger not yet implemented)")
            elif subject == "CANCEL_ALL":
                # In a real scenario, you'd trigger cancellation of all open orders
                print("CANCEL_ALL command received. (Order cancellation not yet implemented)")
            else:
                print(f"Unknown command: '{subject}' (should not happen due to ALLOWED_COMMANDS check)")

            mail.store(e_id, '+FLAGS', '\\Seen') # Mark email as seen after processing

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
