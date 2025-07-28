import imaplib
import email
import subprocess
import sys
import time
import os
import signal # Import signal module for sending signals
# Import the main CONFIG dictionary
from config import CONFIG

def write_restart_flag():
    """Writes a flag file to signal main.py to restart the pipeline."""
    # Access RESTART_FLAG_FILE from the main CONFIG dictionary
    restart_flag_file = CONFIG['RESTART_FLAG_FILE']
    try:
        with open(restart_flag_file, 'w') as f:
            f.write("RESTART")
        print(f"Restart flag file '{restart_flag_file}' created.")
    except Exception as e:
        print(f"Error writing restart flag file: {e}")

def execute_command(command):
    """Executes a shell command or signals for pipeline restart."""
    try:
        print(f"Executing command: {command}")
        
        if command == "signal_restart_ares_bot":
            write_restart_flag()
            print("Signaled main orchestrator to restart ares_pipeline.py.")
            return True # Listener continues running
        
        # For other commands like 'git pull'
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        print(f"Command '{command}' output:\n{result.stdout}")
        if result.stderr:
            print(f"Command '{command}' error:\n{result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command '{command}': {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

def check_emails():
    """Checks the configured email inbox for commands."""
    # Access COMMAND_EMAIL_CONFIG from the main CONFIG dictionary
    command_email_config = CONFIG['COMMAND_EMAIL_CONFIG']

    if not command_email_config.get('enabled', False):
        print("Email command listener is disabled in config.py. Skipping check.")
        return

    try:
        mail = imaplib.IMAP4_SSL(command_email_config['imap_server'], command_email_config['imap_port'])
        mail.login(command_email_config['email_address'], command_email_config['app_password'])
        mail.select('inbox')

        # Search for unread emails from the allowed sender
        status, email_ids = mail.search(None, 'UNSEEN', 'FROM', command_email_config['allowed_sender'])
        
        if status != 'OK':
            print(f"Error searching for emails: {status}")
            return

        email_id_list = email_ids[0].split()
        if not email_id_list:
            return

        print(f"Found {len(email_id_list)} new command emails.")

        for e_id in email_id_list:
            status, msg_data = mail.fetch(e_id, '(RFC822)')
            if status != 'OK':
                print(f"Error fetching email {e_id}: {status}")
                continue

            msg = email.message_from_bytes(msg_data[0][1])
            subject = msg['Subject']
            sender = msg['From']

            print(f"Processing email from '{sender}' with subject: '{subject}'")

            # Basic command parsing from subject
            if "GIT PULL" in subject.upper():
                print("Received 'GIT PULL' command.")
                # Assumes the script is run from the root of the Git repository
                success = execute_command("git pull")
                if success:
                    print("Git pull successful.")
                else:
                    print("Git pull failed.")
            elif "RESTART ARES BOT" in subject.upper():
                print("Received 'RESTART ARES BOT' command.")
                # This will now write a flag for the main orchestrator to handle the restart
                execute_command("signal_restart_ares_bot") 
            else:
                print(f"Unknown command in subject: '{subject}'")

            # Mark email as read after processing
            mail.store(e_id, '+FLAGS', '\\Seen')
            print(f"Email {e_id} marked as read.")

    except imaplib.IMAP4.error as e:
        print(f"IMAP Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during email check: {e}")
    finally:
        try:
            if 'mail' in locals() and mail.state == 'SELECTED':
                mail.close()
            if 'mail' in locals():
                mail.logout()
        except Exception as e:
            print(f"Error during email logout: {e}")


def main_listener():
    """Main loop for the email command listener."""
    print("--- Ares Email Command Listener Starting ---")
    # Access polling_interval_seconds from the main CONFIG dictionary
    polling_interval = CONFIG['COMMAND_EMAIL_CONFIG'].get('polling_interval_seconds', 120)
    print(f"Checking emails every {polling_interval} seconds.")
    print("WARNING: Ensure COMMAND_EMAIL_CONFIG is correctly set up for security.")
    print("This listener now signals the main orchestrator for pipeline restarts.")

    while True:
        check_emails()
        time.sleep(polling_interval)

if __name__ == "__main__":
    main_listener()
