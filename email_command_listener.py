import imaplib
import email
import subprocess
import sys
import time
import os
import signal # Import signal module for sending signals
from config import COMMAND_EMAIL_CONFIG, PIPELINE_PID_FILE, PIPELINE_SCRIPT_NAME # Import the new configuration

def get_pipeline_pid():
    """Reads the PID of the ares_pipeline.py from the PID file."""
    if os.path.exists(PIPELINE_PID_FILE):
        try:
            with open(PIPELINE_PID_FILE, 'r') as f:
                pid = int(f.read().strip())
            return pid
        except (ValueError, FileNotFoundError):
            return None
    return None

def stop_pipeline(pid):
    """Attempts to stop the ares_pipeline.py process."""
    if pid:
        try:
            print(f"Attempting to stop ares_pipeline.py with PID: {pid}")
            os.kill(pid, signal.SIGTERM) # Send a termination signal
            # Give it a moment to shut down gracefully
            time.sleep(5) 
            # Check if it's still running
            try:
                os.kill(pid, 0) # Check if process exists (sends no signal)
                print(f"Process {pid} is still running. Force killing...")
                os.kill(pid, signal.SIGKILL) # Force kill if still alive
            except OSError:
                print(f"Process {pid} stopped successfully.")
            return True
        except OSError as e:
            print(f"Could not stop process {pid}: {e}")
            return False
    print("No pipeline PID found to stop.")
    return False

def start_pipeline():
    """Starts a new instance of ares_pipeline.py."""
    try:
        print(f"Starting a new instance of {PIPELINE_SCRIPT_NAME}...")
        # Use sys.executable to ensure the correct python interpreter is used
        # Use Popen to run it in the background and not block the listener
        # stdout/stderr are redirected to /dev/null or log files to prevent clutter
        # You might want to redirect to actual log files for debugging in production
        subprocess.Popen([sys.executable, PIPELINE_SCRIPT_NAME], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL,
                         preexec_fn=os.setsid) # Detach from current process group
        print(f"{PIPELINE_SCRIPT_NAME} started in background.")
        return True
    except Exception as e:
        print(f"Error starting {PIPELINE_SCRIPT_NAME}: {e}")
        return False

def execute_command(command):
    """Executes a shell command and returns its output."""
    try:
        print(f"Executing command: {command}")
        
        if command == "restart_ares_bot":
            pipeline_pid = get_pipeline_pid()
            if pipeline_pid:
                stop_pipeline(pipeline_pid)
            else:
                print("No active ares_pipeline.py found via PID file. Attempting to start anyway.")
            start_pipeline()
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
    if not COMMAND_EMAIL_CONFIG.get('enabled', False):
        print("Email command listener is disabled in config.py. Skipping check.")
        return

    try:
        mail = imaplib.IMAP4_SSL(COMMAND_EMAIL_CONFIG['imap_server'], COMMAND_EMAIL_CONFIG['imap_port'])
        mail.login(COMMAND_EMAIL_CONFIG['email_address'], COMMAND_EMAIL_CONFIG['app_password'])
        mail.select('inbox')

        # Search for unread emails from the allowed sender
        status, email_ids = mail.search(None, 'UNSEEN', 'FROM', COMMAND_EMAIL_CONFIG['allowed_sender'])
        
        if status != 'OK':
            print(f"Error searching for emails: {status}")
            return

        email_id_list = email_ids[0].split()
        if not email_id_list:
            print("No new commands found.")
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
                # This will now stop and restart the ares_pipeline.py
                execute_command("restart_ares_bot") 
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
    print(f"Checking emails every {COMMAND_EMAIL_CONFIG.get('polling_interval_seconds', 60)} seconds.")
    print("WARNING: Ensure COMMAND_EMAIL_CONFIG is correctly set up for security.")
    print("WARNING: This listener attempts to manage ares_pipeline.py directly. Consider a process manager for production.")

    while True:
        check_emails()
        time.sleep(COMMAND_EMAIL_CONFIG.get('polling_interval_seconds', 60))

if __name__ == "__main__":
    main_listener()