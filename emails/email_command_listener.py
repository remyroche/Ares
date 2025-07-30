import logging
import imaplib
import email
import time
import subprocess
import os
import sys
import pandas as pd # Added import for pandas

# Ensure the source directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.state_manager import StateManager
from src.config import CONFIG
from emails.ares_mailer import AresMailer

class EmailCommandListener:
    """
    ## CHANGE: Fully updated and merged functionality.
    ## This class now provides a robust, object-oriented approach to listening for
    ## remote commands via email. It includes all previous commands (like KILL SWITCH,
    ## GIT PULL) and adds a new 'STATUS REPORT' command for enhanced remote monitoring.
    """
    def __init__(self, config, state_manager: StateManager, mailer: AresMailer):
        self.config = config.get("COMMAND_EMAIL_CONFIG", {})
        self.global_config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.state_manager = state_manager
        self.mailer = mailer
        
        self.imap_server = self.config.get("imap_server")
        self.imap_user = self.config.get("email_address")
        self.imap_password = self.config.get("app_password")
        self.authorized_senders = self.config.get("allowed_sender", [])
        
        # Unified command set
        self.allowed_commands = {
            "STATUS", "SHUTDOWN", "CANCEL_ALL", "PAUSE TRADING", "RESUME TRADING",
            "GIT PULL", "PROMOTE CHALLENGER", "RESTART ARES BOT", "ARES KILL SWITCH", "ARES DEACTIVATE"
        }

    def listen_for_commands(self):
        """Checks the inbox for new commands from authorized senders."""
        if not self.config.get('enabled', False) or not all([self.imap_server, self.imap_user, self.imap_password]):
            self.logger.error("IMAP configuration incomplete or disabled. Cannot listen for email commands.")
            return

        try:
            with imaplib.IMAP4_SSL(self.imap_server, self.config.get('imap_port')) as mail:
                mail.login(self.imap_user, self.imap_password)
                mail.select('inbox')
                
                # Search for unread emails from authorized senders
                for sender in self.authorized_senders:
                    status, data = mail.search(None, 'UNSEEN', 'FROM', sender)
                    if status != 'OK': continue

                    for num in data[0].split():
                        _, msg_data = mail.fetch(num, '(RFC822)')
                        msg = email.message_from_bytes(msg_data[0][1])
                        subject = msg['Subject'].strip().upper()

                        if subject not in self.allowed_commands:
                            self.logger.warning(f"Unauthorized command '{subject}' from {sender}. Ignoring.")
                        else:
                            self.logger.info(f"Processing command '{subject}' from authorized sender {sender}.")
                            self._process_command(subject, sender)
                        
                        # Mark email as read regardless of validity
                        mail.store(num, '+FLAGS', '\\Seen')

        except Exception as e:
            self.logger.error(f"Error checking email for commands: {e}", exc_info=True)

    def _process_command(self, command: str, sender: str):
        """Processes the validated command."""
        if command == "ARES KILL SWITCH":
            self.state_manager.activate_kill_switch(f"Remote command from {sender}")
        elif command == "ARES DEACTIVATE":
            self.state_manager.deactivate_kill_switch()
        elif command == "PAUSE TRADING":
            self.state_manager.pause_trading()
            self.mailer.send_alert("Trading Paused", f"Trading was paused by remote command from {sender}.")
        elif command == "RESUME TRADING":
            self.state_manager.resume_trading()
            self.mailer.send_alert("Trading Resumed", f"Trading was resumed by remote command from {sender}.")
        elif command == "GIT PULL":
            self.logger.info("Executing 'git pull'...")
            result = subprocess.run(["git", "pull"], capture_output=True, text=True)
            output = f"Git Pull Output:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            self.logger.info(output)
            self.mailer.send_alert("Git Pull Executed", output)
        elif command == "RESTART ARES BOT":
            self._write_flag_file(self.global_config.get('RESTART_FLAG_FILE'))
        elif command == "PROMOTE CHALLENGER":
            self._write_flag_file(self.global_config.get('PROMOTE_CHALLENGER_FLAG_FILE'))
        elif command == "STATUS":
            self._send_status_report(sender)
        # Add placeholders for other commands
        elif command in ["SHUTDOWN", "CANCEL_ALL"]:
            self.logger.warning(f"Command '{command}' received but not yet implemented.")

    def _write_flag_file(self, flag_file_path: str):
        """Writes a generic flag file to signal other processes."""
        if not flag_file_path:
            self.logger.error("Flag file path is not configured.")
            return
        try:
            with open(flag_file_path, 'w') as f:
                f.write("1")
            self.logger.info(f"Flag file '{flag_file_path}' created.")
            self.mailer.send_alert("Flag File Created", f"The flag file at {flag_file_path} was created by remote command.")
        except Exception as e:
            self.logger.error(f"Error writing flag file: {e}")

    def _send_status_report(self, sender: str):
        """Fetches the current bot status and emails it back to the sender."""
        self.logger.info("Generating status report...")
        try:
            position = self.state_manager.get_state("current_position", {})
            equity = self.state_manager.get_state("account_equity", "N/A")
            is_paused = self.state_manager.get_state("is_trading_paused", False)
            
            body = (
                f"Ares Status Report @ {pd.Timestamp.now(tz='UTC').isoformat()}\n"
                f"------------------------------------------\n"
                f"Trading Paused: {is_paused}\n"
                f"Account Equity: {equity}\n"
                f"Current Position:\n"
                f"  - Direction: {position.get('direction', 'None')}\n"
                f"  - Size: {position.get('size', 0)}\n"
                f"  - Entry Price: {position.get('entry_price', 'N/A')}\n"
                f"------------------------------------------"
            )
            self.mailer.send_alert("Ares Status Report", body)
        except Exception as e:
            self.logger.error(f"Failed to generate or send status report: {e}")


def main_listener_loop():
    """Main loop to run the email listener periodically."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("EmailListenerLoop")
    system_logger.info("--- Ares Email Command Listener Starting ---")
    
    state_manager = StateManager(CONFIG)
    mailer = AresMailer(CONFIG)
    listener = EmailCommandListener(CONFIG, state_manager, mailer)
    
    polling_interval = listener.config.get('polling_interval_seconds', 60)
    
    while True:
        try:
            listener.listen_for_commands()
            time.sleep(polling_interval)
        except KeyboardInterrupt:
            system_logger.info("Email listener stopped by user.")
            break
        except Exception as e:
            system_logger.error(f"Critical error in listener loop: {e}", exc_info=True)
            time.sleep(polling_interval * 2) # Wait longer after a critical error

if __name__ == "__main__":
    main_listener_loop()
