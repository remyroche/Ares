import subprocess
import time
import os
import sys
import signal
from config import PIPELINE_SCRIPT_NAME, EMAIL_CONFIG, COMMAND_EMAIL_CONFIG, PIPELINE_PID_FILE, RESTART_FLAG_FILE

def get_process_pid(pid_file):
    """Reads the PID from a specified PID file."""
    if os.path.exists(pid_file):
        try:
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())
            return pid
        except (ValueError, FileNotFoundError):
            return None
    return None

def is_process_running(pid):
    """Checks if a process with the given PID is currently running."""
    if pid is None:
        return False
    try:
        os.kill(pid, 0) # Sends no signal, just checks if process exists
        return True
    except OSError:
        return False

def terminate_process(pid, name="process"):
    """Attempts to gracefully terminate a process by PID, then force kills if necessary."""
    if pid is None:
        print(f"No PID provided to terminate {name}.")
        return False
    
    print(f"Attempting to terminate {name} (PID: {pid})...")
    try:
        os.kill(pid, signal.SIGTERM) # Send SIGTERM for graceful shutdown
        time.sleep(5) # Give it some time to shut down
        if is_process_running(pid):
            print(f"{name} (PID: {pid}) did not terminate gracefully. Force killing...")
            os.kill(pid, signal.SIGKILL) # Force kill
            time.sleep(2) # Give it a moment after force kill
        if not is_process_running(pid):
            print(f"{name} (PID: {pid}) terminated.")
            return True
        else:
            print(f"Failed to terminate {name} (PID: {pid}).")
            return False
    except ProcessLookupError:
        print(f"{name} (PID: {pid}) was not found, likely already terminated.")
        return True # Already gone, so consider it terminated
    except Exception as e:
        print(f"Error terminating {name} (PID: {pid}): {e}")
        return False

def start_process(script_name, name="process"): # Removed log_prefix parameter
    """Starts a Python script as a subprocess, redirecting output to the console."""
    print(f"Starting {name} ({script_name})...")
    try:
        # Redirect stdout/stderr to sys.stdout and sys.stderr to print to the console
        process = subprocess.Popen([sys.executable, script_name], 
                                   stdout=sys.stdout, # Redirect to console
                                   stderr=sys.stderr, # Redirect to console
                                   preexec_fn=os.setsid) # Detach from current process group
        print(f"{name} ({script_name}) started with PID {process.pid}.")
        return process
    except Exception as e:
        print(f"Error starting {name} ({script_name}): {e}")
        return None

def main():
    print("--- Ares Main Orchestrator Starting ---")

    # Clean up any old PID or flag files
    if os.path.exists(PIPELINE_PID_FILE):
        os.remove(PIPELINE_PID_FILE)
    if os.path.exists(RESTART_FLAG_FILE):
        os.remove(RESTART_FLAG_FILE)

    # Start the Email Command Listener
    listener_process = start_process("email_command_listener.py", "Email Listener") # No log_prefix
    if listener_process is None:
        print("Failed to start Email Listener. Exiting.")
        sys.exit(1)

    # Start the Ares Pipeline
    pipeline_process = start_process(PIPELINE_SCRIPT_NAME, "Ares Pipeline") # No log_prefix
    if pipeline_process is None:
        print("Failed to start Ares Pipeline. Exiting.")
        # Attempt to terminate listener if pipeline failed to start
        terminate_process(listener_process.pid, "Email Listener")
        sys.exit(1)

    try:
        while True:
            # Check if the pipeline process is still running
            if not is_process_running(pipeline_process.pid):
                print(f"Ares Pipeline (PID: {pipeline_process.pid}) has stopped unexpectedly.")
                # Attempt to restart it
                print("Attempting to restart Ares Pipeline...")
                pipeline_process = start_process(PIPELINE_SCRIPT_NAME, "Ares Pipeline")
                if pipeline_process is None:
                    print("Failed to restart Ares Pipeline. Exiting orchestrator.")
                    break # Exit the main loop

            # Check for restart flag from email listener
            if os.path.exists(RESTART_FLAG_FILE):
                print(f"'{RESTART_FLAG_FILE}' detected. Initiating pipeline restart.")
                # Terminate current pipeline process
                current_pipeline_pid = get_process_pid(PIPELINE_PID_FILE) # Get PID from file for robustness
                if current_pipeline_pid and is_process_running(current_pipeline_pid):
                    terminate_process(current_pipeline_pid, "Ares Pipeline")
                else:
                    print("No active pipeline process found via PID file to stop. Proceeding with restart.")

                # Remove the flag file immediately to avoid re-triggering
                try:
                    os.remove(RESTART_FLAG_FILE)
                    print(f"'{RESTART_FLAG_FILE}' removed.")
                except Exception as e:
                    print(f"Error removing restart flag file: {e}")

                # Start a new pipeline process
                pipeline_process = start_process(PIPELINE_SCRIPT_NAME, "Ares Pipeline")
                if pipeline_process is None:
                    print("Failed to restart Ares Pipeline after flag. Exiting orchestrator.")
                    break # Exit the main loop
                
            time.sleep(10) # Orchestrator checks every 10 seconds

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Shutting down orchestrator.")
    except Exception as e:
        print(f"An unexpected error occurred in orchestrator: {e}")
    finally:
        print("--- Orchestrator Shutting Down ---")
        # Terminate child processes
        if listener_process and is_process_running(listener_process.pid):
            terminate_process(listener_process.pid, "Email Listener")
        if pipeline_process and is_process_running(pipeline_process.pid):
            terminate_process(pipeline_process.pid, "Ares Pipeline")
        
        # Clean up PID and flag files
        if os.path.exists(PIPELINE_PID_FILE):
            os.remove(PIPELINE_PID_FILE)
        if os.path.exists(RESTART_FLAG_FILE):
            os.remove(RESTART_FLAG_FILE)
        print("All processes terminated and cleanup complete.")

if __name__ == "__main__":
    main()
