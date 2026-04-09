import subprocess
import time
import sys
import os
from datetime import datetime

def log(msg):
    t = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{t}] {msg}")
    with open("monitor.log", "a") as f:
        f.write(f"[{t}] {msg}\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 monitor_pipeline.py <command>")
        return

    cmd = " ".join(sys.argv[1:])
    log(f"Starting monitor for command: {cmd}")
    
    # Ensure no stale alert file
    if os.path.exists("FAILURE_ALERT.txt"):
        os.remove("FAILURE_ALERT.txt")

    try:
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy()
        )

        log(f"Process started with PID: {process.pid}")

        with open("pipeline_output.log", "w") as out:
            for line in iter(process.stdout.readline, ""):
                sys.stdout.write(line)
                out.write(line)
                out.flush()

        process.wait()
        
        if process.returncode == 0:
            log("Pipeline completed successfully.")
        else:
            log(f"Pipeline CRASHED with return code {process.returncode}")
            with open("FAILURE_ALERT.txt", "w") as f:
                f.write(f"CRASH detected at {datetime.utcnow()}\n")
                f.write(f"Return code: {process.returncode}\n")
                f.write("Check pipeline_output.log for traceback.\n")
            
    except Exception as e:
        log(f"Monitor error: {e}")
        with open("FAILURE_ALERT.txt", "w") as f:
            f.write(f"Monitor encountered an error: {e}\n")

if __name__ == "__main__":
    main()
