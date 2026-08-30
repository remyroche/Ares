#!/usr/bin/env python3
"""
scripts/supervisor.py

A strict Supervisor script to wrap other commands.
It enforces:
1.  Exit Code 0.
2.  No "Forbidden Patterns" in logs (Warnings, NaNs, Errors that are swallowed).
3.  Timeout constraints (prevents hanging).
4.  Retries (optional).

Usage:
    python scripts/supervisor.py --timeout 3600 --strict -- python scripts/my_script.py arg1
"""

import sys
import subprocess
import time
import re
import argparse
import os
import signal

# Patterns that indicate a failure, even if the script didn't crash.
# Adjust these based on your specific "obvious logical/financial errors".
FORBIDDEN_PATTERNS = [
    r"Traceback \(most recent call last\)",
    r"SettingWithCopyWarning",
    r"FutureWarning",  # Often indicates upcoming breakage
    r"UserWarning: .*Lookahead",  # Catch specific logical warnings
    r"RuntimeWarning: invalid value encountered",  # NaNs/Infs
    r"Error:",
    r"CRITICAL:",
    r"nan",
    r"inf",
    r"Empty DataFrame",
]

def clean_output(line):
    return line.decode('utf-8', errors='replace').strip()

def run_supervised(command, timeout, strict_patterns):
    print(f"[Supervisor] Starting: {' '.join(command)}")
    print(f"[Supervisor] Timeout: {timeout}s")
    print(f"[Supervisor] Strict Patterns: {len(strict_patterns)}")

    # Set environment to force python warnings to stderr
    env = os.environ.copy()
    if strict_patterns:
         env["PYTHONWARNINGS"] = "always" # Ensure warnings are printed so we can catch them

    start_time = time.time()

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env
    )

    stdout_lines = []
    stderr_lines = []

    # Simple polling loop to avoid blocking indefinitely
    while True:
        if process.poll() is not None:
            break

        if time.time() - start_time > timeout:
            print(f"[Supervisor] ❌ TIMEOUT ({timeout}s) exceeded. Killing process.")
            process.kill()
            return False, stdout_lines, stderr_lines + ["TIMEOUT_EXCEEDED"]

        time.sleep(1)

    # Capture output
    out, err = process.communicate()
    stdout_lines = out.decode('utf-8', errors='replace').splitlines()
    stderr_lines = err.decode('utf-8', errors='replace').splitlines()

    # 1. Check Exit Code
    if process.returncode != 0:
        print(f"[Supervisor] ❌ Process exited with code {process.returncode}")
        # Print stderr for debugging
        for line in stderr_lines:
            print(f"  [STDERR] {line}")
        return False, stdout_lines, stderr_lines

    # 2. Check Strict Patterns in Output
    detected_errors = []
    combined_log = stdout_lines + stderr_lines

    for line in combined_log:
        for pattern in strict_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                detected_errors.append((pattern, line))

    if detected_errors:
        print(f"[Supervisor] ❌ Process finished (Exit 0) but violated strict rules:")
        for pat, line in detected_errors:
            print(f"  [VIOLATION] Pattern '{pat}' found in: {line.strip()}")
        return False, stdout_lines, stderr_lines

    print(f"[Supervisor] ✅ Success. No errors or strict violations detected.")
    return True, stdout_lines, stderr_lines

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strict Supervisor for Agent Tasks")
    parser.add_argument("--timeout", type=int, default=1800, help="Timeout in seconds")
    parser.add_argument("--strict", action="store_true", help="Enable strict pattern matching for warnings/NaNs")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="The command to run")

    args = parser.parse_args()

    if not args.command:
        print("Usage: python supervisor.py --strict -- python myscript.py")
        sys.exit(1)

    # Strip the leading '--' if present in the remainder
    cmd = args.command
    if cmd[0] == "--":
        cmd = cmd[1:]

    patterns = FORBIDDEN_PATTERNS if args.strict else []

    success, out, err = run_supervised(cmd, args.timeout, patterns)

    if not success:
        sys.exit(1) # Fail so the Agent knows it failed
    sys.exit(0)
