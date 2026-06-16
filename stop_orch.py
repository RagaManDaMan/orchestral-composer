#!/usr/bin/env python3
"""Stop the Orchestral Composer. Run: ./stop_orch.py"""
import os
import sys
import signal
import time

PID_FILE = os.path.join(os.path.dirname(__file__), ".orch.pid")

def main():
    if not os.path.exists(PID_FILE):
        print("Not running (no PID file found).")
        sys.exit(0)

    try:
        pid = int(open(PID_FILE).read().strip())
    except ValueError:
        os.remove(PID_FILE)
        print("PID file was corrupt — removed.")
        sys.exit(0)

    try:
        os.kill(pid, 0)   # existence check
    except ProcessLookupError:
        os.remove(PID_FILE)
        print(f"Process {pid} was already gone.")
        sys.exit(0)

    print(f"Stopping Orchestral Composer (PID {pid})…")
    os.kill(pid, signal.SIGTERM)

    for _ in range(20):
        time.sleep(0.25)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
    else:
        print("Did not stop cleanly — sending SIGKILL.")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)
    print("Stopped.")

if __name__ == "__main__":
    main()
