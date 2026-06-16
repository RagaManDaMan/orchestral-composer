#!/usr/bin/env python3
"""Start the Orchestral Composer. Run: ./start_orch.py"""
import os
import sys
import subprocess
import signal
import time

PID_FILE = os.path.join(os.path.dirname(__file__), ".orch.pid")
APP      = os.path.join(os.path.dirname(__file__), "app.py")
URL      = "http://127.0.0.1:7861"

def _already_running():
    if not os.path.exists(PID_FILE):
        return None
    try:
        pid = int(open(PID_FILE).read().strip())
        os.kill(pid, 0)   # signal 0 = existence check only
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        os.remove(PID_FILE)
        return None

def main():
    pid = _already_running()
    if pid:
        print(f"Already running (PID {pid})  →  {URL}")
        print("Run ./stop_orch.py first if you want to restart.")
        sys.exit(0)

    print("Starting Orchestral Composer…")
    proc = subprocess.Popen(
        [sys.executable, APP],
        stdout=None, stderr=None,   # inherit terminal — you see logs live
        start_new_session=True,     # detach from this shell's signal group
    )
    with open(PID_FILE, "w") as f:
        f.write(str(proc.pid))

    # Wait up to 8 s for the server to start
    import urllib.request
    for _ in range(16):
        time.sleep(0.5)
        try:
            urllib.request.urlopen(URL, timeout=1)
            break
        except Exception:
            pass

    print(f"Running (PID {proc.pid})  →  {URL}")

if __name__ == "__main__":
    main()
