#!/usr/bin/env python3
"""
safe_run.py - Enforced circuit breaker + logging wrapper for agent-executed commands.

Why this exists:
Rules written in .clinerules are advisory - the model can misjudge "same error"
or lose track under a long context. This script enforces limits in code instead,
so the agent gets an unambiguous, unmissable signal to stop.

Usage:
  Foreground command with retry circuit breaker:
    python safe_run.py run -- <command and args>

  Background job with enforced max runtime + PID tracking:
    python safe_run.py bg --timeout 1800 --name mytrain -- <command and args>

  Check/kill any background jobs that exceeded their timeout:
    python safe_run.py cleanup
"""

import subprocess, sys, os, json, time, hashlib, signal, argparse
from pathlib import Path

STATE_DIR = Path(".cline_state")
STATE_DIR.mkdir(exist_ok=True)
RETRY_FILE = STATE_DIR / "retry_state.json"
LOG_FILE = STATE_DIR / "run_log.csv"
BG_FILE = STATE_DIR / "bg_processes.json"

MAX_RETRIES = 3
OOM_MARKERS = [
    "cuda out of memory",
    "cuda error: out of memory",
    "runtimeerror: cuda out of memory",
]


def log_run(cmd, exit_code, duration, retry_count, note=""):
    is_new = not LOG_FILE.exists()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        if is_new:
            f.write("timestamp,command,exit_code,duration_s,retry_count,note\n")
        safe_cmd = cmd.replace('"', "'")
        f.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")},"{safe_cmd}",{exit_code},{duration:.1f},{retry_count},"{note}"\n')


def error_hash(text, lines=15):
    tail = "\n".join(text.strip().splitlines()[-lines:])
    return hashlib.sha256(tail.encode("utf-8", "ignore")).hexdigest()


def load_retry_state():
    if RETRY_FILE.exists():
        try:
            return json.loads(RETRY_FILE.read_text())
        except Exception:
            pass
    return {"count": 0, "last_hash": None}


def save_retry_state(state):
    RETRY_FILE.write_text(json.dumps(state))


def reset_retry_state():
    save_retry_state({"count": 0, "last_hash": None})


def cmd_run(command):
    start = time.time()
    proc = subprocess.run(command, shell=True, capture_output=True, text=True)
    duration = time.time() - start
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")

    state = load_retry_state()

    if proc.returncode == 0:
        reset_retry_state()
        log_run(command, proc.returncode, duration, 0, "success")
        print(output)
        print(f"\n[safe_run] SUCCESS ({duration:.1f}s)")
        return 0

    # Hard stop: OOM is never worth an automatic retry
    if any(marker in output.lower() for marker in OOM_MARKERS):
        log_run(command, proc.returncode, duration, state["count"], "OOM_HARD_STOP")
        print(output)
        print("\n[safe_run] CIRCUIT_BREAKER_TRIPPED: CUDA OOM detected.")
        print("[safe_run] Do NOT retry automatically. Stop and ask the user to adjust batch size / memory settings.")
        reset_retry_state()
        return 99

    # Hard stop: same error repeated too many times
    h = error_hash(output)
    if h == state.get("last_hash"):
        state["count"] += 1
    else:
        state["count"] = 1
        state["last_hash"] = h
    save_retry_state(state)

    print(output)
    if state["count"] >= MAX_RETRIES:
        log_run(command, proc.returncode, duration, state["count"], "RETRY_LIMIT_HARD_STOP")
        print(f"\n[safe_run] CIRCUIT_BREAKER_TRIPPED: identical error {state['count']}x in a row.")
        print("[safe_run] Do NOT retry automatically. Stop and ask the user.")
        reset_retry_state()
        return 98
    else:
        log_run(command, proc.returncode, duration, state["count"], "failed_retry_allowed")
        print(f"\n[safe_run] Attempt {state['count']}/{MAX_RETRIES} failed. Retry allowed.")
        return proc.returncode


def cmd_bg(command, timeout, name):
    start = time.time()
    proc = subprocess.Popen(command, shell=True)
    bg_state = json.loads(BG_FILE.read_text()) if BG_FILE.exists() else {}
    bg_state[name] = {"pid": proc.pid, "started": start, "timeout": timeout, "command": command}
    BG_FILE.write_text(json.dumps(bg_state))
    print(f"[safe_run] Started background job '{name}' (PID {proc.pid}), max runtime {timeout}s")
    print("[safe_run] Run `python safe_run.py cleanup` periodically to enforce the timeout.")


def _pid_alive(pid):
    try:
        if os.name == "nt":
            out = subprocess.run(f'tasklist /FI "PID eq {pid}"', shell=True, capture_output=True, text=True)
            return str(pid) in out.stdout
        else:
            os.kill(pid, 0)
            return True
    except Exception:
        return False


def _kill_pid(pid):
    try:
        if os.name == "nt":
            subprocess.run(f"taskkill /PID {pid} /F /T", shell=True)
        else:
            os.kill(pid, signal.SIGKILL)
    except Exception as e:
        print(f"[safe_run] Failed to kill PID {pid}: {e}")


def cmd_cleanup():
    if not BG_FILE.exists():
        print("[safe_run] No tracked background jobs.")
        return
    bg_state = json.loads(BG_FILE.read_text())
    now = time.time()
    remaining = {}
    for name, info in bg_state.items():
        elapsed = now - info["started"]
        pid = info["pid"]
        alive = _pid_alive(pid)
        if alive and elapsed > info["timeout"]:
            print(f"[safe_run] Killing '{name}' (PID {pid}) - exceeded timeout ({elapsed:.0f}s > {info['timeout']}s)")
            _kill_pid(pid)
            log_run(info["command"], -9, elapsed, 0, "BG_TIMEOUT_KILLED")
        elif alive:
            print(f"[safe_run] '{name}' (PID {pid}) still running, {elapsed:.0f}s / {info['timeout']}s budget")
            remaining[name] = info
        else:
            print(f"[safe_run] '{name}' (PID {pid}) already finished")
    BG_FILE.write_text(json.dumps(remaining))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("command", nargs=argparse.REMAINDER)

    p_bg = sub.add_parser("bg")
    p_bg.add_argument("--timeout", type=int, default=1800)
    p_bg.add_argument("--name", required=True)
    p_bg.add_argument("command", nargs=argparse.REMAINDER)

    sub.add_parser("cleanup")

    args = parser.parse_args()

    if args.mode == "run":
        command = " ".join(args.command).lstrip("- ").strip()
        sys.exit(cmd_run(command))
    elif args.mode == "bg":
        command = " ".join(args.command).lstrip("- ").strip()
        cmd_bg(command, args.timeout, args.name)
    elif args.mode == "cleanup":
        cmd_cleanup()


if __name__ == "__main__":
    main()