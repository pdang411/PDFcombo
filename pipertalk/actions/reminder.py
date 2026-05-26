"""Reminder action for the Piper agent."""

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _scripts_dir() -> Path:
    d = Path.home() / ".pipertalk" / "reminders"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sanitise(text: str, max_len: int = 200) -> str:
    return text.replace("\\", "").replace('"', "").replace("'", "").replace("\n", " ").replace("\r", "").strip()[:max_len]


def _write_notify_script(task_name: str, message: str) -> Path:
    script_path = _scripts_dir() / f"{task_name}.py"
    msg_literal = json.dumps(message)
    body = f"""# Auto-generated reminder
import json, os, pathlib, subprocess, sys
message = {msg_literal}
try:
    subprocess.run(["notify-send", "--urgency=normal", "--expire-time=15000", "Pipertalk Reminder", message], check=False, timeout=5, capture_output=True)
except Exception:
    pass
try:
    import urllib.request
    urllib.request.urlopen("https://ntfy.sh/pipertalk", data=message.encode("utf-8"), timeout=5)
except Exception:
    pass
try:
    pathlib.Path(__file__).unlink(missing_ok=True)
except Exception:
    pass
"""
    script_path.write_text(body, encoding="utf-8")
    script_path.chmod(0o600)
    return script_path


def _schedule_linux(target_dt: datetime, task_name: str, script_path: Path) -> str:
    if shutil.which("systemd-run"):
        on_calendar = target_dt.strftime("%Y-%m-%d %H:%M:00")
        result = subprocess.run(
            ["systemd-run", "--user", f"--on-calendar={on_calendar}", f"--unit={task_name}", "--", sys.executable, str(script_path)],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return task_name
    if shutil.which("at"):
        at_time = target_dt.strftime("%H:%M %Y-%m-%d")
        cmd_str = f"{sys.executable} {script_path}\n"
        result = subprocess.run(["at", at_time], input=cmd_str, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return task_name
    return ""


def reminder(parameters: dict = None, **kwargs) -> str:
    params = parameters or {}
    dt_str = params.get("datetime", "").strip()
    message = params.get("message", "Reminder").strip()
    if not dt_str:
        return "I need a date and time to set a reminder (YYYY-MM-DD HH:MM)."
    try:
        target_dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
    except ValueError:
        return "Could not parse datetime. Please use YYYY-MM-DD HH:MM format."
    if target_dt <= datetime.now():
        return "That time has already passed."
    safe_msg = _sanitise(message)
    task_name = f"PipertalkReminder_{target_dt.strftime('%Y%m%d_%H%M%S')}"
    try:
        script_path = _write_notify_script(task_name, safe_msg)
    except Exception as e:
        return f"Could not prepare reminder script: {e}"
    try:
        job_id = _schedule_linux(target_dt, task_name, script_path)
    except Exception as e:
        script_path.unlink(missing_ok=True)
        return f"Reminder scheduling failed: {e}"
    if not job_id:
        script_path.unlink(missing_ok=True)
        return "Could not register the reminder with the system scheduler."
    return f"Reminder set for {target_dt.strftime('%B %d at %I:%M %p')}."
