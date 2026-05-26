"""Notification action for the Piper agent."""

import requests

_DEFAULT_TOPIC = "pipertalk"


def send_message(parameters: dict = None, **kwargs) -> str:
    params = parameters or {}
    message = params.get("message", "").strip()
    platform = params.get("platform", "ntfy").strip().lower()
    if not message:
        return "Please specify a message to send."
    if platform == "ntfy":
        topic = params.get("topic", _DEFAULT_TOPIC)
        try:
            resp = requests.post(f"https://ntfy.sh/{topic}", data=message.encode("utf-8"), timeout=10)
            resp.raise_for_status()
            return f"Notification sent: {message[:60]}"
        except Exception as e:
            return f"Failed to send ntfy notification: {e}"
    else:
        return f"Unknown platform: {platform}. Available: ntfy"
