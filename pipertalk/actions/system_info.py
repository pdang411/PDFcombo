"""System information action for the Piper agent."""

import os
import platform
import shutil


def _get_uptime() -> str:
    try:
        with open("/proc/uptime", "r") as f:
            uptime_secs = float(f.read().split()[0])
        days = int(uptime_secs // 86400)
        hours = int((uptime_secs % 86400) // 3600)
        mins = int((uptime_secs % 3600) // 60)
        parts = []
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        parts.append(f"{mins}m")
        return " ".join(parts)
    except Exception:
        return "N/A"


def _get_cpu() -> str:
    try:
        with open("/proc/loadavg", "r") as f:
            parts = f.read().split()
        load1, load5, load15 = parts[:3]
        return f"load: {load1} (1m), {load5} (5m), {load15} (15m)"
    except Exception:
        return "N/A"


def _get_memory() -> str:
    try:
        with open("/proc/meminfo", "r") as f:
            lines = f.readlines()
        mem_info = {}
        for line in lines:
            parts = line.split(":")
            if len(parts) == 2:
                key = parts[0].strip()
                val = parts[1].strip().replace(" kB", "")
                try:
                    mem_info[key] = int(val)
                except ValueError:
                    pass
        total = mem_info.get("MemTotal", 0) // 1024
        available = mem_info.get("MemAvailable", 0) // 1024
        used = total - available
        pct = (used / total * 100) if total else 0
        return f"{used}MB / {total}MB ({pct:.0f}%)"
    except Exception:
        return "N/A"


def _get_disk() -> str:
    try:
        usage = shutil.disk_usage("/")
        total = usage.total // (1024**3)
        used = usage.used // (1024**3)
        free = usage.free // (1024**3)
        pct = usage.used / usage.total * 100
        return f"{used}GB / {total}GB ({pct:.0f}%) used, {free}GB free"
    except Exception:
        return "N/A"


def system_info(parameters: dict = None, **kwargs) -> str:
    query = (parameters or {}).get("query", "all").lower().strip()
    parts = []
    if query in ("all", "uptime"):
        parts.append(f"Uptime: {_get_uptime()}")
    if query in ("all", "cpu"):
        parts.append(f"CPU: {_get_cpu()}")
    if query in ("all", "memory"):
        parts.append(f"Memory: {_get_memory()}")
    if query in ("all", "disk"):
        parts.append(f"Disk: {_get_disk()}")
    if not parts:
        return f"Unknown system info query: {query}"
    return "\n".join([f"System info ({platform.node()} / {platform.system()} {platform.machine()})"] + parts)
