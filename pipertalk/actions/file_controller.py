"""File-management action for the Piper agent."""

import os
import shutil
from pathlib import Path


_SAFE_ROOTS: list[Path] = [Path.home()]


def _is_safe_path(target: Path) -> bool:
    try:
        resolved = target.resolve()
        return any(
            resolved == root.resolve() or resolved.is_relative_to(root.resolve())
            for root in _SAFE_ROOTS
        )
    except Exception:
        return False


def _resolve_path(raw: str) -> Path:
    shortcuts: dict[str, Path] = {
        "desktop": Path.home() / "Desktop",
        "downloads": Path.home() / "Downloads",
        "documents": Path.home() / "Documents",
        "pictures": Path.home() / "Pictures",
        "music": Path.home() / "Music",
        "videos": Path.home() / "Videos",
        "home": Path.home(),
    }
    lower = raw.strip().lower()
    if lower in shortcuts:
        return shortcuts[lower]
    return Path(raw).expanduser()


def _format_size(b: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


def list_files(path: str = "desktop") -> str:
    try:
        target = _resolve_path(path)
        if not _is_safe_path(target):
            return f"Access denied: {target}"
        if not target.exists() or not target.is_dir():
            return f"Path not found: {target}"
        items = []
        for item in sorted(target.iterdir()):
            if item.name.startswith("."):
                continue
            if item.is_dir():
                items.append(f"  {item.name}/")
            else:
                size = _format_size(item.stat().st_size)
                items.append(f"  {item.name} ({size})")
        return f"Contents of {target.name}/ ({len(items)} items):\n" + "\n".join(items) if items else f"Directory is empty: {target.name}/"
    except Exception as e:
        return f"Error listing files: {e}"


def create_file(path: str, name: str = "", content: str = "") -> str:
    try:
        target = _resolve_path(path) / name if name else _resolve_path(path)
        if not _is_safe_path(target):
            return f"Access denied: {target}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"File created: {target.name}"
    except Exception as e:
        return f"Could not create file: {e}"


def read_file(path: str, name: str = "", max_chars: int = 4000) -> str:
    try:
        target = _resolve_path(path) / name if name else _resolve_path(path)
        if not _is_safe_path(target):
            return f"Access denied: {target}"
        if not target.exists() or not target.is_file():
            return f"File not found: {target.name}"
        content = target.read_text(encoding="utf-8", errors="ignore")
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n\n[Truncated \u2014 {len(content)} total chars]"
        return content
    except Exception as e:
        return f"Could not read file: {e}"


def write_file(path: str, name: str = "", content: str = "", append: bool = False) -> str:
    try:
        target = _resolve_path(path) / name if name else _resolve_path(path)
        if not _is_safe_path(target):
            return f"Access denied: {target}"
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with open(target, mode, encoding="utf-8") as f:
            f.write(content)
        action = "Appended to" if append else "Written to"
        return f"{action}: {target.name}"
    except Exception as e:
        return f"Could not write file: {e}"


def delete_file(path: str, name: str = "") -> str:
    try:
        target = _resolve_path(path) / name if name else _resolve_path(path)
        if not _is_safe_path(target):
            return f"Access denied: {target}"
        if not target.exists():
            return f"Not found: {target.name}"
        protected = {Path.home(), Path.home() / "Desktop", Path.home() / "Downloads", Path.home() / "Documents"}
        if target.resolve() in {p.resolve() for p in protected}:
            return f"Protected directory, cannot delete: {target.name}"
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        return f"Deleted: {target.name}"
    except Exception as e:
        return f"Could not delete: {e}"


def move_file(path: str, name: str = "", destination: str = "") -> str:
    try:
        src = _resolve_path(path) / name if name else _resolve_path(path)
        if not src.exists():
            return f"Source not found: {src.name}"
        dst = _resolve_path(destination)
        if dst.is_dir():
            dst = dst / src.name
        src.parent.mkdir(parents=True, exist_ok=True)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return f"Moved: {src.name} -> {dst.parent.name}/"
    except Exception as e:
        return f"Could not move: {e}"


def copy_file(path: str, name: str = "", destination: str = "") -> str:
    try:
        src = _resolve_path(path) / name if name else _resolve_path(path)
        dst = _resolve_path(destination)
        if dst.is_dir():
            dst = dst / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))
        return f"Copied: {src.name} -> {dst.parent.name}/"
    except Exception as e:
        return f"Could not copy: {e}"


def find_files(name: str = "", path: str = "home", max_results: int = 20) -> str:
    try:
        search_path = _resolve_path(path)
        if not search_path.exists():
            return f"Search path not found: {path}"
        results = []
        for item in search_path.rglob("*"):
            if not item.is_file():
                continue
            if name and name.lower() not in item.name.lower():
                continue
            size = _format_size(item.stat().st_size)
            results.append(f"  {item.name} ({size}) \u2014 {item.parent}")
            if len(results) >= max_results:
                break
        return f"Found {len(results)} file(s):\n" + "\n".join(results) if results else f"No files matching '{name}' found."
    except Exception as e:
        return f"Search error: {e}"


def get_disk_usage(path: str = "home") -> str:
    try:
        target = _resolve_path(path)
        usage = shutil.disk_usage(target)
        pct = usage.used / usage.total * 100
        return f"Disk usage ({target}):\n  Total: {_format_size(usage.total)}\n  Used: {_format_size(usage.used)} ({pct:.1f}%)\n  Free: {_format_size(usage.free)}"
    except Exception as e:
        return f"Could not get disk usage: {e}"


def file_controller(parameters: dict = None, **kwargs) -> str:
    params = parameters or {}
    action = params.get("action", "").lower().strip()
    path = params.get("path", "home")
    name = params.get("name", "")

    try:
        if action == "list":
            return list_files(path)
        elif action == "create_file":
            return create_file(path, name=name, content=params.get("content", ""))
        elif action == "read":
            return read_file(path, name=name)
        elif action == "write":
            return write_file(path, name=name, content=params.get("content", ""), append=params.get("append", False))
        elif action == "delete":
            return delete_file(path, name=name)
        elif action == "move":
            return move_file(path, name=name, destination=params.get("destination", ""))
        elif action == "copy":
            return copy_file(path, name=name, destination=params.get("destination", ""))
        elif action == "find":
            return find_files(name=name or params.get("name", ""), path=path)
        elif action == "disk_usage":
            return get_disk_usage(path)
        else:
            return f"Unknown action: '{action}'"
    except Exception as e:
        return f"File controller error ({action}): {e}"
