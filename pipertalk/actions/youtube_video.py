"""YouTube action for the Piper agent."""

import re
import subprocess
from urllib.parse import quote_plus

try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

_YT_VIDEO_FILTER = "EgIQAQ%3D%3D"


def _open_url(url: str) -> None:
    try:
        subprocess.Popen(["xdg-open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"[YouTube] open_url failed: {e}")


def _scrape_first_video_url(query: str) -> str | None:
    if not _REQUESTS_OK:
        return None
    search_url = f"https://www.youtube.com/results?search_query={quote_plus(query)}&sp={_YT_VIDEO_FILTER}"
    try:
        r = requests.get(search_url, headers=HEADERS, timeout=10)
        html = r.text
        video_ids = re.findall(r'"videoId":"([A-Za-z0-9_-]{11})"', html)
        seen = set()
        for vid in video_ids:
            if vid in seen:
                continue
            seen.add(vid)
            if f'/shorts/{vid}' in html:
                continue
            return f"https://www.youtube.com/watch?v={vid}"
    except Exception as e:
        print(f"[YouTube] scrape failed: {e}")
    return None


def _scrape_trending(region: str = "US", max_results: int = 8) -> list[dict]:
    if not _REQUESTS_OK:
        return []
    url = f"https://www.youtube.com/feed/trending?gl={region.upper()}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        html = r.text
        titles = re.findall(r'"title":\{"runs":\[\{"text":"([^"]+)"\}\]', html)
        channels = re.findall(r'"ownerText":\{"runs":\[\{"text":"([^"]+)"', html)
        results, seen = [], set()
        for i, title in enumerate(titles):
            if title in seen or len(title) < 5:
                continue
            seen.add(title)
            channel = channels[i] if i < len(channels) else "Unknown"
            results.append({"rank": len(results) + 1, "title": title, "channel": channel})
            if len(results) >= max_results:
                break
        return results
    except Exception as e:
        print(f"[YouTube] trending scrape failed: {e}")
        return []


def _handle_play(parameters: dict) -> str:
    query = parameters.get("query", "").strip()
    if not query:
        return "Please tell me what you'd like to watch."
    video_url = _scrape_first_video_url(query)
    if video_url:
        _open_url(video_url)
        return f"Playing: {query}"
    fallback_url = f"https://www.youtube.com/results?search_query={quote_plus(query)}&sp={_YT_VIDEO_FILTER}"
    _open_url(fallback_url)
    return f"Opened YouTube search for: {query} (manual selection required)"


def _handle_trending(parameters: dict) -> str:
    region = parameters.get("region", "US").upper()
    trending = _scrape_trending(region=region, max_results=8)
    if not trending:
        return f"Could not fetch trending videos for region {region}."
    lines = [f"Top trending videos in {region}:"]
    lines += [f"  {v['rank']}. {v['title']} \u2014 {v['channel']}" for v in trending]
    return "\n".join(lines)


_ACTION_MAP = {"play": _handle_play, "trending": _handle_trending}


def youtube_video(parameters: dict = None, **kwargs) -> str:
    params = parameters or {}
    action = params.get("action", "play").lower().strip()
    handler = _ACTION_MAP.get(action)
    if handler is None:
        return f"Unknown YouTube action: '{action}'. Available: play, trending."
    try:
        return handler(params) or "Done."
    except Exception as e:
        return f"YouTube {action} failed: {e}"
