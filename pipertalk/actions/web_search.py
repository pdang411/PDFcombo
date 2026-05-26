"""Web search action for the Piper agent."""

import re
import time

import requests


def _ddg_search(query: str, max_results: int = 6) -> list[dict]:
    for mod_name in ("duckduckgo_search", "ddgs"):
        try:
            mod = __import__(mod_name, fromlist=["DDGS"])
            DDGS = mod.DDGS
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "snippet": r.get("body", ""),
                        "url": r.get("href", ""),
                    })
            return results
        except ImportError:
            continue
        except Exception as e:
            print(f"[WebSearch] DDGS({mod_name}) failed: {e}")
            return []
    return []


def _bing_search(query: str, max_results: int = 6) -> list[dict]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        resp = requests.get(
            "https://www.bing.com/search",
            params={"q": query, "count": max_results},
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        html = resp.text
        results = []
        blocks = re.findall(r'<li class="b_algo">.*?</li>', html, re.DOTALL)
        for block in blocks[:max_results]:
            title_m = re.search(r'<a[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>', block, re.DOTALL)
            snippet_m = re.search(r'<p[^>]*>(.*?)</p>', block, re.DOTALL)
            if title_m:
                results.append({
                    "title": re.sub(r'<[^>]+>', '', title_m.group(2)).strip(),
                    "snippet": re.sub(r'<[^>]+>', '', snippet_m.group(1)).strip() if snippet_m else "",
                    "url": title_m.group(1),
                })
        return results
    except Exception as e:
        print(f"[WebSearch] Bing failed: {e}")
        return []


def _format_results(query: str, results: list[dict]) -> str:
    if not results:
        return f"No results found for: {query}"
    lines = [f"Search results for: {query}\n"]
    for i, r in enumerate(results, 1):
        if r.get("title"):
            lines.append(f"{i}. {r['title']}")
        if r.get("snippet"):
            lines.append(f"   {r['snippet']}")
        if r.get("url"):
            lines.append(f"   {r['url']}")
        lines.append("")
    return "\n".join(lines).strip()


def _compare(items: list[str], aspect: str) -> str:
    all_results: dict[str, list] = {}
    for item in items:
        try:
            all_results[item] = _ddg_search(f"{item} {aspect}", max_results=3)
        except Exception:
            try:
                all_results[item] = _bing_search(f"{item} {aspect}")
            except Exception:
                all_results[item] = []
    lines = [f"Comparison \u2014 {aspect.upper()}", "\u2500" * 40]
    for item in items:
        lines.append(f"\n- {item}")
        for r in all_results.get(item, [])[:2]:
            if r.get("snippet"):
                lines.append(f"  \u2022 {r['snippet']}")
    return "\n".join(lines)


def web_search(parameters: dict = None, **kwargs) -> str:
    params = parameters or {}
    query = params.get("query", "").strip()
    mode = params.get("mode", "search").lower().strip()
    items = params.get("items", [])
    aspect = params.get("aspect", "general").strip() or "general"

    if not query and not items:
        return "Please provide a search query."

    if items and mode != "compare":
        mode = "compare"

    try:
        if mode == "compare" and items:
            return _compare(items, aspect)

        results = []
        try:
            results = _ddg_search(query)
        except Exception as e:
            print(f"[WebSearch] DDG error: {e}")

        if not results:
            time.sleep(0.5)
            results = _bing_search(query)

        return _format_results(query, results)

    except Exception as e:
        return f"Search failed: {e}"
