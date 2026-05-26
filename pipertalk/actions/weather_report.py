"""Weather action for the Piper agent."""

import requests


def _get_coords(city: str) -> tuple[float, float] | None:
    try:
        resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("results"):
            r = data["results"][0]
            return (r["latitude"], r["longitude"])
        return None
    except Exception as e:
        print(f"[Weather] Geocoding failed: {e}")
        return None


def weather_action(parameters: dict = None, **kwargs) -> str:
    city = (parameters or {}).get("city", "").strip()
    if not city:
        return "Please specify a city for the weather report."

    coords = _get_coords(city)
    if not coords:
        return f"Could not find coordinates for city: {city}"

    lat, lon = coords
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": "true", "timezone": "auto"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        cw = data.get("current_weather", {})
        temp = cw.get("temperature", "?")
        wspeed = cw.get("windspeed", "?")
        wdir_code = cw.get("weathercode", 0)
        conditions = _code_to_desc(wdir_code)
        return f"Weather in {city}: {conditions}, {temp}\u00b0C, wind {wspeed} km/h"
    except Exception as e:
        return f"Weather fetch failed: {e}"


def _code_to_desc(code: int) -> str:
    if code == 0:
        return "Clear sky"
    elif code <= 3:
        return "Mainly clear"
    elif code <= 19:
        return "Foggy"
    elif code <= 29:
        return "Thunderstorm"
    elif code <= 39:
        return "Drizzle"
    elif code <= 49:
        return "Rain"
    elif code <= 59:
        return "Freezing rain"
    elif code <= 69:
        return "Snow"
    elif code <= 79:
        return "Rain shower"
    elif code <= 89:
        return "Snow shower"
    elif code <= 99:
        return "Thunderstorm"
    return "Unknown"
