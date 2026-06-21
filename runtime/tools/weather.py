# """
# Weather tool — retrieves current weather conditions and optional multi-day forecasts.
# Uses the free, public Open-Meteo API (no API key required).
# """

# import logging
# from typing import Optional
# import httpx

# logger = logging.getLogger(__name__)

# # WMO Weather Interpretation Codes (WMO code)
# WMO_CODES = {
#     0: "Clear sky",
#     1: "Mainly clear",
#     2: "Partly cloudy",
#     3: "Overcast",
#     45: "Fog",
#     48: "Depositing rime fog",
#     51: "Light drizzle",
#     53: "Moderate drizzle",
#     55: "Dense drizzle",
#     56: "Light freezing drizzle",
#     57: "Dense freezing drizzle",
#     61: "Slight rain",
#     63: "Moderate rain",
#     65: "Heavy rain",
#     66: "Light freezing rain",
#     67: "Heavy freezing rain",
#     71: "Slight snow fall",
#     73: "Moderate snow fall",
#     75: "Heavy snow fall",
#     77: "Snow grains",
#     80: "Slight rain showers",
#     81: "Moderate rain showers",
#     82: "Violent rain showers",
#     85: "Slight snow showers",
#     86: "Heavy snow showers",
#     95: "Thunderstorm",
#     96: "Thunderstorm with slight hail",
#     99: "Thunderstorm with heavy hail"
# }

# def get_weather_desc(code: int) -> str:
#     """Return a human-readable description for a WMO weather code."""
#     return WMO_CODES.get(code, f"Unknown ({code})")

# async def weather(location: str, forecast: bool = False) -> str:
#     """
#     Get current weather and optional 3-day forecast for a given location.
    
#     Args:
#         location: City or region name.
#         forecast: If True, returns daily forecast for the next 3 days.
        
#     Returns:
#         A formatted plain text summary of the weather.
#     """
#     location = location.strip()
#     if not location:
#         return "[WEATHER ERROR: Empty location provided]"

#     # 1. Geocoding: Resolve location to latitude/longitude
#     geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
#     try:
#         async with httpx.AsyncClient(timeout=10.0) as client:
#             geo_resp = await client.get(
#                 geocoding_url,
#                 params={"name": location, "count": 1, "language": "en"}
#             )
#             geo_resp.raise_for_status()
#             geo_data = geo_resp.json()
#     except Exception as e:
#         logger.exception("Geocoding API request failed for location: %s", location)
#         return f"[WEATHER ERROR: Geocoding API failed for '{location}' — {type(e).__name__}: {str(e)}]"

#     results = geo_data.get("results")
#     if not results:
#         return f"[WEATHER ERROR: Could not find location '{location}']"

#     best_match = results[0]
#     lat = best_match.get("latitude")
#     lon = best_match.get("longitude")
#     resolved_name = best_match.get("name")
#     country = best_match.get("country", "")
#     admin1 = best_match.get("admin1", "")
    
#     location_parts = [resolved_name]
#     if admin1:
#         location_parts.append(admin1)
#     if country:
#         location_parts.append(country)
#     full_location_str = ", ".join(location_parts)

#     if lat is None or lon is None:
#         return f"[WEATHER ERROR: Geocoding resolved location '{resolved_name}' but coordinates were missing]"

#     # 2. Forecast API: Query weather
#     forecast_url = "https://api.open-meteo.com/v1/forecast"
#     params = {
#         "latitude": lat,
#         "longitude": lon,
#         "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
#         "timezone": "auto"
#     }

#     if forecast:
#         params["daily"] = "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code"

#     try:
#         async with httpx.AsyncClient(timeout=10.0) as client:
#             weather_resp = await client.get(forecast_url, params=params)
#             weather_resp.raise_for_status()
#             weather_data = weather_resp.json()
#     except Exception as e:
#         logger.exception("Forecast API request failed for coordinates: %s, %s", lat, lon)
#         return f"[WEATHER ERROR: Forecast API failed for '{full_location_str}' — {type(e).__name__}: {str(e)}]"

#     # Parse current weather
#     current = weather_data.get("current", {})
#     current_units = weather_data.get("current_units", {})
    
#     temp = current.get("temperature_2m")
#     temp_unit = current_units.get("temperature_2m", "°C")
    
#     humidity = current.get("relative_humidity_2m")
#     humidity_unit = current_units.get("relative_humidity_2m", "%")
    
#     wind = current.get("wind_speed_10m")
#     wind_unit = current_units.get("wind_speed_10m", "km/h")
    
#     weather_code = current.get("weather_code")
#     condition = get_weather_desc(weather_code) if weather_code is not None else "Unknown"

#     lines = [
#         f"Weather for: {full_location_str}",
#         f"Coordinates: Latitude {lat:.4f}, Longitude {lon:.4f}",
#         f"Current Conditions:",
#         f"  - Temperature: {temp} {temp_unit}" if temp is not None else "  - Temperature: N/A",
#         f"  - Condition: {condition}",
#         f"  - Relative Humidity: {humidity}{humidity_unit}" if humidity is not None else "  - Relative Humidity: N/A",
#         f"  - Wind Speed: {wind} {wind_unit}" if wind is not None else "  - Wind Speed: N/A",
#     ]

#     # Parse forecast (optional, next 3 days)
#     if forecast and "daily" in weather_data:
#         daily = weather_data["daily"]
#         daily_units = weather_data.get("daily_units", {})
        
#         times = daily.get("time", [])
#         temp_maxs = daily.get("temperature_2m_max", [])
#         temp_mins = daily.get("temperature_2m_min", [])
#         precip_sums = daily.get("precipitation_sum", [])
#         daily_codes = daily.get("weather_code", [])

#         precip_unit = daily_units.get("precipitation_sum", "mm")
        
#         # We only want next 3 days, standard forecast returns 7 days.
#         forecast_days = min(len(times), 3)
#         if forecast_days > 0:
#             lines.append("")
#             lines.append("3-Day Forecast:")
#             for i in range(forecast_days):
#                 day_date = times[i]
#                 t_max = temp_maxs[i] if i < len(temp_maxs) else None
#                 t_min = temp_mins[i] if i < len(temp_mins) else None
#                 precip = precip_sums[i] if i < len(precip_sums) else None
#                 code = daily_codes[i] if i < len(daily_codes) else None
                
#                 day_cond = get_weather_desc(code) if code is not None else "Unknown"
                
#                 temp_str = f"Min {t_min}{temp_unit}, Max {t_max}{temp_unit}" if (t_min is not None and t_max is not None) else "N/A"
#                 precip_str = f"Precipitation: {precip} {precip_unit}" if precip is not None else "Precipitation: N/A"
                
#                 lines.append(f"  - {day_date}: {temp_str} | {day_cond} | {precip_str}")

#     return "\n".join(lines)
"""
Weather tool — current conditions, hourly outlook, and multi-day forecast.

Uses Open-Meteo API (free, no API key required) + Open-Meteo Geocoding API.

Actions:
    current     — current temperature, feels like, condition, humidity, wind,
                  UV index, visibility, precipitation, pressure. (default)
    forecast    — daily forecast for 1–7 days with min/max temp, condition,
                  precipitation chance, UV index, sunrise/sunset.
    hourly      — hour-by-hour forecast for the next N hours (default 24).

Design rules:
    - One shared httpx.AsyncClient per call.
    - Returns plain strings always. Never raises.
    - Units: metric by default. Pass units='imperial' for °F / mph / inches.
    - WMO weather codes mapped to human-readable descriptions.
    - Wind direction degrees converted to compass bearing (N, NE, E, etc.)
    - Timezone-aware: all times shown in the resolved location's local timezone.
    - Geocoding resolution shown in output so user can confirm the right city.

Env vars required: none.
"""

import asyncio
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ── WMO weather code → description ────────────────────────────────────────────

_WMO_CODES: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    77: "Snow grains",
    80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm + slight hail", 99: "Thunderstorm + heavy hail",
}


def _wmo(code: int | None) -> str:
    if code is None:
        return "Unknown"
    return _WMO_CODES.get(code, f"Code {code}")


# ── compass bearing ────────────────────────────────────────────────────────────

_COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def _compass(degrees: float | None) -> str:
    if degrees is None:
        return "N/A"
    idx = round(degrees / 22.5) % 16
    return _COMPASS[idx]


# ── unit helpers ───────────────────────────────────────────────────────────────

def _temp_unit(imperial: bool) -> str:
    return "°F" if imperial else "°C"


def _wind_unit(imperial: bool) -> str:
    return "mph" if imperial else "km/h"


def _precip_unit(imperial: bool) -> str:
    return "inch" if imperial else "mm"


def _open_meteo_units(imperial: bool) -> dict:
    """Return Open-Meteo unit parameter values for the given system."""
    if imperial:
        return {
            "temperature_unit":   "fahrenheit",
            "wind_speed_unit":    "mph",
            "precipitation_unit": "inch",
        }
    return {}   # metric is the default — no extra params needed


# ── number formatting ──────────────────────────────────────────────────────────

def _f(val: Any, decimals: int = 1, unit: str = "") -> str:
    if val is None:
        return "N/A"
    try:
        formatted = f"{float(val):.{decimals}f}"
        return f"{formatted}{unit}" if unit else formatted
    except (TypeError, ValueError):
        return "N/A"


# ── API constants ─────────────────────────────────────────────────────────────

_GEO_URL     = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

_CURRENT_FIELDS = (
    "temperature_2m,apparent_temperature,relative_humidity_2m,"
    "weather_code,wind_speed_10m,wind_direction_10m,wind_gusts_10m,"
    "precipitation,uv_index,visibility,surface_pressure,is_day"
)

_DAILY_FIELDS = (
    "temperature_2m_max,temperature_2m_min,"
    "apparent_temperature_max,apparent_temperature_min,"
    "weather_code,precipitation_sum,precipitation_probability_max,"
    "wind_speed_10m_max,uv_index_max,sunrise,sunset"
)

_HOURLY_FIELDS = (
    "temperature_2m,apparent_temperature,weather_code,"
    "precipitation_probability,precipitation,wind_speed_10m,"
    "wind_direction_10m,relative_humidity_2m,uv_index,visibility"
)


# ── geocoding ─────────────────────────────────────────────────────────────────

async def _geocode(location: str, client: httpx.AsyncClient) -> dict | None:
    """
    Resolve a location string to coordinates.
    Returns the best-match result dict or None.
    """
    try:
        resp = await client.get(
            _GEO_URL,
            params={"name": location, "count": 1, "language": "en"},
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return results[0] if results else None
    except Exception as e:
        logger.warning("Geocoding failed for %r: %s", location, e)
        return None


def _format_location(geo: dict) -> str:
    """Build a human-readable location string from a geocoding result."""
    parts = [geo.get("name", "")]
    admin1 = geo.get("admin1", "")
    country = geo.get("country", "")
    if admin1:
        parts.append(admin1)
    if country:
        parts.append(country)
    return ", ".join(p for p in parts if p)


# ── action handlers ───────────────────────────────────────────────────────────

async def _action_current(
    geo: dict,
    location_str: str,
    imperial: bool,
    client: httpx.AsyncClient,
) -> str:
    """Fetch and format current weather conditions."""
    params = {
        "latitude":  geo["latitude"],
        "longitude": geo["longitude"],
        "current":   _CURRENT_FIELDS,
        "timezone":  "auto",
        **_open_meteo_units(imperial),
    }

    try:
        resp = await client.get(_FORECAST_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return f"Error: Weather API failed for '{location_str}' — {e}."

    cur   = data.get("current", {})
    units = data.get("current_units", {})
    tz    = data.get("timezone", "UTC")
    time_ = cur.get("time", "")

    tu   = _temp_unit(imperial)
    wu   = _wind_unit(imperial)
    pu   = _precip_unit(imperial)
    code = cur.get("weather_code")
    is_day = cur.get("is_day", 1)

    condition = _wmo(code)
    condition_str = f"{condition} ({'day' if is_day else 'night'})"

    temp        = cur.get("temperature_2m")
    feels_like  = cur.get("apparent_temperature")
    humidity    = cur.get("relative_humidity_2m")
    wind_speed  = cur.get("wind_speed_10m")
    wind_dir    = cur.get("wind_direction_10m")
    wind_gusts  = cur.get("wind_gusts_10m")
    precip      = cur.get("precipitation")
    uv          = cur.get("uv_index")
    visibility  = cur.get("visibility")
    pressure    = cur.get("surface_pressure")

    lines = [
        f"Weather: {location_str}",
        f"Timezone: {tz}  |  As of: {time_}",
        f"Coordinates: {geo['latitude']:.4f}°, {geo['longitude']:.4f}°",
        "",
        f"Condition:    {condition_str}",
        f"Temperature:  {_f(temp)}{tu}  (feels like {_f(feels_like)}{tu})",
        f"Humidity:     {_f(humidity, 0)}%",
        "",
        f"Wind:         {_f(wind_speed)}{wu} from {_compass(wind_dir)} ({wind_dir}°)",
    ]

    if wind_gusts is not None:
        lines.append(f"Gusts:        {_f(wind_gusts)}{wu}")

    if precip is not None:
        lines.append(f"Precipitation:{_f(precip)}{pu} (current hour)")

    if uv is not None:
        uv_label = _uv_label(uv)
        lines.append(f"UV index:     {_f(uv, 0)} ({uv_label})")

    if visibility is not None:
        vis_km = float(visibility) / 1000 if not imperial else float(visibility) / 1609.34
        vis_unit = "mi" if imperial else "km"
        lines.append(f"Visibility:   {vis_km:.1f} {vis_unit}")

    if pressure is not None:
        lines.append(f"Pressure:     {_f(pressure)} hPa")

    return "\n".join(lines)


async def _action_forecast(
    geo: dict,
    location_str: str,
    days: int,
    imperial: bool,
    client: httpx.AsyncClient,
) -> str:
    """Fetch and format a daily forecast for 1–7 days."""
    days = max(1, min(7, days))

    params = {
        "latitude":  geo["latitude"],
        "longitude": geo["longitude"],
        "daily":     _DAILY_FIELDS,
        "timezone":  "auto",
        "forecast_days": days,
        **_open_meteo_units(imperial),
    }

    try:
        resp = await client.get(_FORECAST_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return f"Error: Forecast API failed for '{location_str}' — {e}."

    daily  = data.get("daily", {})
    tz     = data.get("timezone", "UTC")
    tu     = _temp_unit(imperial)
    wu     = _wind_unit(imperial)
    pu     = _precip_unit(imperial)

    times      = daily.get("time", [])
    t_max      = daily.get("temperature_2m_max", [])
    t_min      = daily.get("temperature_2m_min", [])
    feels_max  = daily.get("apparent_temperature_max", [])
    feels_min  = daily.get("apparent_temperature_min", [])
    codes      = daily.get("weather_code", [])
    precip     = daily.get("precipitation_sum", [])
    precip_pct = daily.get("precipitation_probability_max", [])
    wind_max   = daily.get("wind_speed_10m_max", [])
    uv_max     = daily.get("uv_index_max", [])
    sunrises   = daily.get("sunrise", [])
    sunsets    = daily.get("sunset", [])

    if not times:
        return f"Error: No forecast data returned for '{location_str}'."

    lines = [
        f"{days}-Day Forecast: {location_str}",
        f"Timezone: {tz}",
        "",
    ]

    for i in range(min(days, len(times))):
        date      = times[i]
        tmax      = t_max[i]      if i < len(t_max)      else None
        tmin      = t_min[i]      if i < len(t_min)      else None
        fmax      = feels_max[i]  if i < len(feels_max)  else None
        fmin      = feels_min[i]  if i < len(feels_min)  else None
        code      = codes[i]      if i < len(codes)       else None
        prec      = precip[i]     if i < len(precip)      else None
        pct       = precip_pct[i] if i < len(precip_pct) else None
        wmax      = wind_max[i]   if i < len(wind_max)    else None
        uv        = uv_max[i]     if i < len(uv_max)      else None
        sunrise   = sunrises[i]   if i < len(sunrises)    else None
        sunset    = sunsets[i]    if i < len(sunsets)     else None

        condition = _wmo(code)
        uv_str    = f"{_f(uv, 0)} ({_uv_label(uv)})" if uv is not None else "N/A"
        pct_str   = f"{int(pct)}%" if pct is not None else "N/A"

        lines.append(f"── {date} ──────────────────────────────")
        lines.append(f"  Condition:     {condition}")
        lines.append(f"  Temp:          {_f(tmin)}{tu} – {_f(tmax)}{tu}  (feels {_f(fmin)}{tu} – {_f(fmax)}{tu})")
        lines.append(f"  Precipitation: {_f(prec)}{pu}  |  Chance: {pct_str}")
        lines.append(f"  Wind (max):    {_f(wmax)}{wu}")
        lines.append(f"  UV index:      {uv_str}")
        if sunrise and sunset:
            # Strip date prefix from datetime strings (format: 2026-06-20T05:30)
            sr = sunrise.split("T")[-1] if "T" in sunrise else sunrise
            ss = sunset.split("T")[-1]  if "T" in sunset  else sunset
            lines.append(f"  Sunrise/set:   {sr} / {ss}")
        lines.append("")

    return "\n".join(lines).rstrip()


async def _action_hourly(
    geo: dict,
    location_str: str,
    hours: int,
    imperial: bool,
    client: httpx.AsyncClient,
) -> str:
    """Fetch and format an hour-by-hour forecast."""
    hours = max(1, min(48, hours))

    params = {
        "latitude":    geo["latitude"],
        "longitude":   geo["longitude"],
        "hourly":      _HOURLY_FIELDS,
        "timezone":    "auto",
        "forecast_days": 2,   # enough to cover 48h
        **_open_meteo_units(imperial),
    }

    try:
        resp = await client.get(_FORECAST_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return f"Error: Hourly forecast API failed for '{location_str}' — {e}."

    hourly = data.get("hourly", {})
    tz     = data.get("timezone", "UTC")
    tu     = _temp_unit(imperial)
    wu     = _wind_unit(imperial)

    times      = hourly.get("time", [])
    temps      = hourly.get("temperature_2m", [])
    feels      = hourly.get("apparent_temperature", [])
    codes      = hourly.get("weather_code", [])
    precip_pct = hourly.get("precipitation_probability", [])
    precip     = hourly.get("precipitation", [])
    wind       = hourly.get("wind_speed_10m", [])
    wind_dir   = hourly.get("wind_direction_10m", [])
    humidity   = hourly.get("relative_humidity_2m", [])
    uv         = hourly.get("uv_index", [])

    if not times:
        return f"Error: No hourly data returned for '{location_str}'."

    lines = [
        f"Hourly Forecast ({hours}h): {location_str}",
        f"Timezone: {tz}",
        "",
        f"{'Time':<18} {'Temp':>6} {'Feels':>6} {'Condition':<22} {'Rain%':>5} {'Wind':>8} {'Dir':>4} {'UV':>3}",
        "-" * 80,
    ]

    for i in range(min(hours, len(times))):
        t_str   = times[i].split("T")[-1] if "T" in times[i] else times[i]
        date_   = times[i].split("T")[0]  if "T" in times[i] else ""
        temp_v  = temps[i]      if i < len(temps)      else None
        feel_v  = feels[i]      if i < len(feels)      else None
        code_v  = codes[i]      if i < len(codes)       else None
        pct_v   = precip_pct[i] if i < len(precip_pct) else None
        wind_v  = wind[i]       if i < len(wind)        else None
        wdir_v  = wind_dir[i]   if i < len(wind_dir)    else None
        uv_v    = uv[i]         if i < len(uv)          else None

        condition = _wmo(code_v)[:21]   # truncate for column fit
        pct_str   = f"{int(pct_v)}%" if pct_v is not None else "N/A"
        wind_str  = f"{_f(wind_v, 0)}{wu}"
        dir_str   = _compass(wdir_v)
        uv_str    = f"{_f(uv_v, 0)}" if uv_v is not None else "-"

        # Insert date separator when the date changes
        if i == 0 or (i > 0 and "T" in times[i] and times[i].split("T")[0] != times[i-1].split("T")[0]):
            lines.append(f"  ── {date_} ──")

        lines.append(
            f"{t_str:<18} {_f(temp_v)}{tu:>6} {_f(feel_v)}{tu:>6} "
            f"{condition:<22} {pct_str:>5} {wind_str:>8} {dir_str:>4} {uv_str:>3}"
        )

    return "\n".join(lines)


# ── UV label ──────────────────────────────────────────────────────────────────

def _uv_label(uv: Any) -> str:
    if uv is None:
        return "N/A"
    try:
        v = float(uv)
    except (TypeError, ValueError):
        return "N/A"
    if v < 3:
        return "Low"
    if v < 6:
        return "Moderate"
    if v < 8:
        return "High"
    if v < 11:
        return "Very high"
    return "Extreme"


# ── public async entrypoint ───────────────────────────────────────────────────

async def weather(
    location: str,
    action: str = "current",
    days: int = 3,
    hours: int = 24,
    units: str = "metric",
) -> str:
    """
    Get weather information for any location worldwide.

    Args:
        location: City name, region, or country. Be specific for accuracy.
                  Examples: 'Mumbai', 'New Delhi', 'London UK', 'New York City',
                  'Bengaluru Karnataka'. The resolved location is shown in the
                  output so you can confirm the right city was matched.
        action:   which weather data to return (see below). Default 'current'.
        days:     number of forecast days for action='forecast'. Range 1–7. Default 3.
        hours:    number of hours for action='hourly'. Range 1–48. Default 24.
        units:    'metric' (°C, km/h, mm) or 'imperial' (°F, mph, inch). Default 'metric'.

    Actions:
        current   — current conditions: temperature, feels like, humidity,
                    wind speed/direction/gusts, UV index, visibility,
                    precipitation, pressure. (default)
        forecast  — daily min/max temperature, condition, precipitation sum,
                    precipitation chance, max wind, UV index, sunrise/sunset.
                    Configurable 1–7 days via 'days' parameter.
        hourly    — hour-by-hour: temp, feels like, condition, rain chance,
                    wind, direction, UV index. Configurable 1–48h via 'hours'.

    Returns a plain string in all cases. Never raises.
    """
    location = location.strip()
    if not location:
        return "Error: 'location' must not be empty."

    action = action.lower().strip()
    valid_actions = {"current", "forecast", "hourly"}
    if action not in valid_actions:
        return f"Error: Unknown action '{action}'. Valid: {', '.join(sorted(valid_actions))}."

    imperial = units.lower().strip() == "imperial"

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(12.0)) as client:
            geo = await _geocode(location, client)
            if geo is None:
                return f"Error: Could not find location '{location}'. Try a more specific name."

            location_str = _format_location(geo)

            if action == "current":
                return await _action_current(geo, location_str, imperial, client)
            if action == "forecast":
                return await _action_forecast(geo, location_str, days, imperial, client)
            if action == "hourly":
                return await _action_hourly(geo, location_str, hours, imperial, client)

    except Exception as e:
        logger.exception("weather() top-level error for location=%r", location)
        return f"Error: Weather tool failed unexpectedly — {type(e).__name__}: {e}"

    return "Error: Unhandled action path."