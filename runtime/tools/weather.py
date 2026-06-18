"""
Weather tool — retrieves current weather conditions and optional multi-day forecasts.
Uses the free, public Open-Meteo API (no API key required).
"""

import logging
from typing import Optional
import httpx

logger = logging.getLogger(__name__)

# WMO Weather Interpretation Codes (WMO code)
WMO_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail"
}

def get_weather_desc(code: int) -> str:
    """Return a human-readable description for a WMO weather code."""
    return WMO_CODES.get(code, f"Unknown ({code})")

async def weather(location: str, forecast: bool = False) -> str:
    """
    Get current weather and optional 3-day forecast for a given location.
    
    Args:
        location: City or region name.
        forecast: If True, returns daily forecast for the next 3 days.
        
    Returns:
        A formatted plain text summary of the weather.
    """
    location = location.strip()
    if not location:
        return "[WEATHER ERROR: Empty location provided]"

    # 1. Geocoding: Resolve location to latitude/longitude
    geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            geo_resp = await client.get(
                geocoding_url,
                params={"name": location, "count": 1, "language": "en"}
            )
            geo_resp.raise_for_status()
            geo_data = geo_resp.json()
    except Exception as e:
        logger.exception("Geocoding API request failed for location: %s", location)
        return f"[WEATHER ERROR: Geocoding API failed for '{location}' — {type(e).__name__}: {str(e)}]"

    results = geo_data.get("results")
    if not results:
        return f"[WEATHER ERROR: Could not find location '{location}']"

    best_match = results[0]
    lat = best_match.get("latitude")
    lon = best_match.get("longitude")
    resolved_name = best_match.get("name")
    country = best_match.get("country", "")
    admin1 = best_match.get("admin1", "")
    
    location_parts = [resolved_name]
    if admin1:
        location_parts.append(admin1)
    if country:
        location_parts.append(country)
    full_location_str = ", ".join(location_parts)

    if lat is None or lon is None:
        return f"[WEATHER ERROR: Geocoding resolved location '{resolved_name}' but coordinates were missing]"

    # 2. Forecast API: Query weather
    forecast_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
        "timezone": "auto"
    }

    if forecast:
        params["daily"] = "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            weather_resp = await client.get(forecast_url, params=params)
            weather_resp.raise_for_status()
            weather_data = weather_resp.json()
    except Exception as e:
        logger.exception("Forecast API request failed for coordinates: %s, %s", lat, lon)
        return f"[WEATHER ERROR: Forecast API failed for '{full_location_str}' — {type(e).__name__}: {str(e)}]"

    # Parse current weather
    current = weather_data.get("current", {})
    current_units = weather_data.get("current_units", {})
    
    temp = current.get("temperature_2m")
    temp_unit = current_units.get("temperature_2m", "°C")
    
    humidity = current.get("relative_humidity_2m")
    humidity_unit = current_units.get("relative_humidity_2m", "%")
    
    wind = current.get("wind_speed_10m")
    wind_unit = current_units.get("wind_speed_10m", "km/h")
    
    weather_code = current.get("weather_code")
    condition = get_weather_desc(weather_code) if weather_code is not None else "Unknown"

    lines = [
        f"Weather for: {full_location_str}",
        f"Coordinates: Latitude {lat:.4f}, Longitude {lon:.4f}",
        f"Current Conditions:",
        f"  - Temperature: {temp} {temp_unit}" if temp is not None else "  - Temperature: N/A",
        f"  - Condition: {condition}",
        f"  - Relative Humidity: {humidity}{humidity_unit}" if humidity is not None else "  - Relative Humidity: N/A",
        f"  - Wind Speed: {wind} {wind_unit}" if wind is not None else "  - Wind Speed: N/A",
    ]

    # Parse forecast (optional, next 3 days)
    if forecast and "daily" in weather_data:
        daily = weather_data["daily"]
        daily_units = weather_data.get("daily_units", {})
        
        times = daily.get("time", [])
        temp_maxs = daily.get("temperature_2m_max", [])
        temp_mins = daily.get("temperature_2m_min", [])
        precip_sums = daily.get("precipitation_sum", [])
        daily_codes = daily.get("weather_code", [])

        precip_unit = daily_units.get("precipitation_sum", "mm")
        
        # We only want next 3 days, standard forecast returns 7 days.
        forecast_days = min(len(times), 3)
        if forecast_days > 0:
            lines.append("")
            lines.append("3-Day Forecast:")
            for i in range(forecast_days):
                day_date = times[i]
                t_max = temp_maxs[i] if i < len(temp_maxs) else None
                t_min = temp_mins[i] if i < len(temp_mins) else None
                precip = precip_sums[i] if i < len(precip_sums) else None
                code = daily_codes[i] if i < len(daily_codes) else None
                
                day_cond = get_weather_desc(code) if code is not None else "Unknown"
                
                temp_str = f"Min {t_min}{temp_unit}, Max {t_max}{temp_unit}" if (t_min is not None and t_max is not None) else "N/A"
                precip_str = f"Precipitation: {precip} {precip_unit}" if precip is not None else "Precipitation: N/A"
                
                lines.append(f"  - {day_date}: {temp_str} | {day_cond} | {precip_str}")

    return "\n".join(lines)
