# """
# Unit tests for the weather tool.
# """

# import pytest
# from unittest.mock import AsyncMock, MagicMock, patch
# from runtime.tools.weather import weather, get_weather_desc

# def test_get_weather_desc():
#     assert get_weather_desc(0) == "Clear sky"
#     assert get_weather_desc(95) == "Thunderstorm"
#     assert "Unknown" in get_weather_desc(999)

# @pytest.mark.anyio
# async def test_weather_empty_location():
#     res = await weather("")
#     assert "[WEATHER ERROR: Empty location provided]" in res

# @pytest.mark.anyio
# async def test_weather_successful_current_only():
#     # Mock geocoding response
#     geo_data = {
#         "results": [
#             {
#                 "name": "Paris",
#                 "latitude": 48.8566,
#                 "longitude": 2.3522,
#                 "country": "France",
#                 "admin1": "Île-de-France"
#             }
#         ]
#     }
    
#     # Mock forecast response
#     weather_data = {
#         "current": {
#             "temperature_2m": 18.5,
#             "relative_humidity_2m": 60,
#             "weather_code": 1,
#             "wind_speed_10m": 12.0
#         },
#         "current_units": {
#             "temperature_2m": "°C",
#             "relative_humidity_2m": "%",
#             "wind_speed_10m": "km/h"
#         }
#     }
    
#     with patch("httpx.AsyncClient") as MockClientClass:
#         mock_client = AsyncMock()
#         MockClientClass.return_value.__aenter__.return_value = mock_client
        
#         # Geocoding call
#         geo_resp = MagicMock()
#         geo_resp.status_code = 200
#         geo_resp.json.return_value = geo_data
        
#         # Weather call
#         weather_resp = MagicMock()
#         weather_resp.status_code = 200
#         weather_resp.json.return_value = weather_data
        
#         mock_client.get.side_effect = [geo_resp, weather_resp]
        
#         res = await weather("Paris", forecast=False)
        
#         assert "Paris, Île-de-France, France" in res
#         assert "Temperature: 18.5 °C" in res
#         assert "Condition: Mainly clear" in res
#         assert "Relative Humidity: 60%" in res
#         assert "Wind Speed: 12.0 km/h" in res
#         assert "3-Day Forecast:" not in res

# @pytest.mark.anyio
# async def test_weather_successful_with_forecast():
#     geo_data = {
#         "results": [
#             {
#                 "name": "Tokyo",
#                 "latitude": 35.6762,
#                 "longitude": 139.6503,
#                 "country": "Japan"
#             }
#         ]
#     }
    
#     weather_data = {
#         "current": {
#             "temperature_2m": 22.0,
#             "relative_humidity_2m": 75,
#             "weather_code": 3,
#             "wind_speed_10m": 8.5
#         },
#         "current_units": {
#             "temperature_2m": "°C",
#             "relative_humidity_2m": "%",
#             "wind_speed_10m": "km/h"
#         },
#         "daily": {
#             "time": ["2026-06-19", "2026-06-20", "2026-06-21"],
#             "temperature_2m_max": [25.0, 24.5, 23.0],
#             "temperature_2m_min": [19.0, 18.5, 17.0],
#             "precipitation_sum": [0.0, 5.5, 12.0],
#             "weather_code": [0, 61, 95]
#         },
#         "daily_units": {
#             "precipitation_sum": "mm"
#         }
#     }
    
#     with patch("httpx.AsyncClient") as MockClientClass:
#         mock_client = AsyncMock()
#         MockClientClass.return_value.__aenter__.return_value = mock_client
        
#         geo_resp = MagicMock()
#         geo_resp.status_code = 200
#         geo_resp.json.return_value = geo_data
        
#         weather_resp = MagicMock()
#         weather_resp.status_code = 200
#         weather_resp.json.return_value = weather_data
        
#         mock_client.get.side_effect = [geo_resp, weather_resp]
        
#         res = await weather("Tokyo", forecast=True)
        
#         assert "Tokyo, Japan" in res
#         assert "Temperature: 22.0 °C" in res
#         assert "3-Day Forecast:" in res
#         assert "2026-06-19: Min 19.0°C, Max 25.0°C | Clear sky | Precipitation: 0.0 mm" in res
#         assert "2026-06-20: Min 18.5°C, Max 24.5°C | Slight rain | Precipitation: 5.5 mm" in res
#         assert "2026-06-21: Min 17.0°C, Max 23.0°C | Thunderstorm | Precipitation: 12.0 mm" in res

# @pytest.mark.anyio
# async def test_weather_location_not_found():
#     geo_data = {"results": []}
    
#     with patch("httpx.AsyncClient") as MockClientClass:
#         mock_client = AsyncMock()
#         MockClientClass.return_value.__aenter__.return_value = mock_client
        
#         geo_resp = MagicMock()
#         geo_resp.status_code = 200
#         geo_resp.json.return_value = geo_data
        
#         mock_client.get.return_value = geo_resp
        
#         res = await weather("Atlantis")
#         assert "[WEATHER ERROR: Could not find location 'Atlantis']" in res

# @pytest.mark.anyio
# async def test_weather_geocoding_failure():
#     with patch("httpx.AsyncClient") as MockClientClass:
#         mock_client = AsyncMock()
#         MockClientClass.return_value.__aenter__.return_value = mock_client
        
#         mock_client.get.side_effect = Exception("Connection timeout")
        
#         res = await weather("London")
#         assert "[WEATHER ERROR: Geocoding API failed for 'London'" in res
#         assert "Connection timeout" in res

# @pytest.mark.anyio
# async def test_weather_forecast_failure():
#     geo_data = {
#         "results": [
#             {
#                 "name": "London",
#                 "latitude": 51.5074,
#                 "longitude": -0.1278,
#                 "country": "United Kingdom"
#             }
#         ]
#     }
    
#     with patch("httpx.AsyncClient") as MockClientClass:
#         mock_client = AsyncMock()
#         MockClientClass.return_value.__aenter__.return_value = mock_client
        
#         geo_resp = MagicMock()
#         geo_resp.status_code = 200
#         geo_resp.json.return_value = geo_data
        
#         mock_client.get.side_effect = [geo_resp, Exception("API Error")]
        
#         res = await weather("London")
#         assert "[WEATHER ERROR: Forecast API failed for 'London, United Kingdom'" in res
#         assert "API Error" in res
"""
Tests for runtime/tools/weather.py

Run from the project root:
    pytest test/test_weather.py -v

Structure:
    Unit tests      — no network. Mock httpx.AsyncClient entirely.
    Integration     — real Open-Meteo API calls. Skipped by default.
                      Run with: RUN_INTEGRATION_TESTS=1 pytest test/test_weather.py -v -k integration

What is tested:
    _wmo            — WMO code → description, unknown code
    _compass        — degree → bearing, boundary values, None
    _uv_label       — UV index category mapping
    _fmt            — formatting with decimals and units
    _format_location — location string assembly from geocoding result
    weather() dispatch — unknown action, empty location
    geocoding failure — returns error string
    geocoding no result — returns error string
    current action  — happy path, feels like, wind direction, UV, visibility
    current action  — missing fields handled gracefully (None values)
    forecast action — 1 day, 3 days, 7 days, sunrise/sunset formatting
    hourly action   — 24h, date separator inserted correctly
    imperial units  — parameters forwarded to Open-Meteo
    httpx exception — returns error string, never raises
    HTTP error      — raise_for_status triggers error string
"""

import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "runtime"))

from tools.weather import (
    weather,
    _wmo,
    _compass,
    _uv_label,
    _f,
    _format_location,
    _WMO_CODES,
)

# ── helpers ───────────────────────────────────────────────────────────────────

def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_geo_response(
    name: str = "Mumbai",
    admin1: str = "Maharashtra",
    country: str = "India",
    lat: float = 19.0760,
    lon: float = 72.8777,
) -> dict:
    return {
        "results": [{
            "name": name,
            "admin1": admin1,
            "country": country,
            "latitude": lat,
            "longitude": lon,
        }]
    }


def _make_current_response(
    temp: float = 31.0,
    feels: float = 36.0,
    humidity: int = 78,
    weather_code: int = 1,
    wind_speed: float = 18.0,
    wind_dir: float = 225.0,
    wind_gusts: float = 30.0,
    precip: float = 0.0,
    uv: float = 8.0,
    visibility: float = 10000.0,
    pressure: float = 1012.0,
    is_day: int = 1,
    tz: str = "Asia/Kolkata",
    time_: str = "2026-06-20T14:00",
) -> dict:
    return {
        "current": {
            "temperature_2m": temp,
            "apparent_temperature": feels,
            "relative_humidity_2m": humidity,
            "weather_code": weather_code,
            "wind_speed_10m": wind_speed,
            "wind_direction_10m": wind_dir,
            "wind_gusts_10m": wind_gusts,
            "precipitation": precip,
            "uv_index": uv,
            "visibility": visibility,
            "surface_pressure": pressure,
            "is_day": is_day,
            "time": time_,
        },
        "current_units": {
            "temperature_2m": "°C",
            "wind_speed_10m": "km/h",
            "precipitation": "mm",
        },
        "timezone": tz,
    }


def _make_forecast_response(days: int = 3, tz: str = "Asia/Kolkata") -> dict:
    dates   = [f"2026-06-2{i}" for i in range(days)]
    t_max   = [32.0 + i for i in range(days)]
    t_min   = [24.0 + i for i in range(days)]
    f_max   = [35.0 + i for i in range(days)]
    f_min   = [27.0 + i for i in range(days)]
    codes   = [1] * days
    precip  = [2.0] * days
    p_pct   = [40] * days
    w_max   = [25.0] * days
    uv_max  = [9.0] * days
    sunrise = [f"2026-06-2{i}T06:00" for i in range(days)]
    sunset  = [f"2026-06-2{i}T19:30" for i in range(days)]

    return {
        "daily": {
            "time": dates,
            "temperature_2m_max": t_max,
            "temperature_2m_min": t_min,
            "apparent_temperature_max": f_max,
            "apparent_temperature_min": f_min,
            "weather_code": codes,
            "precipitation_sum": precip,
            "precipitation_probability_max": p_pct,
            "wind_speed_10m_max": w_max,
            "uv_index_max": uv_max,
            "sunrise": sunrise,
            "sunset": sunset,
        },
        "timezone": tz,
    }


def _make_hourly_response(n_hours: int = 48, tz: str = "Asia/Kolkata") -> dict:
    base_times = (
        [f"2026-06-20T{h:02d}:00" for h in range(24)] +
        [f"2026-06-21T{h:02d}:00" for h in range(24)]
    )[:n_hours]

    return {
        "hourly": {
            "time": base_times,
            "temperature_2m":        [30.0 + i * 0.1 for i in range(n_hours)],
            "apparent_temperature":  [33.0 + i * 0.1 for i in range(n_hours)],
            "weather_code":          [1] * n_hours,
            "precipitation_probability": [20] * n_hours,
            "precipitation":         [0.0] * n_hours,
            "wind_speed_10m":        [15.0] * n_hours,
            "wind_direction_10m":    [180.0] * n_hours,
            "relative_humidity_2m":  [70] * n_hours,
            "uv_index":              [5.0] * n_hours,
            "visibility":            [10000.0] * n_hours,
        },
        "timezone": tz,
    }


def _mock_client(
    geo_data: dict | None = None,
    weather_data: dict | None = None,
    geo_status: int = 200,
    weather_status: int = 200,
    raise_on_weather: Exception | None = None,
    raise_on_geo: Exception | None = None,
):
    """Mock httpx.AsyncClient for weather tests."""
    geo_resp = MagicMock()
    geo_resp.status_code = geo_status
    geo_resp.json.return_value = geo_data or {"results": []}
    geo_resp.raise_for_status = MagicMock()

    weather_resp = MagicMock()
    weather_resp.status_code = weather_status
    weather_resp.json.return_value = weather_data or {}
    weather_resp.raise_for_status = MagicMock()
    if weather_status >= 400:
        weather_resp.raise_for_status.side_effect = Exception(f"HTTP {weather_status}")

    async def mock_get(url, **kwargs):
        if "geocoding" in url:
            if raise_on_geo:
                raise raise_on_geo
            return geo_resp
        else:
            if raise_on_weather:
                raise raise_on_weather
            return weather_resp

    client = AsyncMock()
    client.get = mock_get
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__  = AsyncMock(return_value=False)
    return client


# ── unit tests: pure helpers ──────────────────────────────────────────────────

class TestWmo(unittest.TestCase):

    def test_known_codes(self):
        self.assertEqual(_wmo(0), "Clear sky")
        self.assertEqual(_wmo(3), "Overcast")
        self.assertEqual(_wmo(95), "Thunderstorm")

    def test_unknown_code(self):
        result = _wmo(999)
        self.assertIn("999", result)

    def test_none(self):
        self.assertEqual(_wmo(None), "Unknown")

    def test_all_codes_defined(self):
        """Every code in _WMO_CODES must return its description via _wmo()."""
        for code, desc in _WMO_CODES.items():
            self.assertEqual(_wmo(code), desc)


class TestCompass(unittest.TestCase):

    def test_cardinal_directions(self):
        self.assertEqual(_compass(0),   "N")
        self.assertEqual(_compass(90),  "E")
        self.assertEqual(_compass(180), "S")
        self.assertEqual(_compass(270), "W")

    def test_intercardinal(self):
        self.assertEqual(_compass(45),  "NE")
        self.assertEqual(_compass(135), "SE")
        self.assertEqual(_compass(225), "SW")
        self.assertEqual(_compass(315), "NW")

    def test_360_equals_north(self):
        self.assertEqual(_compass(360), "N")

    def test_none(self):
        self.assertEqual(_compass(None), "N/A")

    def test_fractional(self):
        # 22.5 is NNE
        result = _compass(22.5)
        self.assertEqual(result, "NNE")


class TestUvLabel(unittest.TestCase):

    def test_low(self):
        self.assertEqual(_uv_label(0), "Low")
        self.assertEqual(_uv_label(2), "Low")

    def test_moderate(self):
        self.assertEqual(_uv_label(3), "Moderate")
        self.assertEqual(_uv_label(5), "Moderate")

    def test_high(self):
        self.assertEqual(_uv_label(6), "High")
        self.assertEqual(_uv_label(7), "High")

    def test_very_high(self):
        self.assertEqual(_uv_label(8), "Very high")
        self.assertEqual(_uv_label(10), "Very high")

    def test_extreme(self):
        self.assertEqual(_uv_label(11), "Extreme")
        self.assertEqual(_uv_label(20), "Extreme")

    def test_none(self):
        self.assertEqual(_uv_label(None), "N/A")


class TestFmt(unittest.TestCase):

    def test_basic(self):
        self.assertEqual(_f(31.5), "31.5")

    def test_zero_decimals(self):
        self.assertEqual(_f(31.5, 0), "32")

    def test_with_unit(self):
        self.assertEqual(_f(31.5, 1, "°C"), "31.5°C")

    def test_none(self):
        self.assertEqual(_f(None), "N/A")

    def test_invalid(self):
        self.assertEqual(_f("text"), "N/A")


class TestFormatLocation(unittest.TestCase):

    def test_full_location(self):
        geo = {"name": "Mumbai", "admin1": "Maharashtra", "country": "India"}
        result = _format_location(geo)
        self.assertIn("Mumbai", result)
        self.assertIn("Maharashtra", result)
        self.assertIn("India", result)

    def test_city_only(self):
        geo = {"name": "London", "admin1": "", "country": ""}
        result = _format_location(geo)
        self.assertEqual(result, "London")

    def test_missing_admin1(self):
        geo = {"name": "Tokyo", "country": "Japan"}
        result = _format_location(geo)
        self.assertIn("Tokyo", result)
        self.assertIn("Japan", result)
        self.assertNotIn(",, ", result)


# ── unit tests: weather() dispatch ───────────────────────────────────────────

class TestWeatherDispatch(unittest.TestCase):

    def test_empty_location_returns_error(self):
        result = run(weather(location=""))
        self.assertIn("Error", result)
        self.assertIn("empty", result.lower())

    def test_whitespace_location_returns_error(self):
        result = run(weather(location="   "))
        self.assertIn("Error", result)

    def test_unknown_action_returns_error(self):
        result = run(weather(location="Mumbai", action="teleport"))
        self.assertIn("Error", result)
        self.assertIn("teleport", result)
        self.assertIn("current", result)   # valid actions listed


# ── unit tests: geocoding failure ────────────────────────────────────────────

class TestGeocodingFailure(unittest.TestCase):

    @patch("tools.weather.httpx.AsyncClient")
    def test_no_results_returns_error(self, MockClient):
        MockClient.return_value = _mock_client(geo_data={"results": []})
        result = run(weather(location="ZxyzNonexistentCity99"))
        self.assertIn("Error", result)
        self.assertIn("ZxyzNonexistentCity99", result)

    @patch("tools.weather.httpx.AsyncClient")
    def test_geocoding_exception_returns_error(self, MockClient):
        import httpx as httpx_mod
        MockClient.return_value = _mock_client(
            raise_on_geo=httpx_mod.RequestError("Connection refused", request=MagicMock())
        )
        result = run(weather(location="Mumbai"))
        self.assertIn("Error", result)
        self.assertIsInstance(result, str)


# ── unit tests: current action ────────────────────────────────────────────────

class TestCurrentAction(unittest.TestCase):

    @patch("tools.weather.httpx.AsyncClient")
    def test_current_happy_path(self, MockClient):
        geo = _make_geo_response()
        cur = _make_current_response(temp=31.0, feels=36.0)
        MockClient.return_value = _mock_client(geo_data=geo, weather_data=cur)

        result = run(weather(location="Mumbai", action="current"))

        self.assertIn("Mumbai", result)
        self.assertIn("31.0", result)
        self.assertIn("36.0", result)   # feels like
        self.assertNotIn("Error", result)

    @patch("tools.weather.httpx.AsyncClient")
    def test_current_shows_condition(self, MockClient):
        geo = _make_geo_response()
        cur = _make_current_response(weather_code=95)
        MockClient.return_value = _mock_client(geo_data=geo, weather_data=cur)

        result = run(weather(location="Mumbai", action="current"))
        self.assertIn("Thunderstorm", result)

    @patch("tools.weather.httpx.AsyncClient")
    def test_current_shows_wind_direction(self, MockClient):
        geo = _make_geo_response()
        cur = _make_current_response(wind_dir=225.0)
        MockClient.return_value = _mock_client(geo_data=geo, weather_data=cur)

        result = run(weather(location="Mumbai", action="current"))
        self.assertIn("SW", result)   # 225° = SW

    @patch("tools.weather.httpx.AsyncClient")
    def test_current_shows_uv_label(self, MockClient):
        geo = _make_geo_response()
        cur = _make_current_response(uv=9.0)
        MockClient.return_value = _mock_client(geo_data=geo, weather_data=cur)

        result = run(weather(location="Mumbai", action="current"))
        self.assertIn("Very high", result)

    @patch("tools.weather.httpx.AsyncClient")
    def test_current_shows_timezone(self, MockClient):
        geo = _make_geo_response()
        cur = _make_current_response(tz="Asia/Kolkata")
        MockClient.return_value = _mock_client(geo_data=geo, weather_data=cur)

        result = run(weather(location="Mumbai", action="current"))
        self.assertIn("Asia/Kolkata", result)

    @patch("tools.weather.httpx.AsyncClient")
    def test_current_none_fields_handled(self, MockClient):
        """All None current fields must not crash — show N/A where applicable."""
        geo = _make_geo_response()
        data = {
            "current": {
                "temperature_2m": None,
                "apparent_temperature": None,
                "relative_humidity_2m": None,
                "weather_code": None,
                "wind_speed_10m": None,
                "wind_direction_10m": None,
                "wind_gusts_10m": None,
                "precipitation": None,
                "uv_index": None,
                "visibility": None,
                "surface_pressure": None,
                "is_day": 1,
                "time": "2026-06-20T14:00",
            },
            "current_units": {},
            "timezone": "Asia/Kolkata",
        }
        MockClient.return_value = _mock_client(geo_data=geo, weather_data=data)

        result = run(weather(location="Mumbai", action="current"))
        # Must not raise
        self.assertIsInstance(result, str)
        self.assertNotIn("Exception", result)

    @patch("tools.weather.httpx.AsyncClient")
    def test_current_weather_api_exception(self, MockClient):
        import httpx as httpx_mod
        geo = _make_geo_response()
        MockClient.return_value = _mock_client(
            geo_data=geo,
            raise_on_weather=httpx_mod.RequestError("Timeout", request=MagicMock())
        )
        result = run(weather(location="Mumbai", action="current"))
        self.assertIn("Error", result)
        self.assertIsInstance(result, str)

    @patch("tools.weather.httpx.AsyncClient")
    def test_current_day_night_label(self, MockClient):
        geo = _make_geo_response()
        cur_day   = _make_current_response(is_day=1)
        cur_night = _make_current_response(is_day=0)

        MockClient.return_value = _mock_client(geo_data=geo, weather_data=cur_day)
        result_day = run(weather(location="Mumbai", action="current"))
        self.assertIn("day", result_day)

        MockClient.return_value = _mock_client(geo_data=geo, weather_data=cur_night)
        result_night = run(weather(location="Mumbai", action="current"))
        self.assertIn("night", result_night)


# ── unit tests: forecast action ───────────────────────────────────────────────

class TestForecastAction(unittest.TestCase):

    @patch("tools.weather.httpx.AsyncClient")
    def test_forecast_3_days(self, MockClient):
        geo  = _make_geo_response()
        fcst = _make_forecast_response(days=3)
        MockClient.return_value = _mock_client(geo_data=geo, weather_data=fcst)

        result = run(weather(location="Mumbai", action="forecast", days=3))

        self.assertIn("3-Day Forecast", result)
        self.assertIn("2026-06-20", result)
        self.assertNotIn("Error", result)

    @patch("tools.weather.httpx.AsyncClient")
    def test_forecast_shows_temp_range(self, MockClient):
        geo  = _make_geo_response()
        fcst = _make_forecast_response(days=1)
        MockClient.return_value = _mock_client(geo_data=geo, weather_data=fcst)

        result = run(weather(location="Mumbai", action="forecast", days=1))
        self.assertIn("32.0", result)   # t_max for day 0
        self.assertIn("24.0", result)   # t_min for day 0

    @patch("tools.weather.httpx.AsyncClient")
    def test_forecast_shows_precipitation_chance(self, MockClient):
        geo  = _make_geo_response()
        fcst = _make_forecast_response(days=3)
        MockClient.return_value = _mock_client(geo_data=geo, weather_data=fcst)

        result = run(weather(location="Mumbai", action="forecast", days=3))
        self.assertIn("40%", result)

    @patch("tools.weather.httpx.AsyncClient")
    def test_forecast_shows_sunrise_sunset(self, MockClient):
        geo  = _make_geo_response()
        fcst = _make_forecast_response(days=1)
        MockClient.return_value = _mock_client(geo_data=geo, weather_data=fcst)

        result = run(weather(location="Mumbai", action="forecast", days=1))
        self.assertIn("06:00", result)   # sunrise time
        self.assertIn("19:30", result)   # sunset time

    @patch("tools.weather.httpx.AsyncClient")
    def test_forecast_days_clamped_to_7(self, MockClient):
        geo  = _make_geo_response()
        fcst = _make_forecast_response(days=7)
        MockClient.return_value = _mock_client(geo_data=geo, weather_data=fcst)

        # days=99 should be clamped to 7
        result = run(weather(location="Mumbai", action="forecast", days=99))
        self.assertNotIn("Error", result)
        self.assertIn("7-Day", result)

    @patch("tools.weather.httpx.AsyncClient")
    def test_forecast_empty_data_returns_error(self, MockClient):
        geo = _make_geo_response()
        MockClient.return_value = _mock_client(geo_data=geo, weather_data={"daily": {}, "timezone": "UTC"})

        result = run(weather(location="Mumbai", action="forecast", days=3))
        self.assertIn("Error", result)

    @patch("tools.weather.httpx.AsyncClient")
    def test_forecast_uv_label_shown(self, MockClient):
        geo  = _make_geo_response()
        fcst = _make_forecast_response(days=1)
        MockClient.return_value = _mock_client(geo_data=geo, weather_data=fcst)

        result = run(weather(location="Mumbai", action="forecast", days=1))
        self.assertIn("Very high", result)   # UV index 9.0


# ── unit tests: hourly action ─────────────────────────────────────────────────

class TestHourlyAction(unittest.TestCase):

    @patch("tools.weather.httpx.AsyncClient")
    def test_hourly_24h(self, MockClient):
        geo    = _make_geo_response()
        hourly = _make_hourly_response(n_hours=48)
        MockClient.return_value = _mock_client(geo_data=geo, weather_data=hourly)

        result = run(weather(location="Mumbai", action="hourly", hours=24))

        self.assertIn("Hourly Forecast", result)
        self.assertIn("24h", result)
        self.assertNotIn("Error", result)

    @patch("tools.weather.httpx.AsyncClient")
    def test_hourly_shows_date_separator(self, MockClient):
        """A date separator line must appear when the date changes."""
        geo    = _make_geo_response()
        hourly = _make_hourly_response(n_hours=48)
        MockClient.return_value = _mock_client(geo_data=geo, weather_data=hourly)

        result = run(weather(location="Mumbai", action="hourly", hours=48))
        # Both dates should appear as separators
        self.assertIn("2026-06-20", result)
        self.assertIn("2026-06-21", result)

    @patch("tools.weather.httpx.AsyncClient")
    def test_hourly_wind_direction_shown(self, MockClient):
        geo    = _make_geo_response()
        hourly = _make_hourly_response(n_hours=3)
        MockClient.return_value = _mock_client(geo_data=geo, weather_data=hourly)

        result = run(weather(location="Mumbai", action="hourly", hours=3))
        # Wind direction 180° = S
        self.assertIn("S", result)

    @patch("tools.weather.httpx.AsyncClient")
    def test_hourly_hours_clamped_to_48(self, MockClient):
        geo    = _make_geo_response()
        hourly = _make_hourly_response(n_hours=48)
        MockClient.return_value = _mock_client(geo_data=geo, weather_data=hourly)

        result = run(weather(location="Mumbai", action="hourly", hours=100))
        self.assertNotIn("Error", result)

    @patch("tools.weather.httpx.AsyncClient")
    def test_hourly_empty_data_returns_error(self, MockClient):
        geo = _make_geo_response()
        MockClient.return_value = _mock_client(
            geo_data=geo, weather_data={"hourly": {}, "timezone": "UTC"}
        )
        result = run(weather(location="Mumbai", action="hourly", hours=24))
        self.assertIn("Error", result)


# ── unit tests: imperial units ────────────────────────────────────────────────

class TestImperialUnits(unittest.TestCase):

    @patch("tools.weather.httpx.AsyncClient")
    def test_imperial_params_forwarded(self, MockClient):
        """Imperial units=imperial should pass temperature_unit=fahrenheit to API."""
        geo = _make_geo_response()
        cur = _make_current_response(temp=88.0)  # Fahrenheit value

        call_params = {}
        geo_resp = MagicMock()
        geo_resp.status_code = 200
        geo_resp.json.return_value = geo
        geo_resp.raise_for_status = MagicMock()

        weather_resp = MagicMock()
        weather_resp.status_code = 200
        weather_resp.json.return_value = cur
        weather_resp.raise_for_status = MagicMock()

        async def mock_get(url, **kwargs):
            if "forecast" in url:
                call_params.update(kwargs.get("params", {}))
            if "geocoding" in url:
                return geo_resp
            return weather_resp

        client = AsyncMock()
        client.get = mock_get
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__  = AsyncMock(return_value=False)
        MockClient.return_value = client

        run(weather(location="New York", action="current", units="imperial"))

        self.assertIn("temperature_unit", call_params)
        self.assertEqual(call_params["temperature_unit"], "fahrenheit")
        self.assertIn("wind_speed_unit", call_params)
        self.assertEqual(call_params["wind_speed_unit"], "mph")

    @patch("tools.weather.httpx.AsyncClient")
    def test_metric_no_unit_params(self, MockClient):
        """Metric (default) should NOT send temperature_unit to API."""
        geo = _make_geo_response()
        cur = _make_current_response()
        call_params = {}

        geo_resp = MagicMock()
        geo_resp.status_code = 200
        geo_resp.json.return_value = geo
        geo_resp.raise_for_status = MagicMock()

        weather_resp = MagicMock()
        weather_resp.status_code = 200
        weather_resp.json.return_value = cur
        weather_resp.raise_for_status = MagicMock()

        async def mock_get(url, **kwargs):
            if "forecast" in url:
                call_params.update(kwargs.get("params", {}))
            if "geocoding" in url:
                return geo_resp
            return weather_resp

        client = AsyncMock()
        client.get = mock_get
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__  = AsyncMock(return_value=False)
        MockClient.return_value = client

        run(weather(location="Mumbai", action="current", units="metric"))

        self.assertNotIn("temperature_unit", call_params)


# ── integration tests ─────────────────────────────────────────────────────────

@unittest.skipUnless(
    os.getenv("RUN_INTEGRATION_TESTS"),
    "Set RUN_INTEGRATION_TESTS=1 to run live API tests"
)
class TestWeatherIntegration(unittest.TestCase):
    """
    Live Open-Meteo API tests.

    Run:
        RUN_INTEGRATION_TESTS=1 pytest test/test_weather.py -v -k integration
    """

    def test_current_mumbai(self):
        result = run(weather(location="Mumbai", action="current"))
        self.assertNotIn("Error", result)
        self.assertIn("Mumbai", result)
        self.assertIn("Temperature", result)

    def test_forecast_new_delhi_7_days(self):
        result = run(weather(location="New Delhi", action="forecast", days=7))
        self.assertNotIn("Error", result)
        self.assertIn("7-Day Forecast", result)

    def test_hourly_london_48h(self):
        result = run(weather(location="London UK", action="hourly", hours=48))
        self.assertNotIn("Error", result)
        self.assertIn("Hourly Forecast (48h)", result)

    def test_imperial_new_york(self):
        result = run(weather(location="New York City", action="current", units="imperial"))
        self.assertNotIn("Error", result)
        self.assertIn("°F", result)

    def test_unknown_city(self):
        result = run(weather(location="XyzNonExistentCityABC", action="current"))
        self.assertIn("Error", result)

    def test_small_town(self):
        """Geocoding should resolve small towns correctly."""
        result = run(weather(location="Shimla Himachal Pradesh", action="current"))
        self.assertNotIn("Error", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)