"""
Unit tests for the weather tool.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from runtime.tools.weather import weather, get_weather_desc

def test_get_weather_desc():
    assert get_weather_desc(0) == "Clear sky"
    assert get_weather_desc(95) == "Thunderstorm"
    assert "Unknown" in get_weather_desc(999)

@pytest.mark.anyio
async def test_weather_empty_location():
    res = await weather("")
    assert "[WEATHER ERROR: Empty location provided]" in res

@pytest.mark.anyio
async def test_weather_successful_current_only():
    # Mock geocoding response
    geo_data = {
        "results": [
            {
                "name": "Paris",
                "latitude": 48.8566,
                "longitude": 2.3522,
                "country": "France",
                "admin1": "Île-de-France"
            }
        ]
    }
    
    # Mock forecast response
    weather_data = {
        "current": {
            "temperature_2m": 18.5,
            "relative_humidity_2m": 60,
            "weather_code": 1,
            "wind_speed_10m": 12.0
        },
        "current_units": {
            "temperature_2m": "°C",
            "relative_humidity_2m": "%",
            "wind_speed_10m": "km/h"
        }
    }
    
    with patch("httpx.AsyncClient") as MockClientClass:
        mock_client = AsyncMock()
        MockClientClass.return_value.__aenter__.return_value = mock_client
        
        # Geocoding call
        geo_resp = MagicMock()
        geo_resp.status_code = 200
        geo_resp.json.return_value = geo_data
        
        # Weather call
        weather_resp = MagicMock()
        weather_resp.status_code = 200
        weather_resp.json.return_value = weather_data
        
        mock_client.get.side_effect = [geo_resp, weather_resp]
        
        res = await weather("Paris", forecast=False)
        
        assert "Paris, Île-de-France, France" in res
        assert "Temperature: 18.5 °C" in res
        assert "Condition: Mainly clear" in res
        assert "Relative Humidity: 60%" in res
        assert "Wind Speed: 12.0 km/h" in res
        assert "3-Day Forecast:" not in res

@pytest.mark.anyio
async def test_weather_successful_with_forecast():
    geo_data = {
        "results": [
            {
                "name": "Tokyo",
                "latitude": 35.6762,
                "longitude": 139.6503,
                "country": "Japan"
            }
        ]
    }
    
    weather_data = {
        "current": {
            "temperature_2m": 22.0,
            "relative_humidity_2m": 75,
            "weather_code": 3,
            "wind_speed_10m": 8.5
        },
        "current_units": {
            "temperature_2m": "°C",
            "relative_humidity_2m": "%",
            "wind_speed_10m": "km/h"
        },
        "daily": {
            "time": ["2026-06-19", "2026-06-20", "2026-06-21"],
            "temperature_2m_max": [25.0, 24.5, 23.0],
            "temperature_2m_min": [19.0, 18.5, 17.0],
            "precipitation_sum": [0.0, 5.5, 12.0],
            "weather_code": [0, 61, 95]
        },
        "daily_units": {
            "precipitation_sum": "mm"
        }
    }
    
    with patch("httpx.AsyncClient") as MockClientClass:
        mock_client = AsyncMock()
        MockClientClass.return_value.__aenter__.return_value = mock_client
        
        geo_resp = MagicMock()
        geo_resp.status_code = 200
        geo_resp.json.return_value = geo_data
        
        weather_resp = MagicMock()
        weather_resp.status_code = 200
        weather_resp.json.return_value = weather_data
        
        mock_client.get.side_effect = [geo_resp, weather_resp]
        
        res = await weather("Tokyo", forecast=True)
        
        assert "Tokyo, Japan" in res
        assert "Temperature: 22.0 °C" in res
        assert "3-Day Forecast:" in res
        assert "2026-06-19: Min 19.0°C, Max 25.0°C | Clear sky | Precipitation: 0.0 mm" in res
        assert "2026-06-20: Min 18.5°C, Max 24.5°C | Slight rain | Precipitation: 5.5 mm" in res
        assert "2026-06-21: Min 17.0°C, Max 23.0°C | Thunderstorm | Precipitation: 12.0 mm" in res

@pytest.mark.anyio
async def test_weather_location_not_found():
    geo_data = {"results": []}
    
    with patch("httpx.AsyncClient") as MockClientClass:
        mock_client = AsyncMock()
        MockClientClass.return_value.__aenter__.return_value = mock_client
        
        geo_resp = MagicMock()
        geo_resp.status_code = 200
        geo_resp.json.return_value = geo_data
        
        mock_client.get.return_value = geo_resp
        
        res = await weather("Atlantis")
        assert "[WEATHER ERROR: Could not find location 'Atlantis']" in res

@pytest.mark.anyio
async def test_weather_geocoding_failure():
    with patch("httpx.AsyncClient") as MockClientClass:
        mock_client = AsyncMock()
        MockClientClass.return_value.__aenter__.return_value = mock_client
        
        mock_client.get.side_effect = Exception("Connection timeout")
        
        res = await weather("London")
        assert "[WEATHER ERROR: Geocoding API failed for 'London'" in res
        assert "Connection timeout" in res

@pytest.mark.anyio
async def test_weather_forecast_failure():
    geo_data = {
        "results": [
            {
                "name": "London",
                "latitude": 51.5074,
                "longitude": -0.1278,
                "country": "United Kingdom"
            }
        ]
    }
    
    with patch("httpx.AsyncClient") as MockClientClass:
        mock_client = AsyncMock()
        MockClientClass.return_value.__aenter__.return_value = mock_client
        
        geo_resp = MagicMock()
        geo_resp.status_code = 200
        geo_resp.json.return_value = geo_data
        
        mock_client.get.side_effect = [geo_resp, Exception("API Error")]
        
        res = await weather("London")
        assert "[WEATHER ERROR: Forecast API failed for 'London, United Kingdom'" in res
        assert "API Error" in res
