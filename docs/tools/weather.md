# weather

**← [Back to Tool Guide](../TOOL_GUIDE.md)**

**File:** `runtime/tools/weather.py`
**Env vars required:** none — Open-Meteo (forecast + geocoding) is keyless
**Execution pattern:** agentic
**Added:** Mark 3

## When it's used

- User asks about current weather, "should I bring an umbrella," or a multi-day/hourly forecast for any location worldwide
- Classifier sets `domains: [weather]` and `tools_needed: [weather]`

## Parameters

| Parameter  | Type    | Required | Constraints                              | Description |
|-------------|---------|----------|---------------------------------------------|--------------|
| `location`  | string  | yes      | 1–200 chars                                  | City, region, or country. Be specific — `"Bengaluru Karnataka"` resolves better than `"Bengaluru"` alone |
| `action`    | string  | no       | enum: `current` (default), `forecast`, `hourly` | Which data to return |
| `days`      | integer | no       | 1–7, default 3                               | Forecast days for `action='forecast'` |
| `hours`     | integer | no       | 1–48, default 24                             | Hours for `action='hourly'` |
| `units`     | string  | no       | enum: `metric` (default), `imperial`         | `metric` = °C/km/h/mm, `imperial` = °F/mph/inch |

## Actions

| Action      | What it returns |
|--------------|-------------------|
| `current`    | Temperature, feels-like, humidity, condition (day/night aware), wind speed/direction/gusts, current-hour precipitation, UV index + category label, visibility, surface pressure. **Default.** |
| `forecast`   | Per-day: min/max temp (+ feels-like range), condition, precipitation sum + chance, max wind, UV index, sunrise/sunset. 1–7 days. |
| `hourly`     | Per-hour table: temp, feels-like, condition, rain chance, wind + direction, UV index. Date separators inserted automatically when the day changes. 1–48 hours. |

## Example LLM calls

**Current conditions:**
```json
{ "name": "weather", "input": { "location": "Mumbai" } }
```

**7-day forecast:**
```json
{ "name": "weather", "input": { "location": "New Delhi", "action": "forecast", "days": 7 } }
```

**48-hour hourly outlook:**
```json
{ "name": "weather", "input": { "location": "London UK", "action": "hourly", "hours": 48 } }
```

**Imperial units:**
```json
{ "name": "weather", "input": { "location": "New York City", "action": "current", "units": "imperial" } }
```

## Return format (current)

```
Weather: Mumbai, Maharashtra, India
Timezone: Asia/Kolkata  |  As of: 2026-06-20T14:00
Coordinates: 19.0760°, 72.8777°

Condition:    Mainly clear (day)
Temperature:  31.0°C  (feels like 36.0°C)
Humidity:     78%

Wind:         18.0km/h from SW (225°)
Gusts:        30.0km/h
Precipitation:0.0mm (current hour)
UV index:     8 (Very high)
Visibility:   10.0 km
Pressure:     1012.0 hPa
```

## Failure modes

| Condition                            | Return value |
|------------------------------------------|-----------------|
| Empty/whitespace-only location              | `Error: 'location' must not be empty.` |
| Unknown action                                | `Error: Unknown action '<action>'. Valid: current, forecast, hourly.` |
| Location not found by geocoding                | `Error: Could not find location '<location>'. Try a more specific name.` |
| Forecast API request fails                       | `Error: Weather API failed for '<location>' — <detail>.` (or `Forecast API failed` / `Hourly forecast API failed` depending on action) |
| Empty daily/hourly payload returned                 | `Error: No forecast data returned for '<location>'.` / `Error: No hourly data returned for '<location>'.` |
| Unexpected exception                                   | `Error: Weather tool failed unexpectedly — <ExceptionType>: <detail>` |

## Notes

- `days`/`hours` are clamped (not rejected) to their valid ranges — passing `days=99` silently becomes `days=7`.
- The resolved location string (city, admin region, country) is always shown in the output so the user/LLM can confirm geocoding matched the right place — useful for ambiguous names like "Springfield."
- Wind direction is converted to a 16-point compass bearing (N, NNE, NE, …) rather than shown as a raw degree value.
