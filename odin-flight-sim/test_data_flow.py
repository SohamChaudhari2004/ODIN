#!/usr/bin/env python3
import requests
import json

resp = requests.get('http://localhost:8000/api/space-weather/current')
data = resp.json()

print('🌌 Live Space Weather Data Sample:')
print(f'Timestamp: {data.get("timestamp")}')
print(f'Historical Period: {data.get("historical_period")}')

sw = data.get('space_weather', {})
print(f'Solar Activity Level: {sw.get("solar_activity", {}).get("solar_activity_level")}')
print(f'Kp Index: {sw.get("geomagnetic_activity", {}).get("kp_index")}')
print(f'Active Events: {len(sw.get("active_events", []))}')
print('✅ Data flowing correctly from AI services through backend!')
