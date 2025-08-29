import aiohttp
import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

from ..models.schemas import TelemetryData, SpaceWeatherData, SolarFlareData, CMEData, TLEData

logger = logging.getLogger(__name__)

class DataIngestionService:
    def __init__(self):
        self.nasa_donki_url = "https://api.nasa.gov/DONKI"
        self.noaa_swpc_url = "https://services.swpc.noaa.gov/json"
        self.celestrak_url = "https://celestrak.org/NORAD/elements"
        self.nasa_api_key = os.getenv("NASA_API_KEY", "DEMO_KEY")
        
        # Cache for offline mode
        self.cache = {
            "solar_data": None,
            "space_weather": None,
            "tle_data": None,
            "last_update": None
        }
    
    async def fetch_solar_data(self) -> Dict:
        """Fetch solar flare and CME data from NASA DONKI API"""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=7)  # Last 7 days
            
            async with aiohttp.ClientSession() as session:
                # Fetch solar flares
                flare_url = f"{self.nasa_donki_url}/FLR"
                flare_params = {
                    "api_key": self.nasa_api_key,
                    "startDate": start_date.strftime("%Y-%m-%d"),
                    "endDate": end_date.strftime("%Y-%m-%d")
                }
                
                # Fetch CMEs
                cme_url = f"{self.nasa_donki_url}/CME"
                cme_params = {
                    "api_key": self.nasa_api_key,
                    "startDate": start_date.strftime("%Y-%m-%d"),
                    "endDate": end_date.strftime("%Y-%m-%d")
                }
                
                async with session.get(flare_url, params=flare_params) as flare_response:
                    flare_data = await flare_response.json() if flare_response.status == 200 else []
                
                async with session.get(cme_url, params=cme_params) as cme_response:
                    cme_data = await cme_response.json() if cme_response.status == 200 else []
                
                # Process and cache data
                processed_data = self._process_solar_data(flare_data, cme_data)
                self.cache["solar_data"] = processed_data
                self.cache["last_update"] = datetime.utcnow()
                
                return processed_data
                
        except Exception as e:
            logger.error(f"Error fetching solar data: {e}")
            # Return cached data or simulated data
            return self._get_simulated_solar_data()
    
    async def fetch_space_weather(self) -> Dict:
        """Fetch space weather data from NOAA SWPC"""
        try:
            async with aiohttp.ClientSession() as session:
                # Fetch current space weather indices
                endpoints = {
                    "planetary_k_index": f"{self.noaa_swpc_url}/planetary_k_index_1m.json",
                    "solar_wind": f"{self.noaa_swpc_url}/rtsw/rtsw_mag_1m.json",
                    "proton_flux": f"{self.noaa_swpc_url}/goes/primary/differential-protons-1-day.json"
                }
                
                results = {}
                for key, url in endpoints.items():
                    try:
                        async with session.get(url) as response:
                            if response.status == 200:
                                results[key] = await response.json()
                            else:
                                results[key] = []
                    except Exception as e:
                        logger.warning(f"Failed to fetch {key}: {e}")
                        results[key] = []
                
                # Process space weather data
                processed_data = self._process_space_weather_data(results)
                self.cache["space_weather"] = processed_data
                
                return processed_data
                
        except Exception as e:
            logger.error(f"Error fetching space weather: {e}")
            return self._get_simulated_space_weather()
    
    async def fetch_tle_data(self) -> List[str]:
        """Fetch TLE data from CelesTrak"""
        try:
            # Fetch active satellites TLE data
            tle_endpoints = [
                f"{self.celestrak_url}/gp.php?GROUP=active&FORMAT=tle",
                f"{self.celestrak_url}/gp.php?GROUP=stations&FORMAT=tle"
            ]
            
            tle_data = []
            async with aiohttp.ClientSession() as session:
                for url in tle_endpoints:
                    try:
                        async with session.get(url) as response:
                            if response.status == 200:
                                content = await response.text()
                                tle_data.extend(self._parse_tle_data(content))
                    except Exception as e:
                        logger.warning(f"Failed to fetch TLE from {url}: {e}")
            
            self.cache["tle_data"] = tle_data
            return tle_data
            
        except Exception as e:
            logger.error(f"Error fetching TLE data: {e}")
            return self._get_simulated_tle_data()
    
    async def get_telemetry_snapshot(self) -> TelemetryData:
        """Generate current telemetry snapshot"""
        # In a real system, this would come from spacecraft systems
        # For simulation, we'll generate realistic data
        
        import numpy as np
        
        # Simulate orbital position (LEO orbit around Earth)
        t = datetime.utcnow().timestamp()
        orbital_period = 5400  # 90 minutes in seconds
        
        # Simple circular orbit simulation
        angle = (t % orbital_period) / orbital_period * 2 * np.pi
        altitude = 400  # km
        earth_radius = 6371  # km
        orbit_radius = earth_radius + altitude
        
        position = [
            orbit_radius * np.cos(angle),
            orbit_radius * np.sin(angle),
            0.0
        ]
        
        # Orbital velocity for circular orbit
        orbital_velocity = 7.66  # km/s for 400km altitude
        velocity = [
            -orbital_velocity * np.sin(angle),
            orbital_velocity * np.cos(angle),
            0.0
        ]
        
        return TelemetryData(
            timestamp=datetime.utcnow().isoformat(),
            spacecraft_position=position,
            spacecraft_velocity=velocity,
            fuel_remaining=85.3,
            solar_panel_efficiency=94.7,
            radiation_level=2.3,
            communication_status="nominal",
            system_health={
                "propulsion": "nominal",
                "power": "nominal",
                "thermal": "nominal",
                "navigation": "nominal",
                "communication": "nominal"
            }
        )
    
    def _process_solar_data(self, flare_data: List, cme_data: List) -> Dict:
        """Process raw solar data into structured format"""
        processed_flares = []
        for flare in flare_data:
            if flare.get("classType"):
                processed_flares.append(SolarFlareData(
                    event_id=flare.get("flrID", "unknown"),
                    start_time=flare.get("beginTime", ""),
                    peak_time=flare.get("peakTime", ""),
                    end_time=flare.get("endTime", ""),
                    flare_class=flare.get("classType", "C"),
                    source_location=flare.get("sourceLocation", "unknown"),
                    intensity=self._parse_flare_intensity(flare.get("classType", "C1.0"))
                ))
        
        processed_cmes = []
        for cme in cme_data:
            if cme.get("activityID"):
                # Extract CME analysis data
                analysis = cme.get("cmeAnalyses", [{}])
                if analysis:
                    analysis = analysis[0]
                    processed_cmes.append(CMEData(
                        event_id=cme.get("activityID", "unknown"),
                        start_time=cme.get("startTime", ""),
                        speed=float(analysis.get("speed", 400)),
                        direction=float(analysis.get("longitude", 0)),
                        half_angle=float(analysis.get("halfAngle", 30)),
                        catalog=analysis.get("catalog", "unknown")
                    ))
        
        return {
            "solar_flares": [flare.dict() for flare in processed_flares],
            "cmes": [cme.dict() for cme in processed_cmes],
            "last_updated": datetime.utcnow().isoformat()
        }
    
    def _process_space_weather_data(self, raw_data: Dict) -> Dict:
        """Process raw space weather data"""
        # Extract latest values from time series data
        kp_data = raw_data.get("planetary_k_index", [])
        kp_index = kp_data[-1].get("kp", 1.0) if kp_data else 1.0
        
        solar_wind_data = raw_data.get("solar_wind", [])
        solar_wind_speed = solar_wind_data[-1].get("speed", 400) if solar_wind_data else 400
        
        proton_data = raw_data.get("proton_flux", [])
        proton_flux = proton_data[-1].get("flux", 1.0) if proton_data else 1.0
        
        return {
            "kp_index": float(kp_index),
            "dst_index": -20.0,  # Simulated
            "solar_wind_speed": float(solar_wind_speed),
            "proton_flux": float(proton_flux),
            "last_updated": datetime.utcnow().isoformat()
        }
    
    def _parse_tle_data(self, tle_content: str) -> List[Dict]:
        """Parse TLE data into structured format"""
        lines = tle_content.strip().split('\n')
        tle_data = []
        
        for i in range(0, len(lines), 3):
            if i + 2 < len(lines):
                tle_data.append({
                    "satellite_name": lines[i].strip(),
                    "line1": lines[i + 1].strip(),
                    "line2": lines[i + 2].strip(),
                    "epoch": self._extract_epoch_from_tle(lines[i + 1])
                })
        
        return tle_data
    
    def _extract_epoch_from_tle(self, line1: str) -> str:
        """Extract epoch from TLE line 1"""
        try:
            epoch_part = line1[18:32]
            # Convert TLE epoch to readable format
            return f"2024-{epoch_part}"  # Simplified
        except:
            return datetime.utcnow().isoformat()
    
    def _parse_flare_intensity(self, class_type: str) -> float:
        """Convert flare class to numerical intensity"""
        try:
            class_letter = class_type[0].upper()
            magnitude = float(class_type[1:])
            
            multipliers = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}
            return multipliers.get(class_letter, 1e-6) * magnitude
        except:
            return 1e-6
    
    def _get_simulated_solar_data(self) -> Dict:
        """Return simulated solar data for offline mode"""
        return {
            "solar_flares": [
                {
                    "event_id": "SIM_001",
                    "start_time": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
                    "peak_time": (datetime.utcnow() - timedelta(hours=1)).isoformat(),
                    "end_time": datetime.utcnow().isoformat(),
                    "flare_class": "M2.1",
                    "source_location": "N15E23",
                    "intensity": 2.1e-5
                }
            ],
            "cmes": [],
            "last_updated": datetime.utcnow().isoformat()
        }
    
    def _get_simulated_space_weather(self) -> Dict:
        """Return simulated space weather data"""
        return {
            "kp_index": 2.0,
            "dst_index": -15.0,
            "solar_wind_speed": 420.0,
            "proton_flux": 1.5,
            "last_updated": datetime.utcnow().isoformat()
        }
    
    def _get_simulated_tle_data(self) -> List[Dict]:
        """Return simulated TLE data"""
        return [
            {
                "satellite_name": "ISS (ZARYA)",
                "line1": "1 25544U 98067A   24240.12345678  .00001234  00000-0  12345-4 0  9999",
                "line2": "2 25544  51.6400 123.4567 0001234  12.3456 123.4567 15.50000000123456",
                "epoch": datetime.utcnow().isoformat()
            }
        ]