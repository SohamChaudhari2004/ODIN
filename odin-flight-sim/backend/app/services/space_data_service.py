"""
NASA DONKI and Space-Track Data Service
Fetches and caches historical space weather and debris data (2012-2018)
"""

import asyncio
import aiohttp
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from spacetrack import SpaceTrackApi
    SPACETRACK_AVAILABLE = True
except ImportError:
    SPACETRACK_AVAILABLE = False
    print("⚠️ SpaceTrack API not available")

logger = logging.getLogger(__name__)

class NASADONKIService:
    """NASA DONKI (Database Of Notifications, Knowledge, Information) Service"""
    
    def __init__(self, api_key: str = "DEMO_KEY"):
        self.api_key = api_key
        self.base_url = "https://api.nasa.gov/DONKI"
        self.cache = {}
        self.session = None
        
        # Historical data range
        self.start_year = 2012
        self.end_year = 2018
        
        logger.info("NASA DONKI Service initialized")
    
    async def initialize_session(self):
        """Initialize aiohttp session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
    
    async def fetch_historical_solar_flares(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Fetch historical solar flare data from NASA DONKI"""
        
        await self.initialize_session()
        
        try:
            url = f"{self.base_url}/FLR"
            params = {
                "startDate": start_date.strftime("%Y-%m-%d"),
                "endDate": end_date.strftime("%Y-%m-%d"),
                "api_key": self.api_key
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return await self._process_solar_flare_data(data)
                else:
                    logger.error(f"NASA DONKI API error: {response.status}")
                    return await self._generate_mock_solar_flare_data(start_date, end_date)
                    
        except Exception as e:
            logger.error(f"Error fetching solar flare data: {e}")
            return await self._generate_mock_solar_flare_data(start_date, end_date)
    
    async def fetch_historical_cmes(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Fetch historical Coronal Mass Ejection data"""
        
        await self.initialize_session()
        
        try:
            url = f"{self.base_url}/CME"
            params = {
                "startDate": start_date.strftime("%Y-%m-%d"),
                "endDate": end_date.strftime("%Y-%m-%d"),
                "api_key": self.api_key
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return await self._process_cme_data(data)
                else:
                    logger.error(f"NASA DONKI CME API error: {response.status}")
                    return await self._generate_mock_cme_data(start_date, end_date)
                    
        except Exception as e:
            logger.error(f"Error fetching CME data: {e}")
            return await self._generate_mock_cme_data(start_date, end_date)
    
    async def fetch_geomagnetic_storms(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Fetch historical geomagnetic storm data"""
        
        await self.initialize_session()
        
        try:
            url = f"{self.base_url}/GST"
            params = {
                "startDate": start_date.strftime("%Y-%m-%d"),
                "endDate": end_date.strftime("%Y-%m-%d"),
                "api_key": self.api_key
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return await self._process_geomagnetic_data(data)
                else:
                    return await self._generate_mock_geomagnetic_data(start_date, end_date)
                    
        except Exception as e:
            logger.error(f"Error fetching geomagnetic storm data: {e}")
            return await self._generate_mock_geomagnetic_data(start_date, end_date)
    
    async def _process_solar_flare_data(self, raw_data: List[Dict]) -> List[Dict[str, Any]]:
        """Process raw NASA DONKI solar flare data"""
        processed_flares = []
        
        for flare in raw_data:
            processed_flare = {
                "hazard_id": f"flare_{flare.get('flrID', 'unknown')}",
                "hazard_type": "solar_flare",
                "flare_class": flare.get("classType", "C"),
                "magnitude": self._parse_flare_magnitude(flare.get("classType", "C1.0")),
                "start_time": self._parse_donki_time(flare.get("beginTime")),
                "peak_time": self._parse_donki_time(flare.get("peakTime")),
                "end_time": self._parse_donki_time(flare.get("endTime")),
                "source_location": flare.get("sourceLocation", "N00W00"),
                "active_region": flare.get("activeRegionNum"),
                "impact_radius_km": self._calculate_flare_impact_radius(flare.get("classType", "C")),
                "severity": self._calculate_flare_severity(flare.get("classType", "C")),
                "source_data": flare
            }
            processed_flares.append(processed_flare)
        
        return processed_flares
    
    async def _process_cme_data(self, raw_data: List[Dict]) -> List[Dict[str, Any]]:
        """Process raw NASA DONKI CME data"""
        processed_cmes = []
        
        for cme in raw_data:
            processed_cme = {
                "hazard_id": f"cme_{cme.get('activityID', 'unknown')}",
                "hazard_type": "cme",
                "start_time": self._parse_donki_time(cme.get("startTime")),
                "speed_km_s": self._extract_cme_speed(cme),
                "direction": self._extract_cme_direction(cme),
                "impact_radius_km": 1000000,  # CMEs affect large areas
                "severity": self._calculate_cme_severity(self._extract_cme_speed(cme)),
                "source_data": cme
            }
            processed_cmes.append(processed_cme)
        
        return processed_cmes
    
    def _parse_flare_magnitude(self, class_type: str) -> float:
        """Parse flare magnitude from class type (e.g., 'X2.1' -> 2.1)"""
        try:
            if not class_type:
                return 1.0
            
            magnitude_str = class_type[1:]  # Remove the class letter
            return float(magnitude_str)
        except:
            return 1.0
    
    def _parse_donki_time(self, time_str: str) -> datetime:
        """Parse NASA DONKI time format"""
        try:
            if not time_str:
                return datetime.utcnow()
            
            # Remove 'Z' and parse
            clean_time = time_str.replace('Z', '')
            return datetime.fromisoformat(clean_time)
        except:
            return datetime.utcnow()
    
    def _calculate_flare_impact_radius(self, flare_class: str) -> float:
        """Calculate flare impact radius based on class"""
        class_multipliers = {
            "X": 500000,  # 500,000 km
            "M": 200000,  # 200,000 km
            "C": 100000,  # 100,000 km
            "B": 50000,   # 50,000 km
            "A": 25000    # 25,000 km
        }
        
        if flare_class and len(flare_class) > 0:
            return class_multipliers.get(flare_class[0], 100000)
        return 100000
    
    def _calculate_flare_severity(self, flare_class: str) -> float:
        """Calculate flare severity (0.0 to 1.0)"""
        if not flare_class:
            return 0.1
        
        class_letter = flare_class[0] if flare_class else "C"
        magnitude = self._parse_flare_magnitude(flare_class)
        
        base_severity = {
            "X": 0.8,
            "M": 0.6,
            "C": 0.3,
            "B": 0.1,
            "A": 0.05
        }.get(class_letter, 0.3)
        
        # Scale by magnitude (typical range 1-9)
        magnitude_factor = min(magnitude / 9.0, 1.0)
        
        return min(base_severity + magnitude_factor * 0.2, 1.0)
    
    def _extract_cme_speed(self, cme_data: Dict) -> float:
        """Extract CME speed from analysis data"""
        try:
            analyses = cme_data.get("cmeAnalyses", [])
            if analyses:
                speed = analyses[0].get("speed", 400)
                return float(speed)
        except:
            pass
        return 400.0  # Default speed km/s
    
    def _extract_cme_direction(self, cme_data: Dict) -> Dict[str, float]:
        """Extract CME direction from analysis data"""
        try:
            analyses = cme_data.get("cmeAnalyses", [])
            if analyses:
                analysis = analyses[0]
                return {
                    "longitude": analysis.get("longitude", 0.0),
                    "latitude": analysis.get("latitude", 0.0),
                    "half_angle": analysis.get("halfAngle", 30.0)
                }
        except:
            pass
        return {"longitude": 0.0, "latitude": 0.0, "half_angle": 30.0}
    
    def _calculate_cme_severity(self, speed: float) -> float:
        """Calculate CME severity based on speed"""
        # Typical CME speeds: 300-2000 km/s
        if speed < 500:
            return 0.2
        elif speed < 800:
            return 0.4
        elif speed < 1200:
            return 0.6
        elif speed < 1600:
            return 0.8
        else:
            return 1.0
    
    # Mock data generators for testing/fallback
    
    async def _generate_mock_solar_flare_data(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Generate mock solar flare data for testing"""
        mock_flares = []
        current_date = start_date
        
        while current_date < end_date:
            # Generate random flares (about 1 per week)
            if current_date.weekday() == 0:  # Monday
                flare = {
                    "hazard_id": f"mock_flare_{current_date.strftime('%Y%m%d')}",
                    "hazard_type": "solar_flare",
                    "flare_class": "M2.5",
                    "magnitude": 2.5,
                    "start_time": current_date + timedelta(hours=12),
                    "peak_time": current_date + timedelta(hours=12, minutes=30),
                    "end_time": current_date + timedelta(hours=13),
                    "source_location": "N15W30",
                    "active_region": "1234",
                    "impact_radius_km": 200000,
                    "severity": 0.6,
                    "source_data": {"mock": True}
                }
                mock_flares.append(flare)
            
            current_date += timedelta(days=1)
        
        return mock_flares
    
    async def _generate_mock_cme_data(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Generate mock CME data for testing"""
        mock_cmes = []
        current_date = start_date
        
        while current_date < end_date:
            # Generate random CMEs (about 1 per month)
            if current_date.day == 15:  # Mid-month
                cme = {
                    "hazard_id": f"mock_cme_{current_date.strftime('%Y%m%d')}",
                    "hazard_type": "cme",
                    "start_time": current_date + timedelta(hours=8),
                    "speed_km_s": 650.0,
                    "direction": {"longitude": 0.0, "latitude": 0.0, "half_angle": 45.0},
                    "impact_radius_km": 1000000,
                    "severity": 0.5,
                    "source_data": {"mock": True}
                }
                mock_cmes.append(cme)
            
            current_date += timedelta(days=1)
        
        return mock_cmes

    async def _process_geomagnetic_data(self, raw_data: List[Dict]) -> List[Dict[str, Any]]:
        """Process raw NASA DONKI geomagnetic storm data"""
        processed_storms = []

        for storm in raw_data:
            processed_storm = {
                "hazard_id": f"gst_{storm.get('gstID', 'unknown')}",
                "hazard_type": "geomagnetic_storm",
                "start_time": self._parse_donki_time(storm.get("startTime")),
                "kp_index": self._extract_kp_index(storm),
                "severity": self._calculate_gst_severity(self._extract_kp_index(storm)),
                "source_data": storm
            }
            processed_storms.append(processed_storm)

        return processed_storms

    def _extract_kp_index(self, storm_data: Dict) -> int:
        """Extract Kp index from geomagnetic storm data"""
        try:
            all_kp_index = storm_data.get("allKpIndex", [])
            if all_kp_index:
                return all_kp_index[0].get("kpIndex", 0)
        except:
            pass
        return 0

    def _calculate_gst_severity(self, kp_index: int) -> float:
        """Calculate geomagnetic storm severity based on Kp-index"""
        if kp_index < 4:
            return 0.1
        elif kp_index < 6:
            return 0.4
        elif kp_index < 8:
            return 0.7
        else:
            return 0.9
    
    async def _generate_mock_geomagnetic_data(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Generate mock geomagnetic storm data"""
        mock_storms = []
        current_date = start_date

        while current_date < end_date:
            # Generate random storms (about 1 per 2 weeks)
            if current_date.day % 14 == 0:
                storm = {
                    "hazard_id": f"mock_gst_{current_date.strftime('%Y%m%d')}",
                    "hazard_type": "geomagnetic_storm",
                    "start_time": current_date + timedelta(hours=18),
                    "kp_index": 5,
                    "severity": 0.4,
                    "source_data": {"mock": True}
                }
                mock_storms.append(storm)

            current_date += timedelta(days=1)

        return mock_storms

class SpaceTrackService:
    """Space-Track.org service for orbital debris data"""
    
    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        self.username = username
        self.password = password
        self.api = None
        
        if SPACETRACK_AVAILABLE and username and password:
            try:
                self.api = SpaceTrackApi(username, password)
                logger.info("Space-Track API initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Space-Track API: {e}")
        else:
            logger.warning("Space-Track API not available - using mock data")
    
    async def fetch_debris_data(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Fetch orbital debris data"""
        
        if self.api:
            try:
                return await self._fetch_real_debris_data(start_date, end_date)
            except Exception as e:
                logger.error(f"Error fetching real debris data: {e}")
        
        return await self._generate_mock_debris_data(start_date, end_date)
    
    async def _fetch_real_debris_data(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Fetch real debris data from Space-Track"""
        # Implementation would use Space-Track API
        # For now, return mock data
        return await self._generate_mock_debris_data(start_date, end_date)
    
    async def _generate_mock_debris_data(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Generate mock debris data for testing"""
        mock_debris = []
        
        # Generate a few high-risk debris objects
        debris_objects = [
            {
                "object_id": "COSMOS-2251-DEB",
                "tle_line1": "1 34454U 93036SX  12001.00000000  .00000000  00000-0  00000-0 0  0000",
                "tle_line2": "2 34454  82.9200 290.0000 0020000  90.0000 270.0000 14.12345678000000",
                "object_size_m": 0.5,
                "collision_probability": 0.001
            },
            {
                "object_id": "IRIDIUM-33-DEB",
                "tle_line1": "1 24946U 97051C   12001.00000000  .00000000  00000-0  00000-0 0  0000",
                "tle_line2": "2 24946  86.4000 180.0000 0001000  45.0000 315.0000 14.34567890000000",
                "object_size_m": 1.2,
                "collision_probability": 0.002
            }
        ]
        
        for obj in debris_objects:
            debris = {
                "hazard_id": f"debris_{obj['object_id']}_{start_date.strftime('%Y%m%d')}",
                "hazard_type": "debris",
                "object_id": obj["object_id"],
                "tle_line1": obj["tle_line1"],
                "tle_line2": obj["tle_line2"],
                "timestamp": start_date + timedelta(hours=12),
                "closest_approach_time": start_date + timedelta(hours=72),
                "miss_distance_km": 5.0,
                "object_size_m": obj["object_size_m"],
                "collision_probability": obj["collision_probability"],
                "impact_radius_km": 10.0,
                "severity": obj["collision_probability"],
                "source_data": obj
            }
            mock_debris.append(debris)
        
        return mock_debris

class SpaceWeatherDataService:
    """Combined space weather data service"""
    
    def __init__(self):
        self.nasa_donki = NASADONKIService()
        self.space_track = SpaceTrackService()
        self.cache = {}
        
        logger.info("Space Weather Data Service initialized")
    
    async def get_historical_data(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get all historical space weather data for date range"""
        
        cache_key = f"{start_date.isoformat()}_{end_date.isoformat()}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Fetch all data types concurrently
            solar_flares_task = self.nasa_donki.fetch_historical_solar_flares(start_date, end_date)
            cmes_task = self.nasa_donki.fetch_historical_cmes(start_date, end_date)
            geomagnetic_task = self.nasa_donki.fetch_geomagnetic_storms(start_date, end_date)
            debris_task = self.space_track.fetch_debris_data(start_date, end_date)
            
            solar_flares, cmes, geomagnetic_storms, debris = await asyncio.gather(
                solar_flares_task, cmes_task, geomagnetic_task, debris_task
            )
            
            data = {
                "solar_flares": solar_flares,
                "cmes": cmes,
                "geomagnetic_storms": geomagnetic_storms,
                "debris": debris
            }
            
            # Cache the result
            self.cache[cache_key] = data
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching historical space weather data: {e}")
            return {
                "solar_flares": [],
                "cmes": [],
                "geomagnetic_storms": [],
                "debris": []
            }
    
    async def get_current_conditions(self, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Get current space weather conditions for given timestamp"""
        
        if not timestamp:
            timestamp = datetime.utcnow()
        
        # Get data for a 24-hour window around the timestamp
        start_time = timestamp - timedelta(hours=12)
        end_time = timestamp + timedelta(hours=12)
        
        data = await self.get_historical_data(start_time, end_time)
        
        return {
            "timestamp": timestamp.isoformat(),
            "active_hazards": self._identify_active_hazards(data, timestamp),
            "hazard_forecast": self._generate_hazard_forecast(data, timestamp),
            "space_weather_summary": self._generate_weather_summary(data)
        }
    
    def _identify_active_hazards(self, data: Dict, timestamp: datetime) -> List[Dict[str, Any]]:
        """Identify hazards active at the given timestamp"""
        active_hazards = []
        
        # Check solar flares
        for flare in data["solar_flares"]:
            if (flare["start_time"] <= timestamp <= flare["end_time"]):
                active_hazards.append(flare)
        
        # Check CMEs (impact window)
        for cme in data["cmes"]:
            impact_window = timedelta(hours=48)  # 48-hour impact window
            if (cme["start_time"] <= timestamp <= cme["start_time"] + impact_window):
                active_hazards.append(cme)
        
        # Check debris (always potentially active)
        for debris in data["debris"]:
            if abs((debris["closest_approach_time"] - timestamp).total_seconds()) < 86400:  # 24 hours
                active_hazards.append(debris)
        
        return active_hazards
    
    def _generate_hazard_forecast(self, data: Dict, timestamp: datetime) -> Dict[str, Any]:
        """Generate hazard forecast for next 24 hours"""
        future_timestamp = timestamp + timedelta(hours=24)
        future_hazards = self._identify_active_hazards(data, future_timestamp)
        
        return {
            "forecast_time": future_timestamp.isoformat(),
            "predicted_hazards": len(future_hazards),
            "risk_level": "LOW" if len(future_hazards) == 0 else "HIGH",
            "hazard_details": future_hazards
        }
    
    def _generate_weather_summary(self, data: Dict) -> Dict[str, Any]:
        """Generate overall space weather summary"""
        total_hazards = (
            len(data["solar_flares"]) + 
            len(data["cmes"]) + 
            len(data["geomagnetic_storms"]) + 
            len(data["debris"])
        )
        
        return {
            "total_events": total_hazards,
            "solar_activity": len(data["solar_flares"]) + len(data["cmes"]),
            "debris_threats": len(data["debris"]),
            "overall_risk": "MODERATE" if total_hazards > 5 else "LOW"
        }
    
    async def close(self):
        """Close all service connections"""
        await self.nasa_donki.close_session()
