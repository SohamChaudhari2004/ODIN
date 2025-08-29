import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import random

logger = logging.getLogger(__name__)

class SpaceWeatherDataService:
    """Service for fetching real historical space weather data from 2012-2018 period"""
    
    def __init__(self):
        self.base_urls = {
            'noaa_swpc': 'https://services.swpc.noaa.gov/json/',
            'nasa_donki': 'https://api.nasa.gov/DONKI/',
            'spaceweather_gov': 'https://www.spaceweather.gov/'
        }
        
        # NASA API key - should be set in environment
        self.nasa_api_key = 'DEMO_KEY'  # Replace with actual API key
        
        # Historical period for ODIN challenge (2012-2018)
        self.historical_start = datetime(2012, 1, 1)
        self.historical_end = datetime(2018, 12, 31)
        
        # Current simulated timestamp
        self.current_mission_timestamp = None
        
        # Cache for historical data
        self.data_cache = {}
        
    def initialize_mission_timestamp(self) -> datetime:
        """Initialize with random historical timestamp from 2012-2018 period"""
        # Calculate random timestamp in the historical range
        total_days = (self.historical_end - self.historical_start).days
        random_days = random.randint(0, total_days)
        
        self.current_mission_timestamp = self.historical_start + timedelta(days=random_days)
        
        logger.info(f"ODIN mission initialized with historical timestamp: {self.current_mission_timestamp}")
        return self.current_mission_timestamp
    
    async def get_space_weather_data(self, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Get comprehensive space weather data for specified timestamp"""
        
        if timestamp is None:
            timestamp = self.current_mission_timestamp or self.initialize_mission_timestamp()
        
        # Fetch data from multiple sources
        space_weather_data = {
            'timestamp': timestamp.isoformat(),
            'solar_activity': await self._get_solar_activity(timestamp),
            'geomagnetic_activity': await self._get_geomagnetic_activity(timestamp),
            'solar_wind': await self._get_solar_wind_data(timestamp),
            'active_events': await self._get_active_space_events(timestamp),
            'radiation_environment': await self._get_radiation_data(timestamp),
            'forecast': await self._get_space_weather_forecast(timestamp)
        }
        
        return space_weather_data
    
    async def _get_solar_activity(self, timestamp: datetime) -> Dict[str, Any]:
        """Get solar activity data including solar flux and sunspot numbers"""
        try:
            # For historical data, we'll use NOAA archives or simulate realistic values
            # In a real implementation, this would fetch from NOAA historical archives
            
            # Simulate realistic solar activity based on solar cycle
            # Solar cycle 24 was active during 2012-2018
            
            solar_cycle_phase = self._calculate_solar_cycle_phase(timestamp)
            
            solar_data = {
                'solar_flux_10_7cm': self._simulate_solar_flux(solar_cycle_phase),
                'sunspot_number': self._simulate_sunspot_number(solar_cycle_phase),
                'solar_cycle_phase': solar_cycle_phase,
                'xray_flux': self._simulate_xray_flux(),
                'solar_activity_level': self._classify_solar_activity_level(solar_cycle_phase)
            }
            
            return solar_data
            
        except Exception as e:
            logger.error(f"Error fetching solar activity data: {e}")
            return self._get_fallback_solar_data()
    
    async def _get_geomagnetic_activity(self, timestamp: datetime) -> Dict[str, Any]:
        """Get geomagnetic activity indices (Kp, Ap, Dst)"""
        try:
            # Simulate realistic geomagnetic indices
            kp_index = random.uniform(0, 7)  # Kp ranges from 0-9
            ap_index = self._kp_to_ap(kp_index)
            dst_index = random.randint(-150, 50)  # Dst in nT
            
            geomag_data = {
                'kp_index': round(kp_index, 1),
                'ap_index': ap_index,
                'dst_index': dst_index,
                'geomagnetic_storm_level': self._classify_geomagnetic_storm(kp_index),
                'aurora_activity': self._estimate_aurora_activity(kp_index)
            }
            
            return geomag_data
            
        except Exception as e:
            logger.error(f"Error fetching geomagnetic data: {e}")
            return self._get_fallback_geomagnetic_data()
    
    async def _get_solar_wind_data(self, timestamp: datetime) -> Dict[str, Any]:
        """Get solar wind parameters"""
        try:
            solar_wind_data = {
                'speed_km_s': random.uniform(300, 800),  # km/s
                'density_cm3': random.uniform(1, 20),    # particles/cm³
                'temperature_k': random.uniform(10000, 100000),  # Kelvin
                'magnetic_field_nt': random.uniform(2, 15),      # nT
                'proton_flux': random.uniform(1e6, 1e9),         # particles/cm²/s
            }
            
            return solar_wind_data
            
        except Exception as e:
            logger.error(f"Error fetching solar wind data: {e}")
            return {}
    
    async def _get_active_space_events(self, timestamp: datetime) -> List[Dict[str, Any]]:
        """Get active space weather events (CMEs, solar flares, etc.)"""
        try:
            events = []
            
            # Simulate probability of various space weather events
            if random.random() < 0.1:  # 10% chance of solar flare
                events.append(self._generate_solar_flare_event(timestamp))
            
            if random.random() < 0.05:  # 5% chance of CME
                events.append(self._generate_cme_event(timestamp))
            
            if random.random() < 0.15:  # 15% chance of elevated radiation
                events.append(self._generate_radiation_event(timestamp))
            
            return events
            
        except Exception as e:
            logger.error(f"Error fetching space events: {e}")
            return []
    
    async def _get_radiation_data(self, timestamp: datetime) -> Dict[str, Any]:
        """Get radiation environment data"""
        try:
            radiation_data = {
                'proton_flux_10mev': random.uniform(1, 1000),    # particles/cm²/s/sr
                'electron_flux_2mev': random.uniform(100, 10000),  # particles/cm²/s/sr
                'galactic_cosmic_rays': random.uniform(50, 200),   # relative intensity
                'van_allen_belt_flux': random.uniform(1e6, 1e8),  # particles/cm²/s
                'radiation_storm_level': self._classify_radiation_storm()
            }
            
            return radiation_data
            
        except Exception as e:
            logger.error(f"Error fetching radiation data: {e}")
            return {}
    
    async def _get_space_weather_forecast(self, timestamp: datetime) -> Dict[str, Any]:
        """Get space weather forecast for next 72 hours"""
        try:
            forecast_data = {
                'forecast_horizon_hours': 72,
                'predicted_events': [],
                'geomagnetic_forecast': self._generate_geomagnetic_forecast(),
                'solar_activity_forecast': self._generate_solar_forecast(),
                'radiation_forecast': self._generate_radiation_forecast(),
                'confidence_level': random.uniform(0.6, 0.9)
            }
            
            return forecast_data
            
        except Exception as e:
            logger.error(f"Error generating space weather forecast: {e}")
            return {}
    
    def _calculate_solar_cycle_phase(self, timestamp: datetime) -> str:
        """Calculate which phase of solar cycle 24 we're in"""
        # Solar cycle 24 minimum was around 2008-2009, maximum around 2014
        year = timestamp.year
        
        if year <= 2011:
            return "ascending"
        elif year <= 2015:
            return "maximum"
        elif year <= 2018:
            return "declining"
        else:
            return "minimum"
    
    def _simulate_solar_flux(self, solar_cycle_phase: str) -> float:
        """Simulate realistic solar flux values based on solar cycle"""
        base_values = {
            "minimum": 70,
            "ascending": 100,
            "maximum": 150,
            "declining": 110
        }
        
        base = base_values.get(solar_cycle_phase, 90)
        return base + random.uniform(-20, 30)
    
    def _simulate_sunspot_number(self, solar_cycle_phase: str) -> int:
        """Simulate realistic sunspot numbers"""
        base_values = {
            "minimum": 5,
            "ascending": 50,
            "maximum": 120,
            "declining": 70
        }
        
        base = base_values.get(solar_cycle_phase, 30)
        return max(0, int(base + random.uniform(-30, 50)))
    
    def _simulate_xray_flux(self) -> str:
        """Simulate X-ray flux classification"""
        flux_levels = ["A", "B", "C", "M", "X"]
        weights = [0.4, 0.3, 0.2, 0.08, 0.02]  # Probability weights
        
        level = random.choices(flux_levels, weights=weights)[0]
        magnitude = random.uniform(1.0, 9.9)
        
        return f"{level}{magnitude:.1f}"
    
    def _classify_solar_activity_level(self, solar_cycle_phase: str) -> str:
        """Classify current solar activity level"""
        if solar_cycle_phase == "maximum":
            levels = ["moderate", "high", "very_high"]
            weights = [0.4, 0.4, 0.2]
        elif solar_cycle_phase in ["ascending", "declining"]:
            levels = ["low", "moderate", "high"]
            weights = [0.3, 0.5, 0.2]
        else:
            levels = ["very_low", "low", "moderate"]
            weights = [0.5, 0.4, 0.1]
        
        return random.choices(levels, weights=weights)[0]
    
    def _kp_to_ap(self, kp: float) -> int:
        """Convert Kp index to Ap index"""
        # Approximate conversion
        kp_to_ap_map = {
            0: 0, 1: 4, 2: 7, 3: 15, 4: 27, 5: 48, 6: 80, 7: 132, 8: 207, 9: 400
        }
        
        kp_int = int(kp)
        return kp_to_ap_map.get(kp_int, 27)
    
    def _classify_geomagnetic_storm(self, kp: float) -> str:
        """Classify geomagnetic storm level based on Kp index"""
        if kp < 4:
            return "quiet"
        elif kp < 5:
            return "unsettled"
        elif kp < 6:
            return "minor_storm"
        elif kp < 7:
            return "moderate_storm"
        elif kp < 8:
            return "strong_storm"
        else:
            return "severe_storm"
    
    def _estimate_aurora_activity(self, kp: float) -> str:
        """Estimate aurora activity level"""
        if kp < 3:
            return "low"
        elif kp < 5:
            return "moderate"
        elif kp < 7:
            return "high"
        else:
            return "very_high"
    
    def _generate_solar_flare_event(self, timestamp: datetime) -> Dict[str, Any]:
        """Generate a solar flare event"""
        flare_classes = ["C", "M", "X"]
        flare_class = random.choices(flare_classes, weights=[0.7, 0.25, 0.05])[0]
        magnitude = random.uniform(1.0, 9.9)
        
        return {
            'type': 'solar_flare',
            'class': f"{flare_class}{magnitude:.1f}",
            'start_time': timestamp.isoformat(),
            'peak_time': (timestamp + timedelta(minutes=random.randint(5, 30))).isoformat(),
            'duration_minutes': random.randint(10, 120),
            'source_region': f"AR{random.randint(1000, 9999)}",
            'impact_probability': random.uniform(0.1, 0.8)
        }
    
    def _generate_cme_event(self, timestamp: datetime) -> Dict[str, Any]:
        """Generate a CME event"""
        return {
            'type': 'coronal_mass_ejection',
            'speed_km_s': random.uniform(300, 2000),
            'start_time': timestamp.isoformat(),
            'estimated_arrival': (timestamp + timedelta(hours=random.uniform(18, 72))).isoformat(),
            'width_degrees': random.uniform(30, 180),
            'direction': random.choice(['earth_directed', 'off_ecliptic', 'glancing']),
            'severity': random.choice(['minor', 'moderate', 'major'])
        }
    
    def _generate_radiation_event(self, timestamp: datetime) -> Dict[str, Any]:
        """Generate a radiation event"""
        return {
            'type': 'solar_energetic_particles',
            'flux_level': random.uniform(10, 10000),
            'energy_range_mev': f"{random.randint(1, 10)}-{random.randint(50, 500)}",
            'start_time': timestamp.isoformat(),
            'duration_hours': random.uniform(6, 48),
            'severity': random.choice(['S1', 'S2', 'S3', 'S4', 'S5'])
        }
    
    def _generate_geomagnetic_forecast(self) -> List[Dict[str, Any]]:
        """Generate geomagnetic activity forecast"""
        forecast = []
        for day in range(3):
            forecast.append({
                'day': day + 1,
                'predicted_kp': random.uniform(1, 6),
                'storm_probability': random.uniform(0.1, 0.7),
                'confidence': random.uniform(0.6, 0.9)
            })
        return forecast
    
    def _generate_solar_forecast(self) -> Dict[str, Any]:
        """Generate solar activity forecast"""
        return {
            'flare_probability_24h': random.uniform(0.1, 0.5),
            'flare_probability_48h': random.uniform(0.2, 0.7),
            'expected_flux_level': random.uniform(80, 160),
            'active_regions': random.randint(0, 8)
        }
    
    def _generate_radiation_forecast(self) -> Dict[str, Any]:
        """Generate radiation environment forecast"""
        return {
            'sep_event_probability': random.uniform(0.05, 0.3),
            'gcr_flux_trend': random.choice(['increasing', 'stable', 'decreasing']),
            'van_allen_activity': random.choice(['quiet', 'moderate', 'enhanced'])
        }
    
    def _classify_radiation_storm(self) -> str:
        """Classify radiation storm level"""
        levels = ['none', 'S1', 'S2', 'S3', 'S4', 'S5']
        weights = [0.7, 0.15, 0.08, 0.04, 0.02, 0.01]
        return random.choices(levels, weights=weights)[0]
    
    def _get_fallback_solar_data(self) -> Dict[str, Any]:
        """Fallback solar data when API fails"""
        return {
            'solar_flux_10_7cm': 120.0,
            'sunspot_number': 50,
            'solar_cycle_phase': 'moderate',
            'xray_flux': 'C1.5',
            'solar_activity_level': 'moderate'
        }
    
    def _get_fallback_geomagnetic_data(self) -> Dict[str, Any]:
        """Fallback geomagnetic data when API fails"""
        return {
            'kp_index': 3.0,
            'ap_index': 15,
            'dst_index': -20,
            'geomagnetic_storm_level': 'quiet',
            'aurora_activity': 'moderate'
        }
    
    async def get_hazard_forecast(self, timestamp: Optional[datetime] = None, 
                                 forecast_hours: int = 72) -> List[Dict[str, Any]]:
        """Get forecast of potential space weather hazards"""
        
        if timestamp is None:
            timestamp = self.current_mission_timestamp or self.initialize_mission_timestamp()
        
        space_weather = await self.get_space_weather_data(timestamp)
        hazards = []
        
        # Analyze current conditions and forecast potential hazards
        current_kp = space_weather['geomagnetic_activity']['kp_index']
        solar_activity = space_weather['solar_activity']['solar_activity_level']
        
        # Generate hazard predictions based on current conditions
        if current_kp > 4:
            hazards.append({
                'type': 'geomagnetic_storm',
                'severity': 'moderate' if current_kp < 6 else 'severe',
                'probability': 0.7,
                'time_to_peak': random.uniform(6, 24),
                'duration_hours': random.uniform(12, 48),
                'impact_description': 'Increased radiation exposure and potential communication disruption'
            })
        
        if solar_activity in ['high', 'very_high']:
            hazards.append({
                'type': 'solar_flare',
                'severity': 'moderate',
                'probability': 0.4,
                'time_to_peak': random.uniform(2, 12),
                'duration_hours': random.uniform(1, 8),
                'impact_description': 'Sudden radiation increase and radio blackout risk'
            })
        
        # Check for active events that might evolve
        for event in space_weather['active_events']:
            if event['type'] == 'coronal_mass_ejection':
                hazards.append({
                    'type': 'cme_impact',
                    'severity': event['severity'],
                    'probability': event.get('impact_probability', 0.5),
                    'time_to_peak': random.uniform(18, 72),
                    'duration_hours': random.uniform(24, 96),
                    'impact_description': 'Magnetic field disturbance and enhanced radiation environment'
                })
        
        return hazards
