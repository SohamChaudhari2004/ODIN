import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import json

from ..models.schemas import HazardEvent

logger = logging.getLogger(__name__)

@dataclass
class HazardEvent:
    event_type: str  # "solar_flare", "cme", "debris_conjunction"
    severity: float
    start_time: str
    duration: float
    affected_regions: List[str]
    mitigation_required: bool

class HazardDetector:
    def __init__(self):
        self.active_hazards: List[HazardEvent] = []
        self.hazard_history: List[HazardEvent] = []
        
        # Detection thresholds
        self.detection_thresholds = {
            "solar_flare": {
                "x_class": 1e-4,  # X-class flare threshold
                "m_class": 1e-5,  # M-class flare threshold
                "duration_hours": 4.0
            },
            "cme": {
                "speed": 500,  # km/s
                "density": 10,  # particles/cm³
                "duration_hours": 24.0
            },
            "debris": {
                "probability": 1e-4,  # Collision probability
                "distance": 5.0,  # km
                "duration_hours": 2.0
            }
        }
        
        # Risk assessment parameters
        self.risk_zones = {
            "LEO": {"altitude_min": 200, "altitude_max": 2000},  # km
            "MEO": {"altitude_min": 2000, "altitude_max": 35000},
            "GEO": {"altitude_min": 35000, "altitude_max": 36000},
            "HEO": {"altitude_min": 36000, "altitude_max": 100000}
        }
        
        self.monitoring_active = False
        self._monitoring_task = None
    
    async def monitor_space_weather(self):
        """Main monitoring loop for space weather hazards"""
        self.monitoring_active = True
        logger.info("Starting space weather monitoring...")
        
        while self.monitoring_active:
            try:
                # Check for solar flares
                await self._check_solar_flares()
                
                # Check for CMEs
                await self._check_cmes()
                
                # Check for debris conjunctions
                await self._check_debris_conjunctions()
                
                # Clean up expired hazards
                await self._cleanup_expired_hazards()
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in space weather monitoring: {e}")
                await asyncio.sleep(30)  # Shorter wait on error
    
    async def _check_solar_flares(self):
        """Check for solar flare hazards"""
        try:
            # In a real system, this would fetch from data ingestion service
            # For simulation, we'll generate occasional random flares
            
            import random
            if random.random() < 0.001:  # 0.1% chance per check
                severity = random.uniform(1.0, 8.0)
                flare_class = self._determine_flare_class(severity)
                
                hazard = HazardEvent(
                    event_type="solar_flare",
                    severity=severity,
                    start_time=datetime.utcnow().isoformat(),
                    duration=self.detection_thresholds["solar_flare"]["duration_hours"],
                    affected_regions=["LEO", "MEO", "GEO"] if severity > 5.0 else ["LEO", "MEO"],
                    mitigation_required=severity > 3.0
                )
                
                await self._add_hazard(hazard)
                logger.warning(f"Solar flare detected: {flare_class} class, severity {severity:.1f}")
                
        except Exception as e:
            logger.error(f"Error checking solar flares: {e}")
    
    async def _check_cmes(self):
        """Check for Coronal Mass Ejection hazards"""
        try:
            import random
            if random.random() < 0.0005:  # 0.05% chance per check
                severity = random.uniform(2.0, 9.0)
                speed = 400 + severity * 200  # km/s
                
                hazard = HazardEvent(
                    event_type="cme",
                    severity=severity,
                    start_time=datetime.utcnow().isoformat(),
                    duration=self.detection_thresholds["cme"]["duration_hours"],
                    affected_regions=["LEO", "MEO", "GEO", "HEO"] if severity > 6.0 else ["MEO", "GEO"],
                    mitigation_required=severity > 4.0
                )
                
                await self._add_hazard(hazard)
                logger.warning(f"CME detected: speed {speed:.0f} km/s, severity {severity:.1f}")
                
        except Exception as e:
            logger.error(f"Error checking CMEs: {e}")
    
    async def _check_debris_conjunctions(self):
        """Check for space debris conjunction hazards"""
        try:
            import random
            if random.random() < 0.002:  # 0.2% chance per check
                severity = random.uniform(1.0, 7.0)
                probability = self.detection_thresholds["debris"]["probability"] * severity
                
                hazard = HazardEvent(
                    event_type="debris_conjunction",
                    severity=severity,
                    start_time=datetime.utcnow().isoformat(),
                    duration=self.detection_thresholds["debris"]["duration_hours"],
                    affected_regions=self._determine_affected_orbital_regions(),
                    mitigation_required=severity > 3.0
                )
                
                await self._add_hazard(hazard)
                logger.warning(f"Debris conjunction detected: probability {probability:.2e}, severity {severity:.1f}")
                
        except Exception as e:
            logger.error(f"Error checking debris conjunctions: {e}")
    
    async def _add_hazard(self, hazard: HazardEvent):
        """Add a new hazard and trigger notifications"""
        self.active_hazards.append(hazard)
        self.hazard_history.append(hazard)
        
        # Trigger hazard response
        await self._trigger_hazard_response(hazard)
    
    async def _trigger_hazard_response(self, hazard: HazardEvent):
        """Trigger appropriate response to hazard"""
        try:
            # Log hazard detection
            logger.info(f"Hazard response triggered for {hazard.event_type} with severity {hazard.severity}")
            
            # If mitigation is required, this would trigger trajectory replanning
            if hazard.mitigation_required:
                logger.info("Mitigation required - trajectory replanning recommended")
                # In the full system, this would interface with the trajectory planner
                
        except Exception as e:
            logger.error(f"Error in hazard response: {e}")
    
    async def _cleanup_expired_hazards(self):
        """Remove expired hazards from active list"""
        current_time = datetime.utcnow()
        
        expired_hazards = []
        for hazard in self.active_hazards:
            start_time = datetime.fromisoformat(hazard.start_time.replace('Z', '+00:00'))
            if current_time > start_time + timedelta(hours=hazard.duration):
                expired_hazards.append(hazard)
        
        for hazard in expired_hazards:
            self.active_hazards.remove(hazard)
            logger.info(f"Hazard expired: {hazard.event_type} with severity {hazard.severity}")
    
    def _determine_flare_class(self, severity: float) -> str:
        """Determine solar flare class based on severity"""
        if severity >= 8.0:
            return "X"
        elif severity >= 5.0:
            return "M"
        elif severity >= 2.0:
            return "C"
        else:
            return "B"
    
    def _determine_affected_orbital_regions(self) -> List[str]:
        """Determine which orbital regions are affected by debris"""
        import random
        all_regions = ["LEO", "MEO", "GEO", "HEO"]
        num_affected = random.randint(1, 3)
        return random.sample(all_regions, num_affected)
    
    async def inject_hazard(self, hazard: HazardEvent):
        """Manually inject a hazard for simulation purposes"""
        await self._add_hazard(hazard)
        logger.info(f"Manually injected hazard: {hazard.event_type} with severity {hazard.severity}")
    
    def get_active_hazards(self) -> List[HazardEvent]:
        """Get list of currently active hazards"""
        return self.active_hazards.copy()
    
    def get_hazard_history(self) -> List[HazardEvent]:
        """Get complete hazard history"""
        return self.hazard_history.copy()
    
    def clear_hazards(self):
        """Clear all active hazards (for simulation reset)"""
        self.active_hazards.clear()
        logger.info("All hazards cleared")
    
    def evaluate_trajectory_risk(self, trajectory, hazards: Optional[List[HazardEvent]] = None) -> float:
        """Evaluate risk for a trajectory given current or specified hazards"""
        if hazards is None:
            hazards = self.active_hazards
        
        total_risk = 0.0
        trajectory_regions = self._get_trajectory_regions(trajectory)
        
        for hazard in hazards:
            # Check if hazard affects trajectory
            affected_regions = set(hazard.affected_regions)
            trajectory_regions_set = set(trajectory_regions)
            
            if affected_regions.intersection(trajectory_regions_set):
                # Calculate risk contribution
                risk_contribution = self._calculate_risk_contribution(hazard, trajectory)
                total_risk += risk_contribution
        
        return total_risk
    
    def _get_trajectory_regions(self, trajectory) -> List[str]:
        """Determine which orbital regions a trajectory passes through"""
        regions = []
        
        if hasattr(trajectory, 'points'):
            for point in trajectory.points:
                altitude = self._calculate_altitude(point.position)
                region = self._altitude_to_region(altitude)
                if region not in regions:
                    regions.append(region)
        
        return regions
    
    def _calculate_altitude(self, position: List[float]) -> float:
        """Calculate altitude from position vector"""
        import math
        earth_radius = 6371.0  # km
        distance_from_center = math.sqrt(sum(x**2 for x in position))
        return distance_from_center - earth_radius
    
    def _altitude_to_region(self, altitude: float) -> str:
        """Map altitude to orbital region"""
        for region, bounds in self.risk_zones.items():
            if bounds["altitude_min"] <= altitude <= bounds["altitude_max"]:
                return region
        return "HEO"  # Default for very high altitudes
    
    def _calculate_risk_contribution(self, hazard: HazardEvent, trajectory) -> float:
        """Calculate how much risk a specific hazard adds to a trajectory"""
        base_risk = hazard.severity
        
        # Adjust based on hazard type
        type_multipliers = {
            "solar_flare": 0.8,
            "cme": 1.0,
            "debris_conjunction": 1.5
        }
        
        multiplier = type_multipliers.get(hazard.event_type, 1.0)
        
        # Adjust based on trajectory duration (longer exposure = higher risk)
        if hasattr(trajectory, 'duration'):
            duration_factor = min(trajectory.duration / 24.0, 2.0)  # Cap at 2x
        else:
            duration_factor = 1.0
        
        return base_risk * multiplier * duration_factor
    
    def get_hazard_statistics(self) -> Dict:
        """Get statistics about detected hazards"""
        total_hazards = len(self.hazard_history)
        active_count = len(self.active_hazards)
        
        type_counts = {}
        severity_sum = 0.0
        
        for hazard in self.hazard_history:
            event_type = hazard.event_type
            type_counts[event_type] = type_counts.get(event_type, 0) + 1
            severity_sum += hazard.severity
        
        avg_severity = severity_sum / total_hazards if total_hazards > 0 else 0.0
        
        return {
            "total_hazards_detected": total_hazards,
            "active_hazards": active_count,
            "hazards_by_type": type_counts,
            "average_severity": avg_severity,
            "monitoring_active": self.monitoring_active
        }
    
    def stop_monitoring(self):
        """Stop the monitoring loop"""
        self.monitoring_active = False
        logger.info("Space weather monitoring stopped")
    
    def get_hazard_forecast(self, hours_ahead: int = 24) -> List[Dict]:
        """Get forecast of potential hazards (simplified prediction)"""
        # This is a placeholder for more sophisticated prediction
        # In a real system, this would use ML models or physics-based forecasting
        
        forecast = []
        import random
        
        # Predict potential solar activity
        if random.random() < 0.3:
            forecast.append({
                "type": "solar_flare",
                "probability": random.uniform(0.1, 0.7),
                "estimated_severity": random.uniform(2.0, 6.0),
                "time_window": f"{random.randint(6, 18)} hours",
                "confidence": random.uniform(0.5, 0.9)
            })
        
        # Predict potential CMEs
        if random.random() < 0.2:
            forecast.append({
                "type": "cme",
                "probability": random.uniform(0.05, 0.4),
                "estimated_severity": random.uniform(3.0, 8.0),
                "time_window": f"{random.randint(12, 48)} hours",
                "confidence": random.uniform(0.3, 0.8)
            })
        
        return forecast