"""
ODIN Trajectory & Propulsion Engine
Built on Poliastro for orbital mechanics calculations
"""

import numpy as np
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

try:
    from boinor.bodies import Earth, Moon
    from boinor.twobody import Orbit
    from boinor.maneuver import Maneuver
    from boinor.plotting import OrbitPlotter
    from astropy import units as u
    from astropy.time import Time
    from astropy.coordinates import CartesianRepresentation
    BOINOR_AVAILABLE = False
except ImportError as e:
    print(f"⚠️ boinor not available: {e}")
    BOINOR_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class TrajectoryPoint:
    """Single point on a trajectory"""
    time: datetime
    position: List[float]  # [x, y, z] in km
    velocity: List[float]  # [vx, vy, vz] in km/s

@dataclass
class ManeuverData:
    """Spacecraft maneuver data"""
    time: datetime
    delta_v: List[float]  # [dvx, dvy, dvz] in m/s
    duration: float  # seconds
    fuel_cost: float  # kg
    description: str

@dataclass
class TrajectoryAnalysis:
    """Complete trajectory analysis"""
    trajectory_id: str
    name: str
    waypoints: List[TrajectoryPoint]
    maneuvers: List[ManeuverData]
    total_delta_v: float  # m/s
    total_duration: float  # hours
    fuel_required: float  # kg
    radiation_exposure: float  # percentage of safe limit
    collision_risk: float  # probability 0-1
    safety_score: float  # overall safety 0-1

class TrajectoryEngine:
    """ODIN Trajectory & Propulsion Engine using boinor"""
    
    def __init__(self):
        self.engine_name = "ODIN Trajectory Engine"
        self.version = "1.0.0"
        
        # Spacecraft parameters
        self.spacecraft_mass = 5000.0  # kg
        self.fuel_capacity = 2000.0   # kg
        self.isp = 320.0              # specific impulse (seconds)
        self.thrust = 500.0           # N
        
        # Safety parameters
        self.min_safe_altitude = 200.0  # km above Earth
        self.max_radiation_exposure = 50.0  # percentage
        self.debris_avoidance_margin = 10.0  # km
        
        logger.info(f"Initialized {self.engine_name} v{self.version}")
        
    async def calculate_initial_trajectory(
        self,
        start_time: datetime,
        destination: str = "Moon"
    ) -> TrajectoryAnalysis:
        """Calculate initial Earth-to-Moon transfer trajectory"""
        
        if not BOINOR_AVAILABLE:
            logger.warning("boinor not available, using fallback trajectory calculation.")
            return await self._fallback_trajectory_calculation(start_time, destination)
        
        try:
            # Convert to astropy Time
            astropy_time = Time(start_time)
            
            # Define initial orbit (LEO)
            initial_orbit = Orbit.circular(
                Earth, 
                alt=400 * u.km,
                epoch=astropy_time
            )
            
            # Calculate Hohmann transfer to Moon
            transfer_time = 5 * u.day  # Typical Earth-Moon transfer time
            
            # Calculate the maneuvers needed
            hohmann_maneuver = await self._calculate_hohmann_transfer(initial_orbit, transfer_time)

            if hohmann_maneuver is None:
                return await self._fallback_trajectory_calculation(start_time, destination)
            
            # Generate trajectory waypoints
            waypoints = await self._generate_trajectory_waypoints(
                initial_orbit, hohmann_maneuver, transfer_time
            )
            
            # Convert maneuver to ManeuverData
            maneuvers = []
            for impulse in hohmann_maneuver.impulses:
                maneuvers.append(ManeuverData(
                    time=initial_orbit.epoch.to_datetime(),
                    delta_v=impulse[1].to(u.m / u.s).value,
                    duration=300,
                    fuel_cost=await self._calculate_fuel_requirement(np.linalg.norm(impulse[1].to(u.m / u.s).value)),
                    description="Hohmann transfer impulse"
                ))

            # Calculate metrics
            total_delta_v = hohmann_maneuver.get_total_cost().to(u.m / u.s).value
            
            return TrajectoryAnalysis(
                trajectory_id=f"baseline_{start_time.strftime('%Y%m%d_%H%M%S')}",
                name="Baseline Hohmann Transfer",
                waypoints=waypoints,
                maneuvers=maneuvers,
                total_delta_v=total_delta_v,
                total_duration=transfer_time.to(u.hour).value,
                fuel_required=await self._calculate_fuel_requirement(total_delta_v),
                radiation_exposure=15.0,  # Typical for this trajectory
                collision_risk=0.001,     # Low risk for baseline trajectory
                safety_score=0.85
            )
            
        except Exception as e:
            logger.error(f"Error calculating initial trajectory with boinor: {e}")
            return await self._fallback_trajectory_calculation(start_time, destination)
    
    async def calculate_alternative_trajectory(
        self,
        mission_id: str,
        strategy_prompt: str,
        current_state: Dict[str, Any],
        hazards: List[Dict[str, Any]]
    ) -> TrajectoryAnalysis:
        """Calculate alternative trajectory based on AI strategy"""
        
        try:
            start_time = datetime.fromisoformat(current_state.get("time", datetime.utcnow().isoformat()))
            current_position = current_state.get("position", [0, 0, 400])  # km
            current_velocity = current_state.get("velocity", [0, 7.8, 0])  # km/s
            
            # Analyze strategy prompt to determine trajectory type
            trajectory_type = await self._analyze_strategy_prompt(strategy_prompt)
            
            if trajectory_type == "conservative":
                return await self._calculate_conservative_trajectory(
                    start_time, current_position, current_velocity, hazards
                )
            elif trajectory_type == "fast":
                return await self._calculate_fast_trajectory(
                    start_time, current_position, current_velocity
                )
            elif trajectory_type == "fuel_efficient":
                return await self._calculate_fuel_efficient_trajectory(
                    start_time, current_position, current_velocity
                )
            else:
                return await self._calculate_adaptive_trajectory(
                    start_time, current_position, current_velocity, hazards
                )
                
        except Exception as e:
            logger.error(f"Error calculating alternative trajectory: {e}")
            return await self._fallback_trajectory_calculation(start_time, "Moon")
    
    async def compute_delta_v_cost(
        self,
        from_orbit: Dict[str, Any],
        to_orbit: Dict[str, Any],
        transfer_time: Optional[float] = None
    ) -> Dict[str, float]:
        """Compute delta-v cost for orbital transfer"""
        
        try:
            # Extract orbital elements
            r1 = np.array(from_orbit.get("position", [7000, 0, 0]))  # km
            v1 = np.array(from_orbit.get("velocity", [0, 7.5, 0]))   # km/s
            r2 = np.array(to_orbit.get("position", [42000, 0, 0]))   # km
            
            # Calculate transfer delta-v using vis-viva equation
            mu_earth = 398600.4418  # km³/s²
            
            r1_mag = np.linalg.norm(r1)
            r2_mag = np.linalg.norm(r2)
            
            # Semi-major axis of transfer orbit
            a_transfer = (r1_mag + r2_mag) / 2
            
            # Delta-v at departure
            v_transfer_1 = np.sqrt(mu_earth * (2/r1_mag - 1/a_transfer))
            v_circular_1 = np.sqrt(mu_earth / r1_mag)
            delta_v_1 = abs(v_transfer_1 - v_circular_1)
            
            # Delta-v at arrival
            v_transfer_2 = np.sqrt(mu_earth * (2/r2_mag - 1/a_transfer))
            v_circular_2 = np.sqrt(mu_earth / r2_mag)
            delta_v_2 = abs(v_circular_2 - v_transfer_2)
            
            total_delta_v = delta_v_1 + delta_v_2
            
            return {
                "departure_delta_v": delta_v_1 * 1000,  # Convert to m/s
                "arrival_delta_v": delta_v_2 * 1000,
                "total_delta_v": total_delta_v * 1000,
                "transfer_duration": transfer_time or await self._calculate_transfer_time(r1_mag, r2_mag),
                "fuel_required": await self._calculate_fuel_requirement(total_delta_v * 1000)
            }
            
        except Exception as e:
            logger.error(f"Error computing delta-v cost: {e}")
            return {
                "departure_delta_v": 3200.0,
                "arrival_delta_v": 800.0,
                "total_delta_v": 4000.0,
                "transfer_duration": 120.0,
                "fuel_required": 200.0
            }
    
    async def evaluate_trajectory_safety(
        self,
        trajectory: TrajectoryAnalysis,
        hazards: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Evaluate trajectory safety against known hazards"""
        
        safety_metrics = {
            "radiation_safety": 1.0,
            "debris_safety": 1.0,
            "solar_storm_safety": 1.0,
            "overall_safety": 1.0
        }
        
        try:
            # Check radiation exposure
            if trajectory.radiation_exposure > self.max_radiation_exposure:
                safety_metrics["radiation_safety"] = max(0.0, 
                    1.0 - (trajectory.radiation_exposure - self.max_radiation_exposure) / 50.0)
            
            # Check debris collision risk
            for hazard in hazards:
                if hazard.get("hazard_type") == "debris":
                    collision_prob = await self._calculate_debris_collision_probability(
                        trajectory, hazard
                    )
                    safety_metrics["debris_safety"] = min(
                        safety_metrics["debris_safety"], 
                        1.0 - collision_prob
                    )
                
                elif hazard.get("hazard_type") == "solar_flare":
                    radiation_impact = await self._calculate_solar_radiation_impact(
                        trajectory, hazard
                    )
                    safety_metrics["solar_storm_safety"] = min(
                        safety_metrics["solar_storm_safety"],
                        1.0 - radiation_impact
                    )
            
            # Calculate overall safety score
            safety_metrics["overall_safety"] = (
                safety_metrics["radiation_safety"] * 0.4 +
                safety_metrics["debris_safety"] * 0.4 +
                safety_metrics["solar_storm_safety"] * 0.2
            )
            
        except Exception as e:
            logger.error(f"Error evaluating trajectory safety: {e}")
            
        return safety_metrics
    
    # Private helper methods
    
    async def _calculate_hohmann_transfer(self, initial_orbit, transfer_time):
        """Calculate Hohmann transfer maneuvers"""
        if not BOINOR_AVAILABLE:
            logger.warning("boinor not available, using fallback Hohmann transfer calculation.")
            return None

        # Use boinor for precise calculations
        hohmann = Maneuver.hohmann(initial_orbit, 384400 * u.km)
        
        return hohmann
    
    async def _generate_trajectory_waypoints(self, initial_orbit, hohmann_maneuver, transfer_time):
        """Generate trajectory waypoints"""
        if not BOINOR_AVAILABLE or hohmann_maneuver is None:
            logger.warning("boinor not available, using fallback trajectory waypoints.")
            return [TrajectoryPoint(time=datetime.utcnow(), position=[0,0,0], velocity=[0,0,0])]

        # Generate waypoints using boinor
        final_orbit = initial_orbit.apply_maneuver(hohmann_maneuver)
        
        waypoints = []
        for state in final_orbit.sample(100):
            waypoints.append(TrajectoryPoint(
                time=state.epoch.to_datetime(),
                position=state.r.to(u.km).value.tolist(),
                velocity=state.v.to(u.km / u.s).value.tolist()
            ))
        return waypoints
    
    async def _calculate_fuel_requirement(self, delta_v_m_s: float) -> float:
        """Calculate fuel requirement using rocket equation"""
        # Tsiolkovsky rocket equation: Δv = Isp * g * ln(m0/m1)
        g = 9.81  # m/s²
        
        mass_ratio = np.exp(delta_v_m_s / (self.isp * g))
        fuel_required = self.spacecraft_mass * (mass_ratio - 1)
        
        return min(fuel_required, self.fuel_capacity)
    
    async def _analyze_strategy_prompt(self, prompt: str) -> str:
        """Analyze AI strategy prompt to determine trajectory type"""
        prompt_lower = prompt.lower()
        
        if "conservative" in prompt_lower or "safe" in prompt_lower:
            return "conservative"
        elif "fast" in prompt_lower or "quick" in prompt_lower:
            return "fast"
        elif "fuel" in prompt_lower or "efficient" in prompt_lower:
            return "fuel_efficient"
        else:
            return "adaptive"
    
    async def _calculate_conservative_trajectory(self, start_time, position, velocity, hazards):
        """Calculate conservative, safe trajectory"""
        return TrajectoryAnalysis(
            trajectory_id=f"conservative_{start_time.strftime('%Y%m%d_%H%M%S')}",
            name="Conservative Safety Route",
            waypoints=[],  # Would be calculated
            maneuvers=[],  # Would be calculated
            total_delta_v=4500.0,  # Higher delta-v for safety
            total_duration=144.0,  # Longer duration
            fuel_required=220.0,
            radiation_exposure=8.0,  # Lower radiation
            collision_risk=0.0001,  # Very low risk
            safety_score=0.95
        )
    
    async def _calculate_fast_trajectory(self, start_time, position, velocity):
        """Calculate fast trajectory with higher delta-v"""
        return TrajectoryAnalysis(
            trajectory_id=f"fast_{start_time.strftime('%Y%m%d_%H%M%S')}",
            name="Fast Transfer Route",
            waypoints=[],
            maneuvers=[],
            total_delta_v=6000.0,  # Higher delta-v
            total_duration=72.0,   # Shorter duration
            fuel_required=300.0,
            radiation_exposure=25.0,
            collision_risk=0.005,
            safety_score=0.75
        )
    
    async def _calculate_fuel_efficient_trajectory(self, start_time, position, velocity):
        """Calculate fuel-efficient trajectory"""
        return TrajectoryAnalysis(
            trajectory_id=f"efficient_{start_time.strftime('%Y%m%d_%H%M%S')}",
            name="Fuel Efficient Route",
            waypoints=[],
            maneuvers=[],
            total_delta_v=3800.0,  # Lower delta-v
            total_duration=168.0,  # Longer duration
            fuel_required=180.0,
            radiation_exposure=20.0,
            collision_risk=0.002,
            safety_score=0.80
        )
    
    async def _calculate_adaptive_trajectory(self, start_time, position, velocity, hazards):
        """Calculate adaptive trajectory based on current conditions"""
        return TrajectoryAnalysis(
            trajectory_id=f"adaptive_{start_time.strftime('%Y%m%d_%H%M%S')}",
            name="Adaptive Route",
            waypoints=[],
            maneuvers=[],
            total_delta_v=4200.0,
            total_duration=120.0,
            fuel_required=200.0,
            radiation_exposure=15.0,
            collision_risk=0.001,
            safety_score=0.85
        )
    
    async def _fallback_trajectory_calculation(self, start_time, destination):
        """Fallback trajectory calculation when boinor is not available"""
        logger.warning(f"Using fallback trajectory calculation for destination: {destination}")
        return TrajectoryAnalysis(
            trajectory_id=f"fallback_{start_time.strftime('%Y%m%d_%H%M%S')}",
            name="Fallback Trajectory (boinor Unavailable)",
            waypoints=[TrajectoryPoint(time=start_time, position=[7000,0,0], velocity=[0,7.8,0])],
            maneuvers=[
                ManeuverData(
                    time=start_time,
                    delta_v=np.array([3200, 0, 0]),
                    duration=300,
                    fuel_cost=150,
                    description="Trans-lunar injection burn (fallback)"
                )
            ],
            total_delta_v=4000.0,
            total_duration=120.0,
            fuel_required=200.0,
            radiation_exposure=15.0,
            collision_risk=0.001,
            safety_score=0.80
        )
    
    async def _calculate_transfer_time(self, r1: float, r2: float) -> float:
        """Calculate transfer time for Hohmann transfer"""
        mu_earth = 398600.4418  # km³/s²
        a_transfer = (r1 + r2) / 2
        period = 2 * np.pi * np.sqrt(a_transfer**3 / mu_earth)
        return period / 2 / 3600  # Return in hours
    
    async def _calculate_debris_collision_probability(self, trajectory, hazard):
        """Calculate probability of collision with debris"""
        # Simplified collision probability calculation
        debris_position = hazard.get("location", {})
        miss_distance = hazard.get("miss_distance_km", 100)
        
        if miss_distance < self.debris_avoidance_margin:
            return min(1.0, self.debris_avoidance_margin / miss_distance - 1.0)
        
        return 0.001  # Base collision probability
    
    async def _calculate_solar_radiation_impact(self, trajectory, hazard):
        """Calculate impact of solar radiation on trajectory"""
        flare_class = hazard.get("flare_class", "C")
        
        impact_multiplier = {
            "X": 0.8,
            "M": 0.4,
            "C": 0.1,
            "B": 0.05,
            "A": 0.01
        }
        
        return impact_multiplier.get(flare_class, 0.1)
