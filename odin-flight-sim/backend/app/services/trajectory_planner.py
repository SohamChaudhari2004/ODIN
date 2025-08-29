import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import asyncio
import logging

try:
    from poliastro.bodies import Earth, Sun
    from poliastro.twobody import Orbit
    from poliastro.maneuver import Maneuver
    from poliastro.iod import lambert
    from astropy import units as u
    from astropy.time import Time
    POLIASTRO_AVAILABLE = True
    print('poliastro dependency loaded')
except ImportError:
    POLIASTRO_AVAILABLE = False
    print('poliastro dependency loading problem there may be error in imports in trajectory planner.py')

    logging.warning("Poliastro not available, using simplified trajectory calculations")

from ..models.schemas import TrajectoryPlan, TrajectoryPoint

logger = logging.getLogger(__name__)

class TrajectoryPlanner:
    def __init__(self):
        self.baseline_trajectories = {}
        self.alternate_trajectories = {}
        self.current_trajectory = None
        self._initialized = False
        
        # Mission parameters
        self.departure_orbit = {
            "altitude": 400,  # km
            "inclination": 51.6,  # degrees (ISS-like)
            "eccentricity": 0.0001
        }
        
        self.destination_orbit = {
            "altitude": 35786,  # GEO altitude
            "inclination": 0.0,
            "eccentricity": 0.0
        }
        
        # Initialize with baseline trajectory synchronously
        self._initialize_sync()
    
    def _initialize_sync(self):
        """Initialize trajectories synchronously"""
        try:
            # Generate baseline trajectory
            self.baseline_trajectories["nominal"] = self._generate_simple_trajectory("nominal")
            
            # Generate alternative trajectories  
            self.alternate_trajectories["bi_elliptic"] = self._generate_simple_trajectory("bi_elliptic")
            self.alternate_trajectories["fast_transfer"] = self._generate_simple_trajectory("fast_transfer")
            self.alternate_trajectories["low_energy"] = self._generate_simple_trajectory("low_energy")
            
            # Set current trajectory to baseline
            self.current_trajectory = self.baseline_trajectories["nominal"]
            self._initialized = True
            
            logger.info("Trajectory planner initialized with baseline and alternative trajectories")
            
        except Exception as e:
            logger.error(f"Error initializing trajectories: {e}")
            # Fallback to simplified trajectory
            self.current_trajectory = self._generate_simple_trajectory()
            self._initialized = True
    
    async def _initialize_trajectories(self):
        """Initialize baseline and alternative trajectories"""
        try:
            # Generate baseline trajectory
            self.baseline_trajectories["nominal"] = await self._generate_hohmann_transfer()
            
            # Generate alternative trajectories
            self.alternate_trajectories["bi_elliptic"] = await self._generate_bi_elliptic_transfer()
            self.alternate_trajectories["fast_transfer"] = await self._generate_fast_transfer()
            self.alternate_trajectories["low_energy"] = await self._generate_low_energy_transfer()
            
            # Set current trajectory to baseline
            self.current_trajectory = self.baseline_trajectories["nominal"]
            
            logger.info("Trajectory planner initialized with baseline and alternative trajectories")
            
        except Exception as e:
            logger.error(f"Error initializing trajectories: {e}")
            # Fallback to simplified trajectory
            self.current_trajectory = self._generate_simple_trajectory()
    
    async def _generate_hohmann_transfer(self) -> TrajectoryPlan:
        """Generate Hohmann transfer trajectory"""
        if POLIASTRO_AVAILABLE:
            try:
                # Create initial and final orbits
                r_initial = (Earth.R + self.departure_orbit["altitude"] * u.km).to(u.km)
                r_final = (Earth.R + self.destination_orbit["altitude"] * u.km).to(u.km)
                
                # Calculate Hohmann transfer
                v_initial = np.sqrt(Earth.k / r_initial).to(u.km / u.s)
                v_transfer_1 = np.sqrt(Earth.k * (2 / r_initial - 2 / (r_initial + r_final))).to(u.km / u.s)
                v_transfer_2 = np.sqrt(Earth.k * (2 / r_final - 2 / (r_initial + r_final))).to(u.km / u.s)
                v_final = np.sqrt(Earth.k / r_final).to(u.km / u.s)
                
                # Calculate delta-V
                delta_v_1 = abs(v_transfer_1 - v_initial)
                delta_v_2 = abs(v_final - v_transfer_2)
                total_delta_v = (delta_v_1 + delta_v_2).to(u.m / u.s).value
                
                # Transfer time
                transfer_time = np.pi * np.sqrt((r_initial + r_final)**3 / (8 * Earth.k)).to(u.hour).value
                
                # Generate trajectory points
                points = self._generate_trajectory_points(r_initial.value, r_final.value, transfer_time)
                
                return TrajectoryPlan(
                    plan_id="hohmann_nominal",
                    name="Hohmann Transfer (Nominal)",
                    departure_time=datetime.utcnow().isoformat(),
                    arrival_time=(datetime.utcnow() + timedelta(hours=transfer_time)).isoformat(),
                    total_delta_v=total_delta_v,
                    duration=transfer_time,
                    points=points,
                    risk_score=2.0,
                    fuel_cost=total_delta_v * 0.1,  # Simplified fuel calculation
                    radiation_exposure=transfer_time * 0.5
                )
                
            except Exception as e:
                logger.error(f"Error in Poliastro Hohmann transfer: {e}")
                return self._generate_simple_trajectory()
        else:
            return self._generate_simple_trajectory()
    
    async def _generate_bi_elliptic_transfer(self) -> TrajectoryPlan:
        """Generate bi-elliptic transfer (more fuel efficient for high altitude changes)"""
        # Simplified bi-elliptic transfer
        r_initial = 6771  # km (400 km altitude)
        r_intermediate = 50000  # km (high apogee)
        r_final = 42164  # km (GEO)
        
        # Simplified delta-V calculation
        total_delta_v = 4200  # m/s (typical for bi-elliptic)
        duration = 16.0  # hours
        
        points = self._generate_trajectory_points(r_initial, r_final, duration, intermediate_r=r_intermediate)
        
        return TrajectoryPlan(
            plan_id="bi_elliptic",
            name="Bi-Elliptic Transfer",
            departure_time=datetime.utcnow().isoformat(),
            arrival_time=(datetime.utcnow() + timedelta(hours=duration)).isoformat(),
            total_delta_v=total_delta_v,
            duration=duration,
            points=points,
            risk_score=3.5,
            fuel_cost=total_delta_v * 0.1,
            radiation_exposure=duration * 0.8
        )
    
    async def _generate_fast_transfer(self) -> TrajectoryPlan:
        """Generate fast transfer (higher delta-V, shorter time)"""
        r_initial = 6771  # km
        r_final = 42164  # km
        
        total_delta_v = 6500  # m/s (higher for fast transfer)
        duration = 3.5  # hours
        
        points = self._generate_trajectory_points(r_initial, r_final, duration)
        
        return TrajectoryPlan(
            plan_id="fast_transfer",
            name="Fast Transfer",
            departure_time=datetime.utcnow().isoformat(),
            arrival_time=(datetime.utcnow() + timedelta(hours=duration)).isoformat(),
            total_delta_v=total_delta_v,
            duration=duration,
            points=points,
            risk_score=1.5,
            fuel_cost=total_delta_v * 0.1,
            radiation_exposure=duration * 0.3
        )
    
    async def _generate_low_energy_transfer(self) -> TrajectoryPlan:
        """Generate low energy transfer using weak stability boundary theory"""
        r_initial = 6771  # km
        r_final = 42164  # km
        
        total_delta_v = 3200  # m/s (lower delta-V)
        duration = 72.0  # hours (much longer)
        
        points = self._generate_trajectory_points(r_initial, r_final, duration)
        
        return TrajectoryPlan(
            plan_id="low_energy",
            name="Low Energy Transfer",
            departure_time=datetime.utcnow().isoformat(),
            arrival_time=(datetime.utcnow() + timedelta(hours=duration)).isoformat(),
            total_delta_v=total_delta_v,
            duration=duration,
            points=points,
            risk_score=4.0,
            fuel_cost=total_delta_v * 0.1,
            radiation_exposure=duration * 1.2
        )
    
    def _generate_simple_trajectory(self, trajectory_type: str = "simple_hohmann") -> TrajectoryPlan:
        """Generate a simple trajectory for fallback"""
        
        # Different trajectory parameters based on type
        trajectory_configs = {
            "nominal": {
                "plan_id": "simple_hohmann",
                "name": "Simple Hohmann Transfer",
                "total_delta_v": 3900,
                "duration": 5.25,
                "risk_score": 2.0
            },
            "bi_elliptic": {
                "plan_id": "simple_bi_elliptic",
                "name": "Simple Bi-Elliptic Transfer",
                "total_delta_v": 4200,
                "duration": 16.0,
                "risk_score": 3.5
            },
            "fast_transfer": {
                "plan_id": "simple_fast",
                "name": "Simple Fast Transfer",
                "total_delta_v": 6500,
                "duration": 3.5,
                "risk_score": 1.5
            },
            "low_energy": {
                "plan_id": "simple_low_energy",
                "name": "Simple Low Energy Transfer",
                "total_delta_v": 3200,
                "duration": 72.0,
                "risk_score": 4.0
            }
        }
        
        config = trajectory_configs.get(trajectory_type, trajectory_configs["nominal"])
        
        r_initial = 6771  # km
        r_final = 42164  # km
        
        total_delta_v = config["total_delta_v"]  # m/s
        duration = config["duration"]  # hours
        
        points = self._generate_trajectory_points(r_initial, r_final, duration)
        
        return TrajectoryPlan(
            plan_id=config["plan_id"],
            name=config["name"],
            departure_time=datetime.utcnow().isoformat(),
            arrival_time=(datetime.utcnow() + timedelta(hours=duration)).isoformat(),
            total_delta_v=total_delta_v,
            duration=duration,
            points=points,
            risk_score=config["risk_score"],
            fuel_cost=total_delta_v * 0.1,
            radiation_exposure=duration * 0.5
        )
    
    def _generate_trajectory_points(self, r_initial: float, r_final: float, duration: float, 
                                  intermediate_r: Optional[float] = None, num_points: int = 100) -> List[TrajectoryPoint]:
        """Generate trajectory points for visualization"""
        points = []
        
        for i in range(num_points):
            t = i / (num_points - 1) * duration  # Time from 0 to duration
            
            if intermediate_r and t < duration / 2:
                # First half of bi-elliptic transfer
                r = r_initial + (intermediate_r - r_initial) * (t / (duration / 2))
            elif intermediate_r and t >= duration / 2:
                # Second half of bi-elliptic transfer
                r = intermediate_r + (r_final - intermediate_r) * ((t - duration / 2) / (duration / 2))
            else:
                # Simple elliptical transfer
                # Simplified: linear interpolation for visualization
                r = r_initial + (r_final - r_initial) * (t / duration)
            
            # Generate position in orbital plane
            angle = 2 * np.pi * t / duration  # Simplified angular progression
            position = [
                r * np.cos(angle),
                r * np.sin(angle),
                0.0
            ]
            
            # Calculate velocity (simplified)
            v = np.sqrt(398600.4418 / r)  # Circular velocity approximation
            velocity = [
                -v * np.sin(angle),
                v * np.cos(angle),
                0.0
            ]
            
            points.append(TrajectoryPoint(
                time=t,
                position=position,
                velocity=velocity
            ))
        
        return points
    
    def calculate_lambert_transfer(self, r1: List[float], r2: List[float], tof: float) -> Dict:
        """Calculate Lambert transfer between two position vectors"""
        if POLIASTRO_AVAILABLE:
            try:
                # Convert to astropy quantities
                r1_vec = np.array(r1) * u.km
                r2_vec = np.array(r2) * u.km
                tof_time = tof * u.hour
                
                # Solve Lambert problem
                v1, v2 = lambert(Earth.k, r1_vec, r2_vec, tof_time)
                
                return {
                    "departure_velocity": v1.to(u.km / u.s).value.tolist(),
                    "arrival_velocity": v2.to(u.km / u.s).value.tolist(),
                    "transfer_time": tof
                }
            except Exception as e:
                logger.error(f"Lambert transfer calculation failed: {e}")
                return self._simple_lambert_solution(r1, r2, tof)
        else:
            return self._simple_lambert_solution(r1, r2, tof)
    
    def _simple_lambert_solution(self, r1: List[float], r2: List[float], tof: float) -> Dict:
        """Simplified Lambert solution"""
        # Very basic approximation
        r1_mag = np.linalg.norm(r1)
        r2_mag = np.linalg.norm(r2)
        
        # Circular orbit velocities
        v1_circular = np.sqrt(398600.4418 / r1_mag)
        v2_circular = np.sqrt(398600.4418 / r2_mag)
        
        return {
            "departure_velocity": [0, v1_circular, 0],
            "arrival_velocity": [0, v2_circular, 0],
            "transfer_time": tof
        }
    
    def optimize_trajectory(self, constraints: Dict) -> TrajectoryPlan:
        """Optimize trajectory based on constraints"""
        optimization_target = constraints.get("optimization_target", "fuel_optimal")
        max_duration = constraints.get("max_duration", 24.0)  # hours
        max_delta_v = constraints.get("max_delta_v", 5000.0)  # m/s
        max_radiation = constraints.get("max_radiation", 20.0)  # arbitrary units
        
        # Get all available trajectories
        all_trajectories = {
            **self.baseline_trajectories,
            **self.alternate_trajectories
        }
        
        # Filter based on constraints
        valid_trajectories = []
        for name, trajectory in all_trajectories.items():
            if (trajectory.duration <= max_duration and 
                trajectory.total_delta_v <= max_delta_v and
                trajectory.radiation_exposure <= max_radiation):
                valid_trajectories.append(trajectory)
        
        if not valid_trajectories:
            logger.warning("No trajectories meet constraints, returning current trajectory")
            return self.current_trajectory
        
        # Optimize based on target
        if optimization_target == "fuel_optimal":
            return min(valid_trajectories, key=lambda t: t.total_delta_v)
        elif optimization_target == "time_optimal":
            return min(valid_trajectories, key=lambda t: t.duration)
        elif optimization_target == "risk_minimal":
            return min(valid_trajectories, key=lambda t: t.risk_score)
        else:
            # Multi-objective optimization (simplified)
            def objective_function(traj):
                # Normalize and combine objectives
                fuel_score = traj.total_delta_v / 5000.0
                time_score = traj.duration / 24.0
                risk_score = traj.risk_score / 5.0
                return fuel_score + time_score + risk_score
            
            return min(valid_trajectories, key=objective_function)
    
    async def replan_trajectory(self, constraints: Optional[Dict] = None) -> TrajectoryPlan:
        """Replan trajectory based on new constraints or hazards"""
        if constraints is None:
            constraints = {}
        
        # Find optimal trajectory
        new_trajectory = self.optimize_trajectory(constraints)
        
        # Update current trajectory
        self.current_trajectory = new_trajectory
        
        logger.info(f"Trajectory replanned: {new_trajectory.name}")
        return new_trajectory
    
    def get_current_trajectory(self) -> TrajectoryPlan:
        """Get the currently active trajectory"""
        return self.current_trajectory
    
    def get_alternative_trajectories(self) -> List[TrajectoryPlan]:
        """Get all alternative trajectories"""
        return list(self.alternate_trajectories.values())
    
    def reset_to_baseline(self):
        """Reset to baseline trajectory"""
        if "nominal" in self.baseline_trajectories:
            self.current_trajectory = self.baseline_trajectories["nominal"]
        
    def evaluate_trajectory_risk(self, trajectory: TrajectoryPlan, hazards: List) -> float:
        """Evaluate risk score for a trajectory given current hazards"""
        base_risk = trajectory.risk_score
        hazard_risk = 0.0
        
        for hazard in hazards:
            # Simple risk calculation based on hazard type and severity
            if hazard.event_type == "solar_flare":
                hazard_risk += hazard.severity * 0.5
            elif hazard.event_type == "cme":
                hazard_risk += hazard.severity * 0.8
            elif hazard.event_type == "debris_conjunction":
                hazard_risk += hazard.severity * 1.2
        
        return base_risk + hazard_risk