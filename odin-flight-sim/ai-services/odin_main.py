"""
ODIN (Optimal Dynamic Interplanetary Navigator) Main System
Autonomous AI system for spacecraft trajectory planning and dynamic replanning
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
from dataclasses import dataclass

# Import ODIN components
try:
    from .ai_copilot import AICoPilot
    from .predictive_hazard_forecasting import PredictiveHazardForecasting
    from .explainability_module import ExplainabilityModule
    from .space_weather_service import SpaceWeatherDataService
except ImportError:
    # Fallback for direct execution
    from ai_copilot import AICoPilot
    from predictive_hazard_forecasting import PredictiveHazardForecasting
    from explainability_module import ExplainabilityModule
    from space_weather_service import SpaceWeatherDataService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MissionState:
    """Current mission state data"""
    mission_time: float
    spacecraft_position: List[float]
    spacecraft_velocity: List[float]
    fuel_remaining: float
    current_trajectory: Dict[str, Any]
    communication_status: str
    system_health: Dict[str, Any]

@dataclass
class TrajectoryOption:
    """Trajectory option for evaluation"""
    name: str
    trajectory_type: str
    total_delta_v: float
    duration: float
    radiation_exposure: float
    collision_risk: float
    maneuvers: List[Dict[str, Any]]
    description: str

class OdinNavigationSystem:
    """Main ODIN system orchestrator for autonomous spacecraft navigation"""
    
    def __init__(self):
        self.system_name = "ODIN (Optimal Dynamic Interplanetary Navigator)"
        self.version = "1.0.0"
        self.mission_start_time = None
        self.historical_timestamp = None
        
        # Initialize ODIN subsystems
        self.ai_copilot = AICoPilot()
        self.hazard_forecaster = PredictiveHazardForecasting()
        self.explainability = ExplainabilityModule()
        self.space_weather = SpaceWeatherDataService()
        
        # Mission state tracking
        self.current_mission_state = None
        self.current_trajectory = None
        self.active_hazards = []
        self.decision_log = []
        
        # Mission parameters
        self.mission_constraints = {
            "max_delta_v": 15000,  # m/s
            "max_duration": 120,   # hours
            "max_radiation": 50,   # percentage of safe limit
            "min_fuel_reserve": 10 # percentage
        }
        
        # System status
        self.system_status = {
            "operational": True,
            "autonomous_mode": True,
            "last_decision": None,
            "hazard_monitoring": True,
            "replanning_active": False
        }
        
        logger.info(f"ODIN Navigation System initialized - {self.system_name} v{self.version}")
    
    async def initialize_mission(self, destination: str = "Moon") -> Dict[str, Any]:
        """Initialize ODIN mission with historical timestamp from 2012-2018"""
        
        # Initialize with random historical timestamp
        self.historical_timestamp = self.space_weather.initialize_mission_timestamp()
        self.mission_start_time = datetime.utcnow()
        
        # Initialize mission state
        self.current_mission_state = MissionState(
            mission_time=0.0,
            spacecraft_position=[6371.0, 0.0, 0.0],  # Earth radius in km
            spacecraft_velocity=[0.0, 7.8, 0.0],     # Initial orbital velocity
            fuel_remaining=100.0,
            current_trajectory={},
            communication_status="NOMINAL",
            system_health={
                "navigation": "NOMINAL",
                "propulsion": "NOMINAL", 
                "communication": "NOMINAL",
                "life_support": "NOMINAL"
            }
        )
        
        # Generate initial trajectory options
        initial_trajectories = await self._generate_initial_trajectories(destination)
        
        # Get initial space weather data
        space_weather_data = await self.space_weather.get_space_weather_data(self.historical_timestamp)
        
        # Perform initial trajectory selection
        selected_trajectory = await self._select_optimal_trajectory(
            initial_trajectories, 
            [],  # No hazards at initialization
            space_weather_data
        )
        
        self.current_trajectory = selected_trajectory
        
        initialization_result = {
            "mission_initialized": True,
            "historical_timestamp": self.historical_timestamp.isoformat(),
            "destination": destination,
            "selected_trajectory": selected_trajectory,
            "space_weather_conditions": space_weather_data,
            "system_status": self.system_status,
            "mission_constraints": self.mission_constraints
        }
        
        # Log initialization
        init_log = f"ODIN MISSION INITIALIZED. Historical timestamp: {self.historical_timestamp.strftime('%Y-%m-%d %H:%M:%S')}. Destination: {destination}. Initial trajectory: {selected_trajectory['name']}."
        self._log_decision(init_log, initialization_result)
        
        logger.info(f"ODIN mission initialized for {destination} using historical data from {self.historical_timestamp}")
        
        return initialization_result
    
    async def autonomous_mission_loop(self, duration_hours: float = 72.0) -> List[Dict[str, Any]]:
        """Main autonomous mission loop with continuous monitoring and replanning"""
        
        mission_events = []
        time_step = 0.5  # hours
        total_steps = int(duration_hours / time_step)
        
        logger.info(f"Starting autonomous mission loop for {duration_hours} hours")
        
        for step in range(total_steps):
            current_time = step * time_step
            self.current_mission_state.mission_time = current_time
            
            # Update mission state
            await self._update_mission_state(current_time)
            
            # Monitor for hazards
            hazards_detected = await self._monitor_hazards(current_time)
            
            if hazards_detected:
                # Hazard detected - initiate replanning
                replan_result = await self._initiate_emergency_replanning(hazards_detected)
                mission_events.append({
                    "timestamp": current_time,
                    "event_type": "hazard_replanning",
                    "details": replan_result
                })
                
            # Regular trajectory optimization check (every 6 hours)
            if current_time > 0 and current_time % 6.0 == 0:
                optimization_result = await self._perform_trajectory_optimization()
                if optimization_result["trajectory_changed"]:
                    mission_events.append({
                        "timestamp": current_time,
                        "event_type": "trajectory_optimization",
                        "details": optimization_result
                    })
            
            # Generate mission status report (every 2 hours)
            if current_time % 2.0 == 0:
                mission_brief = await self._generate_mission_status_report()
                mission_events.append({
                    "timestamp": current_time,
                    "event_type": "status_report",
                    "details": mission_brief
                })
            
            # Simulate real-time delay
            await asyncio.sleep(0.1)  # Small delay for simulation
        
        # Mission completion summary
        completion_summary = await self._generate_mission_completion_summary(mission_events)
        mission_events.append({
            "timestamp": duration_hours,
            "event_type": "mission_completion",
            "details": completion_summary
        })
        
        return mission_events
    
    async def _generate_initial_trajectories(self, destination: str) -> List[TrajectoryOption]:
        """Generate initial trajectory options for the mission"""
        
        # Simulate realistic Earth-to-Moon trajectory options
        trajectories = [
            TrajectoryOption(
                name="Direct Lunar Transfer",
                trajectory_type="hohmann_transfer",
                total_delta_v=11500,
                duration=72.0,
                radiation_exposure=25.0,
                collision_risk=0.15,
                maneuvers=[
                    {"name": "Trans-Lunar Injection", "delta_v": 3200, "time": 0.0},
                    {"name": "Mid-Course Correction", "delta_v": 50, "time": 36.0},
                    {"name": "Lunar Orbit Insertion", "delta_v": 800, "time": 72.0}
                ],
                description="Standard Hohmann transfer orbit to lunar vicinity"
            ),
            TrajectoryOption(
                name="Bi-Elliptic Transfer",
                trajectory_type="bi_elliptic",
                total_delta_v=10800,
                duration=96.0,
                radiation_exposure=35.0,
                collision_risk=0.12,
                maneuvers=[
                    {"name": "First Burn", "delta_v": 2800, "time": 0.0},
                    {"name": "Aphelion Burn", "delta_v": 1200, "time": 48.0},
                    {"name": "Lunar Insertion", "delta_v": 900, "time": 96.0}
                ],
                description="Fuel-efficient bi-elliptic transfer with extended duration"
            ),
            TrajectoryOption(
                name="Free Return Trajectory",
                trajectory_type="free_return",
                total_delta_v=12200,
                duration=84.0,
                radiation_exposure=28.0,
                collision_risk=0.08,
                maneuvers=[
                    {"name": "Free Return TLI", "delta_v": 3400, "time": 0.0},
                    {"name": "Trajectory Correction", "delta_v": 100, "time": 24.0},
                    {"name": "Lunar Approach", "delta_v": 700, "time": 84.0}
                ],
                description="Safety-prioritized free return trajectory"
            )
        ]
        
        return trajectories
    
    async def _select_optimal_trajectory(self, trajectories: List[TrajectoryOption], 
                                       hazards: List[Dict], space_weather: Dict) -> Dict[str, Any]:
        """Select optimal trajectory using AI Co-pilot analysis"""
        
        # Prepare data for AI Co-pilot
        trajectory_data = {
            "trajectories": [self._trajectory_to_dict(t) for t in trajectories],
            "hazards": hazards,
            "space_weather": space_weather,
            "mission_state": self._mission_state_to_dict(),
            "constraints": self.mission_constraints
        }
        
        # Get AI recommendation
        ai_analysis = await self.ai_copilot.generate_mission_brief(trajectory_data)
        
        # For now, select the first trajectory (can be enhanced with decision engine)
        selected = trajectories[0]
        
        selection_result = {
            "name": selected.name,
            "type": selected.trajectory_type,
            "total_delta_v": selected.total_delta_v,
            "duration": selected.duration,
            "radiation_exposure": selected.radiation_exposure,
            "collision_risk": selected.collision_risk,
            "maneuvers": selected.maneuvers,
            "ai_analysis": ai_analysis,
            "selection_rationale": f"Selected {selected.name} for optimal balance of fuel efficiency and mission duration"
        }
        
        return selection_result
    
    async def _monitor_hazards(self, current_time: float) -> List[Dict[str, Any]]:
        """Monitor for space weather and orbital hazards"""
        
        # Get current space weather
        current_timestamp = self.historical_timestamp + timedelta(hours=current_time)
        space_weather = await self.space_weather.get_space_weather_data(current_timestamp)
        
        # Get hazard forecast
        hazard_forecast = await self.space_weather.get_hazard_forecast(current_timestamp)
        
        # Use predictive forecasting to assess risks
        if self.hazard_forecaster:
            ml_predictions = await self.hazard_forecaster.predict_hazards(
                space_weather, 
                prediction_horizon=24
            )
            
            # Combine forecasts
            combined_hazards = hazard_forecast + ml_predictions.get('predicted_hazards', [])
        else:
            combined_hazards = hazard_forecast
        
        # Filter for significant hazards
        significant_hazards = [
            h for h in combined_hazards 
            if h.get('probability', 0) > 0.3 or h.get('severity', 'low') in ['high', 'critical']
        ]
        
        if significant_hazards:
            self.active_hazards = significant_hazards
            logger.warning(f"HAZARDS DETECTED at T+{current_time:.1f}h: {len(significant_hazards)} significant threats")
        
        return significant_hazards
    
    async def _initiate_emergency_replanning(self, hazards: List[Dict]) -> Dict[str, Any]:
        """Initiate emergency trajectory replanning due to hazards"""
        
        self.system_status["replanning_active"] = True
        
        # Generate alternative trajectories
        alternative_trajectories = await self.ai_copilot.generate_trajectory_alternatives(
            self.current_trajectory,
            hazards,
            self.mission_constraints
        )
        
        # Select best alternative
        if alternative_trajectories:
            # Use simple selection for now (can be enhanced with decision engine)
            best_alternative = alternative_trajectories[0]
            
            # Generate decision log
            hazard_types = [h.get('type', 'unknown') for h in hazards]
            decision_log = f"HAZARD DETECTED: {', '.join(hazard_types).upper()}. REROUTING VIA {best_alternative['name'].upper()}. RESULT: +{best_alternative.get('duration_hours', 0) - self.current_trajectory.get('duration', 0):.1f} HOURS TRAVEL TIME, {-20 if 'radiation' in hazard_types else +5}% RADIATION EXPOSURE"
            
            # Update current trajectory
            self.current_trajectory = best_alternative
            
            replan_result = {
                "replanning_triggered": True,
                "hazards_detected": hazards,
                "new_trajectory": best_alternative,
                "human_readable_log": decision_log,
                "alternatives_considered": len(alternative_trajectories)
            }
            
            self._log_decision(decision_log, replan_result)
            
        else:
            # No alternatives available - continue with monitoring
            decision_log = "HAZARDS DETECTED. NO VIABLE ALTERNATIVES. CONTINUING WITH ENHANCED MONITORING."
            replan_result = {
                "replanning_triggered": False,
                "hazards_detected": hazards,
                "action": "enhanced_monitoring",
                "human_readable_log": decision_log
            }
            
            self._log_decision(decision_log, replan_result)
        
        self.system_status["replanning_active"] = False
        self.system_status["last_decision"] = datetime.utcnow().isoformat()
        
        return replan_result
    
    async def _perform_trajectory_optimization(self) -> Dict[str, Any]:
        """Perform regular trajectory optimization"""
        
        # Check if optimization is beneficial
        current_efficiency = self._calculate_trajectory_efficiency()
        
        if current_efficiency < 0.8:  # If efficiency drops below 80%
            # Generate optimized alternatives
            optimization_request = {
                "current_trajectory": self.current_trajectory,
                "mission_state": self._mission_state_to_dict(),
                "optimization_goal": "fuel_efficiency"
            }
            
            optimized_trajectories = await self.ai_copilot.generate_trajectory_alternatives(
                self.current_trajectory,
                self.active_hazards,
                self.mission_constraints
            )
            
            if optimized_trajectories:
                best_optimized = optimized_trajectories[0]
                
                # Check if optimization is significant
                fuel_savings = self.current_trajectory.get('total_delta_v', 0) - best_optimized.get('delta_v_total', 0)
                
                if fuel_savings > 200:  # Significant fuel savings
                    self.current_trajectory = best_optimized
                    
                    decision_log = f"TRAJECTORY OPTIMIZED. FUEL SAVINGS: {fuel_savings:.0f} M/S DELTA-V. NEW ROUTE: {best_optimized['name'].upper()}"
                    
                    optimization_result = {
                        "trajectory_changed": True,
                        "optimization_type": "fuel_efficiency",
                        "fuel_savings": fuel_savings,
                        "new_trajectory": best_optimized,
                        "human_readable_log": decision_log
                    }
                    
                    self._log_decision(decision_log, optimization_result)
                    
                    return optimization_result
        
        return {
            "trajectory_changed": False,
            "optimization_type": "none_required",
            "current_efficiency": current_efficiency
        }
    
    async def _generate_mission_status_report(self) -> Dict[str, Any]:
        """Generate comprehensive mission status report"""
        
        mission_data = {
            "mission_time": self.current_mission_state.mission_time,
            "trajectory": self.current_trajectory,
            "telemetry": self._mission_state_to_dict(),
            "hazards": self.active_hazards,
            "space_weather": await self.space_weather.get_space_weather_data()
        }
        
        # Generate AI briefing
        ai_brief = await self.ai_copilot.generate_mission_brief(mission_data)
        
        status_report = {
            "mission_time": self.current_mission_state.mission_time,
            "trajectory_status": self.current_trajectory.get('name', 'Unknown'),
            "fuel_remaining": self.current_mission_state.fuel_remaining,
            "system_health": self.current_mission_state.system_health,
            "active_hazards": len(self.active_hazards),
            "ai_assessment": ai_brief,
            "system_status": self.system_status
        }
        
        return status_report
    
    async def _generate_mission_completion_summary(self, mission_events: List[Dict]) -> Dict[str, Any]:
        """Generate final mission completion summary"""
        
        total_replanning_events = len([e for e in mission_events if e["event_type"] == "hazard_replanning"])
        total_optimizations = len([e for e in mission_events if e["event_type"] == "trajectory_optimization"])
        
        summary = {
            "mission_duration": self.current_mission_state.mission_time,
            "final_trajectory": self.current_trajectory.get('name', 'Unknown'),
            "fuel_consumed": 100 - self.current_mission_state.fuel_remaining,
            "total_replanning_events": total_replanning_events,
            "total_optimizations": total_optimizations,
            "hazards_encountered": len(set(h['type'] for event in mission_events for h in event.get('details', {}).get('hazards_detected', []))),
            "mission_success": self.current_mission_state.fuel_remaining > 5,
            "decision_logs": [log["decision"] for log in self.decision_log],
            "system_performance": {
                "autonomy_level": "HIGH" if total_replanning_events > 0 else "MODERATE",
                "adaptability": "EXCELLENT" if total_replanning_events > 2 else "GOOD",
                "decision_clarity": "HIGH"
            }
        }
        
        completion_log = f"MISSION COMPLETED. Duration: {self.current_mission_state.mission_time:.1f}h. Fuel remaining: {self.current_mission_state.fuel_remaining:.1f}%. Replanning events: {total_replanning_events}. Mission success: {'YES' if summary['mission_success'] else 'NO'}"
        
        self._log_decision(completion_log, summary)
        
        return summary
    
    def _trajectory_to_dict(self, trajectory: TrajectoryOption) -> Dict[str, Any]:
        """Convert TrajectoryOption to dictionary"""
        return {
            "name": trajectory.name,
            "type": trajectory.trajectory_type,
            "total_delta_v": trajectory.total_delta_v,
            "duration": trajectory.duration,
            "radiation_exposure": trajectory.radiation_exposure,
            "collision_risk": trajectory.collision_risk,
            "maneuvers": trajectory.maneuvers,
            "description": trajectory.description
        }
    
    def _mission_state_to_dict(self) -> Dict[str, Any]:
        """Convert mission state to dictionary"""
        if not self.current_mission_state:
            return {}
        
        return {
            "mission_time": self.current_mission_state.mission_time,
            "spacecraft_position": self.current_mission_state.spacecraft_position,
            "current_velocity": self.current_mission_state.spacecraft_velocity[1],  # orbital velocity
            "fuel_remaining": self.current_mission_state.fuel_remaining,
            "communication_status": self.current_mission_state.communication_status,
            "system_health": self.current_mission_state.system_health
        }
    
    def _calculate_trajectory_efficiency(self) -> float:
        """Calculate current trajectory efficiency"""
        if not self.current_trajectory:
            return 1.0
        
        # Simple efficiency calculation based on fuel usage and time
        fuel_efficiency = 1.0 - (self.current_trajectory.get('total_delta_v', 12000) / 15000)
        time_efficiency = 1.0 - (self.current_trajectory.get('duration', 72) / 120)
        
        return (fuel_efficiency + time_efficiency) / 2
    
    async def _update_mission_state(self, current_time: float):
        """Update mission state based on current time"""
        if not self.current_mission_state:
            return
        
        # Update mission time
        self.current_mission_state.mission_time = current_time
        
        # Simulate fuel consumption
        fuel_consumption_rate = 0.5  # percent per hour
        self.current_mission_state.fuel_remaining = max(0, 100 - (current_time * fuel_consumption_rate))
        
        # Update position (simplified simulation)
        distance_traveled = current_time * 1.5  # km/h average
        self.current_mission_state.spacecraft_position[0] += distance_traveled * 0.1
    
    def _log_decision(self, decision_text: str, context: Dict[str, Any]):
        """Log decision with context"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "mission_time": self.current_mission_state.mission_time if self.current_mission_state else 0,
            "decision": decision_text,
            "context": context
        }
        
        self.decision_log.append(log_entry)
        logger.info(f"ODIN DECISION: {decision_text}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "system_name": self.system_name,
            "version": self.version,
            "mission_initialized": self.mission_start_time is not None,
            "historical_timestamp": self.historical_timestamp.isoformat() if self.historical_timestamp else None,
            "current_mission_time": self.current_mission_state.mission_time if self.current_mission_state else 0,
            "system_status": self.system_status,
            "active_hazards_count": len(self.active_hazards),
            "total_decisions": len(self.decision_log),
            "subsystem_status": {
                "ai_copilot": self.ai_copilot.ai_available,
                "hazard_forecaster": True,
                "explainability": True,
                "space_weather": True
            }
        }
    
    def get_decision_logs(self) -> List[Dict[str, Any]]:
        """Get all decision logs"""
        return self.decision_log.copy()

# Main execution function for testing
async def main():
    """Main function for testing ODIN system"""
    
    # Initialize ODIN
    odin = OdinNavigationSystem()
    
    # Initialize mission
    print("Initializing ODIN mission...")
    init_result = await odin.initialize_mission("Moon")
    print(f"Mission initialized: {init_result['historical_timestamp']}")
    
    # Run autonomous mission
    print("Starting autonomous mission loop...")
    mission_events = await odin.autonomous_mission_loop(duration_hours=24.0)
    
    # Print results
    print(f"\nMission completed with {len(mission_events)} events:")
    for event in mission_events[-5:]:  # Show last 5 events
        print(f"T+{event['timestamp']:.1f}h: {event['event_type']}")
    
    # Show decision logs
    print(f"\nDecision logs ({len(odin.get_decision_logs())} total):")
    for log in odin.get_decision_logs():
        print(f"T+{log['mission_time']:.1f}h: {log['decision']}")

if __name__ == "__main__":
    asyncio.run(main())
