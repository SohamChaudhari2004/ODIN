import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from ..models.schemas import TrajectoryPlan, HazardEvent
import sys
import os

# Add ai-services to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ai-services')))

try:
    from odin_main import OdinNavigationSystem, TrajectoryOption
    ODIN_AVAILABLE = True
    logging.info("Successfully imported OdinNavigationSystem.")
except ImportError as e:
    logging.warning(f"Could not import OdinNavigationSystem: {e}. DecisionEngine will not use AI features.")
    ODIN_AVAILABLE = False
    OdinNavigationSystem = None
    TrajectoryOption = None

logger = logging.getLogger(__name__)

class DecisionEngine:
    """ODIN Decision Engine for evaluating and selecting optimal trajectories with human-readable justification logs"""
    
    def __init__(self):
        if ODIN_AVAILABLE:
            self.odin_system = OdinNavigationSystem()
            logging.info("DecisionEngine initialized with OdinNavigationSystem.")
        else:
            self.odin_system = None
            logging.warning("DecisionEngine initialized without OdinNavigationSystem.")

        self.decision_history = []
        self.mission_constraints = {
            "max_delta_v": 15000,    # m/s
            "max_duration": 120,     # hours
            "max_radiation": 50,     # percentage of safe limit
            "min_success_probability": 0.85
        }
    
    async def evaluate_trajectories(self,
                            trajectories: List[TrajectoryPlan], 
                            hazards: List[HazardEvent],
                            constraints: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Evaluate trajectory options and generate human-readable decision logs
        """
        
        if not self.odin_system:
            raise Exception("OdinNavigationSystem not available. Cannot evaluate trajectories.")

        if not trajectories:
            raise ValueError("No trajectories provided for evaluation")
        
        # Convert TrajectoryPlan to TrajectoryOption
        trajectory_options = []
        for t in trajectories:
            trajectory_options.append(TrajectoryOption(
                name=t.name,
                trajectory_type=t.trajectory_type,
                total_delta_v=t.total_delta_v,
                duration=t.duration_hours,
                radiation_exposure=t.radiation_exposure,
                collision_risk=t.collision_risk,
                maneuvers=t.maneuvers,
                description=t.description
            ))

        # Get space weather data
        space_weather_data = await self.odin_system.space_weather.get_space_weather_data()

        # Call the OdinNavigationSystem to select the optimal trajectory
        selected_trajectory = await self.odin_system._select_optimal_trajectory(
            trajectory_options,
            [h.dict() for h in hazards],
            space_weather_data
        )

        return selected_trajectory
    
    def get_decision_history(self) -> List[Dict[str, Any]]:
        """Get history of recent decisions"""
        if self.odin_system:
            return self.odin_system.get_decision_logs()
        return self.decision_history.copy()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the decision engine"""
        
        if not self.decision_history:
            return {"message": "No decisions recorded yet"}
        
        total_decisions = len(self.decision_history)
        emergency_decisions = len([d for d in self.decision_history if d.get("emergency_decision", False)])
        
        risk_distribution = {}
        for decision in self.decision_history:
            risk_level = decision.get("risk_level", "unknown")
            risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1
        
        return {
            "total_decisions": total_decisions,
            "emergency_decisions": emergency_decisions,
            "emergency_rate": f"{(emergency_decisions / total_decisions * 100):.1f}%",
            "risk_distribution": risk_distribution,
            "recent_decisions": self.decision_history[-5:] if len(self.decision_history) >= 5 else self.decision_history
        }
