"""
ODIN Autonomous Decision Core using LangGraph
Implements the ODIN decision loop: Monitor → Detect → Consult → Evaluate → Decide → Execute
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, TypedDict
from dataclasses import dataclass
import json

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    from langchain_core.messages import HumanMessage, AIMessage
    LANGGRAPH_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ LangGraph not available: {e}")
    LANGGRAPH_AVAILABLE = False

# Local imports
try:
    from .huggingface_llm import HuggingFaceLLMService
    from .space_weather_service import SpaceWeatherDataService
except ImportError:
    # Fallback for direct execution
    from huggingface_llm import HuggingFaceLLMService
    from space_weather_service import SpaceWeatherDataService

logger = logging.getLogger(__name__)

class OdinState(TypedDict):
    """State maintained by the ODIN agent"""
    mission_id: str
    current_time: datetime
    mission_phase: str  # "launch", "transfer", "arrival", "completed"
    spacecraft_position: List[float]
    spacecraft_velocity: List[float]
    fuel_remaining: float
    active_hazards: List[Dict[str, Any]]
    current_trajectory: Optional[Dict[str, Any]]
    decision_history: List[Dict[str, Any]]
    system_alerts: List[str]
    next_action: Optional[str]
    confidence_level: float

@dataclass
class DecisionMetrics:
    """Metrics for evaluating decisions"""
    safety_score: float
    fuel_efficiency: float
    time_efficiency: float
    mission_success_probability: float
    overall_score: float

class OdinDecisionEngine:
    """ODIN Autonomous Decision Core using LangGraph"""
    
    def __init__(self):
        self.engine_name = "ODIN Decision Engine"
        self.version = "1.0.0"
        
        # Initialize services
        self.llm_service = HuggingFaceLLMService()
        self.space_weather_service = SpaceWeatherDataService()
        
        # Decision weights (crew safety > mission time > fuel)
        self.decision_weights = {
            "safety": 0.6,
            "fuel_efficiency": 0.2,
            "time_efficiency": 0.2
        }
        
        # LangGraph components
        self.workflow = None
        self.memory = MemorySaver() if LANGGRAPH_AVAILABLE else None
        
        # Initialize the decision graph
        if LANGGRAPH_AVAILABLE:
            self._initialize_decision_graph()
        
        logger.info(f"Initialized {self.engine_name} v{self.version}")
    
    def _initialize_decision_graph(self):
        """Initialize the LangGraph decision workflow"""
        
        workflow = StateGraph(OdinState)
        
        # Add nodes for the ODIN decision loop
        workflow.add_node("monitor", self._monitor_node)
        workflow.add_node("detect", self._detect_node)
        workflow.add_node("consult", self._consult_node)
        workflow.add_node("evaluate", self._evaluate_node)
        workflow.add_node("decide", self._decide_node)
        workflow.add_node("execute", self._execute_node)
        
        # Define the flow
        workflow.set_entry_point("monitor")
        workflow.add_edge("monitor", "detect")
        workflow.add_edge("detect", "consult")
        workflow.add_edge("consult", "evaluate")
        workflow.add_edge("evaluate", "decide")
        workflow.add_edge("decide", "execute")
        workflow.add_edge("execute", END)
        
        self.workflow = workflow.compile(checkpointer=self.memory)
        
        logger.info("ODIN decision graph initialized")
    
    async def execute_decision_cycle(
        self, 
        initial_state: OdinState,
        mission_id: str
    ) -> Dict[str, Any]:
        """Execute one complete ODIN decision cycle"""
        
        if not LANGGRAPH_AVAILABLE:
            return await self._fallback_decision_cycle(initial_state)
        
        try:
            # Run the LangGraph workflow
            config = {"configurable": {"thread_id": mission_id}}
            
            result = await self.workflow.ainvoke(initial_state, config)
            
            return {
                "success": True,
                "final_state": result,
                "decision_made": result.get("next_action") is not None,
                "confidence": result.get("confidence_level", 0.7),
                "decision_log": self._generate_decision_log(result)
            }
            
        except Exception as e:
            logger.error(f"Error in decision cycle: {e}")
            return await self._fallback_decision_cycle(initial_state)
    
    async def continuous_monitoring(
        self,
        mission_id: str,
        duration_hours: float,
        check_interval_minutes: int = 5
    ) -> List[Dict[str, Any]]:
        """Run continuous ODIN monitoring for specified duration"""
        
        mission_events = []
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(hours=duration_hours)
        
        # Initialize mission state
        current_state = OdinState(
            mission_id=mission_id,
            current_time=start_time,
            mission_phase="transfer",
            spacecraft_position=[7000.0, 0.0, 0.0],  # LEO
            spacecraft_velocity=[0.0, 7.8, 0.0],     # km/s
            fuel_remaining=85.0,
            active_hazards=[],
            current_trajectory=None,
            decision_history=[],
            system_alerts=[],
            next_action=None,
            confidence_level=0.8
        )
        
        logger.info(f"Starting continuous monitoring for {duration_hours} hours")
        
        try:
            while datetime.utcnow() < end_time:
                # Update current time
                current_state["current_time"] = datetime.utcnow()
                
                # Execute decision cycle
                result = await self.execute_decision_cycle(current_state, mission_id)
                
                if result["success"]:
                    current_state = result["final_state"]
                    
                    if result["decision_made"]:
                        event = {
                            "timestamp": datetime.utcnow(),
                            "event_type": "autonomous_decision",
                            "description": result["decision_log"],
                            "confidence": result["confidence"],
                            "state_snapshot": dict(current_state)
                        }
                        mission_events.append(event)
                        logger.info(f"ODIN Decision: {result['decision_log']}")
                
                # Wait for next check interval
                await asyncio.sleep(check_interval_minutes * 60)
                
        except Exception as e:
            logger.error(f"Error in continuous monitoring: {e}")
            mission_events.append({
                "timestamp": datetime.utcnow(),
                "event_type": "system_error",
                "description": f"Monitoring interrupted: {str(e)}",
                "error": True
            })
        
        logger.info(f"Continuous monitoring completed. {len(mission_events)} events recorded.")
        return mission_events
    
    # LangGraph Node Functions
    
    async def _monitor_node(self, state: OdinState) -> OdinState:
        """Monitor mission state and environment"""
        
        try:
            # Update spacecraft state (simplified orbital propagation)
            elapsed_hours = (state["current_time"] - datetime.utcnow()).total_seconds() / 3600
            
            # Simple position update (in real system, would use proper orbital mechanics)
            position = state["spacecraft_position"]
            velocity = state["spacecraft_velocity"]
            
            # Update position based on velocity
            dt = 0.1  # hours
            new_position = [
                position[0] + velocity[0] * dt * 3600,  # Convert to km
                position[1] + velocity[1] * dt * 3600,
                position[2] + velocity[2] * dt * 3600
            ]
            
            # Update fuel consumption
            fuel_consumption_rate = 0.01  # % per hour
            new_fuel = max(0, state["fuel_remaining"] - fuel_consumption_rate)
            
            # Update state
            state["spacecraft_position"] = new_position
            state["fuel_remaining"] = new_fuel
            
            # Clear previous alerts
            state["system_alerts"] = []
            
            # Add alerts based on conditions
            if new_fuel < 20:
                state["system_alerts"].append("LOW_FUEL_WARNING")
            
            if abs(new_position[0]) > 100000:  # Far from Earth
                state["mission_phase"] = "deep_space"
            
            logger.debug("Monitor phase completed")
            
        except Exception as e:
            logger.error(f"Error in monitor node: {e}")
            state["system_alerts"].append(f"MONITOR_ERROR: {str(e)}")
        
        return state
    
    async def _detect_node(self, state: OdinState) -> OdinState:
        """Detect space hazards and threats"""
        
        try:
            # Get current space weather conditions
            current_time = state["current_time"]
            space_conditions = await self.space_weather_service.get_current_conditions(current_time)
            
            # Extract active hazards
            active_hazards = space_conditions.get("active_hazards", [])
            
            # Evaluate hazard threats to current trajectory
            threatening_hazards = []
            for hazard in active_hazards:
                threat_level = await self._evaluate_hazard_threat(hazard, state)
                if threat_level > 0.3:  # Significant threat threshold
                    hazard["threat_level"] = threat_level
                    threatening_hazards.append(hazard)
            
            state["active_hazards"] = threatening_hazards
            
            # Add hazard alerts
            if threatening_hazards:
                hazard_types = [h.get("hazard_type", "unknown") for h in threatening_hazards]
                state["system_alerts"].append(f"HAZARDS_DETECTED: {', '.join(hazard_types)}")
            
            logger.debug(f"Detect phase completed. {len(threatening_hazards)} hazards found.")
            
        except Exception as e:
            logger.error(f"Error in detect node: {e}")
            state["system_alerts"].append(f"DETECT_ERROR: {str(e)}")
        
        return state
    
    async def _consult_node(self, state: OdinState) -> OdinState:
        """Consult AI copilot for strategy recommendations"""
        
        try:
            if not state["active_hazards"]:
                # No hazards, continue nominal operations
                return state
            
            # Prepare mission state for LLM
            mission_state = {
                "position": state["spacecraft_position"],
                "velocity": state["spacecraft_velocity"],
                "fuel_remaining": state["fuel_remaining"],
                "mission_phase": state["mission_phase"],
                "mission_time": state["current_time"].isoformat()
            }
            
            # Get mitigation strategies for each significant hazard
            mitigation_strategies = []
            for hazard in state["active_hazards"]:
                if hazard.get("threat_level", 0) > 0.5:  # High threat
                    strategies = await self.llm_service.generate_mitigation_strategies(
                        hazard, mission_state, "Prioritize crew safety above all else."
                    )
                    mitigation_strategies.extend(strategies)
            
            # Store strategies in state for evaluation
            state["mitigation_options"] = mitigation_strategies
            
            logger.debug(f"Consult phase completed. {len(mitigation_strategies)} strategies generated.")
            
        except Exception as e:
            logger.error(f"Error in consult node: {e}")
            state["system_alerts"].append(f"CONSULT_ERROR: {str(e)}")
            state["mitigation_options"] = []
        
        return state
    
    async def _evaluate_node(self, state: OdinState) -> OdinState:
        """Evaluate trajectory options and mitigation strategies"""
        
        try:
            mitigation_options = state.get("mitigation_options", [])
            
            if not mitigation_options:
                # No options to evaluate
                return state
            
            # Evaluate each option
            evaluated_options = []
            for option in mitigation_options:
                metrics = await self._calculate_option_metrics(option, state)
                option["evaluation_metrics"] = metrics
                option["overall_score"] = self._calculate_overall_score(metrics)
                evaluated_options.append(option)
            
            # Sort by overall score (highest first)
            evaluated_options.sort(key=lambda x: x["overall_score"], reverse=True)
            
            state["evaluated_options"] = evaluated_options
            
            logger.debug(f"Evaluate phase completed. {len(evaluated_options)} options evaluated.")
            
        except Exception as e:
            logger.error(f"Error in evaluate node: {e}")
            state["system_alerts"].append(f"EVALUATE_ERROR: {str(e)}")
            state["evaluated_options"] = []
        
        return state
    
    async def _decide_node(self, state: OdinState) -> OdinState:
        """Make autonomous decision based on evaluation"""
        
        try:
            evaluated_options = state.get("evaluated_options", [])
            
            if not evaluated_options:
                # No options available, continue current trajectory
                state["next_action"] = "continue_nominal"
                state["confidence_level"] = 0.8
                return state
            
            # Choose best option
            best_option = evaluated_options[0]
            
            # Decision threshold - only act if confidence is high enough
            confidence_threshold = 0.7
            
            if best_option["overall_score"] > confidence_threshold:
                state["next_action"] = "execute_mitigation"
                state["chosen_option"] = best_option
                state["confidence_level"] = best_option["overall_score"]
                
                # Create decision record
                decision_record = {
                    "timestamp": state["current_time"],
                    "decision_type": "hazard_mitigation",
                    "chosen_option": best_option,
                    "all_options": evaluated_options,
                    "confidence": best_option["overall_score"],
                    "reasoning": f"Selected {best_option['name']} with score {best_option['overall_score']:.2f}"
                }
                
                state["decision_history"].append(decision_record)
                
            else:
                # Confidence too low, maintain current course
                state["next_action"] = "maintain_course"
                state["confidence_level"] = 0.5
            
            logger.debug(f"Decide phase completed. Action: {state['next_action']}")
            
        except Exception as e:
            logger.error(f"Error in decide node: {e}")
            state["system_alerts"].append(f"DECIDE_ERROR: {str(e)}")
            state["next_action"] = "error_recovery"
            state["confidence_level"] = 0.3
        
        return state
    
    async def _execute_node(self, state: OdinState) -> OdinState:
        """Execute the decided action"""
        
        try:
            action = state.get("next_action", "continue_nominal")
            
            if action == "execute_mitigation":
                chosen_option = state.get("chosen_option", {})
                
                # Simulate execution of mitigation strategy
                delta_v_cost = chosen_option.get("delta_v_cost", 0)
                fuel_cost = delta_v_cost * 0.05  # Simplified fuel calculation
                
                # Update spacecraft state
                state["fuel_remaining"] = max(0, state["fuel_remaining"] - fuel_cost)
                
                # Update trajectory (simplified)
                if "trajectory_adjustment" in chosen_option:
                    # Would implement proper trajectory update here
                    pass
                
                execution_log = f"Executed {chosen_option['name']}: ΔV={delta_v_cost}m/s, Fuel used={fuel_cost:.1f}%"
                
            elif action == "maintain_course":
                execution_log = "Maintaining current trajectory - insufficient confidence in alternatives"
                
            elif action == "continue_nominal":
                execution_log = "Continuing nominal operations - no threats detected"
                
            else:
                execution_log = f"Executed action: {action}"
            
            # Add to system alerts
            state["system_alerts"].append(f"EXECUTED: {execution_log}")
            
            logger.debug(f"Execute phase completed: {execution_log}")
            
        except Exception as e:
            logger.error(f"Error in execute node: {e}")
            state["system_alerts"].append(f"EXECUTE_ERROR: {str(e)}")
        
        return state
    
    # Helper Methods
    
    async def _evaluate_hazard_threat(self, hazard: Dict[str, Any], state: OdinState) -> float:
        """Evaluate how much of a threat a hazard poses to current trajectory"""
        
        try:
            hazard_type = hazard.get("hazard_type", "unknown")
            severity = hazard.get("severity", 0.5)
            
            # Base threat level from severity
            threat_level = severity
            
            # Adjust based on hazard type
            if hazard_type == "solar_flare":
                # Solar flares primarily affect electronics and radiation exposure
                threat_level *= 0.8
            elif hazard_type == "debris":
                # Debris is a direct collision threat
                threat_level *= 1.2
            elif hazard_type == "cme":
                # CMEs affect large areas
                threat_level *= 1.0
            
            # Adjust based on proximity (simplified)
            # In real implementation, would calculate geometric intersection
            position = state["spacecraft_position"]
            spacecraft_distance = (position[0]**2 + position[1]**2 + position[2]**2)**0.5
            
            if spacecraft_distance > 50000:  # Deep space
                threat_level *= 0.7  # Reduced threat in deep space
            
            return min(1.0, max(0.0, threat_level))
            
        except Exception as e:
            logger.error(f"Error evaluating hazard threat: {e}")
            return 0.5  # Default moderate threat
    
    async def _calculate_option_metrics(self, option: Dict[str, Any], state: OdinState) -> DecisionMetrics:
        """Calculate metrics for a mitigation option"""
        
        try:
            # Extract option parameters
            delta_v_cost = option.get("delta_v_cost", 1000)
            time_impact = option.get("time_impact", 0)
            risk_reduction = option.get("risk_reduction", 0.5)
            
            # Calculate metrics (0.0 to 1.0 scale)
            
            # Safety score - higher risk reduction is better
            safety_score = min(1.0, risk_reduction)
            
            # Fuel efficiency - lower delta-v is better
            max_acceptable_deltav = 5000  # m/s
            fuel_efficiency = max(0.0, 1.0 - (delta_v_cost / max_acceptable_deltav))
            
            # Time efficiency - lower time impact is better
            max_acceptable_delay = 48  # hours
            time_efficiency = max(0.0, 1.0 - (time_impact / max_acceptable_delay))
            
            # Mission success probability (combination of factors)
            current_fuel = state["fuel_remaining"]
            fuel_impact = delta_v_cost * 0.05  # Simplified fuel consumption
            fuel_after_maneuver = current_fuel - fuel_impact
            
            if fuel_after_maneuver < 10:  # Critical fuel level
                mission_success = 0.3
            elif fuel_after_maneuver < 30:  # Low fuel
                mission_success = 0.7
            else:
                mission_success = 0.9
            
            return DecisionMetrics(
                safety_score=safety_score,
                fuel_efficiency=fuel_efficiency,
                time_efficiency=time_efficiency,
                mission_success_probability=mission_success,
                overall_score=0.0  # Will be calculated separately
            )
            
        except Exception as e:
            logger.error(f"Error calculating option metrics: {e}")
            return DecisionMetrics(0.5, 0.5, 0.5, 0.5, 0.5)
    
    def _calculate_overall_score(self, metrics: DecisionMetrics) -> float:
        """Calculate overall score using weighted metrics"""
        
        # Weighted score based on ODIN priorities
        overall_score = (
            metrics.safety_score * self.decision_weights["safety"] +
            metrics.fuel_efficiency * self.decision_weights["fuel_efficiency"] +
            metrics.time_efficiency * self.decision_weights["time_efficiency"]
        )
        
        # Apply mission success probability as a multiplier
        overall_score *= metrics.mission_success_probability
        
        return min(1.0, max(0.0, overall_score))
    
    def _generate_decision_log(self, state: OdinState) -> str:
        """Generate human-readable decision log"""
        
        action = state.get("next_action", "unknown")
        confidence = state.get("confidence_level", 0.5)
        
        if action == "execute_mitigation":
            chosen_option = state.get("chosen_option", {})
            return (f"HAZARD MITIGATION: Executing {chosen_option.get('name', 'strategy')} "
                   f"(Confidence: {confidence:.1%})")
        
        elif action == "maintain_course":
            return f"MAINTAINING COURSE: Insufficient confidence in alternatives ({confidence:.1%})"
        
        elif action == "continue_nominal":
            return "NOMINAL OPERATIONS: No threats detected"
        
        else:
            return f"ACTION: {action} (Confidence: {confidence:.1%})"
    
    async def _fallback_decision_cycle(self, state: OdinState) -> Dict[str, Any]:
        """Fallback decision cycle when LangGraph is not available"""
        
        try:
            # Simple rule-based decision making
            active_hazards = state.get("active_hazards", [])
            
            if not active_hazards:
                return {
                    "success": True,
                    "final_state": state,
                    "decision_made": False,
                    "confidence": 0.8,
                    "decision_log": "No hazards detected - continuing nominal operations"
                }
            
            # Simple hazard response
            high_threat_hazards = [h for h in active_hazards if h.get("severity", 0) > 0.7]
            
            if high_threat_hazards:
                action = "execute_conservative_avoidance"
                confidence = 0.7
                decision_log = f"High threat detected - executing conservative avoidance"
            else:
                action = "continue_with_monitoring"
                confidence = 0.6
                decision_log = f"Moderate threats detected - continuing with enhanced monitoring"
            
            state["next_action"] = action
            state["confidence_level"] = confidence
            
            return {
                "success": True,
                "final_state": state,
                "decision_made": True,
                "confidence": confidence,
                "decision_log": decision_log
            }
            
        except Exception as e:
            logger.error(f"Error in fallback decision cycle: {e}")
            return {
                "success": False,
                "final_state": state,
                "decision_made": False,
                "confidence": 0.3,
                "decision_log": f"Decision system error: {str(e)}"
            }
    
    async def get_decision_summary(self, mission_id: str) -> Dict[str, Any]:
        """Get summary of decisions made during mission"""
        
        # In a real implementation, this would query the mission state from storage
        return {
            "mission_id": mission_id,
            "total_decisions": 0,
            "decision_types": [],
            "average_confidence": 0.75,
            "success_rate": 1.0,
            "note": "Decision summary requires mission state persistence"
        }
    
    async def shutdown(self):
        """Shutdown the decision engine and close resources"""
        
        try:
            await self.llm_service.close_session()
            await self.space_weather_service.close()
            logger.info("ODIN Decision Engine shutdown completed")
        
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
