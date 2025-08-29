import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from ..models.schemas import HazardEvent, TrajectoryPlan

logger = logging.getLogger(__name__)

class LogGenerator:
    """Generate human-readable summaries of mission decisions and events"""
    
    def __init__(self):
        self.log_templates = {
            "hazard_detected": {
                "solar_flare": "Solar flare detected: {severity} class event. Radiation levels {impact}. {action}",
                "cme": "CME detected: {details}. Estimated arrival: {arrival}. {action}", 
                "debris_conjunction": "Debris conjunction alert: {details}. Collision probability: {probability}. {action}"
            },
            "trajectory_change": "Trajectory changed: {old_route} → {new_route}. ΔV {delta_v_change:+.0f} m/s, Time {time_change:+.1f} hrs, Radiation {radiation_change:+.0f}%",
            "decision_rationale": "{decision_type}: {rationale}",
            "system_status": "System status update: {component} - {status}. {details}"
        }
    
    def generate_hazard_detection_log(self, hazard: HazardEvent) -> str:
        """Generate log for hazard detection"""
        try:
            template = self.log_templates["hazard_detected"][hazard.event_type]
            
            if hazard.event_type == "solar_flare":
                severity_map = {1: "C", 2: "C", 3: "M", 4: "M", 5: "X", 6: "X", 7: "X", 8: "X", 9: "X", 10: "X"}
                severity_class = severity_map.get(int(hazard.severity), "X")
                
                impact = "elevated" if hazard.severity < 5 else "critical"
                action = "Monitoring radiation levels" if hazard.severity < 5 else "Recommend immediate shelter protocol"
                
                return template.format(
                    severity=severity_class,
                    impact=impact,
                    action=action
                )
            
            elif hazard.event_type == "cme":
                speed = 400 + hazard.severity * 100  # Estimate CME speed
                arrival_hours = 24 + (10 - hazard.severity) * 6  # Higher severity = faster arrival
                
                action = "Monitor trajectory" if hazard.severity < 6 else "Recommend trajectory modification"
                
                return template.format(
                    details=f"Speed {speed:.0f} km/s, Severity {hazard.severity:.1f}/10",
                    arrival=f"T+{arrival_hours:.0f}h",
                    action=action
                )
            
            elif hazard.event_type == "debris_conjunction":
                probability = hazard.severity * 1e-5  # Convert severity to probability
                action = "Continue monitoring" if hazard.severity < 5 else "Recommend avoidance maneuver"
                
                return template.format(
                    details=f"Object size ~{hazard.severity * 0.5:.1f}m",
                    probability=f"{probability:.2e}",
                    action=action
                )
            
        except Exception as e:
            logger.error(f"Error generating hazard log: {e}")
            return f"{hazard.event_type.title()} detected: Severity {hazard.severity:.1f}. Monitoring situation."
    
    def generate_trajectory_change_log(self, 
                                     old_trajectory: TrajectoryPlan,
                                     new_trajectory: TrajectoryPlan,
                                     reason: str = "") -> str:
        """Generate log for trajectory changes"""
        try:
            delta_v_change = new_trajectory.total_delta_v - old_trajectory.total_delta_v
            time_change = new_trajectory.duration - old_trajectory.duration
            radiation_change = new_trajectory.radiation_exposure - old_trajectory.radiation_exposure
            
            log_message = self.log_templates["trajectory_change"].format(
                old_route=old_trajectory.name,
                new_route=new_trajectory.name,
                delta_v_change=delta_v_change,
                time_change=time_change,
                radiation_change=radiation_change
            )
            
            if reason:
                log_message += f". Reason: {reason}"
            
            return log_message
            
        except Exception as e:
            logger.error(f"Error generating trajectory change log: {e}")
            return f"Trajectory changed from {old_trajectory.name} to {new_trajectory.name}"
    
    def generate_decision_rationale_log(self, 
                                      decision_data: Dict[str, Any],
                                      decision_type: str = "Route Selection") -> str:
        """Generate decision rationale log"""
        try:
            rationale = decision_data.get("decision_rationale", "Decision made based on current parameters")
            
            return self.log_templates["decision_rationale"].format(
                decision_type=decision_type,
                rationale=rationale
            )
            
        except Exception as e:
            logger.error(f"Error generating decision rationale log: {e}")
            return f"{decision_type}: Decision made based on current mission parameters"
    
    def generate_cme_reroute_example(self) -> str:
        """Generate the specific example log mentioned in requirements"""
        return "CME detected: rerouting via Plan B. ΔV +15 m/s, Time +6 hrs, Radiation –90%"
    
    def generate_system_status_log(self, 
                                 component: str, 
                                 status: str, 
                                 details: str = "") -> str:
        """Generate system status update log"""
        try:
            return self.log_templates["system_status"].format(
                component=component,
                status=status,
                details=details
            )
        except Exception as e:
            logger.error(f"Error generating system status log: {e}")
            return f"{component}: {status}"
    
    def generate_mission_summary_log(self, 
                                   mission_data: Dict[str, Any]) -> List[str]:
        """Generate comprehensive mission summary logs"""
        summary_logs = []
        
        try:
            # Mission phase log
            current_phase = mission_data.get("current_phase", "Unknown")
            mission_time = mission_data.get("mission_time", 0)
            summary_logs.append(f"Mission Status: {current_phase} at T+{mission_time:.1f}h")
            
            # Trajectory status
            if "trajectory" in mission_data:
                traj = mission_data["trajectory"]
                progress = (mission_time / traj.duration * 100) if traj.duration > 0 else 0
                summary_logs.append(f"Trajectory: {traj.name} - {progress:.1f}% complete")
            
            # Hazard summary
            hazards = mission_data.get("hazards", [])
            if hazards:
                hazard_count = len(hazards)
                max_severity = max(h.severity for h in hazards)
                summary_logs.append(f"Active Hazards: {hazard_count} detected, max severity {max_severity:.1f}")
            else:
                summary_logs.append("Hazard Status: All clear")
            
            # Resource status
            if "telemetry" in mission_data:
                fuel = mission_data["telemetry"].fuel_remaining
                summary_logs.append(f"Fuel Status: {fuel:.1f}% remaining")
            
        except Exception as e:
            logger.error(f"Error generating mission summary: {e}")
            summary_logs.append("Mission summary generation error")
        
        return summary_logs
    
    def generate_ai_recommendation_log(self, 
                                     recommendation: Dict[str, Any]) -> str:
        """Generate log for AI recommendations"""
        try:
            rec_type = recommendation.get("type", "general")
            priority = recommendation.get("priority", "medium")
            title = recommendation.get("title", "AI Recommendation")
            action = recommendation.get("action", "Review current parameters")
            confidence = recommendation.get("confidence", 50)
            
            priority_text = "CRITICAL" if priority == "critical" else priority.upper()
            
            return f"AI-{priority_text}: {title}. Recommended action: {action}. Confidence: {confidence}%"
            
        except Exception as e:
            logger.error(f"Error generating AI recommendation log: {e}")
            return "AI recommendation generated"
    
    def format_log_message(self, 
                          message: str,
                          source: str = "ODIN-SYSTEM",
                          priority: str = "INFO",
                          timestamp: Optional[str] = None) -> Dict[str, str]:
        """Format a complete log message with metadata"""
        
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()
        
        return {
            "timestamp": timestamp,
            "source": source,
            "priority": priority.upper(),
            "message": message,
            "id": f"log-{int(datetime.utcnow().timestamp())}"
        }
