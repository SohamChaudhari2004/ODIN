import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from ..models.schemas import TrajectoryPlan, HazardEvent

logger = logging.getLogger(__name__)

class DecisionEngine:
    """ODIN Decision Engine for evaluating and selecting optimal trajectories with human-readable justification logs"""
    
    def __init__(self):
        self.evaluation_weights = {
            "fuel_cost": 0.30,      # ΔV cost in m/s
            "travel_time": 0.25,    # Time to destination
            "radiation_risk": 0.30, # Crew safety from radiation
            "collision_risk": 0.15  # Debris/collision avoidance
        }
        self.decision_history = []
        self.mission_constraints = {
            "max_delta_v": 15000,    # m/s
            "max_duration": 120,     # hours
            "max_radiation": 50,     # percentage of safe limit
            "min_success_probability": 0.85
        }
    
    def evaluate_trajectories(self, 
                            trajectories: List[TrajectoryPlan], 
                            hazards: List[HazardEvent],
                            constraints: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Evaluate trajectory options and generate human-readable decision logs
        
        Returns:
        - Selected trajectory with detailed justification
        - Trade-off analysis in human-readable format
        - Risk assessment summary
        """
        
        if not trajectories:
            raise ValueError("No trajectories provided for evaluation")
        
        constraints = constraints or self.mission_constraints
        evaluations = []
        
        # Evaluate each trajectory option
        for trajectory in trajectories:
            score_breakdown = self._calculate_trajectory_score(trajectory, hazards, constraints)
            risk_assessment = self._assess_trajectory_risks(trajectory, hazards)
            
            evaluations.append({
                "trajectory": trajectory,
                "score_breakdown": score_breakdown,
                "risk_assessment": risk_assessment,
                "total_score": score_breakdown["total_score"],
                "feasibility": self._check_trajectory_feasibility(trajectory, constraints)
            })
        
        # Filter only feasible trajectories
        feasible_trajectories = [eval for eval in evaluations if eval["feasibility"]["is_feasible"]]
        
        if not feasible_trajectories:
            # Emergency case - select least bad option
            best_evaluation = min(evaluations, key=lambda x: x["total_score"])
            emergency_decision = self._generate_emergency_decision_log(best_evaluation, hazards)
            return emergency_decision
        
        # Sort by total score (lower is better for our scoring system)
        feasible_trajectories.sort(key=lambda x: x["total_score"])
        best_trajectory_eval = feasible_trajectories[0]
        
        # Generate comprehensive decision log
        decision_result = self._generate_decision_log(
            best_trajectory_eval, 
            feasible_trajectories,
            hazards,
            constraints
        )
        
        # Store decision in history
        self._record_decision(decision_result)
        
        return decision_result
    
    def _calculate_trajectory_score(self, trajectory: TrajectoryPlan, hazards: List[HazardEvent], constraints: Dict) -> Dict[str, float]:
        """Calculate weighted score for trajectory with detailed breakdown"""
        
        # Extract trajectory parameters
        delta_v = getattr(trajectory, 'total_delta_v', 12000)
        duration = getattr(trajectory, 'duration', 72.0)
        radiation_exposure = getattr(trajectory, 'radiation_exposure', 20.0)
        
        # Calculate normalized scores (0-1 scale, lower is better)
        fuel_score = min(delta_v / constraints.get("max_delta_v", 15000), 1.0)
        time_score = min(duration / constraints.get("max_duration", 120), 1.0)
        radiation_score = min(radiation_exposure / constraints.get("max_radiation", 50), 1.0)
        
        # Calculate collision risk based on hazards
        collision_score = self._calculate_collision_risk_score(trajectory, hazards)
        
        # Apply weights
        weighted_scores = {
            "fuel_cost": fuel_score * self.evaluation_weights["fuel_cost"],
            "travel_time": time_score * self.evaluation_weights["travel_time"],
            "radiation_risk": radiation_score * self.evaluation_weights["radiation_risk"],
            "collision_risk": collision_score * self.evaluation_weights["collision_risk"]
        }
        
        total_score = sum(weighted_scores.values())
        
        return {
            "fuel_cost": fuel_score,
            "travel_time": time_score,
            "radiation_risk": radiation_score,
            "collision_risk": collision_score,
            "weighted_scores": weighted_scores,
            "total_score": total_score,
            "raw_values": {
                "delta_v": delta_v,
                "duration": duration,
                "radiation_exposure": radiation_exposure
            }
        }
    
    def _calculate_collision_risk_score(self, trajectory: TrajectoryPlan, hazards: List[HazardEvent]) -> float:
        """Calculate collision risk score based on hazards along trajectory"""
        if not hazards:
            return 0.1  # Baseline risk
        
        collision_risk = 0.1
        
        for hazard in hazards:
            hazard_type = getattr(hazard, 'type', 'unknown')
            severity = getattr(hazard, 'severity', 'low')
            
            # Risk multipliers based on hazard type and severity
            risk_multipliers = {
                'debris': {'low': 0.1, 'medium': 0.3, 'high': 0.7, 'critical': 1.0},
                'solar_flare': {'low': 0.05, 'medium': 0.15, 'high': 0.4, 'critical': 0.8},
                'cme': {'low': 0.1, 'medium': 0.25, 'high': 0.6, 'critical': 0.9},
                'radiation': {'low': 0.05, 'medium': 0.2, 'high': 0.5, 'critical': 0.8}
            }
            
            if hazard_type in risk_multipliers:
                collision_risk += risk_multipliers[hazard_type].get(severity, 0.2)
        
        return min(collision_risk, 1.0)
    
    def _assess_trajectory_risks(self, trajectory: TrajectoryPlan, hazards: List[HazardEvent]) -> Dict[str, Any]:
        """Comprehensive risk assessment for trajectory"""
        
        risks = {
            "overall_risk_level": "low",
            "specific_risks": [],
            "mitigation_strategies": [],
            "confidence_level": 0.8
        }
        
        # Assess fuel risk
        delta_v = getattr(trajectory, 'total_delta_v', 12000)
        if delta_v > 13500:
            risks["specific_risks"].append("High fuel consumption - limited margin for contingencies")
            risks["mitigation_strategies"].append("Monitor fuel usage closely, prepare backup maneuvers")
        
        # Assess time risk
        duration = getattr(trajectory, 'duration', 72.0)
        if duration > 96:
            risks["specific_risks"].append("Extended mission duration increases exposure window")
            risks["mitigation_strategies"].append("Enhanced monitoring protocols for extended duration")
        
        # Assess radiation risk
        radiation = getattr(trajectory, 'radiation_exposure', 20.0)
        if radiation > 35:
            risks["specific_risks"].append("Elevated radiation exposure risk to crew")
            risks["mitigation_strategies"].append("Implement radiation shielding protocols")
        
        # Assess hazard-specific risks
        for hazard in hazards:
            hazard_type = getattr(hazard, 'type', 'unknown')
            severity = getattr(hazard, 'severity', 'low')
            
            if severity in ['high', 'critical']:
                risks["specific_risks"].append(f"Critical {hazard_type} hazard detected")
                risks["mitigation_strategies"].append(f"Continuous monitoring of {hazard_type} conditions")
        
        # Determine overall risk level
        if len([r for r in risks["specific_risks"] if "Critical" in r]) > 0:
            risks["overall_risk_level"] = "critical"
        elif len(risks["specific_risks"]) > 2:
            risks["overall_risk_level"] = "high"
        elif len(risks["specific_risks"]) > 0:
            risks["overall_risk_level"] = "moderate"
        
        return risks
    
    def _check_trajectory_feasibility(self, trajectory: TrajectoryPlan, constraints: Dict) -> Dict[str, Any]:
        """Check if trajectory meets mission constraints"""
        
        delta_v = getattr(trajectory, 'total_delta_v', 12000)
        duration = getattr(trajectory, 'duration', 72.0)
        radiation = getattr(trajectory, 'radiation_exposure', 20.0)
        
        feasibility = {
            "is_feasible": True,
            "constraint_violations": [],
            "margin_analysis": {}
        }
        
        # Check ΔV constraint
        max_delta_v = constraints.get("max_delta_v", 15000)
        if delta_v > max_delta_v:
            feasibility["is_feasible"] = False
            feasibility["constraint_violations"].append(f"ΔV exceeds limit: {delta_v:.0f} > {max_delta_v} m/s")
        else:
            feasibility["margin_analysis"]["delta_v_margin"] = f"{((max_delta_v - delta_v) / max_delta_v * 100):.1f}%"
        
        # Check duration constraint
        max_duration = constraints.get("max_duration", 120)
        if duration > max_duration:
            feasibility["is_feasible"] = False
            feasibility["constraint_violations"].append(f"Duration exceeds limit: {duration:.1f} > {max_duration} hours")
        else:
            feasibility["margin_analysis"]["time_margin"] = f"{((max_duration - duration) / max_duration * 100):.1f}%"
        
        # Check radiation constraint
        max_radiation = constraints.get("max_radiation", 50)
        if radiation > max_radiation:
            feasibility["is_feasible"] = False
            feasibility["constraint_violations"].append(f"Radiation exceeds limit: {radiation:.1f}% > {max_radiation}%")
        else:
            feasibility["margin_analysis"]["radiation_margin"] = f"{((max_radiation - radiation) / max_radiation * 100):.1f}%"
        
        return feasibility
    
    def _generate_decision_log(self, best_evaluation: Dict, all_evaluations: List[Dict], 
                              hazards: List[HazardEvent], constraints: Dict) -> Dict[str, Any]:
        """Generate human-readable decision log as specified in problem statement"""
        
        selected = best_evaluation["trajectory"]
        scores = best_evaluation["score_breakdown"]
        risks = best_evaluation["risk_assessment"]
        
        # Generate the human-readable log in the format: 
        # "THREAT DETECTED. REROUTING VIA ALTERNATE TRAJECTORY. RESULT: +X hours travel time, -Y% radiation exposure"
        
        decision_log = {
            "decision_timestamp": datetime.utcnow().isoformat(),
            "selected_trajectory": {
                "name": getattr(selected, 'name', 'Optimal Route'),
                "type": getattr(selected, 'trajectory_type', 'direct_transfer'),
                "delta_v": scores["raw_values"]["delta_v"],
                "duration": scores["raw_values"]["duration"],
                "radiation_exposure": scores["raw_values"]["radiation_exposure"]
            },
            "human_readable_log": self._format_human_readable_decision(selected, scores, hazards, all_evaluations),
            "technical_analysis": {
                "score_breakdown": scores,
                "risk_assessment": risks,
                "feasibility": best_evaluation["feasibility"]
            },
            "alternatives_considered": len(all_evaluations),
            "decision_rationale": self._generate_decision_rationale(best_evaluation, all_evaluations),
            "trade_off_analysis": self._generate_trade_off_analysis(best_evaluation, all_evaluations),
            "monitoring_recommendations": self._generate_monitoring_recommendations(risks, hazards)
        }
        
        return decision_log
    
    def _format_human_readable_decision(self, selected_trajectory: TrajectoryPlan, scores: Dict, 
                                       hazards: List[HazardEvent], all_evaluations: List[Dict]) -> str:
        """Format decision in human-readable format as specified in problem statement"""
        
        # Find baseline/current trajectory for comparison
        baseline = None
        if len(all_evaluations) > 1:
            baseline = all_evaluations[1]  # Assume second option is baseline
        
        threat_description = ""
        if hazards:
            critical_hazards = [h for h in hazards if getattr(h, 'severity', 'low') in ['high', 'critical']]
            if critical_hazards:
                hazard_types = [getattr(h, 'type', 'unknown').upper() for h in critical_hazards]
                threat_description = f"{', '.join(hazard_types)} DETECTED. "
            else:
                threat_description = "SPACE WEATHER HAZARD DETECTED. "
        
        action_description = f"REROUTING VIA {getattr(selected_trajectory, 'name', 'ALTERNATE TRAJECTORY').upper()}. "
        
        # Calculate changes from baseline
        result_parts = []
        if baseline:
            baseline_scores = baseline["score_breakdown"]["raw_values"]
            selected_scores = scores["raw_values"]
            
            # Time change
            time_change = selected_scores["duration"] - baseline_scores["duration"]
            if abs(time_change) > 0.5:
                sign = "+" if time_change > 0 else ""
                result_parts.append(f"{sign}{time_change:.1f} HOURS TRAVEL TIME")
            
            # Radiation change
            radiation_change = selected_scores["radiation_exposure"] - baseline_scores["radiation_exposure"]
            if abs(radiation_change) > 2:
                sign = "+" if radiation_change > 0 else ""
                result_parts.append(f"{sign}{radiation_change:.0f}% RADIATION EXPOSURE")
            
            # Fuel change
            fuel_change = selected_scores["delta_v"] - baseline_scores["delta_v"]
            if abs(fuel_change) > 100:
                sign = "+" if fuel_change > 0 else ""
                result_parts.append(f"{sign}{fuel_change:.0f} M/S DELTA-V")
        else:
            # No baseline comparison
            result_parts = [
                f"{scores['raw_values']['duration']:.1f} HOURS TOTAL TRAVEL TIME",
                f"{scores['raw_values']['radiation_exposure']:.0f}% RADIATION EXPOSURE",
                f"{scores['raw_values']['delta_v']:.0f} M/S TOTAL DELTA-V"
            ]
        
        result_description = "RESULT: " + ", ".join(result_parts)
        
        return threat_description + action_description + result_description
    
    def _generate_decision_rationale(self, best_evaluation: Dict, all_evaluations: List[Dict]) -> str:
        """Generate detailed rationale for trajectory selection"""
        
        selected_scores = best_evaluation["score_breakdown"]
        
        rationale_parts = []
        
        # Identify strongest factors
        weighted_scores = selected_scores["weighted_scores"]
        dominant_factor = min(weighted_scores.keys(), key=lambda k: weighted_scores[k])
        
        factor_descriptions = {
            "fuel_cost": "optimal fuel efficiency",
            "travel_time": "minimal travel time",
            "radiation_risk": "crew safety from radiation",
            "collision_risk": "hazard avoidance"
        }
        
        rationale_parts.append(f"Selected for {factor_descriptions.get(dominant_factor, 'optimal performance')}")
        
        # Risk considerations
        risk_level = best_evaluation["risk_assessment"]["overall_risk_level"]
        if risk_level == "low":
            rationale_parts.append("with acceptable risk profile")
        elif risk_level == "moderate":
            rationale_parts.append("balancing performance with manageable risks")
        else:
            rationale_parts.append("prioritizing mission success despite elevated risks")
        
        return ". ".join(rationale_parts) + "."
    
    def _generate_trade_off_analysis(self, best_evaluation: Dict, all_evaluations: List[Dict]) -> List[str]:
        """Generate trade-off analysis comparing alternatives"""
        
        trade_offs = []
        
        if len(all_evaluations) < 2:
            return ["No alternative trajectories available for comparison"]
        
        best_scores = best_evaluation["score_breakdown"]["raw_values"]
        
        for i, evaluation in enumerate(all_evaluations[1:], 1):
            alt_scores = evaluation["score_breakdown"]["raw_values"]
            alt_name = getattr(evaluation["trajectory"], 'name', f'Alternative {i}')
            
            trade_off = f"{alt_name}: "
            differences = []
            
            if abs(alt_scores["duration"] - best_scores["duration"]) > 1:
                diff = alt_scores["duration"] - best_scores["duration"]
                sign = "+" if diff > 0 else ""
                differences.append(f"{sign}{diff:.1f}h time")
            
            if abs(alt_scores["delta_v"] - best_scores["delta_v"]) > 200:
                diff = alt_scores["delta_v"] - best_scores["delta_v"]
                sign = "+" if diff > 0 else ""
                differences.append(f"{sign}{diff:.0f} m/s ΔV")
            
            if abs(alt_scores["radiation_exposure"] - best_scores["radiation_exposure"]) > 3:
                diff = alt_scores["radiation_exposure"] - best_scores["radiation_exposure"]
                sign = "+" if diff > 0 else ""
                differences.append(f"{sign}{diff:.0f}% radiation")
            
            if differences:
                trade_off += ", ".join(differences)
                trade_offs.append(trade_off)
        
        return trade_offs if trade_offs else ["All alternatives have similar performance characteristics"]
    
    def _generate_monitoring_recommendations(self, risks: Dict, hazards: List[HazardEvent]) -> List[str]:
        """Generate monitoring recommendations based on risks and hazards"""
        
        recommendations = []
        
        # Risk-based recommendations
        for risk in risks["specific_risks"]:
            if "fuel" in risk.lower():
                recommendations.append("Monitor fuel consumption every 2 hours")
            elif "radiation" in risk.lower():
                recommendations.append("Continuous radiation level monitoring with hourly reports")
            elif "duration" in risk.lower():
                recommendations.append("Enhanced system health monitoring for extended mission")
        
        # Hazard-based recommendations
        for hazard in hazards:
            hazard_type = getattr(hazard, 'type', 'unknown')
            if hazard_type == 'solar_flare':
                recommendations.append("Monitor X-ray flux and radio blackout conditions")
            elif hazard_type == 'cme':
                recommendations.append("Track coronal mass ejection arrival time and magnetic field strength")
            elif hazard_type == 'debris':
                recommendations.append("Continuous orbital debris tracking and collision avoidance")
            elif hazard_type == 'radiation':
                recommendations.append("Real-time dosimetry monitoring and crew shelter protocols")
        
        # Default recommendations
        if not recommendations:
            recommendations = [
                "Standard telemetry monitoring every 30 minutes",
                "Space weather condition updates every 6 hours"
            ]
        
        return recommendations
    
    def _generate_emergency_decision_log(self, best_evaluation: Dict, hazards: List[HazardEvent]) -> Dict[str, Any]:
        """Generate decision log for emergency situations where no trajectory meets all constraints"""
        
        selected = best_evaluation["trajectory"]
        scores = best_evaluation["score_breakdown"]
        feasibility = best_evaluation["feasibility"]
        
        emergency_log = {
            "decision_timestamp": datetime.utcnow().isoformat(),
            "emergency_decision": True,
            "selected_trajectory": {
                "name": getattr(selected, 'name', 'Emergency Route'),
                "type": getattr(selected, 'trajectory_type', 'emergency_transfer'),
                "delta_v": scores["raw_values"]["delta_v"],
                "duration": scores["raw_values"]["duration"],
                "radiation_exposure": scores["raw_values"]["radiation_exposure"]
            },
            "human_readable_log": f"CRITICAL SITUATION. ALL TRAJECTORIES EXCEED CONSTRAINTS. SELECTING LEAST-RISK OPTION: {getattr(selected, 'name', 'Emergency Route').upper()}. CONSTRAINT VIOLATIONS: {', '.join(feasibility['constraint_violations'])}",
            "constraint_violations": feasibility["constraint_violations"],
            "emergency_rationale": "Mission continuation prioritized over constraint compliance due to critical situation",
            "emergency_protocols": [
                "Immediate mission control notification required",
                "Enhanced monitoring and abort procedures on standby",
                "Crew briefing on emergency trajectory parameters"
            ],
            "risk_acceptance": "High risk accepted due to absence of compliant alternatives"
        }
        
        return emergency_log
    
    def _record_decision(self, decision_result: Dict[str, Any]):
        """Record decision in history for analysis and learning"""
        
        decision_record = {
            "timestamp": decision_result["decision_timestamp"],
            "selected_trajectory": decision_result["selected_trajectory"]["name"],
            "human_log": decision_result["human_readable_log"],
            "risk_level": decision_result["technical_analysis"]["risk_assessment"]["overall_risk_level"],
            "emergency_decision": decision_result.get("emergency_decision", False)
        }
        
        self.decision_history.append(decision_record)
        
        # Keep only last 50 decisions
        if len(self.decision_history) > 50:
            self.decision_history = self.decision_history[-50:]
        
        logger.info(f"Decision recorded: {decision_record['human_log']}")
    
    def get_decision_history(self) -> List[Dict[str, Any]]:
        """Get history of recent decisions"""
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
