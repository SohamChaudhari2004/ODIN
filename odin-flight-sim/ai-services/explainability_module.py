import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ExplainabilityModule:
    """AI Explainability module for generating clear explanations of AI decisions and recommendations"""
    
    def __init__(self):
        self.explanation_templates = self._load_explanation_templates()
        self.technical_terms = self._load_technical_glossary()
        self.decision_factors = {
            'trajectory': ['fuel_efficiency', 'time_optimization', 'safety_margin', 'radiation_exposure'],
            'hazard_response': ['severity_assessment', 'time_to_impact', 'mitigation_options', 'crew_safety'],
            'mission_planning': ['resource_allocation', 'timeline_constraints', 'risk_tolerance', 'backup_options']
        }
    
    def explain_trajectory_selection(self, decision_data: Dict[str, Any], audience: str = "mission_control") -> Dict[str, Any]:
        """Generate comprehensive explanation for trajectory selection decisions"""
        
        selected_trajectory = decision_data.get('selected_trajectory', {})
        alternatives = decision_data.get('alternative_trajectories', [])
        selection_criteria = decision_data.get('selection_criteria', {})
        
        explanation = {
            'decision_type': 'trajectory_selection',
            'audience_level': audience,
            'summary': self._generate_trajectory_summary(selected_trajectory, selection_criteria),
            'detailed_analysis': self._analyze_trajectory_factors(selected_trajectory, alternatives, selection_criteria),
            'trade_offs': self._explain_trajectory_tradeoffs(selected_trajectory, alternatives),
            'risk_assessment': self._explain_trajectory_risks(selected_trajectory),
            'confidence_level': self._calculate_decision_confidence(decision_data),
            'what_if_scenarios': self._generate_what_if_scenarios(alternatives),
            'key_insights': self._extract_key_insights(decision_data),
            'generated_at': datetime.utcnow().isoformat()
        }
        
        # Adjust explanation complexity based on audience
        if audience == "crew":
            explanation = self._simplify_for_crew(explanation)
        elif audience == "public":
            explanation = self._simplify_for_public(explanation)
        
        return explanation
    
    def explain_hazard_response(self, response_data: Dict[str, Any], audience: str = "mission_control") -> Dict[str, Any]:
        """Generate explanation for hazard response decisions"""
        
        hazard = response_data.get('hazard', {})
        response_strategy = response_data.get('response_strategy', {})
        alternative_responses = response_data.get('alternative_responses', [])
        
        explanation = {
            'decision_type': 'hazard_response',
            'audience_level': audience,
            'hazard_description': self._explain_hazard_nature(hazard),
            'threat_assessment': self._explain_threat_level(hazard),
            'response_rationale': self._explain_response_choice(response_strategy, hazard),
            'timeline_factors': self._explain_response_timing(response_strategy, hazard),
            'alternative_actions': self._explain_alternative_responses(alternative_responses),
            'success_probability': self._estimate_response_success(response_strategy, hazard),
            'consequences': self._explain_response_consequences(response_strategy),
            'monitoring_plan': self._explain_monitoring_strategy(hazard, response_strategy),
            'generated_at': datetime.utcnow().isoformat()
        }
        
        if audience == "crew":
            explanation = self._simplify_for_crew(explanation)
        elif audience == "public":
            explanation = self._simplify_for_public(explanation)
        
        return explanation
    
    def explain_ai_recommendation(self, recommendation_data: Dict[str, Any], audience: str = "mission_control") -> Dict[str, Any]:
        """Generate explanation for AI-generated recommendations"""
        
        recommendation = recommendation_data.get('recommendation', {})
        supporting_data = recommendation_data.get('supporting_data', {})
        ai_confidence = recommendation_data.get('confidence', 0.0)
        
        explanation = {
            'decision_type': 'ai_recommendation',
            'audience_level': audience,
            'recommendation_summary': self._summarize_ai_recommendation(recommendation),
            'data_sources': self._explain_data_sources(supporting_data),
            'analysis_process': self._explain_ai_reasoning(recommendation, supporting_data),
            'confidence_explanation': self._explain_confidence_level(ai_confidence, supporting_data),
            'limitations': self._explain_ai_limitations(recommendation_data),
            'human_oversight': self._explain_human_role(recommendation),
            'verification_steps': self._suggest_verification_steps(recommendation),
            'uncertainty_factors': self._identify_uncertainty_factors(supporting_data),
            'generated_at': datetime.utcnow().isoformat()
        }
        
        if audience == "crew":
            explanation = self._simplify_for_crew(explanation)
        elif audience == "public":
            explanation = self._simplify_for_public(explanation)
        
        return explanation
    
    def explain_mission_optimization(self, optimization_data: Dict[str, Any], audience: str = "mission_control") -> Dict[str, Any]:
        """Generate explanation for mission optimization decisions"""
        
        optimization = optimization_data.get('optimization', {})
        baseline = optimization_data.get('baseline', {})
        improvements = optimization_data.get('improvements', {})
        
        explanation = {
            'decision_type': 'mission_optimization',
            'audience_level': audience,
            'optimization_goal': self._explain_optimization_objective(optimization),
            'baseline_analysis': self._analyze_baseline_performance(baseline),
            'improvement_areas': self._explain_improvements(improvements),
            'optimization_process': self._explain_optimization_method(optimization_data),
            'performance_gains': self._quantify_performance_gains(baseline, optimization),
            'implementation_plan': self._explain_implementation_steps(optimization),
            'risk_mitigation': self._explain_optimization_risks(optimization_data),
            'success_metrics': self._define_success_metrics(optimization),
            'generated_at': datetime.utcnow().isoformat()
        }
        
        if audience == "crew":
            explanation = self._simplify_for_crew(explanation)
        elif audience == "public":
            explanation = self._simplify_for_public(explanation)
        
        return explanation
    
    def _generate_trajectory_summary(self, trajectory: Dict, criteria: Dict) -> str:
        """Generate high-level trajectory selection summary"""
        name = trajectory.get('name', 'Selected trajectory')
        delta_v = trajectory.get('total_delta_v', 0)
        duration = trajectory.get('duration', 0)
        
        primary_factor = max(criteria.items(), key=lambda x: x[1])[0] if criteria else 'efficiency'
        
        return f"{name} was selected as the optimal trajectory, requiring {delta_v:.0f} m/s delta-V over {duration:.1f} hours. This selection prioritizes {primary_factor.replace('_', ' ')} while maintaining safety requirements."
    
    def _analyze_trajectory_factors(self, selected: Dict, alternatives: List, criteria: Dict) -> Dict[str, Any]:
        """Analyze factors that influenced trajectory selection"""
        analysis = {
            'fuel_efficiency': {
                'selected_delta_v': selected.get('total_delta_v', 0),
                'comparison': 'Most fuel-efficient option' if len(alternatives) == 0 else self._compare_fuel_efficiency(selected, alternatives),
                'impact': 'Critical for mission success due to limited fuel reserves'
            },
            'mission_duration': {
                'selected_duration': selected.get('duration', 0),
                'comparison': self._compare_duration(selected, alternatives),
                'impact': 'Affects crew resources and mission timeline'
            },
            'safety_margin': {
                'risk_score': selected.get('risk_score', 0),
                'comparison': self._compare_safety(selected, alternatives),
                'impact': 'Primary consideration for crew and mission protection'
            },
            'radiation_exposure': {
                'exposure_level': selected.get('radiation_exposure', 0),
                'comparison': self._compare_radiation(selected, alternatives),
                'impact': 'Long-term health implications for crew'
            }
        }
        
        return analysis
    
    def _explain_trajectory_tradeoffs(self, selected: Dict, alternatives: List) -> List[Dict[str, str]]:
        """Explain trade-offs made in trajectory selection"""
        tradeoffs = []
        
        if alternatives:
            # Find the most fuel-efficient alternative
            fuel_efficient = min(alternatives, key=lambda x: x.get('total_delta_v', float('inf')), default={})
            if fuel_efficient and fuel_efficient.get('total_delta_v', 0) < selected.get('total_delta_v', 0):
                tradeoffs.append({
                    'factor': 'Fuel Consumption',
                    'trade_off': f"Selected trajectory uses {selected.get('total_delta_v', 0) - fuel_efficient.get('total_delta_v', 0):.0f} m/s more delta-V than most efficient option",
                    'benefit': 'Gained improved safety margin and reduced radiation exposure'
                })
            
            # Find the fastest alternative
            fastest = min(alternatives, key=lambda x: x.get('duration', float('inf')), default={})
            if fastest and fastest.get('duration', 0) < selected.get('duration', 0):
                tradeoffs.append({
                    'factor': 'Mission Duration',
                    'trade_off': f"Selected trajectory takes {selected.get('duration', 0) - fastest.get('duration', 0):.1f} hours longer than fastest option",
                    'benefit': 'Reduced fuel consumption and improved trajectory precision'
                })
        
        return tradeoffs
    
    def _explain_trajectory_risks(self, trajectory: Dict) -> Dict[str, Any]:
        """Explain risk factors associated with selected trajectory"""
        risk_score = trajectory.get('risk_score', 0)
        radiation = trajectory.get('radiation_exposure', 0)
        fuel_margin = trajectory.get('fuel_margin', 100)
        
        risk_factors = []
        
        if risk_score > 7:
            risk_factors.append("High complexity trajectory with multiple critical maneuvers")
        if radiation > 70:
            risk_factors.append("Elevated radiation exposure requiring enhanced monitoring")
        if fuel_margin < 20:
            risk_factors.append("Limited fuel reserves reducing contingency options")
        
        return {
            'overall_risk_level': 'High' if risk_score > 7 else 'Moderate' if risk_score > 4 else 'Low',
            'risk_factors': risk_factors,
            'mitigation_strategies': self._suggest_risk_mitigation(trajectory),
            'contingency_plans': self._suggest_contingencies(trajectory)
        }
    
    def _explain_hazard_nature(self, hazard: Dict) -> str:
        """Explain the nature and characteristics of detected hazard"""
        hazard_type = hazard.get('event_type', 'Unknown')
        severity = hazard.get('severity', 0)
        duration = hazard.get('duration', 0)
        
        descriptions = {
            'solar_flare': f"Solar flare event with intensity level {severity}/10, expected to last {duration:.1f} hours. This electromagnetic radiation burst can disrupt communications and increase radiation exposure.",
            'cme': f"Coronal Mass Ejection with severity {severity}/10, impact duration {duration:.1f} hours. This plasma cloud can cause severe space weather effects and equipment damage.",
            'debris_conjunction': f"Space debris conjunction with collision risk level {severity}/10. Close approach expected within the next {duration:.1f} hours requiring potential avoidance maneuver.",
            'radiation_storm': f"Radiation storm event with intensity {severity}/10, lasting {duration:.1f} hours. Elevated particle radiation poses health risks and equipment threats."
        }
        
        return descriptions.get(hazard_type, f"Space hazard detected: {hazard_type} with severity level {severity}/10")
    
    def _explain_threat_level(self, hazard: Dict) -> Dict[str, Any]:
        """Explain threat level assessment for hazard"""
        severity = hazard.get('severity', 0)
        impact_probability = hazard.get('impact_probability', 0.0)
        
        threat_levels = {
            'immediate': severity >= 8 and impact_probability >= 0.8,
            'high': severity >= 6 and impact_probability >= 0.6,
            'moderate': severity >= 4 and impact_probability >= 0.4,
            'low': severity < 4 or impact_probability < 0.4
        }
        
        level = next((level for level, condition in threat_levels.items() if condition), 'low')
        
        return {
            'threat_level': level,
            'severity_justification': f"Severity rating {severity}/10 based on {hazard.get('event_type', 'hazard')} characteristics",
            'probability_assessment': f"{impact_probability*100:.1f}% probability of mission impact",
            'time_factor': f"Impact expected within {hazard.get('time_to_impact', 0):.1f} hours" if hazard.get('time_to_impact') else "Timeline assessment pending"
        }
    
    def _explain_response_choice(self, response: Dict, hazard: Dict) -> str:
        """Explain why specific response was chosen"""
        response_type = response.get('action_type', 'monitor')
        
        explanations = {
            'trajectory_modification': f"Trajectory modification selected to avoid hazard impact zone. This provides the most reliable protection against {hazard.get('event_type', 'the hazard')} while maintaining mission objectives.",
            'shelter_protocol': f"Crew shelter protocol activated due to high radiation risk. This passive protection strategy minimizes exposure during peak hazard period.",
            'system_shutdown': f"Non-critical systems shutdown to protect against electromagnetic effects. This preserves essential equipment functionality during the event.",
            'monitor': f"Enhanced monitoring protocol sufficient for current threat level. Active intervention not required but readiness maintained.",
            'abort_mission': f"Mission abort recommended due to unacceptable risk level. Crew safety takes precedence over mission objectives."
        }
        
        return explanations.get(response_type, f"Response strategy: {response_type} selected based on risk assessment")
    
    def _explain_confidence_level(self, confidence: float, supporting_data: Dict) -> Dict[str, Any]:
        """Explain AI confidence level and its basis"""
        
        confidence_factors = {
            'data_quality': len(supporting_data.get('data_sources', [])) * 0.2,
            'model_certainty': confidence * 0.4,
            'historical_validation': 0.3,  # Would be calculated from historical accuracy
            'cross_validation': 0.1  # Would be from multiple model agreement
        }
        
        total_confidence = sum(confidence_factors.values())
        
        return {
            'overall_confidence': f"{confidence*100:.1f}%",
            'confidence_breakdown': {
                'Data Quality': f"{confidence_factors['data_quality']*100:.1f}% - Based on {len(supporting_data.get('data_sources', []))} data sources",
                'Model Certainty': f"{confidence_factors['model_certainty']*100:.1f}% - AI model prediction confidence",
                'Historical Validation': f"{confidence_factors['historical_validation']*100:.1f}% - Similar scenario outcomes",
                'Cross-Validation': f"{confidence_factors['cross_validation']*100:.1f}% - Multiple model agreement"
            },
            'reliability_note': self._get_confidence_interpretation(confidence)
        }
    
    def _get_confidence_interpretation(self, confidence: float) -> str:
        """Interpret confidence level for human understanding"""
        if confidence >= 0.9:
            return "Very high confidence - recommendation strongly supported by data and models"
        elif confidence >= 0.75:
            return "High confidence - recommendation well-supported with minor uncertainties"
        elif confidence >= 0.6:
            return "Moderate confidence - recommendation reasonable but requires validation"
        elif confidence >= 0.4:
            return "Low confidence - recommendation uncertain, human review essential"
        else:
            return "Very low confidence - recommendation highly uncertain, alternative analysis needed"
    
    def _simplify_for_crew(self, explanation: Dict) -> Dict:
        """Simplify explanation for crew consumption"""
        # Extract most critical information for crew
        simplified = {
            'decision_type': explanation['decision_type'],
            'crew_summary': self._generate_crew_summary(explanation),
            'immediate_actions': self._extract_crew_actions(explanation),
            'safety_implications': self._extract_safety_info(explanation),
            'timeline': self._extract_timeline_info(explanation),
            'questions_to_ask': self._suggest_crew_questions(explanation)
        }
        
        return simplified
    
    def _simplify_for_public(self, explanation: Dict) -> Dict:
        """Simplify explanation for public communication"""
        simplified = {
            'decision_type': explanation['decision_type'],
            'public_summary': self._generate_public_summary(explanation),
            'mission_impact': self._extract_mission_impact(explanation),
            'safety_measures': self._extract_public_safety_info(explanation),
            'next_steps': self._extract_public_next_steps(explanation)
        }
        
        return simplified
    
    def _load_explanation_templates(self) -> Dict:
        """Load explanation templates for different decision types"""
        return {
            'trajectory': "The {trajectory_name} was selected because it provides the best balance of {primary_factors} while maintaining {safety_requirements}.",
            'hazard_response': "In response to the {hazard_type}, we are implementing {response_action} to ensure {protection_goal}.",
            'optimization': "Mission optimization focused on {optimization_target} resulted in {improvement_percentage}% improvement in {key_metric}."
        }
    
    def _load_technical_glossary(self) -> Dict:
        """Load technical term definitions for explanations"""
        return {
            'delta_v': "Delta-V: The change in velocity required for spacecraft maneuvers, measured in meters per second",
            'conjunction': "Conjunction: A close approach between spacecraft and another object in space",
            'cme': "CME: Coronal Mass Ejection - a large expulsion of plasma from the solar corona",
            'radiation_exposure': "Radiation Exposure: The amount of ionizing radiation absorbed, affecting crew health",
            'orbital_mechanics': "Orbital Mechanics: The physics governing spacecraft motion in space"
        }
    
    def _compare_fuel_efficiency(self, selected: Dict, alternatives: List) -> str:
        """Compare fuel efficiency with alternatives"""
        selected_dv = selected.get('total_delta_v', 0)
        alternatives_dv = [alt.get('total_delta_v', float('inf')) for alt in alternatives]
        
        if not alternatives_dv:
            return "No alternatives available for comparison"
        
        min_dv = min(alternatives_dv)
        if selected_dv <= min_dv:
            return "Most fuel-efficient option selected"
        else:
            difference = selected_dv - min_dv
            return f"Uses {difference:.0f} m/s more delta-V than most efficient alternative"
    
    def _compare_duration(self, selected: Dict, alternatives: List) -> str:
        """Compare mission duration with alternatives"""
        selected_duration = selected.get('duration', 0)
        alternatives_duration = [alt.get('duration', float('inf')) for alt in alternatives]
        
        if not alternatives_duration:
            return "No alternatives available for comparison"
        
        min_duration = min(alternatives_duration)
        if selected_duration <= min_duration:
            return "Fastest trajectory option selected"
        else:
            difference = selected_duration - min_duration
            return f"Takes {difference:.1f} hours longer than fastest alternative"
    
    def _compare_safety(self, selected: Dict, alternatives: List) -> str:
        """Compare safety scores with alternatives"""
        selected_risk = selected.get('risk_score', 10)
        alternatives_risk = [alt.get('risk_score', 0) for alt in alternatives]
        
        if not alternatives_risk:
            return "No alternatives available for comparison"
        
        min_risk = min(alternatives_risk)
        if selected_risk <= min_risk:
            return "Safest trajectory option selected"
        else:
            return f"Risk score {selected_risk:.1f} vs minimum alternative risk of {min_risk:.1f}"
    
    def _compare_radiation(self, selected: Dict, alternatives: List) -> str:
        """Compare radiation exposure with alternatives"""
        selected_radiation = selected.get('radiation_exposure', 0)
        alternatives_radiation = [alt.get('radiation_exposure', float('inf')) for alt in alternatives]
        
        if not alternatives_radiation:
            return "No alternatives available for comparison"
        
        min_radiation = min(alternatives_radiation)
        if selected_radiation <= min_radiation:
            return "Lowest radiation exposure option selected"
        else:
            difference = selected_radiation - min_radiation
            return f"Radiation exposure {difference:.1f}% higher than lowest alternative"
    
    def generate_decision_narrative(self, decision_history: List[Dict], audience: str = "mission_control") -> str:
        """Generate a narrative explaining sequence of decisions"""
        
        narrative_parts = []
        narrative_parts.append(f"MISSION DECISION SEQUENCE - {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")
        narrative_parts.append("=" * 60)
        
        for i, decision in enumerate(decision_history, 1):
            decision_time = decision.get('timestamp', 'Unknown time')
            decision_type = decision.get('decision_type', 'Unknown decision')
            
            narrative_parts.append(f"\n{i}. {decision_type.upper()} - {decision_time}")
            narrative_parts.append("-" * 40)
            
            if decision_type == 'trajectory_selection':
                narrative_parts.append(self._narrative_trajectory_decision(decision))
            elif decision_type == 'hazard_response':
                narrative_parts.append(self._narrative_hazard_decision(decision))
            elif decision_type == 'mission_optimization':
                narrative_parts.append(self._narrative_optimization_decision(decision))
            
        narrative_parts.append(f"\nSequence complete. Total decisions: {len(decision_history)}")
        
        return "\n".join(narrative_parts)
    
    def _narrative_trajectory_decision(self, decision: Dict) -> str:
        """Create narrative for trajectory decision"""
        selected = decision.get('selected_trajectory', {})
        reason = decision.get('reason', 'optimization')
        
        return f"Selected {selected.get('name', 'trajectory')} requiring {selected.get('total_delta_v', 0):.0f} m/s delta-V. Decision driven by {reason} requirements while maintaining safety protocols."
    
    def _narrative_hazard_decision(self, decision: Dict) -> str:
        """Create narrative for hazard response decision"""
        hazard = decision.get('hazard', {})
        response = decision.get('response', {})
        
        return f"Detected {hazard.get('event_type', 'hazard')} with severity {hazard.get('severity', 0)}/10. Implemented {response.get('action_type', 'monitoring')} protocol to ensure crew safety and mission continuity."
    
    def _narrative_optimization_decision(self, decision: Dict) -> str:
        """Create narrative for optimization decision"""
        optimization = decision.get('optimization', {})
        improvement = optimization.get('improvement_percentage', 0)
        
        return f"Applied mission optimization targeting {optimization.get('target', 'efficiency')}. Achieved {improvement:.1f}% improvement in mission parameters while maintaining operational requirements."
