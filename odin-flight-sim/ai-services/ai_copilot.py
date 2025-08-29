import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import asyncio

try:
    from langchain_mistralai import ChatMistralAI
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain Mistral AI not available")

logger = logging.getLogger(__name__)

class AICoPilot:
    """AI Co-pilot using LangChain with Mistral AI for trajectory planning and strategy generation"""
    
    def __init__(self):
        self.mistral_api_key = os.getenv("MISTRAL_API_KEY") or "vbj6ynFqsVLi2MrRu1dIOVBY2RvImn86"
        self.model_name = "mistral-large-latest"  # Use Mistral's most capable model
        self.context_history = []
        self.max_context_history = 10
        
        # Initialize LangChain with Mistral AI if available
        if LANGCHAIN_AVAILABLE and self.mistral_api_key:
            self.llm = ChatMistralAI(
                model=self.model_name,
                mistral_api_key=self.mistral_api_key,
                temperature=0.3,  # Lower temperature for more consistent planning
                max_tokens=1000
            )
            self.ai_available = True
            logger.info("AI Co-pilot initialized with Mistral AI via LangChain")
        else:
            self.llm = None
            self.ai_available = False
            logger.warning("AI Co-pilot running in simulation mode - no Mistral API key provided")
        
        # Initialize prompt templates
        self._setup_prompt_templates()
    
    def _setup_prompt_templates(self):
        """Setup prompt templates for different AI Co-pilot functions"""
        
        # Mission analysis prompt
        self.mission_analysis_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """You are ODIN AI Co-pilot, an expert autonomous spacecraft navigation system for Earth-to-Moon missions.
                You have access to real historical space weather data from 2012-2018 and must provide strategic analysis.
                
                Key responsibilities:
                - Analyze trajectory options based on ΔV cost, time, and safety
                - Assess space weather threats (solar flares, CMEs, radiation)
                - Recommend optimal paths considering fuel efficiency and crew safety
                - Generate clear, actionable decision logs
                
                Always provide responses in this format:
                ASSESSMENT: [Brief status summary]
                RISKS: [Key threats identified] 
                RECOMMENDATION: [Specific action to take]
                RATIONALE: [Why this choice is optimal]"""
            ),
            HumanMessagePromptTemplate.from_template(
                """MISSION STATUS:
                Mission Time: T+{mission_time:.1f} hours
                Current Position: {position}
                Velocity: {velocity:.2f} km/s
                Fuel Remaining: {fuel:.1f}%
                
                TRAJECTORY OPTIONS:
                {trajectory_options}
                
                ACTIVE HAZARDS:
                {hazards}
                
                SPACE WEATHER CONDITIONS:
                {space_weather}
                
                Provide strategic guidance for optimal trajectory selection."""
            )
        ])
        
        # Hazard response prompt
        self.hazard_response_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """You are ODIN's emergency response AI analyzing space hazards for lunar missions.
                Generate immediate response strategies for detected threats.
                
                Focus on:
                - Immediate threat assessment
                - Alternative trajectory options
                - Risk mitigation strategies
                - Clear decision trade-offs
                
                Use this decision log format:
                THREAT: [Hazard type and severity]
                ACTION: [Immediate response required]
                IMPACT: [Mission effects - time, fuel, risk changes]
                JUSTIFICATION: [Why this response is optimal]"""
            ),
            HumanMessagePromptTemplate.from_template(
                """HAZARD DETECTED:
                Type: {hazard_type}
                Severity: {severity}
                Time to Impact: {time_to_impact}
                Affected Trajectory Segments: {affected_segments}
                
                CURRENT MISSION STATE:
                Position: {position}
                Velocity: {velocity}
                Fuel: {fuel_remaining}%
                
                AVAILABLE ALTERNATIVES:
                {alternative_trajectories}
                
                Generate immediate response strategy."""
            )
        ])
    
    async def generate_mission_brief(self, mission_data: Dict[str, Any]) -> str:
        """Generate a comprehensive mission brief based on current state"""
        
        if self.ai_available:
            return await self._generate_ai_mission_brief(mission_data)
        else:
            return self._generate_simulated_mission_brief(mission_data)
    
    async def _generate_ai_mission_brief(self, mission_data: Dict[str, Any]) -> str:
        """Generate mission brief using LangChain with Mistral AI"""
        try:
            # Extract mission data
            trajectory = mission_data.get('trajectory', {})
            hazards = mission_data.get('hazards', [])
            telemetry = mission_data.get('telemetry', {})
            mission_time = mission_data.get('mission_time', 0)
            space_weather = mission_data.get('space_weather', {})
            
            # Format trajectory options
            trajectory_options = self._format_trajectory_options(
                trajectory, 
                mission_data.get('alternative_trajectories', [])
            )
            
            # Format hazards
            hazards_text = self._format_hazards_for_ai(hazards)
            
            # Format space weather
            weather_text = self._format_space_weather(space_weather)
            
            # Generate response using LangChain
            chain = self.mission_analysis_prompt | self.llm | StrOutputParser()
            
            response = await chain.ainvoke({
                "mission_time": mission_time,
                "position": str(telemetry.get('spacecraft_position', [0, 0, 0])),
                "velocity": telemetry.get('current_velocity', 0),
                "fuel": telemetry.get('fuel_remaining', 100),
                "trajectory_options": trajectory_options,
                "hazards": hazards_text,
                "space_weather": weather_text
            })
            
            # Store in context history
            self._add_to_context_history("mission_brief", response, mission_data)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating AI mission brief: {e}")
            return self._generate_simulated_mission_brief(mission_data)
    
    async def generate_hazard_response(self, hazard_data: Dict[str, Any], mission_state: Dict[str, Any]) -> str:
        """Generate immediate response strategy for detected hazards"""
        
        if self.ai_available:
            return await self._generate_ai_hazard_response(hazard_data, mission_state)
        else:
            return self._generate_simulated_hazard_response(hazard_data, mission_state)
    
    async def _generate_ai_hazard_response(self, hazard_data: Dict[str, Any], mission_state: Dict[str, Any]) -> str:
        """Generate hazard response using LangChain with Mistral AI"""
        try:
            hazard = hazard_data.get('hazard', {})
            alternatives = hazard_data.get('alternative_trajectories', [])
            
            # Format alternative trajectories
            alt_text = "\n".join([
                f"- {alt.get('name', 'Unknown')}: ΔV +{alt.get('delta_v_cost', 0):.0f} m/s, "
                f"Time +{alt.get('time_penalty', 0):.1f}h, "
                f"Radiation {alt.get('radiation_exposure', 0):.1f}%"
                for alt in alternatives
            ])
            
            # Generate response using LangChain
            chain = self.hazard_response_prompt | self.llm | StrOutputParser()
            
            response = await chain.ainvoke({
                "hazard_type": hazard.get('type', 'Unknown'),
                "severity": hazard.get('severity', 'Unknown'),
                "time_to_impact": f"{hazard.get('time_to_impact', 0):.1f} hours",
                "affected_segments": str(hazard.get('affected_trajectory_segments', [])),
                "position": str(mission_state.get('position', [0, 0, 0])),
                "velocity": mission_state.get('velocity', 0),
                "fuel_remaining": mission_state.get('fuel_remaining', 100),
                "alternative_trajectories": alt_text or "No alternatives available"
            })
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating AI hazard response: {e}")
            return self._generate_simulated_hazard_response(hazard_data, mission_state)
    
    async def generate_trajectory_alternatives(self, current_trajectory: Dict[str, Any], 
                                             hazards: List[Dict[str, Any]], 
                                             constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alternative trajectory options using AI strategy planning"""
        
        if self.ai_available:
            return await self._generate_ai_alternatives(current_trajectory, hazards, constraints)
        else:
            return self._generate_simulated_alternatives(current_trajectory, hazards, constraints)
    
    async def _generate_ai_alternatives(self, current_trajectory: Dict[str, Any], 
                                      hazards: List[Dict[str, Any]], 
                                      constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trajectory alternatives using Mistral AI strategic planning"""
        try:
            # Create trajectory planning prompt
            planning_prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    """You are ODIN's trajectory planning AI. Generate 3-5 alternative spacecraft paths from Earth to Moon.
                    Consider orbital mechanics, fuel efficiency, and hazard avoidance.
                    
                    Return alternatives in this JSON format:
                    {{
                        "alternatives": [
                            {{
                                "name": "Alternative trajectory name",
                                "description": "Brief description",
                                "delta_v_total": 0000,
                                "duration_hours": 00.0,
                                "radiation_exposure": 00.0,
                                "collision_risk": 0.00,
                                "trajectory_type": "direct/bi-elliptic/lunar_gravity_assist",
                                "key_maneuvers": ["maneuver1", "maneuver2"],
                                "trade_offs": "Explanation of pros/cons"
                            }}
                        ]
                    }}"""
                ),
                HumanMessagePromptTemplate.from_template(
                    """CURRENT TRAJECTORY:
                    {current_traj}
                    
                    HAZARDS TO AVOID:
                    {hazards_list}
                    
                    CONSTRAINTS:
                    Fuel Budget: {fuel_budget} m/s ΔV
                    Time Limit: {time_limit} hours
                    Max Radiation: {max_radiation}%
                    
                    Generate alternative trajectories that avoid these hazards while optimizing for mission success."""
                )
            ])
            
            # Format inputs
            hazards_text = "\n".join([
                f"- {h.get('type', 'Unknown')}: Severity {h.get('severity', 'Unknown')}, "
                f"Location {h.get('location', 'Unknown')}"
                for h in hazards
            ])
            
            chain = planning_prompt | self.llm | StrOutputParser()
            
            response = await chain.ainvoke({
                "current_traj": str(current_trajectory),
                "hazards_list": hazards_text,
                "fuel_budget": constraints.get('max_delta_v', 15000),
                "time_limit": constraints.get('max_duration', 120),
                "max_radiation": constraints.get('max_radiation', 50)
            })
            
            # Parse JSON response (with fallback)
            try:
                parsed = json.loads(response)
                return parsed.get('alternatives', [])
            except json.JSONDecodeError:
                logger.warning("Could not parse AI trajectory alternatives as JSON")
                return self._generate_simulated_alternatives(current_trajectory, hazards, constraints)
                
        except Exception as e:
            logger.error(f"Error generating AI trajectory alternatives: {e}")
            return self._generate_simulated_alternatives(current_trajectory, hazards, constraints)
    
    def _format_trajectory_options(self, current_traj: Dict[str, Any], alternatives: List[Dict[str, Any]]) -> str:
        """Format trajectory options for AI prompt"""
        options = [f"Current: {current_traj.get('name', 'Unknown')} - ΔV: {current_traj.get('total_delta_v', 0):.0f} m/s, Time: {current_traj.get('duration', 0):.1f}h"]
        
        for i, alt in enumerate(alternatives, 1):
            options.append(f"Option {i}: {alt.get('name', 'Unknown')} - ΔV: {alt.get('total_delta_v', 0):.0f} m/s, Time: {alt.get('duration', 0):.1f}h")
        
        return "\n".join(options)
    
    def _format_hazards_for_ai(self, hazards: List[Dict[str, Any]]) -> str:
        """Format hazards list for AI prompt"""
        if not hazards:
            return "No active hazards detected"
        
        hazard_lines = []
        for hazard in hazards:
            severity = hazard.get('severity', 'unknown')
            hazard_type = hazard.get('type', 'unknown')
            location = hazard.get('location', 'unknown')
            time_to_impact = hazard.get('time_to_impact', 0)
            
            hazard_lines.append(f"- {hazard_type.title()} (Severity: {severity}) at {location}, Impact in {time_to_impact:.1f}h")
        
        return "\n".join(hazard_lines)
    
    def _format_space_weather(self, space_weather: Dict[str, Any]) -> str:
        """Format space weather data for AI prompt"""
        if not space_weather:
            return "No space weather data available"
        
        weather_lines = []
        
        # Solar activity
        solar_flux = space_weather.get('solar_flux', 'Unknown')
        kp_index = space_weather.get('kp_index', 'Unknown')
        weather_lines.append(f"Solar Flux: {solar_flux}, Kp Index: {kp_index}")
        
        # Active events
        active_events = space_weather.get('active_events', [])
        if active_events:
            weather_lines.append("Active Events:")
            for event in active_events:
                weather_lines.append(f"  - {event.get('type', 'Unknown')}: {event.get('description', 'No details')}")
        
        return "\n".join(weather_lines)
    
    def _generate_simulated_mission_brief(self, mission_data: Dict[str, Any]) -> str:
        """Generate simulated mission brief when AI is not available"""
        trajectory = mission_data.get('trajectory', {})
        hazards = mission_data.get('hazards', [])
        telemetry = mission_data.get('telemetry', {})
        mission_time = mission_data.get('mission_time', 0)
        
        brief = f"""ODIN AI CO-PILOT MISSION BRIEF (T+{mission_time:.1f}h)

ASSESSMENT: Mission proceeding on {trajectory.get('name', 'unknown')} trajectory. 
Current velocity: {telemetry.get('current_velocity', 0):.2f} km/s, Fuel: {telemetry.get('fuel_remaining', 100):.1f}%

RISKS: {len(hazards)} active hazard(s) detected requiring monitoring.

RECOMMENDATION: {"Continue current trajectory" if len(hazards) == 0 else "Evaluate trajectory alternatives due to hazard threats"}

RATIONALE: {"Nominal mission parameters within acceptable ranges" if len(hazards) == 0 else "Space environment conditions require enhanced monitoring and potential course corrections"}"""
        
        return brief
    
    def _generate_simulated_hazard_response(self, hazard_data: Dict[str, Any], mission_state: Dict[str, Any]) -> str:
        """Generate simulated hazard response when AI is not available"""
        hazard = hazard_data.get('hazard', {})
        hazard_type = hazard.get('type', 'Unknown')
        severity = hazard.get('severity', 'Unknown')
        
        response = f"""THREAT: {hazard_type} detected with {severity} severity level
ACTION: Initiating hazard avoidance protocols and trajectory re-evaluation
IMPACT: Potential +2-6 hours travel time, +500-1500 m/s ΔV cost, -20-50% radiation exposure
JUSTIFICATION: Crew safety takes priority over mission efficiency - avoiding hazard reduces risk by 80-95%"""
        
        return response
    
    def _generate_simulated_alternatives(self, current_trajectory: Dict[str, Any], 
                                       hazards: List[Dict[str, Any]], 
                                       constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate simulated trajectory alternatives when AI is not available"""
        alternatives = [
            {
                "name": "Conservative Safety Route",
                "description": "Extended trajectory avoiding all hazard zones",
                "delta_v_total": current_trajectory.get('total_delta_v', 12000) + 1200,
                "duration_hours": current_trajectory.get('duration', 72) + 8,
                "radiation_exposure": max(0, current_trajectory.get('radiation_exposure', 20) - 15),
                "collision_risk": 0.01,
                "trajectory_type": "extended_transfer",
                "key_maneuvers": ["extended_departure_burn", "mid_course_correction", "lunar_capture"],
                "trade_offs": "Higher fuel cost but maximum safety margin"
            },
            {
                "name": "Efficient Alternative",
                "description": "Optimal route with minimal hazard exposure",
                "delta_v_total": current_trajectory.get('total_delta_v', 12000) + 600,
                "duration_hours": current_trajectory.get('duration', 72) + 3,
                "radiation_exposure": max(0, current_trajectory.get('radiation_exposure', 20) - 8),
                "collision_risk": 0.02,
                "trajectory_type": "bi_elliptic",
                "key_maneuvers": ["modified_departure", "hazard_avoidance_maneuver", "lunar_insertion"],
                "trade_offs": "Balanced approach optimizing time and safety"
            },
            {
                "name": "Rapid Transit Route",
                "description": "Faster trajectory with acceptable risk levels",
                "delta_v_total": current_trajectory.get('total_delta_v', 12000) + 300,
                "duration_hours": current_trajectory.get('duration', 72) + 1,
                "radiation_exposure": current_trajectory.get('radiation_exposure', 20) + 5,
                "collision_risk": 0.03,
                "trajectory_type": "direct_transfer",
                "key_maneuvers": ["high_energy_departure", "direct_injection", "precision_capture"],
                "trade_offs": "Minimal time penalty with managed risk increase"
            }
        ]
        
        return alternatives
    
    def _add_to_context_history(self, action_type: str, result: str, context_data: Dict[str, Any]):
        """Add action to context history for learning"""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action_type": action_type,
            "result": result,
            "context": context_data,
        }
        
        self.context_history.append(entry)
        
        # Keep only recent history
        if len(self.context_history) > self.max_context_history:
            self.context_history = self.context_history[-self.max_context_history:]
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of recent AI Co-pilot actions"""
        return {
            "total_actions": len(self.context_history),
            "recent_actions": [entry["action_type"] for entry in self.context_history[-5:]],
            "ai_available": self.ai_available,
            "model_info": {
                "provider": "Mistral AI" if self.ai_available else "Simulation",
                "model": self.model_name if self.ai_available else "N/A"
            }
        }
    
    def _generate_simulated_mission_brief(self, mission_data: Dict[str, Any]) -> str:
        """Generate simulated mission brief when AI is not available"""
        
        trajectory = mission_data.get('trajectory', {})
        hazards = mission_data.get('hazards', [])
        telemetry = mission_data.get('telemetry', {})
        mission_time = mission_data.get('mission_time', 0)
        
        # Analyze mission status
        fuel_status = "nominal" if telemetry.get('fuel_remaining', 100) > 50 else "concerning"
        hazard_status = "clear" if len(hazards) == 0 else f"{len(hazards)} active"
        
        brief = f"""ODIN AI MISSION BRIEF - T+{mission_time:.1f}h
        
Status: Mission proceeding on {trajectory.get('name', 'current trajectory')}. 
Fuel reserves {fuel_status} at {telemetry.get('fuel_remaining', 100):.1f}%.
Space environment: {hazard_status} hazards detected.

Strategic Assessment:
- Trajectory efficiency: {self._assess_trajectory_efficiency(trajectory)}
- Risk level: {self._assess_overall_risk(hazards, trajectory)}
- Systems status: All nominal

Recommendations:
{self._generate_recommendations(mission_data)}

Next milestone: Continue Trans-Lunar Injection phase. Monitor for trajectory optimization opportunities."""
        
        return brief
    
    async def explain_trajectory_decision(self, decision_data: Dict[str, Any]) -> str:
        """Generate explanation for trajectory decisions"""
        
        if self.ai_available:
            return await self._generate_ai_trajectory_explanation(decision_data)
        else:
            return self._generate_simulated_trajectory_explanation(decision_data)
    
    async def _generate_ai_trajectory_explanation(self, decision_data: Dict[str, Any]) -> str:
        """Generate AI explanation for trajectory decisions"""
        try:
            selected_traj = decision_data.get('selected_trajectory', {})
            rationale = decision_data.get('decision_rationale', '')
            
            prompt = f"""
            As ODIN AI Co-pilot, explain this trajectory decision in clear, actionable language:
            
            SELECTED TRAJECTORY: {selected_traj.get('name', 'Unknown')}
            DECISION RATIONALE: {rationale}
            
            Key Parameters:
            - Delta-V: {selected_traj.get('total_delta_v', 0):.0f} m/s
            - Duration: {selected_traj.get('duration', 0):.1f} hours  
            - Risk Score: {selected_traj.get('risk_score', 0):.1f}
            - Fuel Cost: {selected_traj.get('fuel_cost', 0):.0f} kg
            
            Provide a clear explanation that mission control can understand, focusing on:
            1. Why this trajectory was chosen
            2. Trade-offs made
            3. Mission impact
            
            Keep under 150 words.
            """
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are ODIN AI Co-pilot explaining orbital mechanics decisions to mission control."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.6
            )
            
            explanation = response.choices[0].message.content.strip()
            self._add_to_context_history("trajectory_explanation", explanation, decision_data)
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating AI trajectory explanation: {e}")
            return self._generate_simulated_trajectory_explanation(decision_data)
    
    def _generate_simulated_trajectory_explanation(self, decision_data: Dict[str, Any]) -> str:
        """Generate simulated trajectory explanation"""
        
        selected_traj = decision_data.get('selected_trajectory', {})
        rationale = decision_data.get('decision_rationale', 'Optimal trajectory selected')
        
        return f"""TRAJECTORY DECISION ANALYSIS:

Selected: {selected_traj.get('name', 'Unknown trajectory')}

{rationale}

This selection optimizes mission parameters while maintaining safety margins. 
The trajectory balances fuel efficiency with mission timeline requirements.

Impact: Fuel consumption within acceptable limits, arrival schedule maintained."""
    
    async def generate_hazard_response_strategy(self, hazard_data: Dict[str, Any]) -> str:
        """Generate strategic response to detected hazards"""
        
        if self.ai_available:
            return await self._generate_ai_hazard_strategy(hazard_data)
        else:
            return self._generate_simulated_hazard_strategy(hazard_data)
    
    async def _generate_ai_hazard_strategy(self, hazard_data: Dict[str, Any]) -> str:
        """Generate AI hazard response strategy"""
        try:
            hazard_type = hazard_data.get('event_type', 'unknown')
            severity = hazard_data.get('severity', 0)
            duration = hazard_data.get('duration', 0)
            
            prompt = f"""
            As ODIN AI Co-pilot, provide immediate strategic guidance for this space hazard:
            
            HAZARD TYPE: {hazard_type}
            SEVERITY: {severity}/10
            DURATION: {duration} hours
            DESCRIPTION: {hazard_data.get('description', 'No description')}
            
            Provide concise strategic guidance including:
            1. Immediate actions required
            2. Trajectory modification recommendations
            3. Risk mitigation strategies
            4. Timeline impact
            
            Focus on actionable guidance for mission control. Keep under 120 words.
            """
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are ODIN AI Co-pilot providing emergency hazard response guidance."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.5  # Lower temperature for more focused responses
            )
            
            strategy = response.choices[0].message.content.strip()
            self._add_to_context_history("hazard_strategy", strategy, hazard_data)
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error generating AI hazard strategy: {e}")
            return self._generate_simulated_hazard_strategy(hazard_data)
    
    def _generate_simulated_hazard_strategy(self, hazard_data: Dict[str, Any]) -> str:
        """Generate simulated hazard response strategy"""
        
        hazard_type = hazard_data.get('event_type', 'unknown')
        severity = hazard_data.get('severity', 0)
        
        if hazard_type == "solar_flare":
            if severity > 7:
                return "CRITICAL: Solar flare detected. Recommend immediate crew shelter protocol. Monitor radiation levels. Consider trajectory modification to reduce exposure time."
            else:
                return "Solar flare detected. Increase radiation monitoring. Current trajectory acceptable with enhanced shielding protocols."
        
        elif hazard_type == "cme":
            if severity > 6:
                return "CME approaching. Recommend trajectory adjustment to avoid peak impact. Estimated arrival allows for safe course correction."
            else:
                return "CME detected. Monitor space weather. Current trajectory provides adequate safety margins."
        
        elif hazard_type == "debris_conjunction":
            return "Debris conjunction alert. Calculate avoidance maneuver options. Minimal delta-V adjustment recommended to ensure safe passage."
        
        else:
            return f"Hazard detected: {hazard_type}. Monitoring situation and assessing trajectory implications."
    
    def _format_hazards_for_ai(self, hazards: List[Dict]) -> str:
        """Format hazards list for AI consumption"""
        if not hazards:
            return "No active hazards"
        
        formatted = []
        for hazard in hazards:
            formatted.append(f"- {hazard.get('event_type', 'Unknown')}: Severity {hazard.get('severity', 0)}/10")
        
        return "\n".join(formatted)
    
    def _assess_trajectory_efficiency(self, trajectory: Dict) -> str:
        """Assess trajectory efficiency"""
        delta_v = trajectory.get('total_delta_v', 0)
        
        if delta_v < 3000:
            return "Highly efficient"
        elif delta_v < 4000:
            return "Moderately efficient"
        else:
            return "High delta-V required"
    
    def _assess_overall_risk(self, hazards: List, trajectory: Dict) -> str:
        """Assess overall mission risk"""
        hazard_count = len(hazards)
        radiation = trajectory.get('radiation_exposure', 0)
        
        if hazard_count == 0 and radiation < 30:
            return "Low"
        elif hazard_count <= 2 and radiation < 60:
            return "Moderate"
        else:
            return "Elevated"
    
    def _generate_recommendations(self, mission_data: Dict) -> str:
        """Generate context-appropriate recommendations"""
        hazards = mission_data.get('hazards', [])
        trajectory = mission_data.get('trajectory', {})
        
        recommendations = []
        
        if len(hazards) > 0:
            recommendations.append("Monitor hazard evolution closely")
        
        if trajectory.get('radiation_exposure', 0) > 70:
            recommendations.append("Consider alternative low-radiation trajectory")
        
        if not recommendations:
            recommendations.append("Continue nominal operations")
        
        return ". ".join(recommendations) + "."
    
    def _add_to_context_history(self, request_type: str, response: str, input_data: Dict):
        """Add interaction to context history"""
        self.context_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "request_type": request_type,
            "response": response,
            "input_data": input_data
        })
        
        # Limit context history size
        if len(self.context_history) > self.max_context_history:
            self.context_history.pop(0)
    
    def get_context_history(self) -> List[Dict]:
        """Get AI interaction history"""
        return self.context_history.copy()
    
    async def generate_performance_analysis(self, performance_data: Dict) -> str:
        """Generate performance analysis and optimization suggestions"""
        
        if self.ai_available:
            try:
                prompt = f"""
                As ODIN AI Co-pilot, analyze mission performance and provide optimization recommendations:
                
                PERFORMANCE DATA: {json.dumps(performance_data, indent=2)}
                
                Provide analysis covering:
                1. Performance vs. baseline metrics
                2. Efficiency opportunities
                3. Resource optimization
                4. Strategic improvements
                
                Keep analysis actionable and under 180 words.
                """
                
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are ODIN AI Co-pilot providing mission performance analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=220,
                    temperature=0.6
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                logger.error(f"Error generating performance analysis: {e}")
        
        # Fallback simulated analysis
        return "Performance analysis: Mission proceeding within nominal parameters. Fuel consumption on track. Trajectory efficiency acceptable. No immediate optimizations required."
