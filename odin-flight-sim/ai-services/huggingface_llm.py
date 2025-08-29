"""
Hugging Face Integration for Open-Source LLM Inference
Supports Mistral 7B and Phi-3-mini for ODIN AI decisions
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class HuggingFaceLLMService:
    """Hugging Face Inference API service for open-source LLMs"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        self.base_url = "https://api-inference.huggingface.co/models"
        self.session = None
        
        # Supported models
        self.models = {
            "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.1",
            "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
            "codellama": "codellama/CodeLlama-7b-Instruct-hf"
        }
        
        self.current_model = self.models["mistral-7b"]
        
        # Cache for model responses
        self.response_cache = {}
        
        logger.info(f"Hugging Face LLM Service initialized with model: {self.current_model}")
    
    async def initialize_session(self):
        """Initialize aiohttp session"""
        if not self.session:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            timeout = aiohttp.ClientTimeout(total=60)  # LLM calls can take time
            self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)
    
    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
    
    def set_model(self, model_name: str):
        """Set the current model to use"""
        if model_name in self.models:
            self.current_model = self.models[model_name]
            logger.info(f"Switched to model: {self.current_model}")
        else:
            logger.warning(f"Unknown model: {model_name}. Available models: {list(self.models.keys())}")
    
    async def generate_mitigation_strategies(
        self, 
        hazard_data: Dict[str, Any],
        mission_state: Dict[str, Any],
        context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Generate hazard mitigation strategies using LLM"""
        
        prompt = self._build_mitigation_prompt(hazard_data, mission_state, context)
        
        try:
            response = await self._call_llm(prompt)
            strategies = self._parse_mitigation_response(response)
            
            return strategies
            
        except Exception as e:
            logger.error(f"Error generating mitigation strategies: {e}")
            return self._fallback_mitigation_strategies(hazard_data)
    
    async def explain_decision(
        self,
        decision_data: Dict[str, Any],
        explanation_type: str = "detailed"
    ) -> str:
        """Generate human-readable explanation for ODIN decision"""
        
        prompt = self._build_explanation_prompt(decision_data, explanation_type)
        
        try:
            response = await self._call_llm(prompt)
            explanation = self._parse_explanation_response(response)
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return self._fallback_explanation(decision_data)
    
    async def analyze_trajectory_options(
        self,
        trajectory_options: List[Dict[str, Any]],
        mission_constraints: Dict[str, Any],
        hazard_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze trajectory options and provide recommendations"""
        
        prompt = self._build_trajectory_analysis_prompt(
            trajectory_options, mission_constraints, hazard_context
        )
        
        try:
            response = await self._call_llm(prompt)
            analysis = self._parse_trajectory_analysis_response(response)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing trajectory options: {e}")
            return self._fallback_trajectory_analysis(trajectory_options)
    
    async def generate_mission_commentary(
        self,
        mission_events: List[Dict[str, Any]],
        current_status: Dict[str, Any]
    ) -> str:
        """Generate mission commentary for logs"""
        
        prompt = self._build_mission_commentary_prompt(mission_events, current_status)
        
        try:
            response = await self._call_llm(prompt)
            commentary = self._parse_commentary_response(response)
            
            return commentary
            
        except Exception as e:
            logger.error(f"Error generating mission commentary: {e}")
            return self._fallback_mission_commentary(mission_events)
    
    async def _call_llm(self, prompt: str, max_tokens: int = 512) -> str:
        """Make API call to Hugging Face Inference API"""
        
        await self.initialize_session()
        
        # Check cache first
        cache_key = f"{hash(prompt)}_{self.current_model}"
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        url = f"{self.base_url}/{self.current_model}"
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        try:
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    if isinstance(result, list) and len(result) > 0:
                        generated_text = result[0].get("generated_text", "")
                    else:
                        generated_text = str(result)
                    
                    # Cache the response
                    self.response_cache[cache_key] = generated_text
                    
                    return generated_text
                
                elif response.status == 503:
                    # Model is loading, wait and retry
                    logger.info("Model is loading, waiting...")
                    await asyncio.sleep(10)
                    return await self._call_llm(prompt, max_tokens)
                
                else:
                    error_text = await response.text()
                    logger.error(f"Hugging Face API error {response.status}: {error_text}")
                    return ""
                    
        except Exception as e:
            logger.error(f"Error calling Hugging Face API: {e}")
            return ""
    
    def _build_mitigation_prompt(
        self, 
        hazard_data: Dict[str, Any], 
        mission_state: Dict[str, Any],
        context: Optional[str] = None
    ) -> str:
        """Build prompt for hazard mitigation strategies"""
        
        hazard_type = hazard_data.get("hazard_type", "unknown")
        severity = hazard_data.get("severity", 0.5)
        
        prompt = f"""You are ODIN, an AI system for autonomous spacecraft navigation. Analyze this space hazard and provide mitigation strategies.

MISSION CONTEXT:
- Spacecraft: Earth-to-Moon transfer mission
- Current Position: {mission_state.get('position', 'LEO')}
- Fuel Remaining: {mission_state.get('fuel_remaining', 70)}%
- Mission Time: {mission_state.get('mission_time', 'T+24h')}

DETECTED HAZARD:
- Type: {hazard_type}
- Severity: {severity:.1f}/1.0
- Details: {hazard_data.get('description', 'High-energy space weather event')}

{context or 'Prioritize crew safety over mission timeline.'}

Provide 3 concrete mitigation strategies in this format:
1. STRATEGY_NAME: Brief description focusing on safety and mission success
2. STRATEGY_NAME: Alternative approach with different trade-offs
3. STRATEGY_NAME: Conservative backup option

Each strategy should specify:
- Trajectory adjustments needed
- Estimated fuel cost
- Time impact
- Risk reduction achieved"""
        
        return prompt
    
    def _build_explanation_prompt(
        self, 
        decision_data: Dict[str, Any], 
        explanation_type: str
    ) -> str:
        """Build prompt for decision explanation"""
        
        decision_type = decision_data.get("decision_type", "trajectory_planning")
        chosen_option = decision_data.get("chosen_option", {})
        
        if explanation_type == "brief":
            detail_instruction = "Provide a brief 2-sentence explanation suitable for mission logs."
        elif explanation_type == "technical":
            detail_instruction = "Provide a technical explanation with specific metrics and calculations."
        else:
            detail_instruction = "Provide a detailed explanation that a mission commander would understand."
        
        prompt = f"""You are ODIN explaining your autonomous navigation decision to the mission team.

DECISION CONTEXT:
- Decision Type: {decision_type}
- Timestamp: {decision_data.get('timestamp', 'T+unknown')}
- Threat Analysis: {decision_data.get('threat_analysis', 'Multiple hazards detected')}

CHOSEN OPTION:
- Strategy: {chosen_option.get('name', 'Unknown')}
- Delta-V Cost: {chosen_option.get('delta_v', 0)} m/s
- Time Impact: {chosen_option.get('time_impact', 0)} hours
- Safety Score: {chosen_option.get('safety_score', 0.8)}/1.0

{detail_instruction}

Explain WHY this decision was optimal considering:
- Crew safety (highest priority)
- Mission success probability
- Resource efficiency
- Risk mitigation"""
        
        return prompt
    
    def _build_trajectory_analysis_prompt(
        self,
        trajectory_options: List[Dict[str, Any]],
        mission_constraints: Dict[str, Any],
        hazard_context: Dict[str, Any]
    ) -> str:
        """Build prompt for trajectory analysis"""
        
        options_summary = "\\n".join([
            f"- {opt.get('name', f'Option {i+1}')}: ΔV={opt.get('delta_v', 0)}m/s, "
            f"Duration={opt.get('duration', 0)}h, Safety={opt.get('safety_score', 0.5)}"
            for i, opt in enumerate(trajectory_options[:5])  # Limit to 5 options
        ])
        
        prompt = f"""You are ODIN analyzing trajectory options for an Earth-to-Moon mission under hazardous conditions.

TRAJECTORY OPTIONS:
{options_summary}

MISSION CONSTRAINTS:
- Max ΔV Budget: {mission_constraints.get('max_delta_v', 15000)} m/s
- Max Duration: {mission_constraints.get('max_duration', 120)} hours
- Min Safety Score: {mission_constraints.get('min_safety', 0.7)}

HAZARD CONTEXT:
- Active Threats: {len(hazard_context.get('active_hazards', []))}
- Risk Level: {hazard_context.get('risk_level', 'MODERATE')}

Analyze these options and provide:
1. RECOMMENDED option with clear reasoning
2. RISK ASSESSMENT for top 2 options
3. TRADE-OFF analysis (safety vs efficiency vs time)

Format your analysis clearly for autonomous decision-making."""
        
        return prompt
    
    def _build_mission_commentary_prompt(
        self,
        mission_events: List[Dict[str, Any]],
        current_status: Dict[str, Any]
    ) -> str:
        """Build prompt for mission commentary"""
        
        recent_events = mission_events[-5:] if len(mission_events) > 5 else mission_events
        events_summary = "\\n".join([
            f"T+{event.get('time', '?')}h: {event.get('description', 'Event occurred')}"
            for event in recent_events
        ])
        
        prompt = f"""You are ODIN providing a mission status update for the flight director.

RECENT MISSION EVENTS:
{events_summary}

CURRENT STATUS:
- Position: {current_status.get('position', 'In transit')}
- Fuel: {current_status.get('fuel_remaining', 70)}%
- Next Maneuver: {current_status.get('next_maneuver', 'TBD')}
- System Health: {current_status.get('system_health', 'NOMINAL')}

Provide a concise professional mission update focusing on:
- Key developments since last report
- Current trajectory status
- Any autonomous decisions made
- Upcoming critical events

Keep it brief and mission-focused."""
        
        return prompt
    
    def _parse_mitigation_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response for mitigation strategies"""
        
        if not response:
            return self._fallback_mitigation_strategies({})
        
        strategies = []
        lines = response.split('\n')
        
        current_strategy = None
        for line in lines:
            line = line.strip()
            
            # Look for numbered strategies
            if line and (line[0].isdigit() or line.startswith('-')):
                if ':' in line:
                    strategy_name, description = line.split(':', 1)
                    strategy_name = strategy_name.strip('123456789.- ')
                    
                    current_strategy = {
                        "name": strategy_name,
                        "description": description.strip(),
                        "delta_v_cost": 1000.0,  # Default values
                        "time_impact": 12.0,
                        "risk_reduction": 0.3,
                        "safety_score": 0.8
                    }
                    strategies.append(current_strategy)
            
            # Look for specific metrics in following lines
            elif current_strategy and line:
                if "delta" in line.lower() or "fuel" in line.lower():
                    # Try to extract numeric values
                    import re
                    numbers = re.findall(r'\d+\.?\d*', line)
                    if numbers:
                        current_strategy["delta_v_cost"] = float(numbers[0])
                
                elif "time" in line.lower() or "hour" in line.lower():
                    import re
                    numbers = re.findall(r'\d+\.?\d*', line)
                    if numbers:
                        current_strategy["time_impact"] = float(numbers[0])
        
        return strategies if strategies else self._fallback_mitigation_strategies({})
    
    def _parse_explanation_response(self, response: str) -> str:
        """Parse LLM response for decision explanation"""
        
        if not response:
            return "ODIN autonomous decision executed based on safety protocols."
        
        # Clean up the response
        explanation = response.strip()
        
        # Ensure it starts professionally
        if not explanation.startswith(("ODIN", "The decision", "This choice")):
            explanation = f"ODIN Analysis: {explanation}"
        
        return explanation
    
    def _parse_trajectory_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response for trajectory analysis"""
        
        if not response:
            return self._fallback_trajectory_analysis([])
        
        analysis = {
            "recommended_option": "Option 1",
            "reasoning": response,
            "confidence": 0.8,
            "risk_assessment": "Moderate risk with acceptable trade-offs",
            "trade_offs": "Balancing safety and efficiency"
        }
        
        # Try to extract specific recommendations
        lines = response.lower().split('\n')
        for line in lines:
            if "recommend" in line:
                # Extract option name/number
                import re
                option_match = re.search(r'option\s+(\w+)', line)
                if option_match:
                    analysis["recommended_option"] = f"Option {option_match.group(1)}"
        
        return analysis
    
    def _parse_commentary_response(self, response: str) -> str:
        """Parse LLM response for mission commentary"""
        
        if not response:
            return "ODIN systems operational. Mission proceeding nominally."
        
        # Ensure professional mission commentary format
        commentary = response.strip()
        
        if not commentary.startswith(("Mission", "ODIN", "Status")):
            commentary = f"Mission Update: {commentary}"
        
        return commentary
    
    # Fallback methods for when LLM is unavailable
    
    def _fallback_mitigation_strategies(self, hazard_data: Dict) -> List[Dict[str, Any]]:
        """Fallback mitigation strategies when LLM is unavailable"""
        
        return [
            {
                "name": "Conservative Avoidance",
                "description": "Alter trajectory to avoid hazard zone with maximum safety margin",
                "delta_v_cost": 1500.0,
                "time_impact": 24.0,
                "risk_reduction": 0.8,
                "safety_score": 0.9
            },
            {
                "name": "Timing Adjustment",
                "description": "Adjust departure timing to minimize hazard exposure",
                "delta_v_cost": 500.0,
                "time_impact": 12.0,
                "risk_reduction": 0.5,
                "safety_score": 0.75
            },
            {
                "name": "Shielding Protocol",
                "description": "Maintain course with enhanced radiation shielding procedures",
                "delta_v_cost": 0.0,
                "time_impact": 0.0,
                "risk_reduction": 0.3,
                "safety_score": 0.6
            }
        ]
    
    def _fallback_explanation(self, decision_data: Dict) -> str:
        """Fallback explanation when LLM is unavailable"""
        
        decision_type = decision_data.get("decision_type", "navigation")
        
        return (f"ODIN executed {decision_type} based on autonomous safety protocols. "
                f"Decision prioritized crew safety and mission success using "
                f"pre-programmed hazard avoidance algorithms.")
    
    def _fallback_trajectory_analysis(self, trajectory_options: List) -> Dict[str, Any]:
        """Fallback trajectory analysis when LLM is unavailable"""
        
        return {
            "recommended_option": "Option 1" if trajectory_options else "Baseline",
            "reasoning": "Recommendation based on safety-first autonomous protocols",
            "confidence": 0.7,
            "risk_assessment": "Analysis performed using built-in safety algorithms",
            "trade_offs": "Prioritizing safety over efficiency in autonomous mode"
        }
    
    def _fallback_mission_commentary(self, mission_events: List) -> str:
        """Fallback mission commentary when LLM is unavailable"""
        
        return (f"Mission Status: ODIN autonomous systems operational. "
                f"{len(mission_events)} events processed. "
                f"Continuing nominal flight operations.")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health and model availability"""
        
        try:
            test_prompt = "Hello, ODIN here. Confirm system status."
            response = await self._call_llm(test_prompt, max_tokens=50)
            
            return {
                "service_available": True,
                "model": self.current_model,
                "api_key_configured": bool(self.api_key),
                "cache_size": len(self.response_cache),
                "test_response": bool(response)
            }
            
        except Exception as e:
            return {
                "service_available": False,
                "model": self.current_model,
                "api_key_configured": bool(self.api_key),
                "error": str(e)
            }
