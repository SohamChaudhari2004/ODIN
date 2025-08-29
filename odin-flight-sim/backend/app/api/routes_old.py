from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
import logging
import sys
import os

# Add AI services to path  
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ai-services'))

# Import ODIN AI services
try:
    from odin_main import OdinNavigationSystem
    from ai_copilot import AICoPilot
    from predictive_hazard_forecasting import PredictiveHazardForecasting
    from explainability_module import ExplainabilityModule
    from space_weather_service import SpaceWeatherDataService
    ODIN_AVAILABLE = True
    print("✅ ODIN AI services loaded successfully")
except ImportError as e:
    print(f"❌ ODIN AI services not available: {e}")
    ODIN_AVAILABLE = False

# Import simplified backend schemas
from ..models.schemas import (
    MissionStatus, 
    TelemetryData, 
    TrajectoryPlan
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize ODIN system if available
odin_system = None
ai_copilot = None
hazard_forecaster = None
explainer = None
space_weather_service = None

if ODIN_AVAILABLE:
    try:
        odin_system = OdinNavigationSystem()
        ai_copilot = AICoPilot()
        hazard_forecaster = PredictiveHazardForecasting()
        explainer = ExplainabilityModule()
        space_weather_service = SpaceWeatherDataService()
        logger.info("🚀 ODIN Navigation System initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ODIN system: {e}")
        ODIN_AVAILABLE = False
else:
    logger.warning("⚠️ ODIN system running in fallback mode - AI services not available")
odin_system = None
if ODIN_AVAILABLE:
    odin_system = OdinNavigationSystem()
    ai_copilot = AICoPilot()
    hazard_forecaster = PredictiveHazardForecasting()
    explainer = ExplainabilityModule()
    space_weather_service = SpaceWeatherDataService()
    logger.info("ODIN Navigation System initialized successfully")
else:
    logger.warning("ODIN system running in fallback mode - AI services not available")

# ODIN-specific API endpoints
@router.post("/odin/initialize")
async def initialize_odin_mission(destination: str = "Moon"):
    """Initialize ODIN autonomous navigation system with historical space weather data"""
    if not ODIN_AVAILABLE or not odin_system:
        raise HTTPException(status_code=503, detail="ODIN system not available")
    
    try:
        result = await odin_system.initialize_mission(destination)
        return {
            "success": True,
            "message": "ODIN mission initialized successfully",
            "data": result
        }
    except Exception as e:
        logger.error(f"Error initializing ODIN mission: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize ODIN: {str(e)}")

@router.post("/odin/autonomous-mission")
async def start_autonomous_mission(duration_hours: float = 24.0):
    """Start ODIN autonomous mission loop with continuous monitoring and replanning"""
    if not ODIN_AVAILABLE or not odin_system:
        raise HTTPException(status_code=503, detail="ODIN system not available")
    
    try:
        mission_events = await odin_system.autonomous_mission_loop(duration_hours)
        return {
            "success": True,
            "message": f"Autonomous mission completed - {duration_hours} hours",
            "mission_events": mission_events,
            "decision_logs": odin_system.get_decision_logs()
        }
    except Exception as e:
        logger.error(f"Error during autonomous mission: {e}")
        raise HTTPException(status_code=500, detail=f"Autonomous mission failed: {str(e)}")

@router.get("/odin/status")
async def get_odin_status():
    """Get ODIN system status and capabilities"""
    if not ODIN_AVAILABLE or not odin_system:
        return {
            "odin_available": False,
            "status": "ODIN system not available - running in fallback mode",
            "capabilities": []
        }
    
    try:
        status = odin_system.get_system_status()
        return {
            "odin_available": True,
            "status": status,
            "capabilities": [
                "Autonomous trajectory planning",
                "Real-time hazard detection and avoidance", 
                "Historical space weather analysis (2012-2018)",
                "AI-powered decision making with human-readable logs",
                "Dynamic replanning and optimization",
                "Explainable AI recommendations"
            ]
        }
    except Exception as e:
        logger.error(f"Error getting ODIN status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/odin/decision-logs")
async def get_odin_decision_logs():
    """Get ODIN decision logs with human-readable explanations"""
    if not ODIN_AVAILABLE or not odin_system:
        raise HTTPException(status_code=503, detail="ODIN system not available")
    
    try:
        logs = odin_system.get_decision_logs()
        return {
            "total_decisions": len(logs),
            "decision_logs": logs,
            "latest_decisions": logs[-10:] if len(logs) > 10 else logs
        }
    except Exception as e:
        logger.error(f"Error getting decision logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/space-weather/current")
async def get_current_space_weather():
    """Get current space weather conditions from historical 2012-2018 data"""
    if not ODIN_AVAILABLE:
        raise HTTPException(status_code=503, detail="Space weather service not available")
    
    try:
        weather_data = await space_weather_service.get_space_weather_data()
        return {
            "success": True,
            "space_weather": weather_data,
            "historical_period": "2012-2018",
            "data_sources": ["NOAA SWPC", "NASA DONKI", "Simulated realistic conditions"]
        }
    except Exception as e:
        logger.error(f"Error getting space weather data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/space-weather/hazard-forecast")
async def get_hazard_forecast(hours_ahead: int = 72):
    """Get space weather hazard forecast"""
    if not ODIN_AVAILABLE:
        raise HTTPException(status_code=503, detail="Space weather service not available")
    
    try:
        hazards = await space_weather_service.get_hazard_forecast(forecast_hours=hours_ahead)
        return {
            "success": True,
            "forecast_horizon_hours": hours_ahead,
            "predicted_hazards": hazards,
            "total_hazards": len(hazards)
        }
    except Exception as e:
        logger.error(f"Error getting hazard forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/trajectory/ai-alternatives")
async def generate_ai_trajectory_alternatives(
    current_trajectory: Dict,
    hazards: List[Dict] = [],
    constraints: Dict = {}
):
    """Generate AI-powered trajectory alternatives using Mistral AI via LangChain"""
    if not ODIN_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI services not available")
    
    try:
        alternatives = await ai_copilot.generate_trajectory_alternatives(
            current_trajectory, hazards, constraints
        )
        return {
            "success": True,
            "alternatives_generated": len(alternatives),
            "trajectory_alternatives": alternatives,
            "ai_provider": "Mistral AI via LangChain"
        }
    except Exception as e:
        logger.error(f"Error generating AI alternatives: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/trajectory/evaluate-with-logs")
async def evaluate_trajectories_with_decision_logs(
    trajectories: List[Dict],
    hazards: List[Dict] = [],
    constraints: Dict = {}
):
    """Evaluate trajectory options and generate human-readable decision logs"""
    try:
        # Convert to schema objects (simplified for this example)
        trajectory_plans = [TrajectoryPlan(**traj) for traj in trajectories]
        hazard_events = [HazardEvent(**haz) for haz in hazards]
        
        # Use enhanced decision engine
        decision_result = decision_engine.evaluate_trajectories(
            trajectory_plans, hazard_events, constraints
        )
        
        return {
            "success": True,
            "decision_result": decision_result,
            "human_readable_log": decision_result.get("human_readable_log", ""),
            "technical_analysis": decision_result.get("technical_analysis", {}),
            "alternatives_considered": decision_result.get("alternatives_considered", 0)
        }
    except Exception as e:
        logger.error(f"Error evaluating trajectories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ai/explainability/{decision_id}")
async def get_decision_explanation(decision_id: str, audience: str = "mission_control"):
    """Get explainable AI analysis for a specific decision"""
    if not ODIN_AVAILABLE:
        raise HTTPException(status_code=503, detail="Explainability service not available")
    
    try:
        # This would typically fetch the decision from storage
        # For now, we'll generate a sample explanation
        explanation = {
            "decision_id": decision_id,
            "explanation_type": "trajectory_selection",
            "audience": audience,
            "summary": "AI explanation for trajectory decision",
            "detailed_analysis": "Comprehensive analysis of decision factors",
            "confidence_level": 0.85,
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return {
            "success": True,
            "explanation": explanation,
            "ai_explainability": "Human-readable decision rationale provided"
        }
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/mission/status", response_model=MissionStatus)
async def get_mission_status():
    """Get current mission status including trajectory, hazards, and telemetry"""
    try:
        # Get current trajectory
        current_trajectory = trajectory_planner.get_current_trajectory()
        
        # Get active hazards
        active_hazards = hazard_detector.get_active_hazards()
        
        # Get telemetry data
        telemetry = await data_service.get_telemetry_snapshot()
        
        return MissionStatus(
            mission_id="ODIN-001",
            status="active",
            current_time=datetime.utcnow().isoformat(),
            trajectory=current_trajectory,
            hazards=active_hazards,
            telemetry=telemetry,
            ai_recommendations=await _get_ai_recommendations()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/telemetry", response_model=TelemetryData)
async def get_telemetry():
    """Get current telemetry data"""
    try:
        return await data_service.get_telemetry_snapshot()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/hazards/inject")
async def inject_hazard(hazard_request: HazardEventRequest):
    """Inject a new hazard event for simulation"""
    try:
        hazard_event = HazardEvent(
            event_type=hazard_request.event_type,
            severity=hazard_request.severity,
            start_time=hazard_request.start_time,
            duration=hazard_request.duration,
            affected_regions=hazard_request.affected_regions,
            mitigation_required=hazard_request.mitigation_required
        )
        
        await hazard_detector.inject_hazard(hazard_event)
        return {"status": "success", "message": "Hazard injected successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/hazards", response_model=List[HazardEvent])
async def get_hazards():
    """Get all active hazards"""
    try:
        return hazard_detector.get_active_hazards()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/trajectory/replan", response_model=TrajectoryPlan)
async def replan_trajectory(request: Optional[TrajectoryRequest] = None):
    """Request trajectory replanning"""
    try:
        constraints = request.constraints if request else {}
        new_trajectory = await trajectory_planner.replan_trajectory(constraints)
        return new_trajectory
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trajectory/current", response_model=TrajectoryPlan)
async def get_current_trajectory():
    """Get current active trajectory"""
    try:
        return trajectory_planner.get_current_trajectory()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trajectory/alternatives", response_model=List[TrajectoryPlan])
async def get_alternative_trajectories():
    """Get precomputed alternative trajectories"""
    try:
        return trajectory_planner.get_alternative_trajectories()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/data/solar")
async def get_solar_data():
    """Get latest solar weather data"""
    try:
        return await data_service.fetch_solar_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/data/space-weather")
async def get_space_weather():
    """Get latest space weather data"""
    try:
        return await data_service.fetch_space_weather()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/data/tle")
async def get_tle_data():
    """Get latest TLE data"""
    try:
        return await data_service.fetch_tle_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/simulation/control")
async def simulation_control(action: str):
    """Control simulation (start, pause, reset, accelerate)"""
    try:
        # Implementation for simulation control
        if action == "start":
            # Start simulation
            pass
        elif action == "pause":
            # Pause simulation
            pass
        elif action == "reset":
            # Reset simulation
            trajectory_planner.reset_to_baseline()
            hazard_detector.clear_hazards()
        elif action == "accelerate":
            # Accelerate simulation
            pass
        
        return {"status": "success", "action": action}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def _get_ai_recommendations() -> List[str]:
    """Get AI-generated recommendations (placeholder)"""
    return [
        "Current trajectory optimal for mission parameters",
        "No significant space weather threats detected",
        "Recommend continuing nominal operations"
    ]

# Decision Engine Endpoints
@router.post("/decision/evaluate-trajectories")
async def evaluate_trajectories(request: Dict):
    """Evaluate trajectories and return optimal selection with rationale"""
    try:
        trajectories = [trajectory_planner.get_current_trajectory()]
        trajectories.extend(trajectory_planner.get_alternative_trajectories())
        
        hazards = hazard_detector.get_active_hazards()
        constraints = request.get("constraints", {})
        
        decision = decision_engine.evaluate_trajectories(trajectories, hazards, constraints)
        
        # Generate decision log
        log_message = log_generator.generate_decision_rationale_log(decision)
        
        return {
            "decision": decision,
            "log_message": log_message,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/decision/history")
async def get_decision_history():
    """Get history of trajectory decisions"""
    try:
        return decision_engine.get_decision_history()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Log Generation Endpoints
@router.post("/logs/generate-hazard-log")
async def generate_hazard_log(hazard_data: Dict):
    """Generate human-readable log for hazard detection"""
    try:
        hazard = HazardEvent(**hazard_data)
        log_message = log_generator.generate_hazard_detection_log(hazard)
        
        formatted_log = log_generator.format_log_message(
            log_message,
            source="Hazard Detection",
            priority="WARNING" if hazard.severity > 5 else "INFO"
        )
        
        return formatted_log
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/logs/generate-trajectory-change-log")
async def generate_trajectory_change_log(request: Dict):
    """Generate log for trajectory changes"""
    try:
        old_traj_id = request.get("old_trajectory_id")
        new_traj_id = request.get("new_trajectory_id")
        reason = request.get("reason", "")
        
        # For demo, use current and alternative trajectories
        current_traj = trajectory_planner.get_current_trajectory()
        alternatives = trajectory_planner.get_alternative_trajectories()
        new_traj = alternatives[0] if alternatives else current_traj
        
        log_message = log_generator.generate_trajectory_change_log(current_traj, new_traj, reason)
        
        formatted_log = log_generator.format_log_message(
            log_message,
            source="Navigation",
            priority="INFO"
        )
        
        return formatted_log
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/logs/mission-summary")
async def get_mission_summary():
    """Get comprehensive mission summary logs"""
    try:
        # Gather mission data
        mission_data = {
            "current_phase": "Trans-Lunar Injection",
            "mission_time": time_control.get_current_simulation_time(),
            "trajectory": trajectory_planner.get_current_trajectory(),
            "hazards": hazard_detector.get_active_hazards(),
            "telemetry": await data_service.get_telemetry_snapshot()
        }
        
        summary_logs = log_generator.generate_mission_summary_log(mission_data)
        
        return {
            "summary_logs": summary_logs,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mission Save/Load Endpoints
@router.post("/mission/save")
async def save_mission_state(request: Dict):
    """Save current mission state"""
    try:
        # Gather complete mission state
        state_data = {
            "trajectory": trajectory_planner.get_current_trajectory().dict(),
            "hazards": [h.__dict__ for h in hazard_detector.get_active_hazards()],
            "telemetry": (await data_service.get_telemetry_snapshot()).dict(),
            "time_status": time_control.get_time_status(),
            "decision_history": decision_engine.get_decision_history()
        }
        
        save_id = save_load_system.save_mission_state(
            state_data,
            request.get("save_name", f"mission_{int(datetime.utcnow().timestamp())}"),
            request.get("description", ""),
            request.get("tags", [])
        )
        
        return {"save_id": save_id, "message": "Mission state saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/mission/saves")
async def list_mission_saves(tags: Optional[str] = None):
    """List all saved missions"""
    try:
        tag_list = tags.split(",") if tags else None
        saved_missions = save_load_system.list_saved_missions(tag_list)
        return saved_missions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/mission/load/{save_id}")
async def load_mission_state(save_id: str):
    """Load mission state from save"""
    try:
        mission_state = save_load_system.load_mission_state(save_id)
        return mission_state
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/mission/saves/{save_id}")
async def delete_mission_save(save_id: str):
    """Delete a mission save"""
    try:
        success = save_load_system.delete_mission_save(save_id)
        if success:
            return {"message": "Mission save deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Mission save not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Time Control Endpoints
@router.post("/time/start")
async def start_simulation_time():
    """Start simulation time control"""
    try:
        time_control.start_simulation()
        return {"message": "Simulation time started", "status": time_control.get_time_status()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/time/pause")
async def pause_simulation():
    """Pause simulation"""
    try:
        time_control.pause_simulation()
        return {"message": "Simulation paused", "status": time_control.get_time_status()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/time/resume")
async def resume_simulation():
    """Resume simulation"""
    try:
        time_control.resume_simulation()
        return {"message": "Simulation resumed", "status": time_control.get_time_status()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/time/reset")
async def reset_simulation_time():
    """Reset simulation time"""
    try:
        time_control.reset_simulation()
        return {"message": "Simulation time reset", "status": time_control.get_time_status()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/time/scale")
async def set_time_scale(request: Dict):
    """Set simulation time scale"""
    try:
        scale = request.get("scale")
        preset = request.get("preset")
        
        if preset:
            time_control.set_time_scale_preset(preset)
        elif scale:
            time_control.set_time_scale(float(scale))
        else:
            raise HTTPException(status_code=400, detail="Either scale or preset must be provided")
        
        return {"message": "Time scale updated", "status": time_control.get_time_status()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# AI Services Endpoints
@router.post("/ai/mission-brief")
async def generate_mission_brief(request: Dict):
    """Generate AI-powered mission brief"""
    if not AI_SERVICES_AVAILABLE or ai_copilot is None:
        raise HTTPException(status_code=503, detail="AI services not available")
    
    try:
        mission_data = {
            'trajectory': trajectory_planner.get_current_trajectory(),
            'hazards': hazard_detector.get_active_hazards(),
            'telemetry': await data_service.get_telemetry_snapshot(),
            'mission_time': time_control.get_mission_time()
        }
        
        brief = await ai_copilot.generate_mission_brief(mission_data)
        return {"mission_brief": brief, "generated_at": datetime.utcnow().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating mission brief: {str(e)}")

@router.post("/ai/explain-trajectory")
async def explain_trajectory_decision(request: Dict):
    """Get AI explanation for trajectory decisions"""
    if not AI_SERVICES_AVAILABLE or explainer is None:
        raise HTTPException(status_code=503, detail="AI explainability service not available")
    
    try:
        decision_data = request.get('decision_data', {})
        audience = request.get('audience', 'mission_control')
        
        explanation = explainer.explain_trajectory_selection(decision_data, audience)
        return explanation
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating explanation: {str(e)}")

@router.post("/ai/hazard-response-strategy") 
async def generate_hazard_response_strategy(request: Dict):
    """Generate AI-powered hazard response strategy"""
    if not AI_SERVICES_AVAILABLE or ai_copilot is None:
        raise HTTPException(status_code=503, detail="AI services not available")
    
    try:
        hazard_data = request.get('hazard_data', {})
        strategy = await ai_copilot.generate_hazard_response_strategy(hazard_data)
        return {"response_strategy": strategy, "generated_at": datetime.utcnow().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating hazard strategy: {str(e)}")

@router.post("/ai/predict-solar-activity")
async def predict_solar_activity(request: Dict):
    """Predict solar activity and space weather"""
    if not AI_SERVICES_AVAILABLE or hazard_forecaster is None:
        raise HTTPException(status_code=503, detail="AI forecasting service not available")
    
    try:
        current_data = request.get('current_data', {})
        forecast_hours = request.get('forecast_hours', 72)
        
        prediction = await hazard_forecaster.predict_solar_activity(current_data, forecast_hours)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting solar activity: {str(e)}")

@router.post("/ai/predict-radiation-exposure")
async def predict_radiation_exposure(request: Dict):
    """Predict radiation exposure along trajectory"""
    if not AI_SERVICES_AVAILABLE or hazard_forecaster is None:
        raise HTTPException(status_code=503, detail="AI forecasting service not available")
    
    try:
        current_data = request.get('current_data', {})
        trajectory_data = request.get('trajectory_data', {})
        
        prediction = await hazard_forecaster.predict_radiation_levels(current_data, trajectory_data)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting radiation exposure: {str(e)}")

@router.post("/ai/predict-debris-conjunctions")
async def predict_debris_conjunctions(request: Dict):
    """Predict potential debris conjunctions"""
    if not AI_SERVICES_AVAILABLE or hazard_forecaster is None:
        raise HTTPException(status_code=503, detail="AI forecasting service not available")
    
    try:
        trajectory_data = request.get('trajectory_data', {})
        
        prediction = await hazard_forecaster.predict_debris_conjunctions(trajectory_data)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting debris conjunctions: {str(e)}")

@router.post("/ai/explain-hazard-response")
async def explain_hazard_response(request: Dict):
    """Get AI explanation for hazard response decisions"""
    if not AI_SERVICES_AVAILABLE or explainer is None:
        raise HTTPException(status_code=503, detail="AI explainability service not available")
    
    try:
        response_data = request.get('response_data', {})
        audience = request.get('audience', 'mission_control')
        
        explanation = explainer.explain_hazard_response(response_data, audience)
        return explanation
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating hazard response explanation: {str(e)}")

@router.post("/ai/explain-recommendation")
async def explain_ai_recommendation(request: Dict):
    """Get explanation for AI recommendations"""
    if not AI_SERVICES_AVAILABLE or explainer is None:
        raise HTTPException(status_code=503, detail="AI explainability service not available")
    
    try:
        recommendation_data = request.get('recommendation_data', {})
        audience = request.get('audience', 'mission_control')
        
        explanation = explainer.explain_ai_recommendation(recommendation_data, audience)
        return explanation
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendation explanation: {str(e)}")

@router.get("/ai/context-history")
async def get_ai_context_history():
    """Get AI interaction context history"""
    if not AI_SERVICES_AVAILABLE or ai_copilot is None:
        raise HTTPException(status_code=503, detail="AI services not available")
    
    try:
        history = ai_copilot.get_context_history()
        return {"context_history": history, "retrieved_at": datetime.utcnow().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving AI context: {str(e)}")

@router.post("/ai/performance-analysis")
async def generate_performance_analysis(request: Dict):
    """Generate AI-powered performance analysis"""
    if not AI_SERVICES_AVAILABLE or ai_copilot is None:
        raise HTTPException(status_code=503, detail="AI services not available")
    
    try:
        performance_data = request.get('performance_data', {})
        analysis = await ai_copilot.generate_performance_analysis(performance_data)
        return {"performance_analysis": analysis, "generated_at": datetime.utcnow().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating performance analysis: {str(e)}")

@router.get("/ai/services-status")
async def get_ai_services_status():
    """Get status of AI services"""
    status = {
        "ai_services_available": AI_SERVICES_AVAILABLE,
        "ai_copilot_available": ai_copilot is not None,
        "hazard_forecaster_available": hazard_forecaster is not None,
        "explainer_available": explainer is not None,
        "openai_configured": False,
        "ml_models_available": False
    }
    
    if AI_SERVICES_AVAILABLE and ai_copilot:
        status["openai_configured"] = ai_copilot.ai_available
    
    if AI_SERVICES_AVAILABLE and hazard_forecaster:
        status["ml_models_available"] = hasattr(hazard_forecaster, 'models') and bool(hazard_forecaster.models)
        status["model_performance"] = hazard_forecaster.get_model_performance()
    
    return status
