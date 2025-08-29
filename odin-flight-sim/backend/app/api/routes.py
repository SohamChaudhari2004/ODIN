from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
import logging
import sys
import os
import uuid

# Add ai-services to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ai-services')))

# Import AI services
try:
    from odin_main import OdinNavigationSystem
    from ai_copilot import AICoPilot
    from predictive_hazard_forecasting import PredictiveHazardForecasting
    from explainability_module import ExplainabilityModule
    from space_weather_service import SpaceWeatherDataService

    ODIN_AVAILABLE = True
    logging.info("Successfully imported ODIN AI services.")
except ImportError as e:
    logging.warning(f"Could not import ODIN AI services: {e}. Running in fallback mode.")
    ODIN_AVAILABLE = False

# Import backend services
from ..services.trajectory_engine import TrajectoryEngine
from ..services.space_data_service import SpaceWeatherDataService as BackendSpaceWeatherService
from ..models.odin_models import *
from ..config import get_database

logger = logging.getLogger(__name__)
router = APIRouter()


# Initialize ODIN system components globally
if ODIN_AVAILABLE:
    odin_system = OdinNavigationSystem()
    ai_copilot = AICoPilot()
    hazard_forecaster = PredictiveHazardForecasting()
    explainer = ExplainabilityModule()
    space_weather_service = SpaceWeatherDataService()
    trajectory_engine = TrajectoryEngine()
    backend_space_service = BackendSpaceWeatherService()
    logging.info("ODIN services initialized.")
else:
    # Create fallback dummy services
    class DummyService:
        def __getattr__(self, name):
            def method(*args, **kwargs):
                logging.warning(f"Calling dummy service: {self.__class__.__name__}.{name}")
                if asyncio.iscoroutinefunction(self._dummy_method):
                    return self._dummy_async_method(*args, **kwargs)
                return self._dummy_method(*args, **kwargs)
            return method

        def _dummy_method(self, *args, **kwargs):
            return {"error": "Service not available", "fallback": True}

        async def _dummy_async_method(self, *args, **kwargs):
            return {"error": "Service not available", "fallback": True}

    odin_system = DummyService()
    ai_copilot = DummyService()
    hazard_forecaster = DummyService()
    explainer = DummyService()
    space_weather_service = DummyService()
    trajectory_engine = DummyService()
    backend_space_service = DummyService()
    logging.info("Fallback dummy services initialized.")


# =============================================================================
# ODIN CORE ENDPOINTS
# =============================================================================

@router.post("/odin/initialize", response_model=InitializeMissionResponse)
async def initialize_odin_mission(request: InitializeMissionRequest):
    """Initialize ODIN autonomous navigation system with historical space weather data"""
    if not ODIN_AVAILABLE:
        raise HTTPException(status_code=503, detail="ODIN system not available")
    
    try:
        mission_id = f"odin_mission_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        historical_date = request.historical_date or datetime(2015, 3, 17, 12, 0, 0)
        
        mission_doc = MissionDocument(
            mission_id=mission_id,
            start_time=datetime.utcnow(),
            status=MissionStatus.INITIALIZING,
            destination=request.destination,
            historical_timestamp=historical_date,
            spacecraft_position=[7000.0, 0.0, 0.0],
            spacecraft_velocity=[0.0, 7.8, 0.0],
            fuel_remaining=85.0,
            mission_constraints=request.mission_constraints or {
                "max_delta_v": 15000,
                "max_duration": 120,
                "min_safety_score": 0.7
            }
        )
        
        db = get_database()
        if db is not None:
            await db.missions.insert_one(mission_doc.dict())
        
        initial_trajectory = await trajectory_engine.calculate_initial_trajectory(
            historical_date, request.destination
        )
            
        trajectory_doc = TrajectoryDocument(
            trajectory_id=initial_trajectory.trajectory_id,
            mission_id=mission_id,
            trajectory_type="baseline",
            name=initial_trajectory.name,
            description="Initial Earth-to-Moon transfer trajectory",
            maneuvers=[],
            total_delta_v=initial_trajectory.total_delta_v,
            duration_hours=initial_trajectory.total_duration,
            radiation_exposure=initial_trajectory.radiation_exposure,
            collision_risk=initial_trajectory.collision_risk,
            safety_score=initial_trajectory.safety_score,
            fuel_efficiency=1.0 - (initial_trajectory.fuel_required / 2000.0),
            waypoints=[],
            is_active=True
        )
            
        if db is not None:
            await db.trajectories.insert_one(trajectory_doc.dict())
        
        result = await odin_system.initialize_mission(request.destination)
        
        return {
            "success": True,
            "mission_id": mission_id,
            "message": "ODIN mission initialized successfully",
            "historical_timestamp": historical_date.isoformat(),
            "initial_trajectory": initial_trajectory.trajectory_id,
            "data": result,
            "system_info": {
                "name": "ODIN (Optimal Dynamic Interplanetary Navigator)",
                "version": "1.0.0",
                "destination": request.destination,
                "ai_engine": "LangGraph + Hugging Face (Mistral-7B)",
                "data_source": "Historical space weather (2012-2018)",
                "trajectory_engine": "Poliastro-based orbital mechanics",
                "database": "MongoDB for persistent mission state"
            }
        }
        
    except Exception as e:
        logger.error(f"Error initializing ODIN mission: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize ODIN: {str(e)}")

@router.post("/odin/autonomous-mission")
async def start_autonomous_mission(mission_id: str, duration_hours: float = 24.0):
    """Start ODIN autonomous mission loop with continuous monitoring and replanning"""
    if not ODIN_AVAILABLE:
        raise HTTPException(status_code=503, detail="ODIN decision engine not available")
    
    try:
        mission_events = await odin_system.autonomous_mission_loop(duration_hours)
        
        db = get_database()
        if db is not None:
            await db.missions.update_one(
                {"mission_id": mission_id},
                {"$set": {"status": MissionStatus.COMPLETED, "end_time": datetime.utcnow()}}
            )
        
        return {
            "success": True,
            "mission_id": mission_id,
            "message": f"Autonomous mission completed - {duration_hours} hours",
            "mission_duration": duration_hours,
            "mission_events": mission_events,
        }
        
    except Exception as e:
        logger.error(f"Error during autonomous mission: {e}")
        raise HTTPException(status_code=500, detail=f"Autonomous mission failed: {str(e)}")

@router.get("/odin/status")
async def get_odin_status():
    """Get ODIN system status and capabilities"""
    if not ODIN_AVAILABLE:
        return {
            "odin_available": False,
            "status": "ODIN system not available - running in fallback mode",
            "capabilities": [],
            "fallback_mode": True
        }
    
    try:
        health_checks = {
            "trajectory_engine": {"available": True, "engine": "Poliastro-based"},
            "space_weather": {"available": True, "sources": ["NASA DONKI", "Space-Track"]},
            "database": {"available": get_database() is not None, "type": "MongoDB"}
        }
        
        status = odin_system.get_system_status()
        
        return {
            "odin_available": True,
            "status": status,
            "system_name": "ODIN (Optimal Dynamic Interplanetary Navigator)",
            "version": "1.0.0",
            "capabilities": [
                "🚀 Autonomous Earth-to-Moon trajectory planning with Poliastro",
                "🤖 AI-powered decision making with LangGraph + Mistral-7B", 
                "🌤️ Historical space weather analysis from NASA DONKI (2012-2018)",
                "🛰️ Orbital debris tracking from Space-Track.org",
                "🔮 ML-based hazard prediction and avoidance",
                "📝 Human-readable decision logs and explanations",
                "⚡ Dynamic replanning and route optimization",
                "🧠 Explainable AI recommendations via Hugging Face",
                "📊 Real-time mission monitoring and telemetry",
                "🗄️ Persistent mission state in MongoDB",
                "🎯 Function-calling LLM for strategic planning"
            ],
            "health_checks": health_checks,
            "ai_systems": {
                "ai_copilot": ai_copilot is not None,
                "hazard_forecaster": hazard_forecaster is not None,
                "explainer": explainer is not None
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting ODIN status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# DECISION LOGS & EXPLAINABILITY
# =============================================================================

@router.get("/odin/decision-logs")
async def get_odin_decision_logs():
    """Get ODIN decision logs with human-readable explanations"""
    if not ODIN_AVAILABLE:
        raise HTTPException(status_code=503, detail="ODIN system not available")
    
    try:
        logs = odin_system.get_decision_logs()
        return {
            "total_decisions": len(logs),
            "decision_logs": logs,
            "latest_decisions": logs[-10:] if len(logs) > 10 else logs,
            "log_format_example": "T+2.5h: HAZARD DETECTED: SOLAR_FLARE. REROUTING VIA CONSERVATIVE SAFETY ROUTE. RESULT: +8.0 HOURS TRAVEL TIME, -20% RADIATION EXPOSURE"
        }
    except Exception as e:
        logger.error(f"Error getting decision logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/odin/explainability/{decision_id}")
async def get_decision_explanation(decision_id: str):
    """Get detailed explanation for a specific ODIN decision"""
    if not ODIN_AVAILABLE:
        raise HTTPException(status_code=503, detail="ODIN explainability module not available")
    
    try:
        explanation = await explainer.explain_decision(decision_id)
        return {
            "decision_id": decision_id,
            "explanation": explanation,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting decision explanation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# SPACE WEATHER & HAZARD PREDICTION
# =============================================================================

@router.get("/space-weather/current")
async def get_current_space_weather():
    """Get current space weather conditions from historical 2012-2018 data"""
    if not ODIN_AVAILABLE:
        raise HTTPException(status_code=503, detail="Space weather service not available")
    
    try:
        weather_data = await space_weather_service.get_current_conditions()
        return {
            "timestamp": weather_data.get("timestamp"),
            "historical_period": "2012-2018",
            "space_weather": weather_data,
            "data_source": "Historical NASA/NOAA space weather archives"
        }
    except Exception as e:
        logger.error(f"Error getting space weather data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/hazards/predict")
async def predict_hazards(horizon_hours: int = 72):
    """Predict space weather hazards using ML models trained on historical data"""
    if not ODIN_AVAILABLE:
        raise HTTPException(status_code=503, detail="Hazard forecasting service not available")
    
    try:
        current_weather = await space_weather_service.get_current_conditions()
        predictions = await hazard_forecaster.predict_hazards(current_weather, horizon_hours)
        
        return {
            "prediction_horizon_hours": horizon_hours,
            "predicted_hazards": predictions.get("predicted_hazards", []),
            "confidence_scores": predictions.get("confidence_scores", {}),
            "overall_risk_level": predictions.get("overall_risk_level", "unknown"),
            "model_status": predictions.get("model_status", "unknown"),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error predicting hazards: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# AI CO-PILOT ENDPOINTS
# =============================================================================

@router.post("/ai-copilot/mission-brief")
async def generate_mission_brief(mission_time: float = 0.0):
    """Generate AI mission brief and strategic recommendations"""
    if not ODIN_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI Co-pilot not available")
    
    try:
        mission_status = {
            "mission_time": mission_time,
            "position": "Earth orbit",
            "velocity": 7.8,
            "fuel_remaining": 95.0,
            "destination": "Moon"
        }
        
        brief = await ai_copilot.generate_mission_brief(mission_status)
        return {
            "mission_brief": brief,
            "ai_engine": "LangChain + Mistral AI",
            "generation_time": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error generating mission brief: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ai-copilot/trajectory-alternatives")
async def generate_trajectory_alternatives(
    current_trajectory: Dict[str, Any],
    hazards: List[Dict[str, Any]] = [],
    constraints: Dict[str, Any] = {}
):
    """Generate alternative trajectories using AI strategic planning"""
    if not ODIN_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI Co-pilot not available")
    
    try:
        alternatives = await ai_copilot.generate_trajectory_alternatives(
            current_trajectory, hazards, constraints
        )
        return {
            "current_trajectory": current_trajectory,
            "alternative_trajectories": alternatives,
            "hazards_considered": hazards,
            "constraints": constraints,
            "ai_engine": "LangChain + Mistral AI",
            "generation_time": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error generating trajectory alternatives: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# MISSION TELEMETRY & MONITORING  
# =============================================================================

@router.get("/mission/status")
async def get_mission_status():
    """Get current mission status and telemetry"""
    if not ODIN_AVAILABLE:
        return {
            "mission_active": False,
            "status": "ODIN system not available",
            "fallback_mode": True
        }
    
    try:
        status = odin_system.get_system_status()
        return {
            "mission_active": True,
            "system_status": status,
            "autonomous_mode": True,
            "last_updated": datetime.utcnow().isoformat(),
            "navigation_system": "ODIN v1.0.0"
        }
    except Exception as e:
        logger.error(f"Error getting mission status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# SYSTEM INFORMATION
# =============================================================================

@router.get("/system/info")
async def get_system_info():
    """Get ODIN system information and configuration"""
    return {
        "system_name": "ODIN (Optimal Dynamic Interplanetary Navigator)",
        "version": "1.0.0",
        "description": "AI-powered autonomous spacecraft navigation system for Earth-to-Moon missions",
        "technologies": {
            "ai_framework": "LangChain",
            "llm_provider": "Mistral AI",
            "ml_library": "scikit-learn",
            "backend_framework": "FastAPI",
            "data_period": "2012-2018 historical space weather"
        },
        "capabilities": {
            "autonomous_navigation": True,
            "hazard_prediction": True,
            "dynamic_replanning": True,
            "explainable_ai": True,
            "historical_data_analysis": True,
            "human_readable_logs": True
        },
        "odin_available": ODIN_AVAILABLE,
        "initialization_time": datetime.utcnow().isoformat()
    }

@router.get("/")
async def root():
    """Root endpoint with ODIN system information"""
    return {
        "message": "🚀 ODIN Navigation System API",
        "description": "Optimal Dynamic Interplanetary Navigator - AI-powered autonomous spacecraft navigation",
        "version": "1.0.0",
        "endpoints": {
            "odin": "/api/odin/*",
            "space_weather": "/api/space-weather/*", 
            "hazards": "/api/hazards/*",
            "ai_copilot": "/api/ai-copilot/*",
            "mission": "/api/mission/*",
            "system": "/api/system/*"
        },
        "documentation": "/docs"
    }

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "odin_available": ODIN_AVAILABLE,
        "services": {
            "odin_system": odin_system is not None and not isinstance(odin_system, DummyService),
            "ai_copilot": ai_copilot is not None and not isinstance(ai_copilot, DummyService),
            "hazard_forecaster": hazard_forecaster is not None and not isinstance(hazard_forecaster, DummyService),
            "explainer": explainer is not None and not isinstance(explainer, DummyService),
            "space_weather_service": space_weather_service is not None and not isinstance(space_weather_service, DummyService),
            "trajectory_engine": trajectory_engine is not None and not isinstance(trajectory_engine, DummyService),
            "backend_space_service": backend_space_service is not None and not isinstance(backend_space_service, DummyService),
        }
    }
