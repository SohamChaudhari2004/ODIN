from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
import logging
import sys
import os
import uuid
import importlib.util

# Add AI services to path and handle imports dynamically
import importlib.util
ai_services_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ai-services')
if ai_services_path not in sys.path:
    sys.path.insert(0, ai_services_path)

# Initialize ODIN service classes
ODIN_AVAILABLE = False
OdinNavigationSystem = None
AICoPilot = None
PredictiveHazardForecasting = None
ExplainabilityModule = None
SpaceWeatherDataService = None
HuggingFaceLLMService = None
OdinDecisionEngine = None

def load_ai_service(module_name, class_name):
    """Dynamically load AI service classes"""
    try:
        module_path = os.path.join(ai_services_path, f"{module_name}.py")
        if os.path.exists(module_path):
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return getattr(module, class_name)
        return None
    except Exception as e:
        print(f"Failed to load {module_name}.{class_name}: {e}")
        return None

# Load AI services
try:
    OdinNavigationSystem = load_ai_service("odin_main", "OdinNavigationSystem")
    AICoPilot = load_ai_service("ai_copilot", "AICoPilot")
    PredictiveHazardForecasting = load_ai_service("predictive_hazard_forecasting", "PredictiveHazardForecasting")
    ExplainabilityModule = load_ai_service("explainability_module", "ExplainabilityModule")
    SpaceWeatherDataService = load_ai_service("space_weather_service", "SpaceWeatherDataService")
    HuggingFaceLLMService = load_ai_service("huggingface_llm", "HuggingFaceLLMService")
    OdinDecisionEngine = load_ai_service("langgraph_agent", "OdinDecisionEngine")
    
    # Check if core services loaded
    if OdinNavigationSystem and AICoPilot:
        ODIN_AVAILABLE = True
        print("✅ ODIN AI services loaded successfully")
    else:
        print("❌ Core ODIN services failed to load")
        
except Exception as e:
    print(f"❌ Error loading ODIN AI services: {e}")

# Create fallback dummy service if needed
if not ODIN_AVAILABLE:
    print("   Continuing in fallback mode...")
    
    class DummyService:
        def __init__(self, *args, **kwargs):
            self.available = False
            
        async def __aenter__(self):
            return self
            
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
            
        def __getattr__(self, name):
            async def dummy_async(*args, **kwargs):
                return {"error": "Service not available", "fallback": True}
            def dummy_sync(*args, **kwargs):
                return {"error": "Service not available", "fallback": True}
            return dummy_async if name.startswith(('get_', 'generate_', 'predict_', 'initialize_', 'calculate_')) else dummy_sync
    
    if not OdinNavigationSystem:
        OdinNavigationSystem = DummyService
    if not AICoPilot:
        AICoPilot = DummyService
    if not PredictiveHazardForecasting:
        PredictiveHazardForecasting = DummyService
    if not ExplainabilityModule:
        ExplainabilityModule = DummyService
    if not SpaceWeatherDataService:
        SpaceWeatherDataService = DummyService
    if not HuggingFaceLLMService:
        HuggingFaceLLMService = DummyService
    if not OdinDecisionEngine:
        OdinDecisionEngine = DummyService

# Import backend services
from ..services.trajectory_engine import TrajectoryEngine
from ..services.space_data_service import SpaceWeatherDataService as BackendSpaceWeatherService
from ..models.odin_models import *
from ..config import get_database

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize ODIN system components globally
odin_system = None
ai_copilot = None
hazard_forecaster = None
explainer = None
space_weather_service = None
llm_service = None
decision_engine = None
trajectory_engine = None
backend_space_service = None

async def initialize_odin_services():
    """Initialize all ODIN services asynchronously"""
    global odin_system, ai_copilot, hazard_forecaster, explainer
    global space_weather_service, llm_service, decision_engine
    global trajectory_engine, backend_space_service
    
    if ODIN_AVAILABLE and odin_system is None:
        try:
            # Initialize AI services
            odin_system = OdinNavigationSystem()
            ai_copilot = AICoPilot()
            hazard_forecaster = PredictiveHazardForecasting()
            explainer = ExplainabilityModule()
            space_weather_service = SpaceWeatherDataService()
            llm_service = HuggingFaceLLMService()
            decision_engine = OdinDecisionEngine()
            
            # Initialize backend services
            trajectory_engine = TrajectoryEngine()
            backend_space_service = BackendSpaceWeatherService()
            
            logger.info("🚀 ODIN Navigation System initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize ODIN system: {e}")
            return False
    return ODIN_AVAILABLE

# =============================================================================
# ODIN CORE ENDPOINTS
# =============================================================================

@router.post("/odin/initialize")
async def initialize_odin_mission(request: InitializeMissionRequest):
    """Initialize ODIN autonomous navigation system with historical space weather data"""
    # Initialize services if not already done
    await initialize_odin_services()
    
    if not ODIN_AVAILABLE or not odin_system:
        raise HTTPException(status_code=503, detail="ODIN system not available")
    
    try:
        # Generate unique mission ID
        mission_id = f"odin_mission_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        # Initialize with historical timestamp
        historical_date = request.historical_date or datetime(2015, 3, 17, 12, 0, 0)  # Known solar storm date
        
        # Create mission document
        mission_doc = MissionDocument(
            mission_id=mission_id,
            start_time=datetime.utcnow(),
            status=MissionStatus.INITIALIZING,
            destination=request.destination,
            historical_timestamp=historical_date,
            spacecraft_position=[7000.0, 0.0, 0.0],  # LEO
            spacecraft_velocity=[0.0, 7.8, 0.0],     # km/s
            fuel_remaining=85.0,
            mission_constraints=request.mission_constraints or {
                "max_delta_v": 15000,
                "max_duration": 120,
                "min_safety_score": 0.7
            }
        )
        
        # Store in database
        db = get_database()
        if db is not None:
            await db.missions.insert_one(mission_doc.dict())
        
        # Initialize trajectory
        if trajectory_engine:
            initial_trajectory = await trajectory_engine.calculate_initial_trajectory(
                historical_date, request.destination
            )
            
            # Store trajectory
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
        
        # Initialize ODIN system
        result = await odin_system.initialize_mission(request.destination)
        
        return {
            "success": True,
            "mission_id": mission_id,
            "message": "ODIN mission initialized successfully",
            "historical_timestamp": historical_date.isoformat(),
            "initial_trajectory": initial_trajectory.trajectory_id if trajectory_engine else "baseline",
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
    if not ODIN_AVAILABLE or not decision_engine:
        raise HTTPException(status_code=503, detail="ODIN decision engine not available")
    
    try:
        # Run continuous autonomous monitoring
        mission_events = await decision_engine.continuous_monitoring(
            mission_id, duration_hours, check_interval_minutes=5
        )
        
        # Update mission status
        db = get_database()
        if db is not None:
            await db.missions.update_one(
                {"mission_id": mission_id},
                {"$set": {"status": MissionStatus.COMPLETED, "end_time": datetime.utcnow()}}
            )
        
        # Generate summary
        decision_summary = await decision_engine.get_decision_summary(mission_id)
        
        return {
            "success": True,
            "mission_id": mission_id,
            "message": f"Autonomous mission completed - {duration_hours} hours",
            "mission_duration": duration_hours,
            "mission_events": mission_events,
            "decision_summary": decision_summary,
            "performance_metrics": {
                "total_events": len(mission_events),
                "autonomous_decisions": len([e for e in mission_events if e.get("event_type") == "autonomous_decision"]),
                "hazards_detected": len([e for e in mission_events if "HAZARD" in str(e.get("description", ""))]),
                "system_errors": len([e for e in mission_events if e.get("error", False)])
            }
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
        # Health check all services
        health_checks = {}
        
        if llm_service:
            health_checks["llm_service"] = await llm_service.health_check()
        
        if trajectory_engine:
            health_checks["trajectory_engine"] = {"available": True, "engine": "Poliastro-based"}
        
        if backend_space_service:
            health_checks["space_weather"] = {"available": True, "sources": ["NASA DONKI", "Space-Track"]}
        
        if decision_engine:
            health_checks["decision_engine"] = {"available": True, "framework": "LangGraph"}
        
        # Database status
        db = get_database()
        health_checks["database"] = {"available": db is not None, "type": "MongoDB"}
        
        status = odin_system.get_system_status() if odin_system else "System operational"
        
        return {
            "odin_available": True,
            "status": status,
            "system_name": "ODIN (Optimal Dynamic Interplanetary Navigator)",
            "version": "1.0.0",
            "capabilities": [
                "🚀 Autonomous Earth-to-Moon trajectory planning with Poliastro",
                "🤖 AI-powered decision making with LangGraph + Mistral-7B", 
                "🌤️ Historical space weather analysis from NASA DONKI (2012-2018)",
                "�️ Orbital debris tracking from Space-Track.org",
                "�🔮 ML-based hazard prediction and avoidance",
                "📝 Human-readable decision logs and explanations",
                "⚡ Dynamic replanning and route optimization",
                "🧠 Explainable AI recommendations via Hugging Face",
                "📊 Real-time mission monitoring and telemetry",
                "🗄️ Persistent mission state in MongoDB",
                "🎯 Function-calling LLM for strategic planning"
            ],
            "health_checks": health_checks,
            "ai_systems": {
                "llm_service": llm_service is not None,
                "decision_engine": decision_engine is not None,
                "trajectory_engine": trajectory_engine is not None,
                "space_weather": backend_space_service is not None,
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
    if not ODIN_AVAILABLE or not odin_system:
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
    if not ODIN_AVAILABLE or not explainer:
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
    if not ODIN_AVAILABLE or not space_weather_service:
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
    if not ODIN_AVAILABLE or not hazard_forecaster:
        raise HTTPException(status_code=503, detail="Hazard forecasting service not available")
    
    try:
        # Get current space weather for prediction
        current_weather = await space_weather_service.get_current_conditions()
        
        # Generate hazard predictions
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
    if not ODIN_AVAILABLE or not ai_copilot:
        raise HTTPException(status_code=503, detail="AI Co-pilot not available")
    
    try:
        # Create mock mission status for brief generation
        mission_status = {
            "mission_time": mission_time,
            "position": "Earth orbit",
            "velocity": 7.8,  # km/s
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
    if not ODIN_AVAILABLE or not ai_copilot:
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
    if not ODIN_AVAILABLE or not odin_system:
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
