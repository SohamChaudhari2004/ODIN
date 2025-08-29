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

# =============================================================================
# ODIN CORE ENDPOINTS
# =============================================================================

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
            "data": result,
            "system_info": {
                "name": "ODIN (Optimal Dynamic Interplanetary Navigator)",
                "version": "1.0.0",
                "destination": destination,
                "ai_engine": "LangChain + Mistral AI",
                "data_source": "Historical space weather (2012-2018)"
            }
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
            "mission_duration": duration_hours,
            "mission_events": mission_events,
            "decision_logs": odin_system.get_decision_logs(),
            "performance_metrics": {
                "total_decisions": len(odin_system.get_decision_logs()),
                "hazards_detected": len([e for e in mission_events if "HAZARD" in str(e)]),
                "replanning_events": len([e for e in mission_events if "REROUTING" in str(e)])
            }
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
            "capabilities": [],
            "fallback_mode": True
        }
    
    try:
        status = odin_system.get_system_status()
        return {
            "odin_available": True,
            "status": status,
            "system_name": "ODIN (Optimal Dynamic Interplanetary Navigator)",
            "version": "1.0.0",
            "capabilities": [
                "🚀 Autonomous Earth-to-Moon trajectory planning",
                "🤖 AI-powered decision making with LangChain + Mistral AI", 
                "🌤️ Historical space weather analysis (2012-2018)",
                "🔮 ML-based hazard prediction and avoidance",
                "📝 Human-readable decision logs and explanations",
                "⚡ Dynamic replanning and route optimization",
                "🧠 Explainable AI recommendations",
                "📊 Real-time mission monitoring and telemetry"
            ],
            "ai_systems": {
                "ai_copilot": ai_copilot is not None,
                "hazard_forecaster": hazard_forecaster is not None,
                "explainer": explainer is not None,
                "space_weather": space_weather_service is not None
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
