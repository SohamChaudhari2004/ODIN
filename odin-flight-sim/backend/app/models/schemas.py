from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# =============================================================================
# ODIN MISSION SCHEMAS
# =============================================================================

class OdinMissionConfig(BaseModel):
    """Configuration for ODIN mission initialization"""
    destination: str = Field(default="Moon", description="Mission destination")
    departure_time: Optional[str] = Field(None, description="Planned departure time (ISO format)")
    max_duration_hours: float = Field(default=120.0, description="Maximum mission duration in hours")
    fuel_budget_delta_v: float = Field(default=15000.0, description="Available delta-v budget in m/s")
    max_radiation_exposure: float = Field(default=50.0, description="Maximum allowable radiation exposure")

class OdinMissionStatus(BaseModel):
    """Current ODIN mission status"""
    mission_active: bool
    current_phase: str
    mission_time_hours: float
    destination: str
    autonomous_mode: bool
    system_health: str
    last_decision_time: Optional[str] = None

# =============================================================================
# TRAJECTORY SCHEMAS  
# =============================================================================

class TrajectoryPoint(BaseModel):
    """Single point along a trajectory"""
    time_hours: float
    position_km: List[float] = Field(..., description="[x, y, z] position in km")
    velocity_km_s: List[float] = Field(..., description="[vx, vy, vz] velocity in km/s")

class TrajectoryPlan(BaseModel):
    """Complete trajectory plan with metadata"""
    plan_id: str
    name: str
    trajectory_type: str = Field(..., description="e.g., 'direct', 'bi-elliptic', 'lunar_gravity_assist'")
    departure_time: str
    arrival_time: str
    total_delta_v_ms: float = Field(..., description="Total delta-v requirement in m/s")
    duration_hours: float
    points: List[TrajectoryPoint]
    risk_assessment: Dict[str, float]
    fuel_efficiency_score: float
    radiation_exposure_estimate: float

# =============================================================================
# SPACE WEATHER & HAZARDS
# =============================================================================

class SpaceWeatherConditions(BaseModel):
    """Current space weather conditions from historical 2012-2018 data"""
    timestamp: str
    historical_date: str = Field(..., description="Original date from 2012-2018 period")
    solar_activity: Dict[str, float]
    geomagnetic_activity: Dict[str, float] 
    solar_wind: Dict[str, float]
    radiation_environment: Dict[str, float]
    active_events: List[str]

class HazardPrediction(BaseModel):
    """ML-based hazard prediction"""
    hazard_type: str
    probability: float = Field(..., ge=0.0, le=1.0)
    severity_level: str = Field(..., description="low, moderate, high")
    time_to_peak_hours: float
    duration_hours: float
    confidence_score: float
    recommended_actions: List[str]

# =============================================================================
# AI CO-PILOT SCHEMAS
# =============================================================================

class AICopilotRequest(BaseModel):
    """Request for AI co-pilot analysis"""
    mission_context: Dict[str, Any]
    current_conditions: Dict[str, Any]
    request_type: str = Field(..., description="mission_brief, trajectory_analysis, hazard_response")

class AICopilotResponse(BaseModel):
    """AI co-pilot response with recommendations"""
    analysis_type: str
    recommendations: List[str]
    reasoning: str
    confidence_level: str
    alternative_options: List[Dict[str, Any]]
    ai_engine: str = "LangChain + Mistral AI"

# =============================================================================
# TELEMETRY & MONITORING
# =============================================================================

class TelemetryData(BaseModel):
    """Spacecraft telemetry snapshot"""
    timestamp: str
    mission_time_hours: float
    spacecraft_position_km: List[float]
    spacecraft_velocity_km_s: List[float]
    fuel_remaining_percent: float
    power_generation_watts: float
    radiation_exposure_rate: float
    communication_link_quality: float
    system_health_status: str

class MissionStatus(BaseModel):
    """Overall mission status and health"""
    mission_id: str
    status: str = Field(..., description="active, completed, aborted, planning")
    progress_percent: float
    current_phase: str
    estimated_completion_time: str
    key_metrics: Dict[str, float]
    active_alerts: List[str]
    last_updated: str

# =============================================================================
# DECISION LOGS & EXPLAINABILITY  
# =============================================================================

class DecisionLog(BaseModel):
    """Human-readable decision log entry"""
    timestamp: str
    mission_time: str
    decision_type: str
    decision_text: str = Field(..., description="Human-readable decision description")
    context: Dict[str, Any]
    outcome_metrics: Dict[str, float]

class ExplanationRequest(BaseModel):
    """Request for decision explanation"""
    decision_id: str
    detail_level: str = Field(default="standard", description="basic, standard, detailed")

class DecisionExplanation(BaseModel):
    """Detailed explanation of an AI decision"""
    decision_id: str
    decision_summary: str
    factors_considered: List[Dict[str, Any]]
    alternative_options: List[Dict[str, Any]]
    risk_assessment: Dict[str, float]
    trade_offs_analysis: str
    confidence_metrics: Dict[str, float]