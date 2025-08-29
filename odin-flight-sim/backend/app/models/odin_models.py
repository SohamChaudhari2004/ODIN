"""
MongoDB Data Models for ODIN Navigation System
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum

class MissionStatus(str, Enum):
    """Mission status enumeration"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    COMPLETED = "completed"
    ABORTED = "aborted"
    REPLANNING = "replanning"

class HazardType(str, Enum):
    """Types of space hazards"""
    SOLAR_FLARE = "solar_flare"
    CME = "cme"  # Coronal Mass Ejection
    DEBRIS = "debris"
    GEOMAGNETIC_STORM = "geomagnetic_storm"
    RADIATION_STORM = "radiation_storm"

class DecisionType(str, Enum):
    """Types of ODIN decisions"""
    TRAJECTORY_PLANNING = "trajectory_planning"
    HAZARD_AVOIDANCE = "hazard_avoidance"
    REPLANNING = "replanning"
    MANEUVER_EXECUTION = "maneuver_execution"
    MISSION_ABORT = "mission_abort"

# MongoDB Document Models

class MissionDocument(BaseModel):
    """Mission state document for MongoDB"""
    mission_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: MissionStatus
    destination: str
    historical_timestamp: datetime
    spacecraft_position: List[float]
    spacecraft_velocity: List[float]
    fuel_remaining: float
    current_trajectory_id: Optional[str] = None
    active_hazards: List[str] = []
    mission_constraints: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class HazardDocument(BaseModel):
    """Space hazard document for MongoDB"""
    hazard_id: str
    hazard_type: HazardType
    timestamp: datetime
    historical_timestamp: datetime
    severity: float  # 0.0 to 1.0
    location: Dict[str, Any]  # Geometric zone data
    duration_hours: float
    impact_radius_km: float
    source_data: Dict[str, Any]  # Original NASA/Space-Track data
    affected_trajectories: List[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)

class DecisionLogDocument(BaseModel):
    """Decision log document for MongoDB"""
    decision_id: str
    mission_id: str
    timestamp: datetime
    decision_type: DecisionType
    threat_analysis: Dict[str, Any]
    options_evaluated: List[Dict[str, Any]]
    decision_rationale: str
    chosen_option: Dict[str, Any]
    execution_status: str
    impact_metrics: Dict[str, Any]
    explainability_data: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.utcnow)

class TrajectoryDocument(BaseModel):
    """Trajectory document for MongoDB"""
    trajectory_id: str
    mission_id: str
    trajectory_type: str  # "baseline", "alternative", "optimized"
    name: str
    description: str
    maneuvers: List[Dict[str, Any]]
    total_delta_v: float
    duration_hours: float
    radiation_exposure: float
    collision_risk: float
    safety_score: float
    fuel_efficiency: float
    waypoints: List[Dict[str, Any]]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = False

# API Request/Response Models

class InitializeMissionRequest(BaseModel):
    """Request model for mission initialization"""
    destination: str = "Moon"
    historical_date: Optional[datetime] = None
    mission_constraints: Optional[Dict[str, Any]] = None

class MissionStatusResponse(BaseModel):
    """Response model for mission status"""
    mission_id: str
    status: MissionStatus
    current_time: datetime
    spacecraft_position: List[float]
    spacecraft_velocity: List[float]
    fuel_remaining: float
    active_hazards: List[Dict[str, Any]]
    current_trajectory: Optional[Dict[str, Any]]
    next_maneuver: Optional[Dict[str, Any]]

class HazardForecastRequest(BaseModel):
    """Request model for hazard forecasting"""
    mission_id: str
    forecast_hours: int = 24
    hazard_types: Optional[List[HazardType]] = None

class TrajectoryCalculationRequest(BaseModel):
    """Request model for trajectory calculations"""
    mission_id: str
    strategy_prompt: Optional[str] = None
    constraints: Optional[Dict[str, Any]] = None
    optimization_criteria: List[str] = ["safety", "fuel_efficiency", "time"]

class ExplainabilityRequest(BaseModel):
    """Request model for decision explanation"""
    decision_id: str
    explanation_type: str = "detailed"  # "brief", "detailed", "technical"

# Space Weather Data Models

class SolarFlareData(BaseModel):
    """Solar flare data structure"""
    flare_class: str  # X, M, C, B, A
    magnitude: float
    start_time: datetime
    peak_time: datetime
    end_time: datetime
    location: Dict[str, float]  # longitude, latitude
    source_region: str
    impact_probability: float

class CMEData(BaseModel):
    """Coronal Mass Ejection data structure"""
    speed_km_s: float
    direction: Dict[str, float]  # longitude, latitude, angle
    launch_time: datetime
    arrival_time_estimate: datetime
    impact_probability: float
    magnetic_field_strength: float

class SpaceDebrisData(BaseModel):
    """Space debris data structure"""
    object_id: str
    tle_line1: str
    tle_line2: str
    orbit_epoch: datetime
    collision_probability: float
    closest_approach_time: datetime
    miss_distance_km: float
    object_size_m: float
