"""
ODIN Configuration Module
Database and service configuration for the ODIN Navigation System
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from motor.motor_asyncio import AsyncIOMotorClient

class Settings(BaseSettings):
    # API Configuration
    nasa_api_key: str = "DEMO_KEY"
    huggingface_api_key: Optional[str] = None
    spacetrack_username: Optional[str] = None
    spacetrack_password: Optional[str] = None
    mistral_api_key: Optional[str] = None
    
    # Database Configuration - MongoDB for ODIN
    mongodb_url: str = "mongodb://localhost:27017"
    database_name: str = "odin_navigation"
    
    # Legacy database (keeping for compatibility)
    database_url: str = "sqlite:///./simulation.db"
    redis_url: str = "redis://localhost:6379"
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    
    # CORS Configuration
    allowed_origins: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # ODIN System Configuration
    max_mission_duration_hours: int = 168  # 7 days
    hazard_check_interval_minutes: int = 5
    replanning_threshold_delta_v: float = 1000.0  # m/s
    
    # AI Model Configuration
    llm_model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"
    huggingface_inference_url: str = "https://api-inference.huggingface.co/models"
    
    # Historical Data Range
    historical_data_start_year: int = 2012
    historical_data_end_year: int = 2018
    
    # Simulation Configuration
    simulation_update_interval: float = 1.0
    max_trajectory_points: int = 1000
    
    # Logging Configuration
    log_level: str = "INFO"
    log_file: str = "logs/simulation.log"
    
    class Config:
        env_file = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env')
        env_file_encoding = "utf-8"

# Global settings instance
settings = Settings()

# Global MongoDB client - will be initialized in lifespan
mongodb_client: Optional[AsyncIOMotorClient] = None
database = None

async def init_database():
    """Initialize MongoDB connection"""
    global mongodb_client, database
    
    try:
        mongodb_client = AsyncIOMotorClient(settings.mongodb_url)
        database = mongodb_client[settings.database_name]
        
        # Test connection
        await mongodb_client.admin.command('ping')
        print(f"✅ Connected to MongoDB: {settings.database_name}")
        
        # Create indexes for performance
        await create_indexes()
        
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
        print("🔄 Continuing without database - some features will be limited")

async def create_indexes():
    """Create database indexes for optimal performance"""
    try:
        # Mission collection indexes
        await database.missions.create_index("mission_id")
        await database.missions.create_index("start_time")
        await database.missions.create_index("status")
        
        # Hazard collection indexes
        await database.hazards.create_index("timestamp")
        await database.hazards.create_index("hazard_type")
        await database.hazards.create_index([("timestamp", 1), ("hazard_type", 1)])
        
        # Decision logs indexes
        await database.decision_logs.create_index("mission_id")
        await database.decision_logs.create_index("timestamp")
        await database.decision_logs.create_index("decision_type")
        
        # Trajectory collection indexes
        await database.trajectories.create_index("mission_id")
        await database.trajectories.create_index("trajectory_type")
        await database.trajectories.create_index("created_at")
        
        print("✅ Database indexes created successfully")
        
    except Exception as e:
        print(f"⚠️ Warning: Could not create database indexes: {e}")

async def close_database():
    """Close MongoDB connection"""
    global mongodb_client
    if mongodb_client:
        mongodb_client.close()
        print("🔒 Database connection closed")

def get_database():
    """Get database instance"""
    return database
