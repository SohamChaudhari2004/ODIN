"""
ODIN Configuration Module - Fixed
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
    
    # Database Configuration - Local MongoDB with fallback
    mongodb_url: str = "mongodb://localhost:27017"
    database_name: str = "odin_navigation"
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    
    # CORS Configuration
    allowed_origins: List[str] = [
        "http://localhost:3000", 
        "http://localhost:5173",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080"
    ]
    
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
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global settings instance
settings = Settings()

# Global MongoDB client - will be initialized in lifespan
mongodb_client: Optional[AsyncIOMotorClient] = None
database = None
USE_DATABASE = False

async def init_database():
    """Initialize MongoDB connection with fallback"""
    global mongodb_client, database, USE_DATABASE
    
    try:
        # Try to connect to local MongoDB first
        mongodb_client = AsyncIOMotorClient(settings.mongodb_url, serverSelectionTimeoutMS=5000)
        database = mongodb_client[settings.database_name]
        
        # Test connection
        await mongodb_client.admin.command('ping')
        print(f"✅ Connected to MongoDB: {settings.database_name}")
        USE_DATABASE = True
        
        # Create indexes for performance
        await create_indexes()
        
    except Exception as e:
        print(f"⚠️ MongoDB not available: {e}")
        print("🔄 Running in offline mode - using in-memory storage")
        USE_DATABASE = False
        # Initialize in-memory storage as fallback
        init_memory_storage()

def init_memory_storage():
    """Initialize in-memory storage as database fallback"""
    global database
    database = {
        'missions': [],
        'hazards': [],
        'decision_logs': [],
        'trajectories': [],
        'space_weather': []
    }
    print("✅ In-memory storage initialized")

async def create_indexes():
    """Create database indexes for optimal performance"""
    if not USE_DATABASE:
        return
        
    try:
        # Mission collection indexes
        await database.missions.create_index("mission_id")
        await database.missions.create_index("start_time")
        await database.missions.create_index("status")
        
        # Hazard collection indexes
        await database.hazards.create_index("timestamp")
        await database.hazards.create_index("hazard_type")
        
        # Decision logs indexes
        await database.decision_logs.create_index("mission_id")
        await database.decision_logs.create_index("timestamp")
        
        # Trajectory collection indexes
        await database.trajectories.create_index("mission_id")
        await database.trajectories.create_index("trajectory_type")
        
        print("✅ Database indexes created successfully")
        
    except Exception as e:
        print(f"⚠️ Warning: Could not create database indexes: {e}")

async def close_database():
    """Close MongoDB connection"""
    global mongodb_client
    if mongodb_client and USE_DATABASE:
        mongodb_client.close()
        print("🔒 Database connection closed")

def get_database():
    """Get database instance"""
    return database

def is_database_available():
    """Check if database is available"""
    return USE_DATABASE
