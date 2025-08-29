from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import json
from typing import Dict, List
import uvicorn
import sys
import os

# Add AI services to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ai-services'))

from app.api.routes import router as api_router
from app.websocket.simulation_ws import SimulationWebSocket
from app.config import init_database, close_database

# Global websocket manager
websocket_manager = SimulationWebSocket()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting ODIN (Optimal Dynamic Interplanetary Navigator) Backend...")
    print("🚀 Autonomous AI-powered spacecraft navigation system")
    print("📊 Using LangGraph + Mistral AI for trajectory planning")
    print("🌤️ Historical space weather data (2012-2018)")
    print("🗄️ MongoDB for mission state and decision logs")
    print("🔧 Poliastro for orbital mechanics calculations")
    
    # Initialize database
    await init_database()
    
    # Initialize ODIN services
    from app.api.routes import initialize_odin_services
    try:
        await initialize_odin_services()
        print("✅ ODIN services initialized successfully")
    except Exception as e:
        print(f"⚠️ ODIN services initialization failed: {e}")
    
    # Start background tasks
    asyncio.create_task(websocket_manager.simulation_loop())
    
    yield
    
    # Shutdown
    print("Shutting down ODIN system...")
    await close_database()

# Create FastAPI app
app = FastAPI(
    title="ODIN Navigation API",
    description="Optimal Dynamic Interplanetary Navigator - AI-powered autonomous spacecraft navigation system for Earth-to-Moon missions using historical space weather data and LangGraph/Mistral AI",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:5173", 
        "http://localhost:8080", 
        "http://localhost:8081",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:8081"
    ],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api")

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle ODIN system messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types for ODIN system
            if message.get("type") == "odin_status_request":
                # Get ODIN system status via API routes
                await websocket_manager.broadcast_update({
                    "type": "odin_status", 
                    "status": "operational"
                })
            elif message.get("type") == "mission_update_request":
                # Mission updates will be handled via ODIN system
                await websocket_manager.broadcast_update({
                    "type": "mission_update",
                    "message": "ODIN system monitoring active"
                })
                
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "ODIN Navigation System",
        "description": "Optimal Dynamic Interplanetary Navigator - AI-powered autonomous spacecraft navigation",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )