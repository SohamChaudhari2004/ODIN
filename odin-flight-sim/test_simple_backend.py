#!/usr/bin/env python3
"""
Simple Backend Test
Test basic FastAPI functionality without complex dependencies
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Create simple FastAPI app
app = FastAPI(
    title="ODIN Test API",
    description="Simple test for ODIN backend functionality",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "ODIN Test Backend",
        "description": "Simple test backend for ODIN",
        "version": "1.0.0"
    }

@app.get("/api/test")
async def test_endpoint():
    return {"message": "ODIN backend is working!", "status": "success"}

if __name__ == "__main__":
    print("Starting ODIN Test Backend...")
    uvicorn.run(
        "test_simple_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
