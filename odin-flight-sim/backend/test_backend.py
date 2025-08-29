#!/usr/bin/env python3
"""
Test script for Odin Flight Simulation Backend
"""

import asyncio
import aiohttp
import json
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_backend():
    """Test backend endpoints"""
    base_url = "http://localhost:8000"
    
    async with aiohttp.ClientSession() as session:
        print("🧪 Testing Odin Flight Simulation Backend...")
        
        # Test health endpoint
        try:
            async with session.get(f"{base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Health check: {data}")
                else:
                    print(f"❌ Health check failed: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Cannot connect to backend: {e}")
            print("   Make sure the backend is running with: python run_backend.py")
            return False
        
        # Test mission status endpoint
        try:
            async with session.get(f"{base_url}/api/mission/status") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Mission status: {data.get('status', 'unknown')}")
                else:
                    print(f"❌ Mission status failed: {response.status}")
        except Exception as e:
            print(f"❌ Mission status error: {e}")
        
        # Test telemetry endpoint
        try:
            async with session.get(f"{base_url}/api/telemetry") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Telemetry: Position {data.get('spacecraft_position', 'unknown')}")
                else:
                    print(f"❌ Telemetry failed: {response.status}")
        except Exception as e:
            print(f"❌ Telemetry error: {e}")
        
        # Test trajectory endpoint
        try:
            async with session.get(f"{base_url}/api/trajectory/current") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Current trajectory: {data.get('name', 'unknown')}")
                else:
                    print(f"❌ Trajectory failed: {response.status}")
        except Exception as e:
            print(f"❌ Trajectory error: {e}")
        
        print("\n🎯 Backend test completed!")
        return True

def test_imports():
    """Test if all required modules can be imported"""
    print("📦 Testing imports...")
    
    modules_to_test = [
        "fastapi",
        "uvicorn", 
        "pydantic",
        "numpy",
        "aiohttp",
        "asyncio"
    ]
    
    failed_imports = []
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("   Install missing packages with: pip install -r requirements.txt")
        return False
    
    print("✅ All required modules available!")
    return True

async def main():
    """Main test function"""
    print("🔬 Odin Flight Simulation Backend Test Suite\n")
    
    # Test imports first
    if not test_imports():
        sys.exit(1)
    
    print("\n" + "="*50 + "\n")
    
    # Test backend endpoints
    await test_backend()

if __name__ == "__main__":
    asyncio.run(main())
