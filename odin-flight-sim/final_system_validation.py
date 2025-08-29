#!/usr/bin/env python3
"""
Final System Validation Script for ODIN Flight Simulation System
Tests all major components and integration points
"""

import requests
import json
import time
import asyncio
import websockets
from datetime import datetime

def test_backend_endpoints():
    """Test all major backend API endpoints"""
    base_url = "http://localhost:8000"
    
    endpoints = [
        ("/health", "Health Check"),
        ("/api/system/info", "System Information"),
        ("/api/system/status", "System Status"),
        ("/api/ai/status", "AI Services Status"),
        ("/api/space-weather/current", "Space Weather Data"),
        ("/api/hazards/current", "Current Hazards"),
        ("/api/mission/status", "Mission Status"),
    ]
    
    print("🧪 Testing Backend API Endpoints...")
    print("=" * 50)
    
    results = {}
    for endpoint, description in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            if response.status_code == 200:
                print(f"✅ {description}: {endpoint} - Status: {response.status_code}")
                results[endpoint] = {"status": "SUCCESS", "code": response.status_code}
            else:
                print(f"⚠️  {description}: {endpoint} - Status: {response.status_code}")
                results[endpoint] = {"status": "WARNING", "code": response.status_code}
        except Exception as e:
            print(f"❌ {description}: {endpoint} - Error: {str(e)}")
            results[endpoint] = {"status": "ERROR", "error": str(e)}
    
    return results

def test_frontend_access():
    """Test frontend accessibility"""
    frontend_url = "http://localhost:8081"
    
    print("\n🌐 Testing Frontend Access...")
    print("=" * 30)
    
    try:
        response = requests.get(frontend_url, timeout=10)
        if response.status_code == 200:
            print(f"✅ Frontend accessible at {frontend_url}")
            return True
        else:
            print(f"⚠️  Frontend returned status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Frontend access error: {str(e)}")
        return False

async def test_websocket_connection():
    """Test WebSocket connection"""
    ws_url = "ws://localhost:8000/ws"
    
    print("\n🔌 Testing WebSocket Connection...")
    print("=" * 35)
    
    try:
        async with websockets.connect(ws_url) as websocket:
            print(f"✅ WebSocket connected to {ws_url}")
            
            # Send a test message
            test_message = {"type": "ping", "timestamp": datetime.now().isoformat()}
            await websocket.send(json.dumps(test_message))
            print("✅ Test message sent")
            
            # Wait for response (with timeout)
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                print(f"✅ WebSocket response received: {response[:100]}...")
                return True
            except asyncio.TimeoutError:
                print("⚠️  WebSocket response timeout (but connection established)")
                return True
                
    except Exception as e:
        print(f"❌ WebSocket connection error: {str(e)}")
        return False

def test_ai_services():
    """Test AI services functionality"""
    base_url = "http://localhost:8000"
    
    print("\n🧠 Testing AI Services...")
    print("=" * 25)
    
    # Test AI status
    try:
        response = requests.get(f"{base_url}/api/ai/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ AI Services Status:")
            for service, status in data.items():
                status_icon = "✅" if status else "❌"
                print(f"  {status_icon} {service}: {status}")
            return True
        else:
            print(f"⚠️  AI status check returned: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ AI services test error: {str(e)}")
        return False

def test_space_weather_service():
    """Test space weather data generation"""
    base_url = "http://localhost:8000"
    
    print("\n🌤️  Testing Space Weather Service...")
    print("=" * 35)
    
    try:
        response = requests.get(f"{base_url}/api/space-weather/current", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Space Weather Data Retrieved:")
            print(f"  📅 Timestamp: {data.get('timestamp', 'N/A')}")
            print(f"  📊 Historical Period: {data.get('historical_period', 'N/A')}")
            
            weather = data.get('space_weather', {})
            if weather:
                solar = weather.get('solar_activity', {})
                print(f"  ☀️  Solar Flux: {solar.get('solar_flux_10_7cm', 'N/A')}")
                print(f"  🌟 Sunspot Number: {solar.get('sunspot_number', 'N/A')}")
            return True
        else:
            print(f"⚠️  Space weather returned: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Space weather test error: {str(e)}")
        return False

def main():
    """Run complete system validation"""
    print("🚀 ODIN Flight Simulation System - Final Validation")
    print("=" * 60)
    print(f"⏰ Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run all tests
    backend_results = test_backend_endpoints()
    frontend_accessible = test_frontend_access()
    ai_services_working = test_ai_services()
    space_weather_working = test_space_weather_service()
    
    # WebSocket test (async)
    websocket_working = asyncio.run(test_websocket_connection())
    
    # Summary
    print("\n📋 VALIDATION SUMMARY")
    print("=" * 25)
    
    total_endpoints = len(backend_results)
    successful_endpoints = sum(1 for r in backend_results.values() if r["status"] == "SUCCESS")
    
    print(f"🔗 Backend Endpoints: {successful_endpoints}/{total_endpoints} working")
    print(f"🌐 Frontend Access: {'✅ Working' if frontend_accessible else '❌ Failed'}")
    print(f"🔌 WebSocket Connection: {'✅ Working' if websocket_working else '❌ Failed'}")
    print(f"🧠 AI Services: {'✅ Working' if ai_services_working else '❌ Failed'}")
    print(f"🌤️  Space Weather: {'✅ Working' if space_weather_working else '❌ Failed'}")
    
    # Overall status
    all_working = (
        successful_endpoints == total_endpoints and
        frontend_accessible and
        websocket_working and
        ai_services_working and
        space_weather_working
    )
    
    print("\n🎯 OVERALL SYSTEM STATUS")
    print("=" * 30)
    if all_working:
        print("🎉 ✅ ALL SYSTEMS OPERATIONAL!")
        print("🚀 ODIN Flight Simulation System is ready for use")
    else:
        print("⚠️  Some components need attention")
        print("📝 Review the detailed test results above")
    
    print(f"\n⏰ Test Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
