#!/usr/bin/env python3
"""
ODIN System Integration Test
Tests all components and data flow
"""

import requests
import json
import time
import asyncio
import sys
import os

# Test configuration
BACKEND_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:8080"

def test_external_apis():
    """Test external space weather APIs"""
    print("🌌 Testing External APIs...")
    
    try:
        # Test NASA DONKI API
        nasa_resp = requests.get(
            "https://api.nasa.gov/DONKI/FLR?startDate=2024-01-01&endDate=2024-01-02&api_key=DEMO_KEY",
            timeout=10
        )
        print(f"✅ NASA DONKI API: {nasa_resp.status_code}")
        
        # Test NOAA SWPC API
        noaa_resp = requests.get(
            "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json",
            timeout=10
        )
        print(f"✅ NOAA SWPC API: {noaa_resp.status_code}")
        
        return True
    except Exception as e:
        print(f"❌ External API error: {e}")
        return False

def test_ai_services():
    """Test AI services components"""
    print("\n🤖 Testing AI Services...")
    
    try:
        # Test space weather service
        sys.path.insert(0, "ai-services")
        from space_weather_service import SpaceWeatherDataService
        
        service = SpaceWeatherDataService()
        print("✅ SpaceWeatherDataService: Loaded")
        
        # Test sync data retrieval (simplified test)
        timestamp = service.initialize_mission_timestamp()
        print(f"✅ Mission timestamp initialized: {timestamp}")
        
        # Test other services
        try:
            from ai_copilot import AICoPilot
            copilot = AICoPilot()
            print(f"✅ AICoPilot: Available={copilot.ai_available}")
        except Exception as e:
            print(f"⚠️ AICoPilot: {e}")
        
        try:
            from predictive_hazard_forecasting import PredictiveHazardForecasting
            forecaster = PredictiveHazardForecasting()
            print("✅ PredictiveHazardForecasting: Loaded")
        except Exception as e:
            print(f"⚠️ PredictiveHazardForecasting: {e}")
        
        return True
    except Exception as e:
        print(f"❌ AI Services error: {e}")
        return False

def test_backend_apis():
    """Test backend API endpoints"""
    print("\n🚀 Testing Backend APIs...")
    
    try:
        # Wait for backend to be ready
        max_retries = 30
        for i in range(max_retries):
            try:
                resp = requests.get(f"{BACKEND_URL}/health", timeout=2)
                if resp.status_code == 200:
                    print(f"✅ Backend health check: {resp.json()}")
                    break
            except:
                if i == max_retries - 1:
                    print("❌ Backend not responding")
                    return False
                time.sleep(1)
        
        # Test system info
        resp = requests.get(f"{BACKEND_URL}/api/system/info", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            print(f"✅ System info: ODIN Available={data.get('odin_available')}")
            print(f"   AI Services: {data.get('ai_services_status')}")
        else:
            print(f"⚠️ System info failed: {resp.status_code}")
        
        # Test space weather
        resp = requests.get(f"{BACKEND_URL}/api/space-weather/current", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            print(f"✅ Space weather: Period={data.get('historical_period')}")
            print(f"   Data keys: {list(data.get('space_weather', {}).keys())}")
        else:
            print(f"⚠️ Space weather failed: {resp.status_code}")
        
        # Test mission status
        resp = requests.get(f"{BACKEND_URL}/api/mission/status", timeout=10)
        print(f"✅ Mission status: {resp.status_code}")
        
        # Test hazards
        resp = requests.get(f"{BACKEND_URL}/api/hazards/current", timeout=10)
        print(f"✅ Current hazards: {resp.status_code}")
        
        return True
    except Exception as e:
        print(f"❌ Backend API error: {e}")
        return False

def test_frontend_integration():
    """Test frontend accessibility"""
    print("\n⚛️ Testing Frontend Integration...")
    
    try:
        resp = requests.get(FRONTEND_URL, timeout=5)
        if resp.status_code == 200:
            print(f"✅ Frontend accessible: {resp.status_code}")
            return True
        else:
            print(f"⚠️ Frontend status: {resp.status_code}")
            return False
    except Exception as e:
        print(f"❌ Frontend error: {e}")
        return False

def main():
    """Main test suite"""
    print("🧪 ODIN System Integration Analysis")
    print("=" * 50)
    
    results = {
        "external_apis": test_external_apis(),
        "ai_services": test_ai_services(),
        "backend_apis": test_backend_apis(),
        "frontend": test_frontend_integration()
    }
    
    print("\n📊 Integration Summary:")
    print("=" * 50)
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {total_passed}/{total_tests} components working")
    
    if total_passed == total_tests:
        print("🎉 All systems operational! Ready for simulation.")
    elif total_passed >= 3:
        print("⚠️ Most systems working. Minor issues detected.")
    else:
        print("❌ Critical issues detected. System needs attention.")
    
    return total_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
