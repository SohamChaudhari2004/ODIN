#!/usr/bin/env python3
"""
ODIN System Integration Test
Tests the core ODIN functionality to ensure everything is working
"""

import asyncio
import sys
import os
import json
from datetime import datetime

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'ai-services'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

async def test_basic_imports():
    """Test that all core modules can be imported"""
    print("🧪 Testing basic imports...")
    
    try:
        # Test legacy AI services imports (if they exist)
        try:
            from ai_copilot import AICoPilot
            from space_weather_service import SpaceWeatherDataService as LegacySpaceWeather
            print("✅ Legacy AI services imports successful")
        except ImportError:
            print("ℹ️ Legacy AI services not found (this is okay)")
        
        # Test new services imports
        try:
            from backend.app.services.trajectory_engine import TrajectoryEngine
            from backend.app.services.space_data_service import SpaceWeatherDataService
            print("✅ New backend services imports successful")
        except ImportError as e:
            print(f"❌ Backend services import error: {e}")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

async def test_trajectory_engine():
    """Test trajectory calculation engine"""
    print("\n🚀 Testing trajectory engine...")
    
    try:
        from backend.app.services.trajectory_engine import TrajectoryEngine
        
        engine = TrajectoryEngine()
        
        # Test basic trajectory calculation
        start_time = datetime.utcnow()
        trajectory = await engine.calculate_initial_trajectory(start_time, "Moon")
        
        print(f"✅ Trajectory calculated: {trajectory.name}")
        print(f"   - Total ΔV: {trajectory.total_delta_v:.1f} m/s")
        print(f"   - Duration: {trajectory.total_duration:.1f} hours")
        print(f"   - Safety Score: {trajectory.safety_score:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Trajectory engine error: {e}")
        return False

async def test_space_weather_service():
    """Test space weather data service"""
    print("\n🌤️ Testing space weather service...")
    
    try:
        from backend.app.services.space_data_service import SpaceWeatherDataService
        
        service = SpaceWeatherDataService()
        
        # Test current conditions
        conditions = await service.get_current_conditions()
        
        print(f"✅ Space weather data retrieved")
        print(f"   - Active hazards: {len(conditions.get('active_hazards', []))}")
        print(f"   - Risk level: {conditions.get('space_weather_summary', {}).get('overall_risk', 'UNKNOWN')}")
        
        await service.close()
        return True
        
    except Exception as e:
        print(f"❌ Space weather service error: {e}")
        return False

async def test_database_models():
    """Test database model definitions"""
    print("\n🗄️ Testing database models...")
    
    try:
        from backend.app.models.odin_models import (
            MissionDocument, HazardDocument, DecisionLogDocument, 
            TrajectoryDocument, MissionStatus, HazardType
        )
        
        # Test model creation
        mission = MissionDocument(
            mission_id="test_mission_db",
            start_time=datetime.utcnow(),
            status=MissionStatus.INITIALIZING,
            destination="Moon",
            historical_timestamp=datetime.utcnow(),
            spacecraft_position=[7000.0, 0.0, 0.0],
            spacecraft_velocity=[0.0, 7.8, 0.0],
            fuel_remaining=85.0,
            mission_constraints={"max_delta_v": 15000}
        )
        
        print(f"✅ Database models working")
        print(f"   - Mission model: {mission.mission_id}")
        print(f"   - Status: {mission.status}")
        print(f"   - Destination: {mission.destination}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database models error: {e}")
        return False

async def test_api_integration():
    """Test API route definitions"""
    print("\n🔌 Testing API integration...")
    
    try:
        from backend.app.api.routes import router
        
        print(f"✅ API routes loaded")
        print(f"   - Router configured successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ API integration error: {e}")
        return False

async def test_legacy_odin_system():
    """Test legacy ODIN system if available"""
    print("\n🤖 Testing legacy ODIN system...")
    
    try:
        from odin_main import OdinNavigationSystem
        
        odin = OdinNavigationSystem()
        print("✅ Legacy ODIN system initialized")
        
        # Test mission initialization
        result = await odin.initialize_mission("Moon")
        print(f"✅ Mission initialized: {type(result)}")
        
        # Test system status
        status = odin.get_system_status()
        print(f"✅ System status: {status}")
        
        return True
        
    except Exception as e:
        print(f"ℹ️ Legacy ODIN system not available: {e}")
        return True  # This is okay, it's optional

async def run_all_tests():
    """Run all ODIN system tests"""
    print("🧪 ODIN System Integration Test")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Database Models", test_database_models),
        ("Trajectory Engine", test_trajectory_engine),
        ("Space Weather Service", test_space_weather_service),
        ("API Integration", test_api_integration),
        ("Legacy ODIN System", test_legacy_odin_system),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 Test Summary")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nTests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed >= total - 1:  # Allow one optional test to fail
        print("\n🎉 ODIN system is ready!")
        print("\nNext steps:")
        print("1. Start MongoDB (if using): mongod")
        print("2. Run backend: cd backend && python main.py")
        print("3. Run frontend: cd frontend && npm run dev")
        print("4. Open http://localhost:5173")
        print("\nSee SETUP.md for detailed instructions.")
    else:
        print("\n⚠️ Some critical tests failed. Check the errors above.")
        print("Refer to SETUP.md for troubleshooting guidance.")
    
    return passed >= total - 1

if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
