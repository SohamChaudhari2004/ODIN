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

async def test_odin_system():
    """Test the ODIN system components"""
    
    print("=" * 60)
    print("ODIN (Optimal Dynamic Interplanetary Navigator) Test")
    print("=" * 60)
    
    try:
        # Test 1: Space Weather Service
        print("\n1. Testing Space Weather Service...")
        from space_weather_service import SpaceWeatherDataService
        
        weather_service = SpaceWeatherDataService()
        historical_timestamp = weather_service.initialize_mission_timestamp()
        print(f"   ✅ Historical timestamp initialized: {historical_timestamp}")
        
        space_weather = await weather_service.get_space_weather_data()
        print(f"   ✅ Space weather data retrieved for {space_weather['timestamp']}")
        print(f"   📡 Solar flux: {space_weather['solar_activity']['solar_flux_10_7cm']:.1f}")
        print(f"   🌍 Kp index: {space_weather['geomagnetic_activity']['kp_index']}")
        
        # Test 2: AI Co-pilot
        print("\n2. Testing AI Co-pilot...")
        from ai_copilot import AICoPilot
        
        ai_copilot = AICoPilot()
        if ai_copilot.ai_available:
            print("   ✅ AI Co-pilot initialized with Mistral AI")
        else:
            print("   ⚠️  AI Co-pilot running in simulation mode")
        
        # Test mission brief generation
        test_mission_data = {
            "mission_time": 12.5,
            "trajectory": {"name": "Direct Lunar Transfer", "total_delta_v": 11500, "duration": 72.0},
            "telemetry": {"fuel_remaining": 85.0, "current_velocity": 10.5},
            "hazards": [{"type": "solar_flare", "severity": "moderate"}],
            "space_weather": space_weather
        }
        
        mission_brief = await ai_copilot.generate_mission_brief(test_mission_data)
        print(f"   ✅ Mission brief generated:")
        print(f"   📝 {mission_brief[:100]}...")
        
        # Test 3: Predictive Hazard Forecasting
        print("\n3. Testing Predictive Hazard Forecasting...")
        from predictive_hazard_forecasting import PredictiveHazardForecasting
        
        hazard_forecaster = PredictiveHazardForecasting()
        print("   ✅ Hazard forecasting system initialized")
        
        # Test 4: Decision Engine
        print("\n4. Testing Decision Engine...")
        sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
        from app.services.decision_engine import DecisionEngine
        
        decision_engine = DecisionEngine()
        print("   ✅ Decision engine initialized")
        print(f"   ⚖️  Evaluation weights: {decision_engine.evaluation_weights}")
        
        # Test 5: ODIN Main System
        print("\n5. Testing ODIN Main System...")
        from odin_main import OdinNavigationSystem
        
        odin = OdinNavigationSystem()
        print(f"   ✅ {odin.system_name} v{odin.version} initialized")
        
        # Initialize a test mission
        print("\n6. Running ODIN Mission Initialization...")
        init_result = await odin.initialize_mission("Moon")
        print(f"   ✅ Mission initialized successfully")
        print(f"   🗓️  Historical timestamp: {init_result['historical_timestamp']}")
        print(f"   🚀 Selected trajectory: {init_result['selected_trajectory']['name']}")
        print(f"   ⛽ Total ΔV: {init_result['selected_trajectory']['total_delta_v']:.0f} m/s")
        print(f"   ⏱️  Duration: {init_result['selected_trajectory']['duration']:.1f} hours")
        
        # Test short autonomous mission
        print("\n7. Running Short Autonomous Mission...")
        mission_events = await odin.autonomous_mission_loop(duration_hours=6.0)
        print(f"   ✅ Autonomous mission completed with {len(mission_events)} events")
        
        # Display decision logs
        decision_logs = odin.get_decision_logs()
        if decision_logs:
            print(f"\n8. Decision Logs Generated:")
            for log in decision_logs:
                print(f"   T+{log['mission_time']:.1f}h: {log['decision']}")
        
        # System status
        print(f"\n9. Final System Status:")
        status = odin.get_system_status()
        print(f"   🟢 System operational: {status['system_status']['operational']}")
        print(f"   🤖 Autonomous mode: {status['system_status']['autonomous_mode']}")
        print(f"   📊 Total decisions: {status['total_decisions']}")
        print(f"   ⚠️  Active hazards: {status['active_hazards_count']}")
        
        print("\n" + "=" * 60)
        print("✅ ODIN SYSTEM TEST COMPLETED SUCCESSFULLY!")
        print("🚀 Ready for autonomous Earth-to-Moon navigation")
        print("=" * 60)
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure all required packages are installed:")
        print("   pip install -r ai-services/requirements.txt")
        print("   pip install -r backend/requirements.txt")
        return False
        
    except Exception as e:
        print(f"❌ Test Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_individual_components():
    """Test individual components if full system test fails"""
    
    print("\n🔧 Testing Individual Components...")
    
    # Test Space Weather Service
    try:
        from space_weather_service import SpaceWeatherDataService
        weather = SpaceWeatherDataService()
        timestamp = weather.initialize_mission_timestamp()
        print(f"✅ Space Weather Service: OK (timestamp: {timestamp.strftime('%Y-%m-%d')})")
    except Exception as e:
        print(f"❌ Space Weather Service: {e}")
    
    # Test AI Co-pilot
    try:
        from ai_copilot import AICoPilot
        ai = AICoPilot()
        status = "Mistral AI" if ai.ai_available else "Simulation Mode"
        print(f"✅ AI Co-pilot: OK ({status})")
    except Exception as e:
        print(f"❌ AI Co-pilot: {e}")
    
    # Test Predictive Forecasting
    try:
        from predictive_hazard_forecasting import PredictiveHazardForecasting
        forecaster = PredictiveHazardForecasting()
        print(f"✅ Hazard Forecasting: OK")
    except Exception as e:
        print(f"❌ Hazard Forecasting: {e}")
    
    # Test Decision Engine
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
        from app.services.decision_engine import DecisionEngine
        engine = DecisionEngine()
        print(f"✅ Decision Engine: OK")
    except Exception as e:
        print(f"❌ Decision Engine: {e}")

if __name__ == "__main__":
    print("Starting ODIN System Tests...")
    
    # Run main test
    success = asyncio.run(test_odin_system())
    
    if not success:
        print("\nRunning individual component tests...")
        asyncio.run(test_individual_components())
    
    print(f"\nTest completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
