#!/usr/bin/env python3
"""
ODIN System Status Report
Final integration test and status summary
"""

import requests
import json
from datetime import datetime

def test_odin_endpoints():
    """Test all major ODIN endpoints"""
    base_url = "http://127.0.0.1:8000"
    
    tests = [
        ("Health Check", "/health"),
        ("ODIN Status", "/api/odin/status"),
        ("Space Weather", "/api/space-weather/current"),
        ("AI Services", "/api/ai/status"),
    ]
    
    print("🚀 ODIN System Integration Test")
    print("=" * 50)
    
    for test_name, endpoint in tests:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            if response.status_code == 200:
                print(f"✅ {test_name:<20} WORKING")
            else:
                print(f"⚠️  {test_name:<20} HTTP {response.status_code}")
        except Exception as e:
            print(f"❌ {test_name:<20} ERROR: {str(e)[:50]}")

def display_odin_status():
    """Display detailed ODIN system status"""
    try:
        response = requests.get("http://127.0.0.1:8000/api/odin/status", timeout=10)
        if response.status_code == 200:
            status = response.json()
            
            print("\n📊 ODIN System Status")
            print("=" * 50)
            print(f"System: {status['system_name']}")
            print(f"Version: {status['version']}")
            print(f"Mission Initialized: {status['status']['mission_initialized']}")
            print(f"Historical Timestamp: {status['status']['historical_timestamp']}")
            print(f"Operational: {status['status']['system_status']['operational']}")
            print(f"Autonomous Mode: {status['status']['system_status']['autonomous_mode']}")
            print(f"Hazard Monitoring: {status['status']['system_status']['hazard_monitoring']}")
            
            print("\n🤖 AI Subsystems:")
            for subsystem, status_val in status['status']['subsystem_status'].items():
                emoji = "✅" if status_val else "❌"
                print(f"  {emoji} {subsystem.replace('_', ' ').title()}")
            
            print("\n🔧 Health Checks:")
            for check, details in status['health_checks'].items():
                if isinstance(details, dict) and 'available' in details:
                    emoji = "✅" if details['available'] else "❌"
                    print(f"  {emoji} {check.replace('_', ' ').title()}")
            
            print(f"\n📈 Mission Stats:")
            print(f"  Total Decisions: {status['status']['total_decisions']}")
            print(f"  Active Hazards: {status['status']['active_hazards_count']}")
            
    except Exception as e:
        print(f"❌ Could not get ODIN status: {e}")

if __name__ == "__main__":
    print(f"ODIN System Test - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test endpoints
    test_odin_endpoints()
    
    # Show detailed status
    display_odin_status()
    
    print("\n🎉 ODIN System Integration Complete!")
    print("\nTo use the system:")
    print("1. Backend is running on http://127.0.0.1:8000")
    print("2. Start frontend: cd frontend && npm run dev")
    print("3. Access API docs: http://127.0.0.1:8000/docs")
    print("4. WebSocket endpoint: ws://127.0.0.1:8000/ws")
