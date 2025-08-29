#!/usr/bin/env python3
"""
ODIN System Startup Script
Launches backend and frontend servers
"""

import subprocess
import sys
import os
import time
import signal
import requests
from pathlib import Path

class OdinLauncher:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.base_dir = Path(__file__).parent
        self.backend_dir = self.base_dir / "backend"
        self.frontend_dir = self.base_dir / "frontend"
        
    def check_python_env(self):
        """Check if we're in the right Python environment"""
        try:
            import fastapi
            import uvicorn
            print("✅ Python environment ready")
            return True
        except ImportError:
            print("❌ Required Python packages not found")
            print("💡 Install requirements: pip install -r backend/requirements.txt")
            return False
    
    def check_node_env(self):
        """Check if Node.js environment is ready"""
        try:
            result = subprocess.run(["npm", "--version"], 
                                  capture_output=True, text=True, cwd=self.frontend_dir)
            if result.returncode == 0:
                print("✅ Node.js environment ready")
                return True
            else:
                print("❌ npm not available")
                return False
        except FileNotFoundError:
            print("❌ Node.js/npm not found")
            print("💡 Install Node.js and run: cd frontend && npm install")
            return False
    
    def start_backend(self):
        """Start the FastAPI backend"""
        print("🚀 Starting ODIN Backend...")
        try:
            self.backend_process = subprocess.Popen(
                [sys.executable, "main.py"],
                cwd=self.backend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for backend to start
            for i in range(30):  # 30 second timeout
                try:
                    response = requests.get("http://localhost:8000/api/health", timeout=1)
                    if response.status_code == 200:
                        print("✅ Backend started successfully")
                        return True
                except:
                    time.sleep(1)
                    
            print("❌ Backend failed to start within 30 seconds")
            return False
            
        except Exception as e:
            print(f"❌ Failed to start backend: {e}")
            return False
    
    def start_frontend(self):
        """Start the React frontend"""
        print("🎮 Starting ODIN Frontend...")
        try:
            self.frontend_process = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=self.frontend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for frontend to start
            for i in range(60):  # 60 second timeout for npm
                try:
                    response = requests.get("http://localhost:5173", timeout=1)
                    if response.status_code == 200:
                        print("✅ Frontend started successfully")
                        return True
                except:
                    time.sleep(1)
                    
            print("✅ Frontend is starting (this may take a moment)")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start frontend: {e}")
            return False
    
    def stop_services(self):
        """Stop both services"""
        print("\n🛑 Stopping ODIN services...")
        
        if self.frontend_process:
            self.frontend_process.terminate()
            try:
                self.frontend_process.wait(timeout=5)
                print("✅ Frontend stopped")
            except subprocess.TimeoutExpired:
                self.frontend_process.kill()
                print("⚠️ Frontend force stopped")
        
        if self.backend_process:
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=5)
                print("✅ Backend stopped")
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
                print("⚠️ Backend force stopped")
    
    def run(self):
        """Main run method"""
        print("🛸 ODIN Navigation System Launcher")
        print("=" * 50)
        
        # Check environments
        if not self.check_python_env():
            return False
            
        if not self.check_node_env():
            print("ℹ️ Frontend will not be started")
            
        # Start services
        try:
            # Start backend first
            if not self.start_backend():
                return False
            
            # Start frontend if available
            if self.frontend_dir.exists() and (self.frontend_dir / "package.json").exists():
                self.start_frontend()
            
            print("\n" + "=" * 50)
            print("🎉 ODIN SYSTEM READY!")
            print("=" * 50)
            print("🔧 Backend API: http://localhost:8000")
            print("🎮 Frontend UI: http://localhost:5173")
            print("📚 API Docs: http://localhost:8000/docs")
            print("🧪 Run tests: python test_integration.py")
            print("=" * 50)
            print("\nPress Ctrl+C to stop all services")
            
            # Keep running until interrupted
            while True:
                time.sleep(1)
                
                # Check if processes are still running
                if self.backend_process and self.backend_process.poll() is not None:
                    print("❌ Backend process stopped unexpectedly")
                    break
                    
        except KeyboardInterrupt:
            print("\n⏹️ Shutdown requested")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            self.stop_services()
            
        return True

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n⏹️ Shutdown signal received")
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    launcher = OdinLauncher()
    success = launcher.run()
    sys.exit(0 if success else 1)
