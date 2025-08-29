#!/usr/bin/env python3
"""
Startup script for Odin Flight Simulation Backend
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from app.config import settings

def setup_logging():
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(settings.log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def check_dependencies():
    """Check if required dependencies are available"""
    required_packages = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "numpy",
        "aiohttp"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True

def find_available_port(start_port=8000, max_attempts=10):
    """Find an available port starting from start_port"""
    import socket
    
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    
    raise RuntimeError(f"No available ports found in range {start_port}-{start_port + max_attempts}")

def main():
    """Main entry point"""
    print("🚀 Starting Odin Flight Simulation Backend...")
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    logger.info("All dependencies available")
    
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Import and run the application
    try:
        import uvicorn
        from main import app
        
        # Find available port
        try:
            available_port = find_available_port(settings.port)
            if available_port != settings.port:
                logger.warning(f"Port {settings.port} is busy, using port {available_port} instead")
        except RuntimeError as e:
            logger.error(f"Port finding error: {e}")
            available_port = settings.port  # Try original port anyway
        
        logger.info(f"Starting server on {settings.host}:{available_port}")
        
        uvicorn.run(
            "main:app",
            host=settings.host,
            port=available_port,
            reload=settings.reload,
            log_level=settings.log_level.lower()
        )
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
