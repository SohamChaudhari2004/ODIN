from fastapi import WebSocket, WebSocketDisconnect
import json
import asyncio
import logging
from typing import Dict, List, Set
from datetime import datetime

logger = logging.getLogger(__name__)

class SimulationWebSocket:
    """WebSocket manager for ODIN real-time mission monitoring and updates"""
    
    def __init__(self):
        self.connections: Set[WebSocket] = set()
        self.odin_active = False
        self.mission_data = {}
        self._broadcast_task = None
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection for ODIN monitoring"""
        await websocket.accept()
        self.connections.add(websocket)
        logger.info(f"ODIN WebSocket connected. Total connections: {len(self.connections)}")
        
        # Send initial ODIN state
        await self._send_to_client(websocket, {
            "type": "odin_connection_established",
            "message": "🚀 Connected to ODIN Navigation System",
            "system": "ODIN (Optimal Dynamic Interplanetary Navigator)",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        self.connections.discard(websocket)
        logger.info(f"ODIN WebSocket disconnected. Total connections: {len(self.connections)}")
    
    async def broadcast_update(self, data: Dict):
        """Broadcast ODIN system updates to all connected clients"""
        if not self.connections:
            return
        
        # Add ODIN system metadata
        odin_message = {
            **data,
            "source": "ODIN Navigation System",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        message = json.dumps(odin_message)
        disconnected = set()
        
        for websocket in self.connections.copy():
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to send ODIN update to WebSocket: {e}")
                disconnected.add(websocket)
        
        # Clean up disconnected clients
        for websocket in disconnected:
            self.disconnect(websocket)
    
    async def _send_to_client(self, websocket: WebSocket, data: Dict):
        """Send data to a specific client"""
        try:
            await websocket.send_text(json.dumps(data))
        except Exception as e:
            logger.warning(f"Failed to send to specific client: {e}")
            self.disconnect(websocket)
    
    async def simulation_loop(self):
        """ODIN mission monitoring loop that broadcasts periodic updates"""
        logger.info("🚀 Starting ODIN mission monitoring WebSocket loop...")
        
        while True:
            try:
                if self.odin_active and self.connections:
                    # Get current ODIN system state
                    odin_state = await self._get_odin_state()
                    
                    # Broadcast to all clients
                    await self.broadcast_update({
                        "type": "odin_mission_update",
                        "data": odin_state,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
                # Update every 2 seconds for real-time monitoring
                await asyncio.sleep(2.0)
                
            except Exception as e:
                logger.error(f"Error in ODIN monitoring loop: {e}")
                await asyncio.sleep(5.0)  # Wait longer on error
    
    async def _get_odin_state(self) -> Dict:
        """Get current ODIN system state for broadcasting"""
        return {
            "system_name": "ODIN Navigation System",
            "status": "operational" if self.odin_active else "standby",
            "mission_time": self.mission_data.get("mission_time", 0.0),
            "autonomous_mode": self.odin_active,
            "active_hazards": self.mission_data.get("active_hazards", 0),
            "decisions_made": self.mission_data.get("decisions_made", 0),
            "last_decision": self.mission_data.get("last_decision", "No recent decisions"),
            "fuel_remaining": self.mission_data.get("fuel_remaining", 100.0),
            "navigation_status": "Active AI navigation" if self.odin_active else "Standby"
        }
    
    async def handle_client_message(self, websocket: WebSocket, message: str):
        """Handle incoming messages from ODIN clients"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "odin_control":
                await self._handle_odin_control(data.get("action"), data.get("parameters"))
            elif message_type == "mission_status_request":
                await self._handle_mission_status_request(websocket)
            elif message_type == "decision_logs_request":
                await self._handle_decision_logs_request(websocket)
            elif message_type == "ping":
                await self._send_to_client(websocket, {
                    "type": "pong", 
                    "source": "ODIN Navigation System",
                    "timestamp": datetime.utcnow().isoformat()
                })
            else:
                logger.warning(f"Unknown ODIN message type: {message_type}")
                
        except json.JSONDecodeError:
            logger.error("Invalid JSON received from ODIN client")
        except Exception as e:
            logger.error(f"Error handling ODIN client message: {e}")
    
    async def _handle_odin_control(self, action: str, parameters: Dict = None):
        """Handle ODIN system control commands"""
        if parameters is None:
            parameters = {}
            
        if action == "start_mission":
            self.odin_active = True
            await self.broadcast_update({
                "type": "odin_mission_started",
                "message": "🚀 ODIN autonomous mission started",
                "timestamp": datetime.utcnow().isoformat()
            })
            
        elif action == "pause_mission":
            self.odin_active = False
            await self.broadcast_update({
                "type": "odin_mission_paused", 
                "message": "⏸️ ODIN mission paused",
                "timestamp": datetime.utcnow().isoformat()
            })
            
        elif action == "abort_mission":
            self.odin_active = False
            self.mission_data = {}
            await self.broadcast_update({
                "type": "odin_mission_aborted",
                "message": "🛑 ODIN mission aborted",
                "timestamp": datetime.utcnow().isoformat()
            })
    
    async def _handle_mission_status_request(self, websocket: WebSocket):
        """Handle mission status request from specific client"""
        status = await self._get_odin_state()
        await self._send_to_client(websocket, {
            "type": "mission_status_response",
            "data": status,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def _handle_decision_logs_request(self, websocket: WebSocket):
        """Handle decision logs request from specific client"""
        logs = self.mission_data.get("decision_logs", [])
        await self._send_to_client(websocket, {
            "type": "decision_logs_response",
            "logs": logs,
            "total_count": len(logs),
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def broadcast_odin_decision(self, decision_data: Dict):
        """Broadcast ODIN decision to all clients"""
        await self.broadcast_update({
            "type": "odin_decision_made",
            "decision": decision_data,
            "message": f"📝 ODIN Decision: {decision_data.get('summary', 'Decision made')}",
            "priority": "high" if "HAZARD" in str(decision_data) else "normal",
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def broadcast_hazard_alert(self, hazard_data: Dict):
        """Broadcast hazard alert to all ODIN clients"""
        await self.broadcast_update({
            "type": "odin_hazard_alert",
            "hazard": hazard_data,
            "message": f"⚠️ ODIN Hazard Alert: {hazard_data.get('type', 'unknown')}",
            "priority": "critical" if hazard_data.get("severity", 0) > 0.7 else "warning",
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def broadcast_trajectory_update(self, trajectory_data: Dict):
        """Broadcast ODIN trajectory update to all clients"""
        await self.broadcast_update({
            "type": "odin_trajectory_updated",
            "trajectory": trajectory_data,
            "message": f"🛰️ ODIN Trajectory Update: {trajectory_data.get('name', 'New trajectory')}",
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def update_mission_data(self, data: Dict):
        """Update internal mission data for broadcasting"""
        self.mission_data.update(data)
    
    def get_connection_status(self) -> Dict:
        """Get WebSocket connection status"""
        return {
            "active_connections": len(self.connections),
            "simulation_running": self.simulation_running,
            "simulation_speed": self.simulation_speed,
            "last_update": datetime.utcnow().isoformat()
        }
