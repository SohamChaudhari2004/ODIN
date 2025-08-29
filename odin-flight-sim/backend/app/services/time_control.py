import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class TimeControlSystem:
    """System for controlling simulation time scale and pausing/resuming"""
    
    def __init__(self):
        self.time_scale = 1.0  # 1x real time by default
        self.is_paused = False
        self.start_time = None
        self.pause_start_time = None
        self.total_pause_duration = 0.0
        self.simulation_start_real_time = None
        self.listeners = []
        
        # Time scale presets
        self.time_scale_presets = {
            "real_time": 1.0,
            "2x": 2.0,
            "5x": 5.0,
            "10x": 10.0,
            "30x": 30.0,
            "60x": 60.0,
            "300x": 300.0,  # 5 minutes = 1 second
            "3600x": 3600.0  # 1 hour = 1 second
        }
    
    def start_simulation(self):
        """Start the simulation timer"""
        if self.simulation_start_real_time is None:
            self.simulation_start_real_time = datetime.utcnow()
            self.start_time = datetime.utcnow()
            logger.info("Simulation time control started")
        else:
            logger.warning("Simulation already started")
    
    def pause_simulation(self):
        """Pause the simulation"""
        if not self.is_paused:
            self.is_paused = True
            self.pause_start_time = datetime.utcnow()
            logger.info("Simulation paused")
            self._notify_listeners("paused")
        else:
            logger.warning("Simulation already paused")
    
    def resume_simulation(self):
        """Resume the simulation"""
        if self.is_paused:
            self.is_paused = False
            if self.pause_start_time:
                pause_duration = (datetime.utcnow() - self.pause_start_time).total_seconds()
                self.total_pause_duration += pause_duration
                self.pause_start_time = None
            logger.info("Simulation resumed")
            self._notify_listeners("resumed")
        else:
            logger.warning("Simulation not paused")
    
    def reset_simulation(self):
        """Reset simulation time"""
        self.is_paused = False
        self.start_time = None
        self.pause_start_time = None
        self.total_pause_duration = 0.0
        self.simulation_start_real_time = None
        logger.info("Simulation time reset")
        self._notify_listeners("reset")
    
    def set_time_scale(self, scale: float):
        """Set simulation time scale"""
        if scale <= 0:
            raise ValueError("Time scale must be positive")
        
        old_scale = self.time_scale
        self.time_scale = scale
        
        logger.info(f"Time scale changed from {old_scale}x to {scale}x")
        self._notify_listeners("time_scale_changed", {"old_scale": old_scale, "new_scale": scale})
    
    def set_time_scale_preset(self, preset_name: str):
        """Set time scale using preset name"""
        if preset_name not in self.time_scale_presets:
            available = ", ".join(self.time_scale_presets.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
        
        self.set_time_scale(self.time_scale_presets[preset_name])
    
    def get_current_simulation_time(self) -> float:
        """
        Get current simulation time in hours
        
        Returns:
            Simulation time accounting for pauses and time scale
        """
        if self.start_time is None:
            return 0.0
        
        current_time = datetime.utcnow()
        
        if self.is_paused:
            # Calculate time up to pause point
            if self.pause_start_time:
                elapsed_real_time = (self.pause_start_time - self.start_time).total_seconds()
            else:
                elapsed_real_time = 0.0
        else:
            # Calculate total elapsed time minus pause time
            elapsed_real_time = (current_time - self.start_time).total_seconds() - self.total_pause_duration
        
        # Convert to simulation time using time scale
        simulation_time_seconds = elapsed_real_time * self.time_scale
        return simulation_time_seconds / 3600.0  # Convert to hours
    
    def get_real_time_elapsed(self) -> float:
        """Get real time elapsed since simulation start (in seconds)"""
        if self.simulation_start_real_time is None:
            return 0.0
        
        return (datetime.utcnow() - self.simulation_start_real_time).total_seconds()
    
    def get_time_status(self) -> Dict[str, Any]:
        """Get comprehensive time status"""
        return {
            "simulation_time_hours": self.get_current_simulation_time(),
            "real_time_elapsed_seconds": self.get_real_time_elapsed(),
            "time_scale": self.time_scale,
            "is_paused": self.is_paused,
            "total_pause_duration": self.total_pause_duration,
            "is_running": self.start_time is not None,
            "time_scale_presets": self.time_scale_presets
        }
    
    def calculate_estimated_completion(self, total_mission_duration_hours: float) -> Dict[str, Any]:
        """Calculate estimated completion time"""
        if self.start_time is None:
            return {"error": "Simulation not started"}
        
        current_sim_time = self.get_current_simulation_time()
        remaining_sim_time = max(0, total_mission_duration_hours - current_sim_time)
        
        if self.is_paused:
            return {
                "status": "paused",
                "progress_percent": (current_sim_time / total_mission_duration_hours) * 100,
                "remaining_simulation_hours": remaining_sim_time
            }
        
        # Calculate real time needed to complete
        remaining_real_seconds = remaining_sim_time * 3600 / self.time_scale
        estimated_completion = datetime.utcnow().timestamp() + remaining_real_seconds
        
        return {
            "status": "running",
            "progress_percent": (current_sim_time / total_mission_duration_hours) * 100,
            "remaining_simulation_hours": remaining_sim_time,
            "remaining_real_seconds": remaining_real_seconds,
            "estimated_completion_timestamp": estimated_completion,
            "estimated_completion_iso": datetime.fromtimestamp(estimated_completion).isoformat()
        }
    
    def add_time_listener(self, callback):
        """Add listener for time control events"""
        self.listeners.append(callback)
    
    def remove_time_listener(self, callback):
        """Remove time control event listener"""
        if callback in self.listeners:
            self.listeners.remove(callback)
    
    def _notify_listeners(self, event_type: str, data: Dict = None):
        """Notify all listeners of time control events"""
        event_data = {
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "time_status": self.get_time_status()
        }
        
        if data:
            event_data.update(data)
        
        for listener in self.listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    asyncio.create_task(listener(event_data))
                else:
                    listener(event_data)
            except Exception as e:
                logger.error(f"Error notifying time listener: {e}")
    
    def format_simulation_time(self, hours: Optional[float] = None) -> str:
        """Format simulation time as human-readable string"""
        if hours is None:
            hours = self.get_current_simulation_time()
        
        total_seconds = int(hours * 3600)
        hours_part = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        return f"T+{hours_part:02d}:{minutes:02d}:{seconds:02d}"
    
    def get_time_scale_recommendations(self, mission_duration_hours: float) -> List[Dict[str, Any]]:
        """Get recommended time scales based on mission duration"""
        recommendations = []
        
        # Real time for short missions or critical phases
        if mission_duration_hours <= 2:
            recommendations.append({
                "preset": "real_time",
                "scale": 1.0,
                "description": "Real-time for detailed monitoring",
                "estimated_duration_minutes": mission_duration_hours * 60
            })
        
        # 5x for moderate missions
        if mission_duration_hours > 1:
            recommendations.append({
                "preset": "5x",
                "scale": 5.0,
                "description": "5x speed for balanced monitoring",
                "estimated_duration_minutes": mission_duration_hours * 60 / 5
            })
        
        # 60x for long missions
        if mission_duration_hours > 12:
            recommendations.append({
                "preset": "60x", 
                "scale": 60.0,
                "description": "1 minute = 1 hour simulation",
                "estimated_duration_minutes": mission_duration_hours * 60 / 60
            })
        
        # 3600x for very long missions
        if mission_duration_hours > 72:
            recommendations.append({
                "preset": "3600x",
                "scale": 3600.0,
                "description": "1 second = 1 hour simulation",
                "estimated_duration_minutes": mission_duration_hours * 60 / 3600
            })
        
        return recommendations
