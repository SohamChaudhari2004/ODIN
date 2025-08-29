import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class MissionSaveLoadSystem:
    """System for saving and loading simulation states for different scenarios"""
    
    def __init__(self, save_directory: str = "data/mission_saves"):
        self.save_directory = Path(save_directory)
        self.save_directory.mkdir(parents=True, exist_ok=True)
        
        # Metadata for saved missions
        self.metadata_file = self.save_directory / "mission_metadata.json"
        self.load_metadata()
    
    def save_mission_state(self, 
                          state_data: Dict[str, Any],
                          save_name: str,
                          description: str = "",
                          tags: List[str] = None) -> str:
        """
        Save current mission state to file
        
        Args:
            state_data: Complete mission state including trajectory, hazards, telemetry
            save_name: Name for the saved mission
            description: Optional description
            tags: Optional tags for categorization
            
        Returns:
            save_id: Unique identifier for the saved mission
        """
        try:
            # Generate unique save ID
            save_id = f"{save_name}_{int(datetime.utcnow().timestamp())}"
            
            # Prepare save data
            save_data = {
                "save_id": save_id,
                "save_name": save_name,
                "description": description,
                "tags": tags or [],
                "timestamp": datetime.utcnow().isoformat(),
                "mission_state": state_data,
                "version": "1.0"
            }
            
            # Save to file
            save_file = self.save_directory / f"{save_id}.json"
            with open(save_file, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
            
            # Update metadata
            self._update_metadata(save_id, save_name, description, tags)
            
            logger.info(f"Mission state saved: {save_id}")
            return save_id
            
        except Exception as e:
            logger.error(f"Error saving mission state: {e}")
            raise
    
    def load_mission_state(self, save_id: str) -> Dict[str, Any]:
        """
        Load mission state from file
        
        Args:
            save_id: Unique identifier of the saved mission
            
        Returns:
            Complete mission state data
        """
        try:
            save_file = self.save_directory / f"{save_id}.json"
            
            if not save_file.exists():
                raise FileNotFoundError(f"Save file not found: {save_id}")
            
            with open(save_file, 'r') as f:
                save_data = json.load(f)
            
            logger.info(f"Mission state loaded: {save_id}")
            return save_data["mission_state"]
            
        except Exception as e:
            logger.error(f"Error loading mission state: {e}")
            raise
    
    def list_saved_missions(self, tags: List[str] = None) -> List[Dict[str, Any]]:
        """
        List all saved missions, optionally filtered by tags
        
        Args:
            tags: Optional list of tags to filter by
            
        Returns:
            List of mission metadata
        """
        try:
            filtered_missions = []
            
            for mission in self.metadata["missions"]:
                if tags:
                    # Check if mission has any of the specified tags
                    if not any(tag in mission.get("tags", []) for tag in tags):
                        continue
                
                filtered_missions.append({
                    "save_id": mission["save_id"],
                    "save_name": mission["save_name"],
                    "description": mission["description"],
                    "tags": mission["tags"],
                    "timestamp": mission["timestamp"],
                    "file_size": self._get_file_size(mission["save_id"])
                })
            
            # Sort by timestamp (newest first)
            filtered_missions.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return filtered_missions
            
        except Exception as e:
            logger.error(f"Error listing saved missions: {e}")
            return []
    
    def delete_mission_save(self, save_id: str) -> bool:
        """
        Delete a saved mission
        
        Args:
            save_id: Unique identifier of the mission to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            save_file = self.save_directory / f"{save_id}.json"
            
            if save_file.exists():
                save_file.unlink()
                
                # Remove from metadata
                self.metadata["missions"] = [
                    m for m in self.metadata["missions"] 
                    if m["save_id"] != save_id
                ]
                self._save_metadata()
                
                logger.info(f"Mission save deleted: {save_id}")
                return True
            else:
                logger.warning(f"Save file not found for deletion: {save_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting mission save: {e}")
            return False
    
    def create_scenario_template(self, 
                               scenario_name: str,
                               initial_conditions: Dict[str, Any],
                               hazard_schedule: List[Dict[str, Any]] = None) -> str:
        """
        Create a reusable scenario template
        
        Args:
            scenario_name: Name of the scenario
            initial_conditions: Initial mission parameters
            hazard_schedule: Scheduled hazards for the scenario
            
        Returns:
            Template ID
        """
        try:
            template_data = {
                "template_id": f"scenario_{scenario_name}_{int(datetime.utcnow().timestamp())}",
                "scenario_name": scenario_name,
                "created": datetime.utcnow().isoformat(),
                "initial_conditions": initial_conditions,
                "hazard_schedule": hazard_schedule or [],
                "type": "scenario_template"
            }
            
            template_file = self.save_directory / f"template_{template_data['template_id']}.json"
            with open(template_file, 'w') as f:
                json.dump(template_data, f, indent=2, default=str)
            
            logger.info(f"Scenario template created: {template_data['template_id']}")
            return template_data['template_id']
            
        except Exception as e:
            logger.error(f"Error creating scenario template: {e}")
            raise
    
    def load_scenario_template(self, template_id: str) -> Dict[str, Any]:
        """Load a scenario template"""
        try:
            template_file = self.save_directory / f"template_{template_id}.json"
            
            if not template_file.exists():
                raise FileNotFoundError(f"Template not found: {template_id}")
            
            with open(template_file, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Error loading scenario template: {e}")
            raise
    
    def export_mission_data(self, save_id: str, export_format: str = "json") -> str:
        """
        Export mission data in specified format
        
        Args:
            save_id: Mission save to export
            export_format: Export format ('json', 'csv', etc.)
            
        Returns:
            Path to exported file
        """
        try:
            mission_data = self.load_mission_state(save_id)
            
            if export_format.lower() == "json":
                export_file = self.save_directory / f"export_{save_id}.json"
                with open(export_file, 'w') as f:
                    json.dump(mission_data, f, indent=2, default=str)
            
            elif export_format.lower() == "csv":
                # Export telemetry data as CSV
                import pandas as pd
                
                telemetry = mission_data.get("telemetry", {})
                df = pd.DataFrame([telemetry])
                
                export_file = self.save_directory / f"export_{save_id}.csv"
                df.to_csv(export_file, index=False)
            
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
            
            logger.info(f"Mission data exported: {export_file}")
            return str(export_file)
            
        except Exception as e:
            logger.error(f"Error exporting mission data: {e}")
            raise
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics for saved missions"""
        try:
            total_files = len(list(self.save_directory.glob("*.json")))
            total_size = sum(f.stat().st_size for f in self.save_directory.glob("*.json"))
            
            # Convert bytes to MB
            total_size_mb = total_size / (1024 * 1024)
            
            return {
                "total_saves": len(self.metadata["missions"]),
                "total_files": total_files,
                "total_size_mb": round(total_size_mb, 2),
                "save_directory": str(self.save_directory)
            }
            
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {}
    
    def load_metadata(self):
        """Load mission metadata from file"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {"missions": [], "version": "1.0"}
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            self.metadata = {"missions": [], "version": "1.0"}
    
    def _save_metadata(self):
        """Save metadata to file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def _update_metadata(self, save_id: str, save_name: str, description: str, tags: List[str]):
        """Update metadata with new mission save"""
        mission_meta = {
            "save_id": save_id,
            "save_name": save_name,
            "description": description,
            "tags": tags,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.metadata["missions"].append(mission_meta)
        self._save_metadata()
    
    def _get_file_size(self, save_id: str) -> int:
        """Get file size in bytes"""
        try:
            save_file = self.save_directory / f"{save_id}.json"
            return save_file.stat().st_size if save_file.exists() else 0
        except:
            return 0
